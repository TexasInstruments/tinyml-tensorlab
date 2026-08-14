"""
architect.py — NAS architecture parameter optimizer.

Manages bilevel optimization of architecture parameters (alphas) for
differentiable NAS (CNN mode).  Supports both standard and unrolled
second-order gradient estimation, plus differentiable resource-aware
penalties (Memory / Compute) that bias the search toward smaller or
cheaper architectures.
"""
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


def _concat(xs):
    """
    Concatenate a list of tensors into a single 1D tensor.
    Args:
        xs (list of torch.Tensor): List of tensors to concatenate.
    Returns:
        torch.Tensor: Flattened concatenated tensor.
    """
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):
    """
    Manages the optimization of architecture parameters for CNN-based
    neural architecture search.
    """

    def __init__(self, model, args):
        """
        Initialize the Architect.
        Args:
            model: The search-phase network (must implement arch_parameters,
                   _loss, new, and expose .cells with MixedOps).
            args: Namespace containing hyperparameters.
        """
        self.model = model
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        # Adam optimizer for architecture parameters
        self.optimizer = torch.optim.Adam(
            self.model.arch_parameters(),
            lr=args.arch_learning_rate,
            betas=(0.5, 0.999),
            weight_decay=args.arch_weight_decay
        )
        # Pre-compute per-operation parameter counts for resource penalty
        self._op_param_counts = self._compute_op_param_counts()

    # -----------------------------------------------------------------
    # Differentiable resource penalty
    # -----------------------------------------------------------------

    def _compute_op_param_counts(self):
        """
        Pre-compute the parameter count for each primitive operation in each
        MixedOp of the search model.  Returns a dict of tensors keyed by cell
        type ('normal' / 'reduce').

        Each tensor has shape (num_edges, num_ops) containing the parameter
        count for that (edge, op) pair.
        """
        counts = {}  # keyed by 'normal' / 'reduce'
        for cell in self.model.cells:
            cell_type = 'reduce' if cell.reduction else 'normal'
            if cell_type in counts:
                continue
            edge_counts = []
            for mixed_op in cell._ops:
                op_counts = []
                for op in mixed_op._ops:
                    n = sum(p.numel() for p in op.parameters())
                    op_counts.append(float(n))
                edge_counts.append(op_counts)
            device = self.model.arch_parameters()[0].device
            counts[cell_type] = torch.tensor(edge_counts, device=device)
        return counts

    def _differentiable_resource_penalty(self, model):
        """
        Compute a differentiable expected-parameter-count penalty.

        For each edge in each cell, the expected param count is
            E[params] = sum_i softmax(alpha_i) * params_i
        This is differentiable w.r.t. the architecture parameters (alphas)
        because the softmax weights create a smooth weighting.

        The total is normalised to [0, 1] range by dividing by the maximum
        possible parameter count (if every edge picked the heaviest op).

        Returns:
            torch.Tensor: Scalar penalty in [0, 1], differentiable w.r.t. alphas.
        """
        weights_normal = F.softmax(model.alphas_normal, dim=-1)
        weights_reduce = F.softmax(model.alphas_reduce, dim=-1)

        counts_normal = self._op_param_counts.get('normal')
        counts_reduce = self._op_param_counts.get('reduce')

        expected = torch.tensor(0.0, device=weights_normal.device)
        max_possible = 0.0

        if counts_normal is not None:
            expected = expected + (weights_normal * counts_normal).sum()
            max_possible += counts_normal.max(dim=-1).values.sum().item()

        if counts_reduce is not None:
            expected = expected + (weights_reduce * counts_reduce).sum()
            max_possible += counts_reduce.max(dim=-1).values.sum().item()

        if max_possible > 0:
            return expected / max_possible
        return expected

    # -----------------------------------------------------------------
    # Unrolled model construction
    # -----------------------------------------------------------------

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        """
        Compute the unrolled model after a single SGD step on training data.
        Includes momentum and weight decay.
        """
        loss = self.model._loss(input, target)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(
                network_optimizer.state[v]['momentum_buffer']
                for v in self.model.parameters()
            ).mul_(self.network_momentum)
        except Exception:
            moment = torch.zeros_like(theta)
        dtheta = (
            _concat(torch.autograd.grad(loss, self.model.parameters())).data
            + self.network_weight_decay * theta
        )
        unrolled_model = self._construct_model_from_theta(
            theta - eta * (moment + dtheta)
        )
        return unrolled_model

    # -----------------------------------------------------------------
    # Architecture parameter update
    # -----------------------------------------------------------------

    def step(self, input_train, target_train, input_valid, target_valid,
             optimize, *, eta, network_optimizer, unrolled):
        """
        Perform a single optimization step for architecture parameters.

        Args:
            input_train, target_train: Training batch.
            input_valid, target_valid: Validation batch.
            optimize: Resource penalty mode ('Memory', 'Compute', or None).
            eta: Learning rate (overridden by network_optimizer lr).
            network_optimizer: Optimizer for network weights.
            unrolled: Whether to use unrolled (second-order) optimization.
        """
        self.optimizer.zero_grad()
        eta = network_optimizer.param_groups[0]['lr']

        if unrolled:
            self._backward_step_unrolled(
                input_train, target_train, input_valid, target_valid,
                eta, network_optimizer, optimize,
            )
        else:
            self._backward_step(input_valid, target_valid, optimize)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid, optimize):
        """
        Standard backward step: compute validation loss (+ optional resource
        penalty) and backpropagate to architecture parameters.
        """
        loss = self.model._loss(input_valid, target_valid)

        if optimize in ('Memory', 'Compute'):
            resource_weight = 0.1
            loss = loss + resource_weight * self._differentiable_resource_penalty(self.model)

        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train,
                                input_valid, target_valid, eta,
                                network_optimizer, optimize):
        """
        Unrolled backward step: compute the unrolled model, evaluate on
        validation data (+ optional resource penalty), and compute implicit
        gradients via Hessian-vector product.
        """
        unrolled_model = self._compute_unrolled_model(
            input_train, target_train, eta, network_optimizer,
        )
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)

        if optimize in ('Memory', 'Compute'):
            resource_weight = 0.1
            unrolled_loss = unrolled_loss + resource_weight * self._differentiable_resource_penalty(unrolled_model)

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(
            vector, input_train, target_train,
        )

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(ig.data, alpha=eta)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _construct_model_from_theta(self, theta):
        """
        Construct a new model instance with parameters set from a flattened
        tensor *theta*.
        """
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new  # Already on correct device via model.new()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        """
        Finite-difference approximation of the Hessian-vector product for
        implicit gradient calculation.
        """
        R = r / _concat(vector).norm()
        # Positive perturbation
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(v, alpha=R)
        loss = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        # Negative perturbation (2*R from positive)
        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(v, alpha=2 * R)
        loss = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        # Restore original parameters
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(v, alpha=R)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
