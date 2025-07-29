"""
architect.py
This module implements the `Architect` class, which is responsible for managing the architecture optimization process
in Neural Architecture Search (NAS) for both CNN and RNN models. The class supports unrolled optimization and 
incorporates resource-aware penalties (such as memory and compute) into the architecture search process.
Dependencies:
    - torch: PyTorch deep learning framework.
    - numpy: For numerical operations.
    - torch.autograd.Variable: For autograd variable handling.
    - torchinfo.summary: For model summary and MACs computation.
Functions:
    - _concat(xs): Concatenates a list of tensors into a single 1D tensor.
    - _clip(grads, max_norm): Clips gradients to a maximum norm for stability.
Classes:
    Architect:
        Handles architecture parameter optimization for both CNN and RNN models, supporting unrolled optimization,
        resource-aware penalties, and Hessian-vector product computation for implicit gradients.
    Methods:
        __init__(self, model, args):
            Initializes the Architect with the given model and arguments.
            Sets up the optimizer and relevant hyperparameters based on the mode ('cnn' or 'rnn').
        _compute_unrolled_model(self, *args):
            Computes the unrolled model parameters after a single optimization step on the training data.
            For CNNs, includes momentum and weight decay in the update.
            For RNNs, applies gradient clipping and weight decay.
        step(self, *args, eta, network_optimizer, unrolled):
            Performs a single optimization step for the architecture parameters.
            Supports both unrolled and standard backward steps.
            Applies resource-aware penalties if specified.
        _backward_step(self, *args):
            Computes the backward step for architecture parameters using validation data.
            Optionally adds memory or compute penalties to the loss.
        _backward_step_unrolled(self, *args):
            Computes the backward step for architecture parameters using the unrolled model.
            Handles implicit gradient computation via Hessian-vector products.
            Optionally adds memory or compute penalties to the unrolled loss.
        _construct_model_from_theta(self, theta):
            Constructs a new model instance with parameters set to the given flattened tensor `theta`.
            Ensures the new model has the same architecture as the original.
        _hessian_vector_product(self, vector, input, target, r=1e-2):
            Computes the Hessian-vector product for implicit gradient calculation using finite differences.
            Perturbs model parameters in the direction of the vector and computes the difference in gradients.
Details:
- _concat(xs):
    Concatenates a list of tensors into a single 1D tensor. Used to flatten model parameters or gradients for
    vectorized operations.
- _clip(grads, max_norm):
    Computes the total norm of a list of gradients and scales them down if the norm exceeds `max_norm`.
    Returns the scaling coefficient applied. Used for gradient clipping in RNNs to prevent exploding gradients.
- Architect.__init__:
    Initializes the Architect object. Sets up the optimizer for architecture parameters (`arch_parameters`)
    and stores relevant hyperparameters (momentum, weight decay, clipping) based on the model type (CNN or RNN).
- Architect._compute_unrolled_model:
    For CNNs:
        - Computes the loss on training data.
        - Calculates the parameter update using momentum and weight decay.
        - Constructs a new model with updated parameters (unrolled model).
    For RNNs:
        - Computes the loss and next hidden state on training data.
        - Applies gradient clipping and weight decay.
        - Constructs a new model with updated parameters (unrolled model).
        - Returns the unrolled model and the clipping coefficient.
- Architect.step:
    Orchestrates a single architecture optimization step.
    For CNNs:
        - Accepts training and validation data, and an optimization mode ('Memory' or 'Compute').
        - Performs either a standard or unrolled backward step.
        - Applies the optimizer step to update architecture parameters.
    For RNNs:
        - Accepts hidden states and input/target data for both training and validation.
        - Performs either a standard or unrolled backward step.
        - Applies the optimizer step and returns the next hidden state.
- Architect._backward_step:
    For CNNs:
        - Computes the validation loss.
        - If optimizing for 'Memory', adds a penalty proportional to the number of model parameters.
        - If optimizing for 'Compute', adds a penalty proportional to the number of multiply-accumulate operations (MACs).
        - Backpropagates the loss.
    For RNNs:
        - Computes the validation loss and next hidden state.
        - Backpropagates the loss and returns the next hidden state.
- Architect._backward_step_unrolled:
    For CNNs:
        - Computes the unrolled model after a training step.
        - Computes the validation loss on the unrolled model.
        - Adds resource-aware penalties if specified.
        - Computes gradients with respect to architecture parameters.
        - Computes implicit gradients using the Hessian-vector product.
        - Updates the gradients of the original model's architecture parameters.
    For RNNs:
        - Similar to CNNs, but also handles hidden states and gradient clipping.
        - Returns the next hidden state.
- Architect._construct_model_from_theta:
    Constructs a new model instance with parameters set from a flattened tensor `theta`.
    Ensures the parameter shapes match the original model.
    Loads the new parameters into the model and moves it to CUDA.
- Architect._hessian_vector_product:
    Computes the Hessian-vector product for implicit gradient calculation.
    Perturbs the model parameters in the direction of the input vector by a small amount `r`.
    Computes the difference in gradients with respect to architecture parameters for positive and negative perturbations.
    Returns the finite-difference approximation of the Hessian-vector product.
Usage:
    - Instantiate the Architect with a model and argument namespace.
    - Call `step` during the architecture search process to update architecture parameters.
    - Supports both standard and unrolled optimization, as well as resource-aware search.
Note:
    - The code assumes the model implements `arch_parameters()`, `_loss()`, and `new()` methods.
    - Resource-aware penalties require torchinfo for MACs computation.
"""
import torch
import numpy as np
from torch.autograd import Variable
from torchinfo import summary


def _concat(xs):
    """
    Concatenate a list of tensors into a single 1D tensor.
    Args:
        xs (list of torch.Tensor): List of tensors to concatenate.
    Returns:
        torch.Tensor: Flattened concatenated tensor.
    """
    return torch.cat([x.view(-1) for x in xs])

def _clip(grads, max_norm):
    """
    Clip gradients to a maximum norm for stability.
    Args:
        grads (list of torch.Tensor): Gradients to be clipped.
        max_norm (float): Maximum allowed norm.
    Returns:
        float: The scaling coefficient applied to gradients.
    """
    total_norm = 0  # Initialize total norm
    for g in grads:
        param_norm = g.data.norm(2)  # Compute L2 norm of each gradient tensor
        total_norm += param_norm ** 2  # Accumulate squared norms
    total_norm = total_norm ** 0.5  # Take square root to get total norm
    clip_coef = max_norm / (total_norm + 1e-6)  # Compute scaling coefficient
    if clip_coef < 1:
        for g in grads:
            g.data.mul_(clip_coef)  # Scale gradients if norm exceeds max_norm
    return clip_coef  # Return the coefficient used for clipping

class Architect(object):
    """
    The Architect class manages the optimization of architecture parameters
    for neural architecture search (NAS), supporting both CNN and RNN models.
    """

    def __init__(self, model, args):
        """
        Initialize the Architect.
        Args:
            model: The neural network model (must implement arch_parameters, _loss, new).
            args: Namespace containing hyperparameters and mode.
        """
        self.model = model  # Store the model
        self.mode = args.mode  # Mode: 'cnn' or 'rnn'

        if self.mode == 'cnn':
            # For CNNs, set momentum and weight decay for the network optimizer
            self.network_momentum = args.momentum
            self.network_weight_decay = args.weight_decay
            # Adam optimizer for architecture parameters
            self.optimizer = torch.optim.Adam(
                self.model.arch_parameters(),
                lr=args.arch_learning_rate,
                betas=(0.5, 0.999),
                weight_decay=args.arch_weight_decay
            )
        elif self.mode == 'rnn':
            # For RNNs, set weight decay and gradient clipping
            self.network_weight_decay = args.wdecay
            self.network_clip = args.clip
            # Adam optimizer for architecture parameters
            self.optimizer = torch.optim.Adam(
                self.model.arch_parameters(),
                lr=args.arch_lr,
                weight_decay=args.arch_wdecay
            )

    def _compute_unrolled_model(self, *args):
        """
        Compute the unrolled model after a single optimization step on training data.
        For CNNs, includes momentum and weight decay.
        For RNNs, applies gradient clipping and weight decay.
        Returns:
            Unrolled model (and clip coefficient for RNNs).
        """
        if self.mode == 'cnn':
            input, target, eta, network_optimizer = args
            # Compute training loss
            loss = self.model._loss(input, target)
            # Flatten current model parameters
            theta = _concat(self.model.parameters()).data
            try:
                # Try to get momentum buffer from optimizer state
                moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
            except:
                # If not available, use zeros
                moment = torch.zeros_like(theta)
            # Compute gradient of loss w.r.t. model parameters and add weight decay
            dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
            # Construct new model with updated parameters (unrolled step)
            unrolled_model = self._construct_model_from_theta(theta - eta * (moment + dtheta))
            return unrolled_model
        elif self.mode == 'rnn':
            hidden, input, target, eta = args
            # Compute training loss and next hidden state
            loss, hidden_next = self.model._loss(hidden, input, target)
            # Flatten current model parameters
            theta = _concat(self.model.parameters()).data
            # Compute gradients of loss w.r.t. model parameters
            grads = torch.autograd.grad(loss, self.model.parameters())
            # Clip gradients and get coefficient
            clip_coef = _clip(grads, self.network_clip)
            # Add weight decay to gradients
            dtheta = _concat(grads).data + self.network_weight_decay*theta
            # Construct new model with updated parameters (unrolled step)
            unrolled_model = self._construct_model_from_theta(theta - eta * dtheta)
            return unrolled_model, clip_coef

    def step(self, *args, eta, network_optimizer, unrolled):
        """
        Perform a single optimization step for architecture parameters.
        Supports both unrolled and standard backward steps.
        Args:
            *args: Data and hidden states for training/validation.
            eta: Learning rate for unrolled step.
            network_optimizer: Optimizer for network weights.
            unrolled: Whether to use unrolled optimization.
        Returns:
            For RNNs, returns next hidden state.
        """
        self.optimizer.zero_grad()  # Zero gradients for architecture parameters
        eta = network_optimizer.param_groups[0]['lr']  # Get current learning rate
        
        if self.mode == 'cnn':
            input_train, target_train, input_valid, target_valid, optimize = args
            if unrolled:
                # Use unrolled backward step
                self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer, optimize)
            else:
                # Use standard backward step
                self._backward_step(input_valid, target_valid, optimize)
            self.optimizer.step()  # Update architecture parameters
        elif self.mode == 'rnn':
            hidden_train, input_train, target_train, hidden_valid, input_valid, target_valid = args
            if unrolled:
                # Use unrolled backward step and get next hidden state
                hidden = self._backward_step_unrolled(hidden_train, input_train, target_train, hidden_valid, input_valid, target_valid, eta)
            else:
                # Use standard backward step and get next hidden state
                hidden = self._backward_step(hidden_valid, input_valid, target_valid)
            self.optimizer.step()  # Update architecture parameters
            return hidden, None  # Return next hidden state (None for compatibility)

    def _backward_step(self, *args):
        """
        Compute the backward step for architecture parameters using validation data.
        Optionally adds memory or compute penalties to the loss.
        Args:
            *args: Validation data (and optimization mode).
        Returns:
            For RNNs, returns next hidden state.
        """
        if self.mode == 'cnn':
            input_valid, target_valid, optimize = args
            # Compute validation loss
            loss = self.model._loss(input_valid, target_valid)

            if optimize == 'Memory':
                # Add penalty proportional to number of model parameters
                param_penalty = 0.1
                num_params = sum(p.numel() for p in self.model.parameters())
                loss = loss + param_penalty * num_params
            elif optimize == 'Compute':
                # Add penalty proportional to number of MACs (multiply-accumulate operations)
                model_summary = summary(self.model, input_size=(input_valid.size(0), *input_valid.shape[1:]), verbose=0)
                macs = model_summary.total_mult_adds if hasattr(model_summary, 'total_mult_adds') else 0
                macs_penalty = 0.1
                loss = loss + macs_penalty * macs

            loss.backward()  # Backpropagate loss
        elif self.mode == 'rnn':
            hidden_valid, input_valid, target_valid = args
            # Compute validation loss and next hidden state
            loss , hidden_next = self.model._loss(hidden_valid, input_valid, target_valid)
            loss.backward()  # Backpropagate loss
            return hidden_next  # Return next hidden state

    def _backward_step_unrolled(self, *args):
        """
        Compute the backward step for architecture parameters using the unrolled model.
        Handles implicit gradient computation via Hessian-vector products.
        Optionally adds memory or compute penalties to the unrolled loss.
        Args:
            *args: Training and validation data, learning rate, optimizer, and optimization mode.
        Returns:
            For RNNs, returns next hidden state.
        """
        if self.mode == 'cnn':
            input_train, target_train, input_valid, target_valid, eta, network_optimizer, optimize = args
            # Compute unrolled model after a training step
            unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
            # Compute validation loss on unrolled model
            unrolled_loss = unrolled_model._loss(input_valid, target_valid)

            if optimize == 'Memory':
                # Add penalty proportional to number of model parameters
                param_penalty = 0.1
                num_params = sum(p.numel() for p in unrolled_model.parameters())
                unrolled_loss = unrolled_loss + param_penalty * num_params
            elif optimize == 'Compute':
                # Add penalty proportional to number of MACs
                model_summary = summary(unrolled_model, input_size=(input_valid.size(0), *input_valid.shape[1:]), verbose=0)
                macs = model_summary.total_mult_adds if hasattr(model_summary, 'total_mult_adds') else 0
                macs_penalty = 0.1
                unrolled_loss = unrolled_loss + macs_penalty * macs

            unrolled_loss.backward()  # Backpropagate unrolled loss
            dalpha = [v.grad for v in unrolled_model.arch_parameters()]  # Gradients w.r.t. architecture parameters
            vector = [v.grad.data for v in unrolled_model.parameters()]  # Gradients w.r.t. model parameters
            # Compute implicit gradients using Hessian-vector product
            implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

            # Subtract implicit gradients from dalpha (scaled by eta)
            for g, ig in zip(dalpha, implicit_grads):
                g.data.sub_(ig.data, alpha=eta)

            # Copy gradients to original model's architecture parameters
            for v, g in zip(self.model.arch_parameters(), dalpha):
                if v.grad is None:
                    v.grad = Variable(g.data)
                else:
                    v.grad.data.copy_(g.data)
        elif self.mode == 'rnn':
            hidden_train, input_train, target_train, hidden_valid, input_valid, target_valid, eta = args
            # Compute unrolled model and clip coefficient
            unrolled_model, clip_coef = self._compute_unrolled_model(hidden_train, input_train, target_train, eta)
            # Compute validation loss and next hidden state on unrolled model
            unrolled_loss, hidden_next = unrolled_model._loss(hidden_valid, input_valid, target_valid)

            unrolled_loss.backward()  # Backpropagate unrolled loss
            dalpha = [v.grad for v in unrolled_model.arch_parameters()]  # Gradients w.r.t. architecture parameters
            dtheta = [v.grad for v in unrolled_model.parameters()]  # Gradients w.r.t. model parameters
            _clip(dtheta, self.network_clip)  # Clip gradients for stability
            vector = [dt.data for dt in dtheta]  # Convert gradients to data
            # Compute implicit gradients using Hessian-vector product
            implicit_grads = self._hessian_vector_product(vector, hidden_train, input_train, target_train, r=1e-2)

            # Subtract implicit gradients from dalpha (scaled by eta and clip_coef)
            for g, ig in zip(dalpha, implicit_grads):
                g.data.sub_(eta * clip_coef, ig.data)

            # Copy gradients to original model's architecture parameters
            for v, g in zip(self.model.arch_parameters(), dalpha):
                if v.grad is None:
                    v.grad = Variable(g.data)
                else:
                    v.grad.data.copy_(g.data)
            return hidden_next  # Return next hidden state

    def _construct_model_from_theta(self, theta):
        """
        Construct a new model instance with parameters set from a flattened tensor theta.
        Args:
            theta (torch.Tensor): Flattened parameter tensor.
        Returns:
            model_new: New model instance with updated parameters.
        """
        model_new = self.model.new()  # Create new model instance
        model_dict = self.model.state_dict()  # Get current state dict

        params, offset = {}, 0  # Initialize parameter dict and offset
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())  # Number of elements in parameter
            params[k] = theta[offset: offset+v_length].view(v.size())  # Slice and reshape
            offset += v_length  # Update offset

        assert offset == len(theta)  # Ensure all elements are used
        model_dict.update(params)  # Update state dict with new parameters
        model_new.load_state_dict(model_dict)  # Load updated state dict
        return model_new.cuda()  # Move model to CUDA and return

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        """
        Compute the Hessian-vector product for implicit gradient calculation using finite differences.
        Args:
            vector (list of torch.Tensor): Vector to multiply with Hessian.
            input: Input data for loss computation.
            target: Target data for loss computation.
            r (float): Small scalar for finite difference approximation.
        Returns:
            list of torch.Tensor: Hessian-vector product for each architecture parameter.
        """
        R = r / _concat(vector).norm()  # Scale for finite difference
        # Add perturbation in the direction of vector
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(v, alpha=R)
        loss = self.model._loss(input, target)  # Compute loss with positive perturbation
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())  # Gradients w.r.t. arch params

        # Subtract perturbation in the direction of vector (2*R from original)
        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(v, alpha=2*R)
        loss = self.model._loss(input, target)  # Compute loss with negative perturbation
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())  # Gradients w.r.t. arch params

        # Restore original parameters by adding R back
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(v, alpha=R)

        # Compute finite difference approximation of Hessian-vector product
        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]