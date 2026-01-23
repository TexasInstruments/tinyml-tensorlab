import torch
import torch.nn as nn
import torch.nn.functional as F
from .operations import *  # Import all operation definitions (custom layers)
from torch.autograd import Variable
from .genotypes import PRIMITIVES_CNN  # List of primitive operations for CNN search space
from .genotypes import Genotype_CNN    # Namedtuple for CNN genotype

class MixedOp(nn.Module):
    """
    MixedOp represents a weighted sum of all possible operations (from PRIMITIVES_CNN)
    on a single edge in the cell. The weights are learned as architecture parameters.
    """
    def __init__(self, C, stride):
        """
        Args:
            C (int): Number of channels.
            stride (int): Stride for the operation.
        """
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()  # List to hold all possible operations
        for primitive in PRIMITIVES_CNN:
            op = OPS[primitive](C, stride, False)  # Instantiate operation from OPS dict
            if 'pool' in primitive:
                # For pooling, add BatchNorm for normalization
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)  # Add operation to the list
    
    def forward(self, x, weights):
        """
        Forward pass: weighted sum of all operations.
        Args:
            x (Tensor): Input tensor.
            weights (Tensor): Architecture weights for this edge.
        Returns:
            Tensor: Weighted sum of operation outputs.
        """
        return sum(w * op(x) for w, op in zip(weights, self._ops))
    
class Cell(nn.Module):
    """
    Cell represents a building block of the network, consisting of several nodes.
    Each node is connected to all previous nodes via MixedOps.
    """
    def __init__(self, steps, multiplier, C_prev, C, reduction, reduction_prev):
        """
        Args:
            steps (int): Number of intermediate nodes in the cell.
            multiplier (int): Number of outputs to concatenate.
            C_prev (int): Number of channels in previous cell.
            C (int): Number of output channels.
            reduction (bool): Whether this cell is a reduction cell.
            reduction_prev (bool): Whether previous cell was a reduction cell.
        """
        super(Cell, self).__init__()
        self.reduction = reduction  # Store reduction flag
        
        # Preprocess input depending on whether previous cell was reduction
        if reduction_prev:
            self.preprocess = FactorizedReduce(C_prev, C, affine=False)
        else:
            self.preprocess = ConvBNReLU(C_prev, C, 1, 1, 0, affine=False)
        
        self._steps = steps
        self._multiplier = multiplier
        
        self._ops = nn.ModuleList()  # List to hold all MixedOps for this cell
        for i in range(self._steps):
            for j in range(1+i):
                # For reduction cells, first node uses stride 2
                stride = 2 if reduction and j < 1 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)
                
    def forward(self, s0, weights):
        """
        Forward pass through the cell.
        Args:
            s0 (Tensor): Input tensor from previous cell.
            weights (Tensor): Architecture weights for all edges in this cell.
        Returns:
            Tensor: Output tensor after concatenating selected node outputs.
        """
        s0 = self.preprocess(s0)  # Preprocess input
        
        states = [s0]  # List to hold outputs of all nodes
        offset = 0     # Offset for indexing weights and ops
        for i in range(self._steps):
            # Each node sums over all previous node outputs, each through a MixedOp
            s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)  # Add new node output to states
            
        # Concatenate outputs of the last 'multiplier' nodes along channel dimension
        return torch.cat(states[-self._multiplier:], dim=1)
    
class Network(nn.Module):
    """
    Network is the full model for architecture search, composed of multiple cells.
    """
    def __init__(self, C, num_classes, layers, criterion, in_channels, steps=4, multiplier=4, stem_multiplier=3):
        """
        Args:
            C (int): Initial number of channels.
            num_classes (int): Number of output classes.
            layers (int): Number of cells in the network.
            criterion: Loss function.
            in_channels (int): Number of input channels.
            steps (int): Number of intermediate nodes per cell.
            multiplier (int): Number of outputs to concatenate per cell.
            stem_multiplier (int): Multiplier for initial stem channels.
        """
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self._in_channels = in_channels
        self._stem_multiplier = stem_multiplier

        # Input batchnorm
        self.input_batchnorm = nn.BatchNorm2d(self._in_channels)
        
        # Initial stem convolution to increase channel dimension
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(self._in_channels, C_curr, (3, 1), padding=(1, 0)),
            nn.BatchNorm2d(C_curr),
            nn.ReLU(inplace=False),
        )
        
        C_prev, C_curr = C_curr, C  # Track previous and current channel counts
        self.cells = nn.ModuleList()  # List to hold all cells
        reduction_prev = False  # Track if previous cell was reduction
        for i in range(layers):
            # Insert reduction cells at 1/3 and 2/3 of total layers
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev = multiplier * C_curr  # Update for next cell
            
        # Global pooling and classifier for final output
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(C_prev, num_classes)
        
        self._initialize_alphas()  # Initialize architecture parameters
    
    def _initialize_alphas(self):
        """
        Initialize architecture parameters (alphas) for normal and reduction cells.
        """
        k = sum(1 for i in range(self._steps) for n in range(1+i))  # Number of edges per cell
        num_ops = len(PRIMITIVES_CNN)  # Number of possible operations

        # Architecture weights for normal and reduction cells (learnable)
        self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]
        
    def arch_parameters(self):
        """
        Return architecture parameters (alphas) as a list.
        """
        return self._arch_parameters
    
    def new(self):
        """
        Create a new instance of this network with the same architecture parameters.
        Returns:
            Network: New network instance with copied architecture parameters.
        """
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion, self._in_channels, self._steps, self._multiplier, self._stem_multiplier).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new
    
    def forward(self, input):
        """
        Forward pass through the network.
        Args:
            input (Tensor): Input tensor.
        Returns:
            Tensor: Logits for each class.
        """
        input = self.input_batchnorm(input)  # Optional batchnorm
        s0 = self.stem(input)  # Initial stem convolution
        for i, cell in enumerate(self.cells):
            # Use reduction or normal architecture weights depending on cell type
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            s0 = cell(s0, weights)  # Forward through cell
        out = self.global_pooling(s0)  # Global average pooling
        logits = self.classifier(out.view(out.size(0), -1))  # Linear classifier
        return logits
    
    def _loss(self, input, target):
        """
        Compute loss for a batch.
        Args:
            input (Tensor): Input data.
            target (Tensor): Target labels.
        Returns:
            Tensor: Loss value.
        """
        logits = self(input)
        return self._criterion(logits, target)
    
    def genotype(self):
        """
        Parse the learned architecture weights and return the discrete genotype.
        Returns:
            Genotype_CNN: The architecture genotype.
        """
        # Helper function to parse weights for a cell
        def _parse(weights):
            gene = []
            n = 1
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                # For each node, select the edge with the highest non-'none' op weight
                edges = sorted(
                    range(i+1),
                    key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES_CNN.index('none'))
                )[:1]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES_CNN.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES_CNN[k_best], j))
                start = end
                n += 1
            return gene
        
        # Parse normal and reduction cell weights
        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
        
        # Indices of nodes to concatenate for output
        concat = range(1 + self._steps - self._multiplier, self._steps + 1)
        genotype = Genotype_CNN(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        
        return genotype