import torch
import torch.nn as nn
from .operations import *

class Cell(nn.Module):
    def __init__(self, genotype, C_prev, C, reduction, reduction_prev):
        """
        Cell represents a block in the network, constructed from a genotype.
        Args:
            genotype: Genotype object specifying the cell structure.
            C_prev (int): Number of channels in previous cell.
            C (int): Number of output channels for this cell.
            reduction (bool): Whether this cell is a reduction cell.
            reduction_prev (bool): Whether previous cell was a reduction cell.
        """
        super(Cell, self).__init__()
        
        # Preprocess input depending on whether previous cell was reduction
        if reduction_prev:
            self.preprocess = FactorizedReduce(C_prev, C)
        else:
            self.preprocess = ConvBNReLU(C_prev, C, 1, 1, 0)

        # Select operations and connections based on cell type (normal/reduction)
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        
        # Compile the cell structure (ops, indices, concat)
        self._compile(C, op_names, indices, concat, reduction)
    
    def _compile(self, C, op_names, indices, concat, reduction):
        """
        Compile the cell by creating the operations and storing connection info.
        Args:
            C (int): Number of output channels.
            op_names (list): List of operation names.
            indices (list): List of indices for input nodes.
            concat (list): Indices of nodes to concatenate for output.
            reduction (bool): Whether this cell is a reduction cell.
        """
        assert len(op_names) == len(indices)  # Ensure each op has an input index
        self._steps = len(op_names)           # Number of operations (nodes)
        self._concat = concat                 # Indices for concatenation
        self.multiplier = len(concat)         # Number of outputs to concatenate

        self._ops = nn.ModuleList()           # List to hold all operations
        for name, index in zip(op_names, indices):
            # For reduction cells, first node uses stride 2
            stride = 2 if reduction and index < 1 else 1
            op = OPS[name](C, stride, True)   # Instantiate operation from OPS dict
            self._ops += [op]
        self._indices = indices               # Store input indices for each node

    def forward(self, s0):
        """
        Forward pass through the cell.
        Args:
            s0 (Tensor): Input tensor from previous cell.
        Returns:
            Tensor: Output tensor after concatenating selected node outputs.
        """
        s0 = self.preprocess(s0)  # Preprocess input

        states = [s0]             # List to hold outputs of all nodes
        for i in range(self._steps):
            h1 = states[self._indices[i]]  # Select input for this node
            op1 = self._ops[i]             # Select operation for this node
            h1 = op1(h1)                   # Apply operation
            s = h1                         # Node output
            states += [s]                  # Add node output to states
        # Concatenate outputs of selected nodes along channel dimension
        return torch.cat([states[i] for i in self._concat], dim=1)

class Network(nn.Module):
    def __init__(self, C, num_classes, layers, genotype, in_channels, with_input_batchnorm):
        """
        Network is the full model, composed of multiple cells.
        Args:
            C (int): Initial number of channels.
            num_classes (int): Number of output classes.
            layers (int): Number of cells in the network.
            genotype: Genotype object specifying the cell structure.
            in_channels (int): Number of input channels.
            with_input_batchnorm: Whether to use batchnorm on input ('True' or not).
        """
        super(Network, self).__init__()
        self._layers = layers
        self._in_channels = in_channels

        # Optional batchnorm on input
        self.input_batchnorm = nn.BatchNorm2d(self._in_channels) if with_input_batchnorm == 'True' else nn.Identity()
        
        stem_multiplier = 3
        C_curr = stem_multiplier * C
        # Initial stem convolution to increase channel dimension
        self.stem = nn.Sequential(
            nn.Conv2d(self._in_channels, C_curr, (3, 1), padding=(1,0)),
            nn.BatchNorm2d(C_curr),
            nn.ReLU(inplace=False)
        )

        C_prev, C_curr = C_curr, C  # Track previous and current channel counts
        self.cells = nn.ModuleList()  # List to hold all cells
        reduction_prev = False        # Track if previous cell was reduction
        for i in range(layers):
            # Insert reduction cells at 1/3 and 2/3 of total layers
            if i in [layers//3, 2*layers//3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev = cell.multiplier * C_curr  # Update for next cell
        
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.flat = nn.Flatten()                            # Flatten for classifier
        self.classifier = nn.Linear(C_prev, num_classes)     # Linear classifier
    
    def forward(self, input):
        """
        Forward pass through the network.
        Args:
            input (Tensor): Input tensor.
        Returns:
            Tensor: Logits for each class.
        """
        input = self.input_batchnorm(input)  # Optional batchnorm
        s0 = self.stem(input)                # Initial stem convolution
        for i, cell in enumerate(self.cells):
            s0 = cell(s0)                    # Forward through each cell
        out = self.global_pooling(s0)        # Global average pooling
        out = self.flat(out)                 # Flatten for classifier
        logits = self.classifier(out)        # Linear classifier
        return logits

