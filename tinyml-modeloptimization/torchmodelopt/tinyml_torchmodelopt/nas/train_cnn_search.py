import time
import torch
import logging
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from .model_search_cnn import Network as TrainNetwork  # Import the search-phase network
from .model import Network                            # Import the final evaluation network
from .architect import Architect                      # Import the NAS architect
from .utils import count_parameters_in_MB, AvgrageMeter, accuracy, get_device  # Utility functions


def search_and_get_model(args):
    """
    Runs the neural architecture search (NAS) process and returns the best found model.
    Args:
        args: Namespace containing all hyperparameters and data loaders.
            args.gpu (int): GPU index (used for CUDA). Ignored for MPS/CPU.
    Returns:
        eval_model: The final model with the best found architecture.
    """
    logger = logging.getLogger("root.modelopt.nas.search")

    # Resolve the compute device (cuda, mps, or cpu)
    device = get_device(getattr(args, 'gpu', 0))
    args._nas_device = device  # Store for use in architect / train / infer

    if device.type == 'cpu':
        logger.warning(
            'NAS running on CPU. This is extremely slow — '
            'consider using a CUDA or MPS-capable machine.'
        )

    if device.type == 'cuda':
        torch.cuda.set_device(device)
        cudnn.benchmark = True
        cudnn.enabled = True

    criterion = nn.CrossEntropyLoss().to(device)  # Define + move loss
    # Instantiate the search-phase network (with architecture parameters)
    model = TrainNetwork(
        args.nas_init_channels,
        args.num_classes,
        args.nas_layers,
        criterion,
        args.in_channels,
        args.nas_nodes_per_layer,
        args.nas_multiplier,
        args.nas_stem_multiplier,
        device=device,
    )
    model = model.to(device)  # Move model to device
    logger.info("param size = %fMB", count_parameters_in_MB(model))

    # Optimizer for model weights (not architecture parameters)
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    train_loader = args.train_loader  # Training data loader
    valid_loader = args.valid_loader  # Validation data loader
    
    # Learning rate scheduler (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        args.nas_budget,
        eta_min=args.lr/100
    )
    
    architect = Architect(model, args)  # Instantiate the architect for NAS

    best_genotype = None   # Track the best found genotype
    best_valid_acc = 0.0   # Track the best validation accuracy

    # Main NAS loop
    for epoch in range(args.nas_budget):
        lr = scheduler.get_last_lr()[0]  # Get current learning rate

        genotype = model.genotype()      # Get current architecture genotype
        logger.info('genotype = %s', genotype)

        # Training step (updates model weights and architecture parameters)
        train_acc = train(args, epoch, train_loader, valid_loader, model, architect, criterion, optimizer, lr)
        logger.info('Train:  Acc@1 %f', train_acc)

        # Validation step (evaluate current architecture)
        valid_acc = infer(args, epoch, valid_loader, model, criterion)
        logger.info('Test:  Acc@1 %f', valid_acc)

        # Keep the genotype with the best validation accuracy
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_genotype = genotype
            logger.info('New best genotype at epoch %d (Acc@1 %f)', epoch, valid_acc)

        scheduler.step()  # Update learning rate

    # Instantiate the final evaluation model with the best found genotype,
    # passing the same structural parameters used during search to ensure
    # the final model matches the searched architecture exactly.
    eval_model = Network(
        args.nas_init_channels,
        args.num_classes,
        args.nas_layers,
        best_genotype,
        args.in_channels,
        steps=args.nas_nodes_per_layer,
        multiplier=args.nas_multiplier,
        stem_multiplier=args.nas_stem_multiplier,
    )

    return eval_model

def train(args, epoch, train_loader, valid_loader, model, architect, criterion, optimizer, lr):
    """
    Performs one epoch of training for NAS.
    Args:
        args: Namespace with hyperparameters.
        train_loader: DataLoader for training data.
        valid_loader: DataLoader for validation data.
        model: The search-phase network.
        architect: The NAS architect.
        criterion: Loss function.
        optimizer: Optimizer for model weights.
        lr: Current learning rate.
    Returns:
        float: Top-1 training accuracy for the epoch.
    """
    logger = logging.getLogger("root.modelopt.nas.train")
    objs = AvgrageMeter()  # Tracks average loss
    top1 = AvgrageMeter()  # Tracks average top-1 accuracy

    device = args._nas_device

    start_time = time.time()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    # Create a persistent iterator over the validation set so that
    # successive architecture-update steps cycle through different batches
    # instead of always reusing the first batch.
    valid_iter = iter(valid_loader)

    for step, (input_raw, input, target) in enumerate(train_loader):
        model.train()  # Set model to training mode
        n = input.size(0)  # Batch size

        # Move input and target to device and set types
        input = Variable(input, requires_grad=False).to(device).float()
        target = Variable(target, requires_grad=False).to(device).long()

        # Get a batch from the validation set for architecture step,
        # cycling back to the start when the validation set is exhausted.
        try:
            input_raw, input_search, target_search = next(valid_iter)
        except StopIteration:
            valid_iter = iter(valid_loader)
            input_raw, input_search, target_search = next(valid_iter)
        input_search = Variable(input_search, requires_grad=False).to(device).float()
        target_search = Variable(target_search, requires_grad=False).to(device).long()
        
        # Update architecture parameters (unrolled or standard)
        architect.step(
            input, target, input_search, target_search, args.nas_optimization_mode,
            eta=lr, network_optimizer=optimizer, unrolled=args.unrolled
        )
        
        optimizer.zero_grad()  # Zero gradients for model weights
        logits = model(input)  # Forward pass
        loss = criterion(logits, target)  # Compute loss

        loss.backward()  # Backpropagate
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # Clip gradients
        optimizer.step()  # Update model weights
        
        prec1, prec5 = accuracy(logits, target, topk=(1, 1))  # Compute accuracy
        objs.update(loss.item(), n)  # Update average loss
        top1.update(prec1.item(), n) # Update average accuracy
        
        if step % 50 == 0:
            elapsed = time.time() - start_time
            samples = (step + 1) * input.size(0)
            samples_per_sec = samples / elapsed
            step_time = elapsed / (step + 1)
            max_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == 'cuda' else 0
            estimated_total = elapsed / (step + 1) * len(train_loader)
            eta_seconds = max(0, estimated_total - elapsed)
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
            logger.info('Epoch: [%d]  [%03d/%03d]  eta: %s  lr: %f  samples/s: %f  loss: %.2f  acc1: %.2f  time: %f  max_mem: %f', epoch, step, len(train_loader), eta_str, lr, samples_per_sec, objs.avg, top1.avg, step_time, max_mem_mb)

    return top1.avg  # Return average top-1 accuracy

def infer(args, epoch, valid_loader, model, criterion):
    """
    Evaluates the model on the validation set.
    Args:
        args: Namespace with hyperparameters.
        valid_loader: DataLoader for validation data.
        model: The network to evaluate.
        criterion: Loss function.
    Returns:
        float: Top-1 validation accuracy.
    """
    logger = logging.getLogger("root.modelopt.nas.infer")
    device = args._nas_device
    objs = AvgrageMeter()  # Tracks average loss
    top1 = AvgrageMeter()  # Tracks average top-1 accuracy
    model.eval()           # Set model to evaluation mode

    start_time = time.time()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    for step, (input_raw, input, target) in enumerate(valid_loader):
        with torch.no_grad():
            input = input.to(device).float()   # Move input to device
            target = target.to(device).long()  # Move target to device

        logits = model(input)              # Forward pass
        loss = criterion(logits, target)   # Compute loss

        prec1, prec5 = accuracy(logits, target, topk=(1, 1))  # Compute accuracy
        n = input.size(0)
        objs.update(loss.item(), n)        # Update average loss
        top1.update(prec1.item(), n)       # Update average accuracy

        if step % 50 == 0:
            elapsed = time.time() - start_time
            samples = (step + 1) * input.size(0)
            samples_per_sec = samples / elapsed
            step_time = elapsed / (step + 1)
            max_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == 'cuda' else 0
            estimated_total = elapsed / (step + 1) * len(valid_loader)
            eta_seconds = max(0, estimated_total - elapsed)
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
            logger.info('Epoch: [%d]  [%03d/%03d]  eta: %s  samples/s: %f  loss: %.2f  acc1: %.2f  time: %f  max_mem: %f', epoch, step, len(valid_loader), eta_str, samples_per_sec, objs.avg, top1.avg, step_time, max_mem_mb)

    return top1.avg  # Return average top-1 accuracy

