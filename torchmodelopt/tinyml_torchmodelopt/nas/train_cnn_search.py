import time
import torch
import logging
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from .model_search_cnn import Network as TrainNetwork  # Import the search-phase network
from .model import Network                            # Import the final evaluation network
from .architect import Architect                      # Import the NAS architect
from .utils import count_parameters_in_MB, AvgrageMeter, accuracy  # Utility functions

def search_and_get_model(args):
    """
    Runs the neural architecture search (NAS) process and returns the best found model.
    Args:
        args: Namespace containing all hyperparameters and data loaders.
    Returns:
        eval_model: The final model with the best found architecture.
    """
    logger = logging.getLogger("root.modelopt.nas.search")

    # Check for GPU availability
    if not torch.cuda.is_available():
        logger.error('Since no GPU is available, NAS will not be performed. NAS is a highly compute intensive operation, and might completely clog your CPU')
        # print('no GPU available')
        return None
    
    torch.cuda.set_device(args.gpu)  # Set the CUDA device
    cudnn.benchmark = True           # Enable cudnn autotuner for faster training
    cudnn.enabled = True             # Enable cudnn

    # logger.info('gpu device = %d' % args.gpu)
    # logger.info("args = %s", args)
    
    criterion = nn.CrossEntropyLoss()    # Define the loss function
    criterion = criterion.cuda()         # Move loss to GPU
    # Instantiate the search-phase network (with architecture parameters)
    model = TrainNetwork(
        args.nas_init_channels,
        args.num_classes,
        args.nas_layers,
        criterion,
        args.in_channels,
        args.nas_nodes_per_layer,
        args.nas_multiplier,
        args.nas_stem_multiplier
    )
    model = model.cuda()  # Move model to GPU
    logger.info("param size = %fMB", count_parameters_in_MB(model))  # Log model size
    # print(f"param size = {count_parameters_in_MB(model)}MB")
    
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

    best_genotype = None  # Track the best found genotype
    
    # Main NAS loop
    for epoch in range(args.nas_budget):
        lr = scheduler.get_last_lr()[0]  # Get current learning rate
        # logger.info('Epoch %d lr %f', epoch, lr)
        # print(f'Epoch: {epoch} \t LR: {lr}')
        
        genotype = model.genotype()      # Get current architecture genotype
        logger.info('genotype = %s', genotype)
        # print(f'genotype = {genotype}')
        
        # print(F.softmax(model.alphas_normal, dim=-1))
        # print(F.softmax(model.alphas_reduce, dim=-1))
        
        # Training step (updates model weights and architecture parameters)
        train_acc = train(args, epoch, train_loader, valid_loader, model, architect, criterion, optimizer, lr)
        logger.info('Train:  Acc@1 %f', train_acc)
        # print('train_acc:', train_acc)
        
        # Validation step (evaluate current architecture)
        valid_acc = infer(args, epoch, valid_loader, model, criterion)
        logger.info('Test:  Acc@1 %f', valid_acc)
        # print('valid_acc: ', valid_acc)

        best_genotype = genotype  # Update best genotype (could add selection logic)

        scheduler.step()  # Update learning rate
        
        # save(model, os.path.join(args.save, 'weights.pt'))

    # Instantiate the final evaluation model with the best found genotype
    eval_model = Network(
        args.nas_init_channels,
        args.num_classes,
        args.nas_layers,
        best_genotype,
        args.in_channels
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

    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()
    
    for step, (input_raw, input, target) in enumerate(train_loader):
        model.train()  # Set model to training mode
        n = input.size(0)  # Batch size
        
        # Move input and target to GPU and set types
        input = Variable(input, requires_grad=False).cuda().float()
        target = Variable(target, requires_grad=False).cuda().long()
        
        # Get a batch from the validation set for architecture step
        input_raw, input_search, target_search = next(iter(valid_loader))
        input_search = Variable(input_search, requires_grad=False).cuda().float()
        target_search = Variable(target_search, requires_grad=False).cuda().long()
        
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
            max_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            estimated_total = elapsed / (step + 1) * len(train_loader)
            eta_seconds = max(0, estimated_total - elapsed)
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
            logger.info('Epoch: [%d]  [%03d/%03d]  eta: %s  lr: %f  samples/s: %f  loss: %.2f  acc1: %.2f  time: %f  max_mem: %f', epoch, step, len(train_loader), eta_str, lr, samples_per_sec, objs.avg, top1.avg, step_time, max_mem_mb)
            # print(f'train {round(step, 3)} {objs.avg} {top1.avg}')
        
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
    objs = AvgrageMeter()  # Tracks average loss
    top1 = AvgrageMeter()  # Tracks average top-1 accuracy
    model.eval()           # Set model to evaluation mode

    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    for step, (input_raw, input, target) in enumerate(valid_loader):
        with torch.no_grad():
            input = input.cuda().float()   # Move input to GPU
            target = target.cuda().long()  # Move target to GPU

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
            max_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            estimated_total = elapsed / (step + 1) * len(valid_loader)
            eta_seconds = max(0, estimated_total - elapsed)
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
            logger.info('Epoch: [%d]  [%03d/%03d]  eta: %s  samples/s: %f  loss: %.2f  acc1: %.2f  time: %f  max_mem: %f', epoch, step, len(valid_loader), eta_str, samples_per_sec, objs.avg, top1.avg, step_time, max_mem_mb)
        # print(f'valid {round(step, 3)} {objs.avg} {top1.avg}')

    return top1.avg  # Return average top-1 accuracy

