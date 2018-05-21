import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler

def train_resnet(dataloaders, dataset, classes, num_epochs, FixConv=False, use_gpu=False):
    from train_loop import train_model

    model_conv = models.resnet18(pretrained=True)
    if FixConv:
        # freeze the conv layers
        for param in model_conv.parameters():
            param.requires_grad = False
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, len(classes))
    if use_gpu:
        model_conv = model_conv.cuda()
    criterion = nn.CrossEntropyLoss()
    if FixConv:
        # only parameters of final layer are being optimized
        optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
    else:
        # all parameters are being optimized
        optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    # Train and evaluate
    model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler,
                             dataloaders, dataset, num_epochs=num_epochs, use_gpu=use_gpu)
    return model_conv