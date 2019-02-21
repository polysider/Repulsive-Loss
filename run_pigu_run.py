"""
using pytorch
mnist
"""

# -*- coding: utf-8 -*-

import os
import time
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from datasets.load_dataset import load_dataset
from losses.center_loss import CenterLoss
from losses.repulsive_loss import RepulsiveLoss
from models.custom_models import MnistModel, Net
from testing import test_one_batch, test, test_classwise, test_retrieval
from utils.show_images import plot_data_better, visualize_better, visualize_mnist
from utils.tsne import pca

from train import train_epoch

########################################################################
# CONSTANTS
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TORCH_SEED = 1
NUMPY_SEED = 1
GPU_ID = 2

########################################################################
# OPTIONS
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

home_dir = os.path.expanduser('~')
root_dir = os.path.join(home_dir, 'sharedLocal/Datasets')
mnist_dir = os.path.join(root_dir, 'MNIST')
model_save_path = 'checkpoints/saved_torch_model'

train = True
# fix_conv = True
show_plots = True
show_misclassified = False
log = False

########################################################################
# MODEL HYPERPARAMETERS
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

dataset_choice = 'MNIST'
# dataset_choice = 'ONLINE_PRODUCTS'
# dataset_choice = 'FASHION'

num_epochs = 1
batch_size_train = 50
batch_size_test = 1000
embedding_dim = 256

# center loss params
cl_weight = 0.01

# repulsive loss params
rl_weight = 0.05
rl_margin = 10.0

# learning rates
cross_entropy_lr = 0.0001
center_loss_lr = 0.5

########################################################################


def main():

    ########################################################################
    # Set-up
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    print(sys.executable)

    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed(TORCH_SEED)
    np.random.seed(NUMPY_SEED)

    use_gpu = torch.cuda.is_available()

    device = torch.device("cuda" if use_gpu else "cpu")
    print('GPU id: {}, name: {}'.format(GPU_ID, torch.cuda.get_device_name(torch.cuda.current_device())))

    trainloader, testloader, trainset, testset, num_classes = load_dataset(dataset_choice, batch_size_train, batch_size_test)

    classes = np.arange(0, 10)
    # classes = [1,2,3,4,5,6,7,8,9,0]

    if train:

        since = time.time()
        # Define a Convolution Neural Network
        net = MnistModel(embedding_dim)
        if dataset_choice == 'ONLINE_PRODUCTS':
            net = Net(embedding_dim)

        net = net.to(device)

        # Define a Loss function and optimizer
        # cross_entropy = nn.CrossEntropyLoss()
        cross_entropy = nn.NLLLoss()

        center_loss_weight = cl_weight
        center_loss_module = CenterLoss(num_classes, embedding_dim, center_loss_weight)
        center_loss_module = center_loss_module.to(device)
        if use_gpu:
            center_loss_module = center_loss_module.cuda()

        repulsive_loss_weight = rl_weight
        repulsive_loss_margin = rl_margin
        repulsive_loss_module = RepulsiveLoss(num_classes, embedding_dim, repulsive_loss_margin, repulsive_loss_weight)
        repulsive_loss_module = repulsive_loss_module.to(device)
        if use_gpu:
            repulsive_loss_module = repulsive_loss_module.cuda()

        criterion = [cross_entropy, center_loss_module, repulsive_loss_module]

        optimizer_net = optim.Adam(net.parameters(), lr=cross_entropy_lr)
        optimizer_center = optim.SGD(center_loss_module.parameters(), lr=center_loss_lr)
        optimizer = [optimizer_net, optimizer_center]

        for epoch in range(num_epochs):  # loop over the dataset multiple times
            _, centers = train_epoch(net, trainloader, criterion, optimizer, epoch, num_classes, batch_size_train, device, use_gpu, show_plots, embedding_dim)

        print('Finished Training')
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        torch.save(net.state_dict(), model_save_path)

    else:

        net = MnistModel()
        if use_gpu:
            net = net.cuda()
        net.load_state_dict(torch.load(model_save_path))

    ########################################################################
    # Run the network on the test data
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # Test for one batch:
    embeddings_one_batch, labels_one_batch = test_one_batch(net, testloader, classes, use_gpu, Show=show_misclassified)

    # Test on the whole dataset:
    accuracy = test(net, testloader, use_gpu)

    # Classes that performed well, and the classes that did not:
    test_classwise(net, testloader, classes, use_gpu)

    # Test for retrieval
    k = 3
    test_retrieval(net, testloader, k, use_gpu)

    ########################################################################
    # Show embeddings
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # load up data
    x_data = embeddings_one_batch.data.cpu().numpy()
    y_data = labels_one_batch.cpu().numpy()

    # convert image data to float64 matrix. float64 is need for bh_sne
    x_data = np.asarray(x_data).astype('float64')
    x_data = x_data.reshape((x_data.shape[0], -1))

    # perform t-SNE embedding
    # vis_data = tsne(x_data)
    vis_data = pca(x_data, 2)

    # plot the result
    if show_plots:
        visualize_better(vis_data, y_data)

    # logging
    if log:
        from utils.my_logging import save_run_info, prepare_log_dir
        log_dir = prepare_log_dir('logs')
        log_string = 'Dataset: {}\tEpochs: {}\tBatch size: {}\tEmbedding dim: {}\tCenter loss weigth: {:.3f}' \
                     '\tRepulsive loss weigth: {:.3f}\tCross entropy learning rate: {:.5f}\t' \
                     'Center loss learning rate: {:.4f}\tRepulsive loss margin: {:.2f}\tAccuracy: {:.3f}'. \
            format(dataset_choice, num_epochs, batch_size_train, embedding_dim, cl_weight, rl_weight, cross_entropy_lr, center_loss_lr,
                   rl_margin, accuracy)
        save_run_info(log_string, log_dir)
        plot_data_better(vis_data, y_data, log_dir=log_dir)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    with torch.cuda.device(GPU_ID):
        main()
