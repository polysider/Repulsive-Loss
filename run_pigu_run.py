"""
using pytorch
mnist
"""

# -*- coding: utf-8 -*-

import os
import time

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


def train_epoch(model, trainloader, criterion, optimizer, epoch, num_classes, use_gpu=False, show=False):

    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs
        inputs, labels = data

        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer[0].zero_grad()
        optimizer[1].zero_grad()

        outputs, embeddings = model(inputs)

        grads = {}

        def save_grad(name):
            def hook(grad):
                grads[name] = grad

            return hook

        h = embeddings.register_hook(save_grad('embeddings'))

        # cross_entropy_loss = criterion[0](outputs, labels)
        cross_entropy_loss = criterion[0](F.log_softmax(outputs), labels)

        center_loss = criterion[1](labels, embeddings)

        # centers = criterion[1].state_dict()['centers']
        centers = criterion[1].centers

        feat_dim = centers.size()[1]

        repulsive_loss_module = criterion[2]
        repulsive_loss_module.update_centers(centers)
        repulsive_loss = repulsive_loss_module(labels, embeddings)

        loss = cross_entropy_loss + center_loss + repulsive_loss
        # loss = cross_entropy_loss + center_loss
        # loss = repulsive_loss

        # backward
        loss.backward()

        # loss = loss + repulsive_loss

        # updates net parameters
        optimizer[0].step()

        # updates centers
        optimizer[1].step()

        # second '[1]' output of torch.max() is an argmax, i.e. outputs index location of each maximum value found
        prediction = outputs.data.max(1)[1]  # first column has actual prob.
        accuracy = float(prediction.eq(labels.data).sum()) / batch_size_train * 100

        # print statistics
        running_loss += loss.item()  # loss.data[0]
        if i % 500 == 0:  # print every 1000 mini-batches
            # print('Epoch: {:<3}\tMinibatch No: {:<5}\tLoss: {:.3f}\tCross entropy: {:.3f}\tCenter Loss: {:.3f}\tRepulsive Loss: {:.3f}\tRunning Loss: {:<4.3f}\tAccuracy: {:.3f}'.
            #       format(epoch, i, loss.data[0], cross_entropy_loss.data[0], center_loss.data[0], repulsive_loss.data[0], running_loss, accuracy))
            print(
                'Epoch: {:<3}\tMinibatch No: {:<5}\tLoss: {:.3f}\tCross entropy: {:.3f}\tCenter Loss: {:.3f}\tRepulsive Loss: {:.3f}\tRunning Loss: {:<4.3f}\tAccuracy: {:.3f}'.
                format(epoch, i, loss.item(), cross_entropy_loss.item(), center_loss.item(), repulsive_loss.item(),
                       running_loss, accuracy))
            # print('Centers: {}'.format(centers.data.cpu()))
            # print('Centers grad: {}'.format(centers.grad))
            # print('Embeddings grad: {}'.format(embeddings.grad))
            # print('Embeddings grad: {}'.format(grads['embeddings']))
            running_loss = 0.0

    if show:
        if embedding_dim == 2:
            plot_data_better(embeddings.data.cpu().numpy(), labels.data.cpu().numpy(), centers.data.cpu().numpy(), epoch)
        else:
            # do PCA and show
            visualize_mnist(embeddings, labels, centers, epoch, num_classes)

    return model, centers


def main():

    ########################################################################
    # Set-up
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed(TORCH_SEED)
    np.random.seed(NUMPY_SEED)

    use_gpu = torch.cuda.is_available()

    device = torch.device("cuda" if use_gpu else "cpu")
    print('GPU id: {}, name: {}'.format(GPU_ID, torch.cuda.get_device_name(torch.cuda.current_device())))

    # if use_gpu:
        # this doesn't work. Setting it in the configuration menu does
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
        # print("GPU: {}".format(torch.cuda.current_device()))
    # use_gpu = False

    # plt.interactive(False) # an attempt to make plt work

    ########################################################################
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1]

    #   transform = transforms.Compose(
    #       [transforms.Scale((32, 32)),
    #       transforms.ToTensor(),
    #       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # transform = transforms.ToTensor()
    #
    # trainset = datasets.MNIST(MNIST_DIR, train=True, download=True, transform=transform)
    # trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
    #
    # testset = datasets.MNIST(MNIST_DIR, train=False, transform=transform)
    # testloader = DataLoader(testset, batch_size=1000)

    trainloader, testloader, trainset, testset, num_classes = load_dataset(dataset_choice, batch_size_train, batch_size_test)

    dataloaders = {'train': trainloader, 'val': testloader}
    dataset = {'train': trainset, 'val': testset}

    classes = np.arange(0, 10)

    if train:

        since = time.time()

        # Define a Convolution Neural Network
        net = MnistModel(embedding_dim)
        if dataset_choice == 'ONLINE_PRODUCTS':
            net = Net(embedding_dim)

        if use_gpu:
            net = net.cuda()

        # Define a Loss function and optimizer
        # criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(net.parameters(), lr=0.0001)

        # cross_entropy = nn.CrossEntropyLoss()
        cross_entropy = nn.NLLLoss()

        center_loss_weight = cl_weight
        center_loss_module = CenterLoss(num_classes, embedding_dim, center_loss_weight)
        if use_gpu:
            center_loss_module = center_loss_module.cuda()

        repulsive_loss_weight = rl_weight
        repulsive_loss_margin = rl_margin
        repulsive_loss_module = RepulsiveLoss(num_classes, embedding_dim, repulsive_loss_margin, repulsive_loss_weight)
        if use_gpu:
            repulsive_loss_module = repulsive_loss_module.cuda()

        criterion = [cross_entropy, center_loss_module, repulsive_loss_module]

        optimizer_net = optim.Adam(net.parameters(), lr=cross_entropy_lr)
        optimizer_center = optim.SGD(center_loss_module.parameters(), lr=center_loss_lr)
        optimizer = [optimizer_net, optimizer_center]

        for epoch in range(num_epochs):  # loop over the dataset multiple times
            _, centers = train_epoch(net, trainloader, criterion, optimizer, epoch, num_classes, use_gpu, show_plots)

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
