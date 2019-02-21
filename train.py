import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.show_images import plot_data_better, visualize_mnist



def train_epoch(model, trainloader, criterion, optimizer, epoch, num_classes, batch_size_train, device, use_gpu=False, show=False, embedding_dim=2):

    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        # get the inputs
        inputs, labels = inputs.to(device), labels.to(device)

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
        # should I put softmax in the network itself?
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