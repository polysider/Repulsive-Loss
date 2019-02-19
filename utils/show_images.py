import os

import numpy as np
from matplotlib import pyplot as plt

from utils.tsne import pca


def imshow(img, title=None):
    # img = img / 2 + 0.5     # unnormalize
    plt.figure()
    npimg = img.numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title is not None:
        plt.title(title)
    # plt.pause(0.01)  # pause a bit so that plots are updated
    plt.show(block=False)


def showgrid(imgs, labels, classes, title=None):
    """
    inputs:
    imgs: list of Torch tensors of NumImgs x 1 x 32 x 32
    labels: list of class labels of length = NumImgs
    classes: list of all possible class labels
    """
    plt.figure()
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    npimgs = np.asarray([img.numpy() for img in imgs])
    labels = np.asarray(labels)
    plt.axis('off')

    num_classes = len(classes)
    samples_per_class = 8
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(labels == y)
        samples_per_class = len(idxs) if len(idxs) > samples_per_class else samples_per_class
        #         if not len(idxs)==0:
        #             idxs = np.random.choice(idxs, samples_per_class, replace=True)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            img = npimgs[idx]
            img = np.squeeze(img)
            img = np.stack((img,) * 3)
            img = np.transpose(img, (1, 2, 0))
            plt.imshow(img)
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    if title is not None:
        plt.suptitle(title, fontsize=16)
    plt.show(block=False)


def plot_nice_grid(images, labels_true, preds=None):

    fig_size = [10, 10]
    plt.rcParams["figure.figsize"] = fig_size

    fig, axes = plt.subplots(5, 5)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    npimages = np.asarray([image.numpy() for image in images])

    for i, ax in enumerate(axes.flat):

        ax.set_xticks([])
        ax.set_yticks([])

    for i, npimage in enumerate(npimages):

        if i < 25:
            # Plot image.
            axes.flat[i].imshow(npimage.reshape(28, 28), cmap='binary')

            # True vs predicted labels
            if preds is None:
                xlabel = "True: {0}".format(labels_true[i])
            else:
                xlabel = "True: {0}, Pred: {1}".format(labels_true[i], preds[i])

            axes.flat[i].set_xlabel(xlabel)
            axes.flat[i].set_xticks([])
            axes.flat[i].set_yticks([])

    # Draw the plot
    plt.show()

def visualize_MNIST_original(feat, labels, epoch):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    plt.xlim(xmin=-5, xmax=5)
    plt.ylim(ymin=-5, ymax=5)
    plt.text(-4.8, 4.6, "epoch=%d" % epoch)
    # plt.savefig('./images/epoch=%d.jpg' % epoch)
    plt.draw()
    plt.pause(0.001)


def visualize_mnist(embeddings, labels, centers=None, epoch=None, num_classes=10):

    features_data = embeddings.data.cpu().numpy()
    labels_data = labels.data.cpu().numpy()
    centers_data = centers.data.cpu().numpy()

    all_data = np.concatenate((features_data, centers_data), axis=0)

    # convert image data to float64 matrix. float64 is need for bh_sne
    all_data = np.asarray(all_data).astype('float64')
    all_data = all_data.reshape((all_data.shape[0], -1))

    pca_data = pca(all_data, 2)

    pca_features_data = pca_data[:-num_classes]
    pca_centers_data = pca_data[-num_classes:]

    # plot the result
    plot_data_better(pca_features_data, labels_data, pca_centers_data, epoch)


def plot_data(points, labels, center_points=None, epoch=None):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(10):
        plt.plot(points[labels == i, 0], points[labels == i, 1], '.', c=c[i])
    for i in range(10):
        if center_points is not None:
            plt.plot(center_points[i, 0], center_points[i, 1], '*', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    plt.text(-4.8, 4.6, "epoch=%d" % epoch)
    # plt.savefig('./images/epoch=%d.jpg' % epoch)
    plt.draw()
    plt.pause(0.001)


def plot_data_better(points, labels, center_points=None, epoch=None, log_dir=None):
    plt.ion()
    plt.clf()
    plt.scatter(points[:, 0], points[:, 1], c=labels, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    if center_points is not None:
        plt.scatter(center_points[:, 0], center_points[:, 1], c=np.arange(10), cmap=plt.cm.get_cmap("jet", 10), marker='*')

    if epoch is not None:
        plt.text(-4.8, 4.6, "epoch=%d" % epoch)
    if log_dir is not None:
        plot_name = 'embeddings.png'
        img_path = os.path.join(log_dir, plot_name)
        plt.savefig(img_path)
    plt.draw()
    plt.pause(0.001)


def visualize_better(features, labels, num_classes=10, epoch=None):
    features_x = features[:, 0]
    features_y = features[:, 1]
    # plt.ion()
    plt.figure()
    plt.scatter(features_x, features_y, c=labels, cmap=plt.cm.get_cmap("jet", num_classes))
    plt.colorbar(ticks=range(num_classes))
    plt.clim(-0.5, 9.5)
    plt.show(block=True)
    if epoch is not None:
        plt.text(-4.8, 4.6, "epoch=%d" % epoch)
    # plt.savefig('./images/epoch=%d.jpg' % epoch)
    # plt.draw()
    # plt.pause(0.001)
    plt.show(block=False)