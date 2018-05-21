import torch
from torch.autograd import Variable
import torchvision
from utils.show_images import showgrid, imshow, plot_nice_grid


def test_one_batch(model, testloader, classes, use_gpu=False, Show=False):
    print("Testing on one batch")

    dataiter = iter(testloader)
    images, labels = dataiter.next()  # images: torch.FloatTensor of [1000, 1, 28, 28]

    if False:
        num = 16
        if images.shape[0] > num:
            imgs_to_show = images[:num]
            lbls_to_show = labels[:num]
        else:
            imgs_to_show = images
            lbls_to_show = labels

        title = ' '.join('%5s' % label for label in lbls_to_show)
        imshow(torchvision.utils.make_grid(imgs_to_show), title)

    misclassified_images = []
    misclassified_labels = []

    if use_gpu:
        outputs, embeddings = model(Variable(images).cuda())
        labels = labels.cuda()
    else:
        outputs, embeddings = model(Variable(images))

    _, predictions = torch.max(outputs.data, 1)

    # both predictions and labels are torch tensors here (cpu or cuda)
    fails = predictions != labels

    print("Number of misclassifications: {}".format(fails.sum()))
    misclassified_idxs = [i for i, x in enumerate(fails) if x]

    misclassified_images = [images[idx] for idx in misclassified_idxs]
    misclassified_true_labels = [labels[idx] for idx in misclassified_idxs]
    misclassified_predictions = [predictions[idx] for idx in misclassified_idxs]
    # print("Misclassified predictions: {}".format(misclassified_predictions))

    if Show:
        title = 'Misclassified images'
        # showgrid(misclassified_images, misclassified_predictions, classes, title)
        plot_nice_grid(misclassified_images, misclassified_true_labels, misclassified_predictions)

    # print('GroundTruth: ', ' '.join('%s' % classes[label] for label in labels))
    # print('Predicted: ', ' '.join('%s' % classes[pred] for pred in predictions))

    return embeddings, labels


def test(model, testloader, use_gpu=False):
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        if use_gpu:
            outputs, _ = model(Variable(images).cuda())
            labels = labels.cuda()
        else:
            outputs, _ = model(Variable(images))

        _, predicted = torch.max(outputs.data, 1)
        # print(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    accuracy = 100.0 * correct / total

    print('Accuracy of the network on the test images: {:.3f}; correct: {} out of {}'.format(
        accuracy, correct, total))

    return accuracy


def test_classwise(model, testloader, classes, use_gpu=False):
    ########################################################################
    # Classes that performed well, and the classes that did not:

    class_correct = list(0. for class_name in classes)
    class_total = list(0. for class_name in classes)

    for data in testloader:
        images, labels = data
        if use_gpu:
            outputs, _ = model(Variable(images).cuda())
            labels = labels.cuda()
        else:
            outputs, _ = model(Variable(images))

        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for i in range(labels.size(0)):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    for i in range(classes.__len__()):
        if not class_total[i] == 0:
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
        else:
            print('Accuracy of %5s is not defined' % classes[i])
