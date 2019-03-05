import torch
import torchvision
from torch.autograd import Variable
import numpy as np

from evaluation.knn import KNearestNeighbor
from evaluation.evaluation_metrics import recall_at_k
from utils.show_images import imshow, plot_nice_grid


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


def test(model, testloader, device, use_gpu=False):

    correct = 0
    total = 0

    for (inputs, labels) in testloader:

        inputs, labels = inputs.to(device), labels.to(device)
        outputs, _ = model(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total

    print('Accuracy of the network on the test images: {:.3f}%; correct: {} out of {}'.format(
        accuracy, correct, total))

    return accuracy


def test_classwise(model, testloader, classes, device, use_gpu=False):
    ##################################################################
    # Classes that performed well, and the classes that did not:

    class_correct = list(0 for class_name in classes)
    class_total = list(0 for class_name in classes)

    for (inputs, labels) in testloader:

        inputs, labels = inputs.to(device), labels.to(device)
        outputs, _ = model(inputs)

        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()

        for i in range(labels.size(0)):
            label = labels[i].item()
            class_correct[label] += c[i].item()
            class_total[label] += 1

    for i in range(classes.__len__()):
        if not class_total[i] == 0:
            class_accuracy = 100.0 * class_correct[i] / class_total[i]
            print('Accuracy of {:3} : {:.1f}%'.format(classes[i], class_accuracy))
        else:
            print('Accuracy of {:3} is not defined'.format(classes[i]))


def test_retrieval(model, testloader, device, k=1, use_gpu=False):

    labels_list = []
    retrieved_labels_list = []

    for inputs, labels in testloader:

        inputs, labels = inputs.to(device), labels.to(device)
        outputs, embeddings = model(inputs)

        knn_classifier = KNearestNeighbor()
        knn_classifier.train(embeddings.cpu().data, labels.cpu())
        retrieved_labels = knn_classifier.get_nearest_labels(embeddings.cpu().data, k)

        labels_list.extend(labels.cpu().numpy())
        retrieved_labels_list.extend(retrieved_labels)

    labels_list = np.asarray(labels_list)
    retrieved_labels_list = np.asarray(retrieved_labels_list)

    #print(labels_list)
    #print(retrieved_labels_list)
    #print(zip(labels_list, retrieved_labels_list))
    #print([label in label_pred for label, label_pred in zip(labels_list, retrieved_labels_list)])
    recall = recall_at_k(labels_list, retrieved_labels_list, k)
    print("Recall at {} is: {}".format(k, recall))








