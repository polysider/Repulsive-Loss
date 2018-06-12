import numpy as np
import torch
import time
from evaluation.evaluation_metrics import recall_at_k


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        dists = self.compute_distances(X)
        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.
        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.
        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in xrange(num_test):
            for j in xrange(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension.                                    #
                #####################################################################
                dists[i, j] = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2))
                #####################################################################
                #                       END OF YOUR CODE                            #
                #####################################################################
        return dists

    def compute_distances(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        assert X.shape[1] == self.X_train.shape[1], "Input should have same feature dimensions"

        since = time.time()

        # Compute the l2 distance between all test points and all training
        # points using matrix multiplication and two broadcast sums:

        # squared_dists = np.sum(X ** 2, axis=1, keepdims=True) + np.sum(self.X_train ** 2, axis=1, keepdims=True) - 2 * np.dot(X, self.X_train.T)
        # my_squared_dists = np.dot(np.sum(np.square(X), axis = 1, keepdims = True), np.ones((1, num_train))) + np.dot(np.ones((num_test, 1)), np.transpose(np.sum(np.square(self.X_train), axis = 1, keepdims = True))) + (-2)*np.dot(X, np.transpose(self.X_train))
        # dists = np.sqrt(np.abs(my_squared_dists))

        XX = np.sum(np.square(X), axis=1, keepdims=True)
        #print("XX: {}".format(XX))
        XX_broadcasted = np.dot(XX, np.ones((1, num_train)))
        #print("XX_broadcasted: {}".format(XX_broadcasted))
        YY = np.transpose(np.sum(np.square(self.X_train), axis=1, keepdims=True))
        #print("YY: {}".format(YY))
        YY_broadcasted = np.dot(np.ones((num_test, 1)), YY)
        #print("YY_broadcasted: {}".format(YY_broadcasted))
        XY = (-2)*np.dot(X, np.transpose(self.X_train))
        #print("XY: {}".format(XY))
        dists = np.sqrt(np.abs(XX_broadcasted + YY_broadcasted + XY))
        #print("dists: {}".format(dists))
        #assert (self.compute_distances_two_loops(X) == dists).all(), "Something is wrong in distance calculation"
        #print("error: {}".format(self.compute_distances_two_loops(X) - dists))
        # print([[dist for dist in row if dist != dist] for row in dists]) #checking for NaNs

        time_elapsed = time.time() - since
        #print('Distance calculation in numpy complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        return dists

    def compute_distances_pytorch(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Inputs:
        - X: A torch tensor of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.size(0)
        num_train = self.X_train.size(0)
        dists = np.zeros((num_test, num_train))

        assert X.size(1) == self.X_train.size(1), "Input should have same feature dimensions"

        since = time.time()

        # Compute the l2 distance between all test points and all training
        # points using matrix multiplication and two broadcast sums:

        XX = torch.sum(X ** 2, dim=1, keepdim=True)
        #print("XX: {}".format(XX))
        XX_broadcasted = torch.mm(XX, torch.ones((1, num_train)))
        #print("XX_broadcasted: {}".format(XX_broadcasted))
        YY = torch.transpose(torch.sum(self.X_train ** 2, dim=1, keepdim=True), 0, 1)
        #print("YY: {}".format(YY))
        YY_broadcasted = torch.mm(torch.ones((num_test, 1)), YY)
        #print("YY_broadcasted: {}".format(YY_broadcasted))
        XY = (-2) * torch.mm(X, torch.t(self.X_train))
        #print("XY: {}".format(XY))

        squared_dists = XX_broadcasted + YY_broadcasted + XY

        dists = torch.sqrt(torch.abs(squared_dists))

        time_elapsed = time.time() - since
        #print('Distance calculation with torch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        return dists.numpy()

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in xrange(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []

            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors.
            closest_y = self.y_train[np.argsort(dists[i, :])[:k]]

            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            y_pred[i] = np.bincount(closest_y).argmax()

        return y_pred

    def get_nearest_neighbors(self, X, k=1):
        """
        Get the k nearest neighbors for each test point in X

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.
        - k: The number of nearest neighbors retrieved

        Returns:
        - closest_X: A numpy array of shape (num_test, k, d) containing coordinates (d-dimensional) of k nearest
          neighbors for the test data.
        - closest_y: A numpy array of shape (num_test, k) containing labels of k nearest neighbors for the
          test data.
        """

        # an ugly way of choosing between pytorch and numpy distance matrix calculation
        if isinstance(X, (np.ndarray, np.generic)):
            dists = self.compute_distances(X)
        else:
            dists = self.compute_distances_pytorch(X)
        num_test = dists.shape[0]

        # closest_X = np.zeros((num_test, k, self.X_train.shape[1]))
        # closest_y = np.zeros((num_test, k))
        #
        # for i in xrange(num_test):
        #     positions_of_closest = np.argsort(dists[i, :])[:k]
        #     closest_X[i] = self.X_train[positions_of_closest]
        #     closest_y[i] = self.y_train[positions_of_closest]
        #
        # return closest_X, closest_y

        indices = np.argsort(dists, axis=1)[:, 1:k + 1]
        return indices

    def get_nearest_labels(self, X, k=1):

        indices = self.get_nearest_neighbors(X, k)
        retrieved_labels = np.array([[self.y_train[i] for i in ii] for ii in indices])
        return retrieved_labels


def main():
    np.random.seed(1)

    embeddings = np.asarray([[0., 0.003], [1.01, 0.], [0., 1.032], [5.041, 0.03], [4.01, 1.034], [0.034, 5.05], [-1.04, 6.01]])
    labels = np.asarray([1, 1, 3, 2, 2, 3, 3])

    print(embeddings)
    print(labels)

    knn_classifier = KNearestNeighbor()
    knn_classifier.train(embeddings, labels)

    retrieved_labels = knn_classifier.get_nearest_labels(embeddings, 5)
    print(retrieved_labels)
    print(recall_at_k(labels, retrieved_labels, 3))

    embeddings_tensor = torch.Tensor([[0., 0.003], [1.01, 0.], [0., 1.032], [5.041, 0.03], [4.01, 1.034], [0.034, 5.05], [-1.04, 6.01]])
    labels_tensor = torch.Tensor([1, 1, 3, 2, 2, 3, 3])

    knn_classifier = KNearestNeighbor()
    knn_classifier.train(embeddings_tensor, labels_tensor)
    retrieved_labels_tensor = knn_classifier.get_nearest_labels(embeddings_tensor, 5)
    print(retrieved_labels_tensor)
    print(recall_at_k(labels_tensor, retrieved_labels_tensor, 4))



if __name__ == '__main__':
    main()
