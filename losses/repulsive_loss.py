import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable

from losses.compute_distance_matrix import compute_distance_matrix


class RepulsiveLoss(nn.Module):

    def __init__(self, num_classes, feat_dim, margin, loss_weight=1.0):
        super(RepulsiveLoss, self).__init__()
        self.loss_weight = loss_weight
        self.centers = Variable(torch.zeros(num_classes, feat_dim), requires_grad=False)
        self.feat_dim = feat_dim
        self.margin = margin
        self.use_cuda = False
        # self.broadcasted_margin = nn.Parameter(torch.Tensor([self.margin]), requires_grad=False)
        # self.broadcasted_zeros = nn.Parameter(torch.Tensor([0.0]), requires_grad=False)

    def update_centers(self, centers):
        self.centers = Variable(centers.data,
                                requires_grad=False)  # torch Tensor of [num_centers x feat_dim], not a trainable variable

    def forward(self, labels, embeddings):

        batch_size = embeddings.size()[0]
        center_size = self.centers.size()[0]
        num_classes = center_size
        # To check the dim of centers and features
        if embeddings.size()[1] != self.feat_dim:
            raise ValueError(
                "Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim, embeddings.size()[1]))

        # get a corresponding center for each sample in the minibatch
        centers_pred = self.centers.index_select(0, labels.long())

        # Now we need to compute the Squared Euclidean Distances Between Two Sets of Vectors: embeddings and centers
        distances = compute_distance_matrix(embeddings, self.centers)
        # print(distances)

        # ones are assigned only to the other classes, zeros for the distances to own class
        other_classes_mask = torch.ones([batch_size, num_classes])
        # other_classes_mask[torch.from_numpy(np.arange(batch_size, dtype=int)), labels.data.long()] = 0
        other_classes_mask[np.arange(batch_size), labels.data.long().cpu().numpy()] = 0
        other_classes_mask = Variable(other_classes_mask, requires_grad=False)

        # distances = torch.mul(distances, other_classes_mask)
        # broadcasted_margin = Variable(torch.Tensor([self.margin]).expand_as(distances), requires_grad=False)

        broadcasted_margin = Variable(torch.Tensor([self.margin]), requires_grad=False)
        broadcasted_zeros = Variable(torch.Tensor([0.0]).expand_as(distances), requires_grad=False)

        if self.use_cuda:
            other_classes_mask = other_classes_mask.cuda()
            broadcasted_margin = broadcasted_margin.cuda()
            broadcasted_zeros = broadcasted_zeros.cuda()

        # print(broadcasted_margin - distances)
        penalties = torch.max(broadcasted_zeros, broadcasted_margin - distances)
        # print(penalties)
        penalties_to_others = torch.mul(penalties, other_classes_mask)
        # print(penalties_to_others)

        # TODO: write the proper expression for it
        loss = self.loss_weight * (1.0/batch_size) * (1.0/center_size) * torch.sum(penalties_to_others.pow(2))
        return loss

    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU.
        Arguments:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        self.use_cuda = True
        return self._apply(lambda t: t.cuda(device_id))