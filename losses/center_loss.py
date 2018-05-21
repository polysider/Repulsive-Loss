import torch
from torch import nn as nn
from torch.autograd import Variable


class CenterLoss(nn.Module):

    def __init__(self, num_classes, feat_dim, loss_weight=1.0):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        # self.feat_dim = centers.size()[1]
        self.loss_weight = loss_weight
        self.centers = nn.Parameter(torch.zeros(num_classes, feat_dim))
        # self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        # self.register_parameter('centers', self.centers) # no need to register manually. See nn.Module.__setattr__(...)
        self.use_cuda = False

    def forward(self, y, feat):
        # torch.histc can only be implemented on CPU
        # feat requires gradient and thus will be changed by the center loss
        # To calculate the total number of every class in one mini-batch. See Equation 4 in the paper
        if self.use_cuda:
            hist = Variable(
                torch.histc(y.cpu().data.float(), bins=self.num_classes, min=0, max=self.num_classes) + 1).cuda()
        else:
            hist = Variable(torch.histc(y.data.float(), bins=self.num_classes, min=0, max=self.num_classes) + 1)

        centers_count = hist.index_select(0, y.long())

        # To squeeze the Tensor
        batch_size = feat.size()[0]
        # feat = feat.view(batch_size, 1, 1, -1).squeeze()
        # To check the dim of centers and features
        if feat.size()[1] != self.feat_dim:
            raise ValueError(
                "Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim, feat.size()[1]))

        # get a corresponding center for each sample in the minibatch
        centers_pred = self.centers.index_select(0, y.long())
        diff = feat - centers_pred
        loss = self.loss_weight * 1 / 2.0 * (diff.pow(2).sum(1) / centers_count).sum()
        return loss

    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU.
        Arguments:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        self.use_cuda = True
        return self._apply(lambda t: t.cuda(device_id))