import torch
from torch.autograd import Variable

from losses.center_loss import CenterLoss
from losses.repulsive_loss import RepulsiveLoss


def main():

    torch.manual_seed(2)

    ct = CenterLoss(10, 2)
    ct = ct.cuda()
    print list(ct.parameters())

    # print ct.centers.grad

    y = Variable(torch.Tensor([0, 0, 0, 1])).cuda()
    # print y
    feat = Variable(torch.randn([4, 2]), requires_grad=True).cuda()
    # print feat

    out = ct(y, feat)
    out.backward()
    # print ct.centers.grad


    rp = RepulsiveLoss(10, 2, margin=2)
    rp.update_centers(ct.centers)
    rp = rp.cuda()

    outie = rp(y, feat)
    print('output: {}'.format(outie))
    outie.backward()
    print('gradient of the features: {}'.format(feat.grad))


if __name__ == '__main__':
    main()
