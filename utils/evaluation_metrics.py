import numpy as np


class average_meter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_results(res, labelq, labelx, n=10):
    pr, rec = prec_recall(res, labelq, labelx)
    ap = 0
    for i in range(labelq.shape[0]):
        ap += compute_ap(pr[i], rec[i], n)

    return ap / labelq.shape[0]


def compute_ap(prec, rec, n):
    ap = 0

    for t in np.linspace(0, 1, n):

        all_p = prec[rec >= t]
        if all_p.size == 0:
            p = 0
        else:
            p = all_p.max()
        ap = 1.0 * ap + 1.0 * p / n

    return ap


def prec_recall(dhamm, labelq, labelx):
    prec = np.zeros_like(dhamm, dtype=float)
    recall = np.zeros_like(dhamm, dtype=float)
    n = np.arange(dhamm.shape[1]) + 1.0

    res = np.argsort(dhamm, axis=1)
    for i in range(res.shape[0]):
        label = labelq[i, :] * 1
        label[label == 0] = -1

        imatch = np.sum(labelx[res[i, :], :] * 1. == label, 1) > 0
        x = np.cumsum(imatch.astype(float))
        prec[i] = x / n
        recall[i] = x / x[-1]

    return prec, recall


def getHammingDist(Bquery, Bgallery):
    nquery, _ = np.shape(Bquery)
    ngallery, _ = np.shape(Bgallery)

    dhamm = np.zeros((nquery, ngallery), dtype=int)
    i = 0
    for Bq in Bquery:
        dhamm[i, :] = np.sum(Bq ^ Bgallery, axis=1)
        i += 1

    return dhamm
