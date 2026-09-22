"""VOC2010-2012 interpolated all-points AP (not sklearn average_precision).

Specification: official VOC2012 development kit documentation section 3.4.1.
https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/htmldoc/index.html
"""
import numpy as np


def voc_ap(labels, scores):
    """Raw labels {-1,0,+1}; higher score is more positive; zero ignored.

    Ties retain input order (stable sort). No-positive class returns NaN, since
    recall is undefined. Evaluate a full split, never average minibatch APs.
    """
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or labels.size == 0:
        raise ValueError('nonempty matching 1D labels and scores required')
    if not np.isin(labels, [-1, 0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError('raw labels must be -1/0/1 and scores finite')
    keep = labels != 0
    labels, scores = labels[keep], scores[keep]
    positives = np.count_nonzero(labels == 1)
    if positives == 0:
        return float('nan')
    ordered = labels[np.argsort(-scores, kind='stable')]
    tp, fp = np.cumsum(ordered == 1), np.cumsum(ordered == -1)
    recall = tp / positives
    precision = tp / (tp + fp)
    recall = np.r_[0., recall, 1.]
    precision = np.r_[0., precision, 0.]
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changes = np.flatnonzero(recall[1:] != recall[:-1])
    return float(np.sum((recall[changes+1]-recall[changes])*precision[changes+1]))


def classification_map(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.ndim != 2 or scores.shape != labels.shape or min(labels.shape) == 0:
        raise ValueError('matching nonempty [N,C] arrays required')
    ap = np.array([voc_ap(labels[:, c], scores[:, c]) for c in range(labels.shape[1])])
    # Strict policy: do not silently exclude missing classes from macro mean.
    return {'ap': ap, 'map': float(ap.mean())}
