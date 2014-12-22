import numpy as np

from sklearn.metrics import mutual_info_score


def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi / np.log(2)


def FDbinSize(X):
    """Calculates the Freedman-Diaconis bin size for
    a data set for use in making a histogram

    Arguments:
    X:  1D Data set

    Returns:
    h:  F-D bin size
    """

    # First Calculate the interquartile range
    X = np.sort(X)
    maxmin_range = X.max() - X.min()
    IQR = np.subtract(*np.percentile(X, [75, 25]))

    # Find the F-D bin size
    h = np.ceil(maxmin_range / (2.*IQR/len(X)**(1./3.)))
    return h
