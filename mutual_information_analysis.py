import numpy as np
import networkx as nx
import os
import socket
import mne

from sklearn.metrics import mutual_info_score


def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi /np.log(2)


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

# Setup paths and prepare raw data
hostname = socket.gethostname()

if hostname == "wintermute":
    data_path = "/home/mje/mnt/Hyp_meg/scratch/Tone_task_MNE/"
    subjects_dir = "/home/mje/mnt/Hyp_meg/scratch/fs_subjects_dir/"
else:
    data_path = "/scratch1/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                "Tone_task_MNE/"
    subjects_dir = "/scratch1/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                   "fs_subjects_dir"

# change dir to save files the rigth place
os.chdir(data_path)

# load numpy files; crop
epochs_fnormal = data_path + "tone_task_normal-epo.fif"
epochs_normal = mne.read_epochs(epochs_fnormal)

# crop zscored TS
fromTime = np.argmax(epochs_normal.times == -0.5)
toTime = np.argmax(epochs_normal.times == -0.01)


labelTsHypZscore = np.load("labelTsHypZscore.npy")
labelTsNormalZscore = np.load("labelTsNormalZscore.npy")

labelTsNormalZscoreCrop = labelTsNormalZscore[:, :, fromTime:toTime]
labelTsHypZscoreCrop = labelTsHypZscore[:, :, fromTime:toTime]


n_trials_normal = labelTsNormalZscoreCrop.shape[0]
n_labels_normal = labelTsNormalZscoreCrop.shape[1]
MI_results_normal = np.empty([n_labels_normal, n_labels_normal, n_trials_normal])

n_trials_hyp = labelTsHypZscoreCrop.shape[0]
n_labels_hyp = labelTsHypZscoreCrop.shape[1]
MI_results_hyp = np.empty([n_labels_hyp, n_labels_hyp, n_trials_hyp])

# calculate the number of bins
bins = np.empty(0)
for t in range(n_trials_normal):
    for l in range(n_labels_normal):
        bins = np.append(bins, FDbinSize(labelTsNormalZscoreCrop[t, l, :]))
for t in range(n_trials_hyp):
    for l in range(n_labels_hyp):
        bins = np.append(bins, FDbinSize(labelTsHypZscoreCrop[t, l, :]))

bestBinsize = np.ceil(np.mean(bins))

# calc MI for normal
for h in range(n_trials_normal):
    counter = 0
    tmpResult = np.empty([n_labels_normal * n_labels_normal])
    for j in range(n_labels_normal):
        for k in range(n_labels_normal):
            tmpResult[counter] = (calc_MI(labelTsNormalZscoreCrop[h, j, :],
                                          labelTsNormalZscoreCrop[h, k, :],
                                          bestBinsize))
            counter += 1
    MI_results_normal[:, :, h] = np.reshape(tmpResult, [n_labels_normal,
                                            n_labels_normal])

# calc MI for Hyp
for h in range(n_trials_hyp):
    counter = 0
    tmpResult = np.empty([n_labels_hyp * n_labels_hyp])
    for j in range(n_labels_hyp):
        for k in range(n_labels_hyp):
            tmpResult[counter] = (calc_MI(labelTsHypZscoreCrop[h, j, :],
                                          labelTsHypZscoreCrop[h, k, :],
                                          bestBinsize))
            counter += 1
    MI_results_hyp[:, :, h] = np.reshape(tmpResult, [n_labels_hyp,
                                            n_labels_hyp])

