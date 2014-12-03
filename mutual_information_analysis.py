import numpy as np
import networkx as nx
import os
import socket
import mne

from sklearn.metrics import mutual_info_score


def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


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
fromTime = np.argmax(epochs_normal.times == 0)
toTime = np.argmax(epochs_normal.times == 0.7)


labelTsHypZscore = np.load("labelTsHypZscore.npy")
labelTsNormalZscore = np.load("labelTsNormalZscore.npy")

labelTsNormalZscoreCrop = labelTsNormalZscore[:, :, fromTime:toTime]
labelTsHypZscoreCrop = labelTsHypZscore[:, :, fromTime:toTime]

n_trials = labelTsNormalZscoreCrop.shape[0]
n_labels = labelTsNormalZscoreCrop.shape[1]
MI_results_normal = np.empty([n_labels, n_labels, n_trials])

for h in range(n_trials):
    counter = 0
    tmpResult = np.empty([n_labels * n_labels])
    for j in range(n_labels):
        for k in range(n_labels):
            tmpResult[counter] = (calc_MI(labelTsNormalZscoreCrop[h, j, :],
                                          labelTsNormalZscoreCrop[h, k, :], 20)) \
                / np.log(2)
            counter += 1
    MI_results_normal[:, :, h] = np.reshape(tmpResult, [n_labels, n_labels])

for j in range(n_trials):
    FDbinSize(labelTsNormalZscoreCrop[0, j, :])
