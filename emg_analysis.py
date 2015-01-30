import os
import socket
import numpy as np
from scipy import stats
# from sklearn import preprocessing
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.cross_validation import ShuffleSplit, permutation_test_score

# Setup paths and prepare raw data
hostname = socket.gethostname()

if hostname == "Wintermute":
    data_path = "/home/mje/mnt/Hyp_meg/scratch/Tone_task_MNE/"
    script_path = "/home/mje/mnt/Hyp_meg/scripts/MNE_analysis/"
    subjects_dir = "/home/mje/mnt/Hyp_meg/scratch/fs_subjects_dir/"
    n_jobs = 3
else:
    data_path = "/scratch1/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                "Tone_task_MNE/"
    script_path = "/projects/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                  "scripts/MNE_analysis/"
    subjects_dir = "/scratch1/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                   "fs_subjects_dir"
    n_jobs = 6


os.chdir(data_path)


def peak_to_vally(data):
    """ Find the difference between the max and min value
    for a time series.
    """
    result = np.empty(len(data))
    for j in range(len(data)):
        result[j] = np.max(data[j, :]) - np.min(data[j, :])

    return result


# load data
emg_normal = np.load("emg_normal.npy")
emg_hyp = np.load("emg_hyp.npy")

p2v_normal = peak_to_vally(emg_normal)
p2v_hyp = peak_to_vally(emg_hyp)

t_val, p_val = stats.ttest_ind(p2v_normal, p2v_hyp, equal_var=False)


