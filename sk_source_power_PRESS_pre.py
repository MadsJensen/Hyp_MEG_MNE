"""
Created on Wed June 20 2015

@author: mje
"""
import os
import socket
import numpy as np
import mne
import csv
from mne.minimum_norm import (read_inverse_operator,
                              source_band_induced_power)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import (StratifiedShuffleSplit,
                                      permutation_test_score)

# Setup paths and prepare raw data
hostname = socket.gethostname()

if hostname == "Wintermute":
    data_path = "/home/mje/mnt/Hyp_meg/scratch/Tone_task_MNE/"
    subjects_dir = "/home/mje/mnt/Hyp_meg/scratch/fs_subjects_dir/"
    n_jobs = 1
elif hostname == "isis":
    data_path = "/scratch1/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                "Tone_task_MNE/"
    subjects_dir = "/scratch1/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                   "fs_subjects_dir"
    n_jobs = 1
else:
    raise RuntimeWarning('Unknown host')

# change dir to save files the rigth place
os.chdir(data_path)

###############################################################################
epochs_fnormal = data_path + "tone_task_normal-epo.fif"
epochs_fhyp = data_path + "tone_task_hyp-epo.fif"
inverse_fnormal = data_path + "tone_task_normal-inv.fif"
inverse_fhyp = data_path + "tone_task_hyp-inv.fif"

inverse_normal = read_inverse_operator(inverse_fnormal)
inverse_hyp = read_inverse_operator(inverse_fhyp)

epochs_normal = mne.read_epochs(epochs_fnormal)
epochs_hyp = mne.read_epochs(epochs_fhyp)

epochs_normal = epochs_normal["press"]
epochs_hyp = epochs_hyp["press"]

epochs_normal.resample(250)
epochs_hyp.resample(250)


label_dir = subjects_dir + "/subject_1/label/"
labels = mne.read_labels_from_annot('subject_1', parc='aparc.a2009s',
                                    regexp="[G|S]",
                                    subjects_dir=subjects_dir)
label_single = [labels[3]]

#
snr = 1.0  # Standard assumption for average data but using it for single trial
lambda2 = 1.0 / snr ** 2
method = "dSPM"
bands = dict(alpha=[9, 11], beta=[18, 22], gamme_low=[30, 48],
             gamma_high=[52, 88])

band = {bands.keys()[0]: bands.values()[0]}

stcs_normal = []
for j in range(len(epochs_normal)):
    print "*********************\n"
    print "working on %d of %d\n" % (j, len(epochs_normal))
    print "*********************"

    stcs_normal += [source_band_induced_power(epochs_normal[j],
                                              inverse_normal,
                                              band,
                                              n_cycles=2,
                                              use_fft=False,
                                              baseline=(None, -0.7),
                                              baseline_mode="zscore",
                                              n_jobs=n_jobs)]

stcs_hyp = []
for j in range(len(epochs_hyp)):
    print "*********************\n"
    print "working on %d of %d" % (j, len(epochs_hyp))
    print "*********************\n"

    stcs_hyp += [source_band_induced_power(epochs_hyp[j],
                                           inverse_hyp,
                                           band,
                                           n_cycles=2,
                                           use_fft=False,
                                           baseline=(None, -0.7),
                                           baseline_mode="zscore",
                                           n_jobs=n_jobs)]


# Classification setting
n_splits = 10
LR = LogisticRegression()
gnb = GaussianNB()

classifiers = [LR]
clf_names = ["LR"]
# score_results[label.name] = score
#     p_results[label.name] = pvalue
#
#     outfile_p_name = "p_results_DA_press_TFR_" +\
#         "dSPM_-02-0_%s_nostd_mean_flip.csv" % clf_names[h]
#     outfile_score_name = "score_results_DA_press_TFR_" +\
#         "dSPM_-02-0_%s_nostd_mean_flip.csv" % clf_names[h]
#
#     with open(outfile_p_name, "w") as outfile:
#         writer = csv.writer(outfile)
#         for key, val in p_results.items():
#             writer.writerow([key, val])
#
#     with open(outfile_score_name, "w") as outfile:
#         writer = csv.writer(outfile)
#         for key, val in score_results.items():
#             writer.writerow([key, val])
