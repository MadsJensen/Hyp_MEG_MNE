# -*- coding: utf-8 -*-
"""
Created on Wed May 21 15:21:02 2014

@author: mje
"""

import mne
import os
import csv
import socket
import numpy as np

from mne.minimum_norm import read_inverse_operator, apply_inverse_epochs
from mne.baseline import rescale
# from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import (ShuffleSplit, permutation_test_score)

# Setup paths and prepare raw data
hostname = socket.gethostname()

if hostname == "Wintermute":
    data_path = "/home/mje/mnt/Hyp_meg/scratch/Tone_task_MNE/"
    script_path = "/home/mje/mnt/Hyp_meg/scripts/MNE_analysis/"
    subjects_dir = "/home/mje/mnt/Hyp_meg/scratch/fs_subjects_dir/"
    n_jobs = 1
else:
    data_path = "/scratch1/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                "Tone_task_MNE/"
    script_path = "/projects/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                  "scripts/MNE_analysis/"
    subjects_dir = "/scratch1/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                   "fs_subjects_dir"
    n_jobs = 4

result_dir = data_path + "/class_result"

# setup clf
n_splits = 10
ngb = GaussianNB()
LR = LogisticRegression()

os.chdir(data_path)

epochs_fnormal = data_path + "tone_task_normal-epo.fif"
epochs_fhyp = data_path + "tone_task_hyp-epo.fif"
inverse_fnormal = data_path + "tone_task_normal-inv.fif"
inverse_fhyp = data_path + "tone_task_hyp-inv.fif"

epochs_normal = mne.read_epochs(epochs_fnormal)
epochs_hyp = mne.read_epochs(epochs_fhyp)

epochs_normal = epochs_normal["press"]
epochs_hyp = epochs_hyp["press"]


snr = 1.0  # Standard assumption for average data but using it for single trial
lambda2 = 1.0 / snr ** 2
method = "MNE"

# Load data
inverse_normal = read_inverse_operator(inverse_fnormal)
inverse_hyp = read_inverse_operator(inverse_fhyp)
src_normal = inverse_normal['src']
src_hyp = inverse_hyp['src']

stcs_normal = apply_inverse_epochs(epochs_normal, inverse_normal,
                                   lambda2, method,
                                   pick_ori="normal",
                                   return_generator=False)

stcs_hyp = apply_inverse_epochs(epochs_hyp, inverse_hyp,
                                lambda2, method,
                                pick_ori="normal",
                                return_generator=False)

# resample
[stc.resample(250) for stc in stcs_normal]
[stc.resample(250) for stc in stcs_hyp]

label_dir = subjects_dir + "/subject_1/label/"

# labels = mne.read_labels_from_annot('subject_1', parc='aparc.DKTatlas40',
#                                     subjects_dir=subjects_dir)
labels = mne.read_labels_from_annot('subject_1', parc='PALS_B12_Brodmann',
                                    regexp="Bro",
                                    subjects_dir=subjects_dir)
labels_name = [label.name for label in labels]

classifiers = [LR]
clf_names = ["LR"]

for h, clf in enumerate(classifiers):
    p_results = {}
    score_results = {}

    for label in labels:
        labelTsNormal = mne.extract_label_time_course(stcs_normal,
                                                      labels=label,
                                                      src=src_normal,
                                                      mode='mean_flip',
                                                      return_generator=False)

        labelTsHyp = mne.extract_label_time_course(stcs_hyp,
                                                   labels=label,
                                                   src=src_hyp,
                                                   mode='mean_flip',
                                                   return_generator=False)

        labelTsNormalRescaled = []
        for j in range(len(labelTsNormal)):
            labelTsNormalRescaled += [rescale(labelTsNormal[j],
                                              stcs_normal[0].times,
                                              baseline=(None, -0.7),
                                              mode="zscore")]

        labelTsHypRescaled = []
        for j in range(len(labelTsHyp)):
            labelTsHypRescaled += [rescale(labelTsHyp[j],
                                           stcs_hyp[0].times,
                                           baseline=(None, -0.7),
                                           mode="zscore")]

        fromTime = np.argmax(stcs_normal[0].times == 0)
        toTime = np.argmax(stcs_normal[0].times == 0.5)

        labelTsNormalRescaledCrop = []
        for j in range(len(labelTsNormal)):
            labelTsNormalRescaledCrop +=\
                [labelTsNormalRescaled[j][:, fromTime:toTime]]

        labelTsHypRescaledCrop = []
        for j in range(len(labelTsHyp)):
            labelTsHypRescaledCrop +=\
                [labelTsHypRescaled[j][:, fromTime:toTime]]

        X = np.vstack([labelTsNormalRescaledCrop, labelTsHypRescaledCrop])
        X = X[:, 0, :]
        y = np.concatenate([np.zeros(len(labelTsNormalRescaledCrop)),
                            np.ones(len(labelTsHypRescaledCrop))])

#        X = preprocessing.scale(X)
        cv = ShuffleSplit(len(X), n_splits, test_size=0.2)
        print "Working on: ", label.name

        score, permutation_scores, pvalue =\
            permutation_test_score(
                clf, X, y, scoring="accuracy",
                cv=cv, n_permutations=5000,
                n_jobs=n_jobs)

        score_results[label.name] = score
        p_results[label.name] = pvalue

    outfile_p_name = "p_results_BA_press_surf-normal_" +\
        "MNE_zscore_0-05_%s_no-std.csv" % clf_names[h]
    outfile_score_name = "score_results_BA_press_surf-normal_" +\
        "MNE_zscore_0-05_%s_no-std.csv" % clf_names[h]

    with open(outfile_p_name, "w") as outfile:
        writer = csv.writer(outfile)
        for key, val in p_results.items():
            writer.writerow([key, val])

    with open(outfile_score_name, "w") as outfile:
        writer = csv.writer(outfile)
        for key, val in score_results.items():
            writer.writerow([key, val])
