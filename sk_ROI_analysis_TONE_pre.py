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
# from mne.baseline import rescale
# from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
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
    n_jobs = 3

result_dir = data_path + "/class_result"

# setup clf
n_splits = 10
LR = LogisticRegression()

os.chdir(data_path)

epochs_fnormal = data_path + "tone_task_normal-epo.fif"
epochs_fhyp = data_path + "tone_task_hyp-epo.fif"
inverse_fnormal = data_path + "tone_task_normal-inv.fif"
inverse_fhyp = data_path + "tone_task_hyp-inv.fif"

epochs_normal = mne.read_epochs(epochs_fnormal)
epochs_hyp = mne.read_epochs(epochs_fhyp)

epochs_normal = epochs_normal["Tone"]
epochs_hyp = epochs_hyp["Tone"]


snr = 1.0  # Standard assumption for average data but using it for single trial
lambda2 = 1.0 / snr ** 2
method = "dSPM"

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

# Resample
[stc.resample(250) for stc in stcs_normal]
[stc.resample(250) for stc in stcs_hyp]

# Crop
[stc.crop(-0.5, 0) for stc in stcs_normal]
[stc.crop(-0.5, 0) for stc in stcs_hyp]

label_dir = subjects_dir + "/subject_1/label/"

labels = mne.read_labels_from_annot('subject_1', parc='aparc.a2009s',
                                    regexp="[G|S]",
                                    subjects_dir=subjects_dir)
# labels = mne.read_labels_from_annot('subject_1', parc='PALS_B12_Brodmann',
#                                     regexp="Bro",
#                                     subjects_dir=subjects_dir)

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

        X = np.vstack([labelTsNormal, labelTsHyp])
        X = X[:, 0, :]
        y = np.concatenate([np.zeros(len(labelTsNormal)),
                            np.ones(len(labelTsHyp))])

        # X = X * 1e11
        # X_pre = preprocessing.scale(X)
        cv = ShuffleSplit(len(X), n_splits, test_size=0.2)
        print "Working on: ", label.name

        score, permutation_scores, pvalue =\
            permutation_test_score(
                clf, X, y, scoring="accuracy",
                cv=cv, n_permutations=5000,
                n_jobs=n_jobs)

        score_results[label.name] = score
        p_results[label.name] = pvalue

    outfile_p_name = "p_results_DA_tone_surf-normal_" +\
        "dSPM_-05-0_%s_nostd_mean_flip.csv" % clf_names[h]
    outfile_score_name = "score_results_DA_tone_surf-normal_" +\
        "dSPM_-05-0_%s_nostd_mean_flip.csv" % clf_names[h]

    with open(outfile_p_name, "w") as outfile:
        writer = csv.writer(outfile)
        for key, val in p_results.items():
            writer.writerow([key, val])

    with open(outfile_score_name, "w") as outfile:
        writer = csv.writer(outfile)
        for key, val in score_results.items():
            writer.writerow([key, val])
