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
# from sklearn import preprocessing

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

epochs_normal = epochs_normal["Tone"]
epochs_hyp = epochs_hyp["Tone"]

epochs_normal.resample(250)
epochs_hyp.resample(250)

src_normal = inverse_normal['src']
src_hyp = inverse_hyp['src']

label_dir = subjects_dir + "/subject_1/label/"
# labels = mne.read_labels_from_annot('subject_1', parc='aparc.a2009s',
#                                     regexp="[G|S]",
#                                     subjects_dir=subjects_dir)
labels = mne.read_labels_from_annot('subject_1', parc='PALS_B12_Brodmann',
                                    regexp="Bro",
                                    subjects_dir=subjects_dir)

#
snr = 1.0  # Standard assumption for average data but using it for single trial
lambda2 = 1.0 / snr ** 2
method = "MNE"
bands = dict(alpha=[8, 13], beta=[13, 30], gamma_low=[30, 48],
             gamma_high=[52, 88])

band = {bands.keys()[0]: bands.values()[0]}

for h in range(len(bands)):
    band = {bands.keys()[h]: bands.values()[h]}

    stcs_normal = []
    for j in range(len(epochs_normal)):
        print "\n*********************"
        print "working on %d of %d" % (j+1, len(epochs_normal))
        print "*********************\n"

        stcs_normal += [source_band_induced_power(epochs_normal[j],
                                                  inverse_normal,
                                                  band,
                                                  n_cycles=2,
                                                  lambda2=lambda2,
                                                  method=method,
                                                  use_fft=False,
                                                  baseline=(None, -0.7),
                                                  baseline_mode="zscore",
                                                  n_jobs=n_jobs)]

    stcs_hyp = []
    for j in range(len(epochs_hyp)):
        print "\n*********************"
        print "working on %d of %d" % (j+1, len(epochs_hyp))
        print "*********************\n"

        stcs_hyp += [source_band_induced_power(epochs_hyp[j],
                                               inverse_hyp,
                                               band,
                                               n_cycles=2,
                                               lambda2=lambda2,
                                               method=method,
                                               use_fft=False,
                                               baseline=(None, -0.7),
                                               baseline_mode="zscore",
                                               n_jobs=n_jobs)]

    [stc[band.keys()[0]].crop(-0.5, 0) for stc in stcs_normal]
    [stc[band.keys()[0]].crop(-0.5, 0) for stc in stcs_hyp]

    # Classification setting
    n_splits = 10
    LR = LogisticRegression()
    gnb = GaussianNB()

    clf = gnb
    clf_names = ["GNB"]

    p_results = {}
    score_results = {}

    stcs_normal = [stc[band.keys()[0]] for stc in stcs_normal]
    stcs_hyp = [stc[band.keys()[0]] for stc in stcs_hyp]

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
        # X_pre = prepro;cessing.scale(X)
        cv = StratifiedShuffleSplit(y, n_splits)
        print "Working on: %s in band: %s" % (label.name, band.keys()[0])

        score, permutation_scores, pvalue =\
            permutation_test_score(
                clf, X, y, scoring="accuracy",
                cv=cv, n_permutations=5000,
                n_jobs=n_jobs)

        score_results[label.name] = score
        p_results[label.name] = pvalue

        outfile_p_name = "p_results_DA_tone_power" +\
            "_%s_MNE_-05-0_%s_nostd_mean_flip.csv" % (band.keys()[0],
                                                      clf_names[0])
        outfile_score_name = "score_results_DA_tone_power" +\
            "_%s_MNE_-05-0_%s_nostd_mean_flip.csv" % (band.keys()[0],
                                                      clf_names[0])

        with open(outfile_p_name, "w") as outfile:
            writer = csv.writer(outfile)
            for key, val in p_results.items():
                writer.writerow([key, val])

        with open(outfile_score_name, "w") as outfile:
            writer = csv.writer(outfile)
            for key, val in score_results.items():
                writer.writerow([key, val])
