# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 08:41:17 2015

@author: mje
"""
import numpy as np
import numpy.random as npr
import os
import socket
import mne
import pandas as pd

from mne.connectivity import spectral_connectivity


# from mne.stats import fdr_correction


# %% Permutation test.
def permutation_resampling(case, control, num_samples, statistic):
    """Returns p-value that statistic for case is different
    from statistc for control."""

    observed_diff = abs(statistic(case) - statistic(control))
    num_case = len(case)

    combined = np.concatenate([case, control])
    diffs = []
    for i in range(num_samples):
        xs = npr.permutation(combined)
        diff = np.mean(xs[:num_case]) - np.mean(xs[num_case:])
        diffs.append(diff)

    pval = (np.sum(diffs > observed_diff) +
            np.sum(diffs < -observed_diff))/float(num_samples)
    return pval, observed_diff, diffs


def permutation_test(a, b, num_samples, statistic):

    """Returns p-value that statistic for a is different
    from statistc for b."""

    observed_diff = abs(statistic(b) - statistic(a))
    num_a = len(a)

    combined = np.concatenate([a, b])
    diffs = []
    for i in range(num_samples):
        xs = npr.permutation(combined)
        diff = np.mean(xs[:num_a]) - np.mean(xs[num_a:])
        diffs.append(diff)

    pval = np.sum(np.abs(diffs) >= np.abs(observed_diff)) / float(num_samples)
    return pval, observed_diff, diffs


# %% Setup paths and prepare raw data
hostname = socket.gethostname()

if hostname == "Wintermute":
    data_path = "/home/mje/mnt/Hyp_meg/scratch/Tone_task_MNE/"
    subjects_dir = "/home/mje/mnt/Hyp_meg/scratch/fs_subjects_dir/"
else:
    data_path = "/scratch1/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                "Tone_task_MNE/"
    subjects_dir = "/scratch1/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                   "fs_subjects_dir"

# change dir to save files the rigth place
os.chdir(data_path)

# load numpy files
labelTsHypCrop =\
    np.load("labels_ts_hyp_press_post_mean-flip_zscore_resample_crop_BA.npy")
labelTsNormalCrop =\
    np.load("labels_ts_normal_press_post_mean-flip_" +
            "zscore_resample_crop_BA.npy")

# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
labels = mne.read_labels_from_annot('subject_1', parc='PALS_B12_Brodmann',
                                    regexp="Brodmann",
                                    subjects_dir=subjects_dir)

# labels = mne.read_labels_from_annot('subject_1', parc='aparc.DKTatlas40',
#                                     subjects_dir=subjects_dir)

labels_name = []
for label in labels:
    labels_name += [label.name]


ts_all = np.vstack([labelTsNormalCrop,labelTsHypCrop])

number_of_permutations = 2
number_of_repititions = 2
index = np.arange(0, 154)
permutations_results = np.empty(number_of_repititions)
fmin, fmax = 8, 12


for i in range(number_of_repititions):
    np.random.shuffle(index)
    tmp_ctl = ts_all[index[:80]]
    tmp_case = ts_all[index[80:]]
    
    con_ctl, freqs_ctl, times_ctl, n_epochs_ctl, n_tapers_ctl =\
        spectral_connectivity(
             tmp_ctl, method='plv', 
             mode='multitaper', 
             sfreq=250,
             fmin=fmin, fmax=fmax,
             faverage=True,
             tmin=0, tmax=0.5,
             mt_adaptive=False,
             n_jobs=1)
             
    con_case, freqs_case, times_case, n_epochs_case, n_tapers_case =\
        spectral_connectivity(
             tmp_case, method='plv', 
             mode='multitaper', 
             sfreq=250,
             fmin=fmin, fmax=fmax,
             faverage=True,
             tmin=0, tmax=0.5,
             mt_adaptive=False,
             n_jobs=1)
             
    pval, obs_diff, diffs =\
        permutation_resampling(con_case, con_ctl,
                               number_of_permutations,
                               np.mean)
                               
   permutation_results[i] = pval
                              

