# -*- coding: utf-8 -*-
"""
@author: mje
"""

from mne.minimum_norm import read_inverse_operator, apply_inverse_epochs
# from mne.baseline import rescale
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import roc_curve, auc
from scipy import interp
from mne.stats import fdr_correction

import pandas as pd
import mne
import os
import socket
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def load_result(fname):
    """
    Keyword Arguments:
    name -- the file to be loaded.
        """

    result_clf = pd.read_csv(
        fname,
        header=None)
    result_clf.columns = ["ROI", "pval"]  # rename columns
    result_clf = result_clf.sort("ROI")

    res_score = pd.read_csv(
        "score_" + fname[2:],
        header=None)

    result_clf["score"] = res_score[1]
    result_clf["rejected"], result_clf["pval_corr"] =\
        fdr_correction(result_clf["pval"])
    result_clf.index = range(0, len(result_clf))

    result_clf["rejected"], result_clf["pval_corr"] =\
        fdr_correction(result_clf["pval"])

    ROIs = [roi[:-3] for roi in result_clf.ROI]
    hemi = [roi[-2:] for roi in result_clf.ROI]
    result_clf["hemi"] = hemi
    result_clf.ROI = ROIs

    return result_clf


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
    n_jobs = 1

result_dir = data_path + "/class_result"

# setup clf
n_folds = 10
LR = LogisticRegression()

Cs = np.logspace(-4, 4, 100)

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

# Resample
[stc.resample(250) for stc in stcs_normal]
[stc.resample(250) for stc in stcs_hyp]

# Crop
[stc.crop(0, 0.5) for stc in stcs_normal]
[stc.crop(0, 0.5) for stc in stcs_hyp]

label_dir = subjects_dir + "/subject_1/label/"
labels = mne.read_labels_from_annot('subject_1', parc='aparc.a2009s',
                                    regexp="[G|S]",
                                    subjects_dir=subjects_dir)
labels_single = [labels[26]]

# Load results from classification
press_pre_clf = load_result(
    "p_results_DA_press_surf-normal_MNE_-02-0_LR_std_mean.csv")
press_pre_index =\
    press_pre_clf[press_pre_clf["rejected"] == True].index.get_values()
press_pre_labels = [labels[index] for index in press_pre_index]

press_post_clf = load_result(
    "p_results_DA_press_surf-normal_MNE_0-05_LR_std_mean.csv")
press_post_index =\
    press_post_clf[press_post_clf["rejected"] == True].index.get_values()
press_post_labels = [labels[index] for index in press_post_index]


# CLassifier
classifiers = [LR]
clf_names = ["LR"]

for h, clf in enumerate(classifiers):
    p_results = {}
    score_results = {}
    for label in press_post_labels:
        labelTsNormal = mne.extract_label_time_course(stcs_normal,
                                                      labels=label,
                                                      src=src_normal,
                                                      mode='mean',
                                                      return_generator=False)

        labelTsHyp = mne.extract_label_time_course(stcs_hyp,
                                                   labels=label,
                                                   src=src_hyp,
                                                   mode='mean',
                                                   return_generator=False)

        X = np.vstack([labelTsNormal, labelTsHyp])
        X = X[:, 0, :]
        y = np.concatenate([np.zeros(len(labelTsNormal)),
                            np.ones(len(labelTsHyp))])

        X = X * 1e11
        X = preprocessing.scale(X)
        cv = StratifiedShuffleSplit(y, test_size=0.2)
        print "Working on: ", label.name

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = []
        plt.figure()

        # logistic = LogisticRegression(C=grid.best_params_["C"])
        for i, (train, test) in enumerate(cv):
            probas_ = LR.fit(X[train], y[train]).predict_proba(X[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1,
                     label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

        mean_tpr /= len(cv)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, 'k--',
                 label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        # FIXME: make sure plot is not plotting on top of each other
#        plt.savefig(result_dir + label.name + "_MNE_press_pre.jpg")
