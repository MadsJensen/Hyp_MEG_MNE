import numpy as np
import os
import socket
import mne
import networkx as nx
from mne.stats import fdr_correction
import cPickle as pickle

# Setup paths and prepare raw data
hostname = socket.gethostname()

if hostname == "Wintermute":
    data_path = "/home/mje/mnt/Hyp_meg/scratch/Tone_task_MNE/"
    script_path = "/home/mje/mnt/Hyp_meg/scripts/MNE_analysis/"
    subjects_dir = "/home/mje/mnt/Hyp_meg/scratch/fs_subjects_dir/"
else:
    data_path = "/scratch1/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                "Tone_task_MNE/"
    script_path = "/projects/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                  "scripts/MNE_analysis/"
    subjects_dir = "/scratch1/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                   "fs_subjects_dir"

# Import MI functions
os.chdir(script_path)
from MI_functions import calc_MI, FDbinSize
from permTest import permutation_resampling

# change dir to save files the rigth place
os.chdir(data_path)

# load numpy files; crop
# epochs_fnormal = data_path + "tone_task_normal-epo.fif"
# epochs_normal = mne.read_epochs(epochs_fnormal)
# epochs_normal = epochs_normal["press"]

# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
# labels = mne.read_labels_from_annot('subject_1', parc='PALS_B12_Brodmann',
#                                     regexp="Brodmann",
#                                     subjects_dir=subjects_dir)

labels = mne.read_labels_from_annot('subject_1', parc='aparc.DKTatlas40',
                                    subjects_dir=subjects_dir)

labels_name = []
for label in labels:
    labels_name += [label.name]

# crop zscored TS
# fromTime = np.argmax(epochs_normal.times == -0.5)
# toTime = np.argmax(epochs_normal.times == 0)

label_ts_hyp = np.load("labelTsHypToneMean-flipZscore_resample_crop.npy")

label_ts_normal =\
    np.load("labelTsNormalToneMean-flipZscore_resample_crop.npy")
# label_ts_normal_crop = label_ts_normal[:, :, fromTime:toTime]
# label_ts_hyp_crop = label_ts_hyp[:, :, fromTime:toTime]


n_trials_normal = 80  # label_ts_normal_crop.shape[0]
n_labels = len(labels)  # label_ts_normal_crop.shape[1]
MI_results_normal = np.empty([n_labels, n_labels,
                              n_trials_normal])

n_trials_hyp = 74  # label_ts_hyp_crop.shape[0]
MI_results_hyp = np.empty([n_labels, n_labels, n_trials_hyp])

# calculate the number of bins
bins = np.empty(0)
for t in range(80):
    for l in range(82):
        bins = np.append(bins, FDbinSize(label_ts_normal[t][l, :]))
for t in range(74):
    for l in range(82):
        bins = np.append(bins, FDbinSize(label_ts_hyp[t][l, :]))

bestBinsize = np.ceil(np.mean(bins))

# calc MI for normal
for h in range(n_trials_normal):
    counter = 0
    print "Normal #: ", h
    tmpResult = np.empty([n_labels * n_labels])
    for j in range(n_labels):
        for k in range(n_labels):
            tmpResult[counter] = (calc_MI(label_ts_normal[h][j, :],
                                          label_ts_normal[h][k, :],
                                          bestBinsize))
            counter += 1
    MI_results_normal[:, :, h] = np.reshape(tmpResult, [n_labels,
                                            n_labels])

# calc MI for Hyp
for h in range(n_trials_hyp):
    counter = 0
    print "Hyp #: ", h
    tmpResult = np.empty([n_labels * n_labels])
    for j in range(n_labels):
        for k in range(n_labels):
            tmpResult[counter] = (calc_MI(label_ts_hyp[h][j, :],
                                          label_ts_hyp[h][k, :],
                                          bestBinsize))
            counter += 1
    MI_results_hyp[:, :, h] = np.reshape(tmpResult, [n_labels,
                                                     n_labels])

# Convert to networkx classes
fullMatrix = np.concatenate([MI_results_normal, MI_results_hyp], axis=2)

threshold = np.median(fullMatrix[np.nonzero(fullMatrix)]) + \
    (np.std(fullMatrix[np.nonzero(fullMatrix)]))

binMatrixNormal = MI_results_normal > threshold
binMatrixHyp = MI_results_hyp > threshold

# %%
print "\n************* \nMaking network classes\n*************"
nxNormal = []
for j in range(binMatrixNormal.shape[2]):
    nxNormal += [nx.from_numpy_matrix(binMatrixNormal[:, :, j])]

nxHyp = []
for j in range(binMatrixHyp.shape[2]):
    nxHyp += [nx.from_numpy_matrix(binMatrixHyp[:, :, j])]

degreesNormal = []
for j, trial in enumerate(nxNormal):
    degreesNormal += [trial.degree()]

degreesHyp = []
for j, trial in enumerate(nxHyp):
    degreesHyp += [trial.degree()]

ccNormal = []
for j, trial in enumerate(nxNormal):
    ccNormal += [nx.cluster.clustering(trial)]
ccHyp = []
for j, trial in enumerate(nxHyp):
    ccHyp += [nx.cluster.clustering(trial)]

# %% Degress
print "\n************* \nTesting degrees\n*************"
pvalList = []
for degreeNumber in range(binMatrixHyp.shape[0]):

    postHyp = np.empty(len(degreesHyp))
    for j in range(len(postHyp)):
        postHyp[j] = degreesHyp[j][degreeNumber]

    postNormal = np.empty(len(degreesNormal))
    for j in range(len(postNormal)):
        postNormal[j] = degreesNormal[j][degreeNumber]

    pval, observed_diff, diffs = \
        permutation_resampling(postHyp, postNormal,
                               10000, np.mean)

    pvalList  += [{'area': labels_name[degreeNumber],
                   'pval': pval,
                   "obsDiff": observed_diff,
                   "diffs": diffs}]

    pickle.dump(pvalList,
                open("MI_tone_zscore_DKT_-05-0_resample_crop_deg.p", "wb"))
# %% for CC
print "\n************* \nTesting cluster-coefficient\n*************"
pvalListCC = []
for ccNumber in range(binMatrixHyp.shape[0]):

    postHyp = np.empty(len(ccHyp))
    for j in range(len(ccHyp)):
        postHyp[j] = ccHyp[j][ccNumber]

    postNormal = np.empty(len(ccNormal))
    for j in range(len(postNormal)):
        postNormal[j] = ccNormal[j][ccNumber]

    pval, observed_diff, diffs = \
        permutation_resampling(postNormal, postHyp,
                               10000, np.mean)

    pvalListCC  += [{'area': labels_name[degreeNumber],
                      'pval': pval,
                      "obsDiff": observed_diff,
                      "diffs": diffs}]
        
    pickle.dump(pvalListCC,
                open("MI_tone_zscore_DKT_-05-0_resample_crop_CC.p", "wb"))
    

# pvals = np.empty(len(pvalList))
# for j in range(len(pvals)):
#     pvals[j] = pvalList[j]["pval"]

# rejected, pvals_corrected = fdr_correction(pvals)

# print "\nSignificient regions for Degrees:"
# for i in range(len(labels_name)):
#     if rejected[i] and pvalList[i]["obsDiff"] != 0:
#         print "\n", labels_name[i], \
#             "pval:", pvals_corrected[i], \
#             "observed differnce:", pvalList[i]["obsDiff"], \

# results_degrees = []
# for i in range(len(labels_name)):
#     if rejected[i] and pvalList[i]["obsDiff"] != 0:
#         results_degrees += [{"label": labels_name[i],
#                             "pval_corr": pvals_corrected[i],
#                              "obs_diff":
#                              pvalListCC[i]["obsDiff"],
#                              "mean_random_diff":
#                              np.asarray(pvalListCC[i]["diffs"]).mea# ficient (CC)
# pvalsCC = np.empty(len(pvalListCC))
# for j in range(len(pvalsCC)):
#     pvalsCC[j] = pvalListCC[j]["pval"]

# rejectedCC, pvals_correctedCC = fdr_correction(pvalsCC)

# print "\nSignificient regions for CC:"
# for i in range(len(labels_name)):
#     if rejectedCC[i] and pvalListCC[i]["obsDiff"] != 0:
#         print "\n", labels_name[i], \
#             "pval:", pvals_correctedCC[i], \
#             "observed differnce:", pvalListCC[i]["obsDiff"]

# results_CC = []
# for i in range(len(labels_name)):
#     if rejectedCC[i] and pvalListCC[i]["obsDiff"] != 0:
#         results_CC += [{"label": labels_name[i],
#                         "pval_corr": pvals_correctedCC[i],
#                         "obs_diff": pvalListCC[i]["obsDiff"],
#                         "mean_random_diff:":
#                         np.asarray(pvalListCC[i]["diffs"]).mean()}]

# results_all = [results_degrees, results_CC]
# pickle.dump(results_all,
#             open("MI_press_0-05_fdr_resample_crop_perm-resample.p", "wb"))
