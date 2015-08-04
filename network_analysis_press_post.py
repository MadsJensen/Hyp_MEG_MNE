import numpy as np
import networkx as nx
import numpy.random as npr
import os
import socket
import mne
import pandas as pd

from nitime.analysis import MTCoherenceAnalyzer
from nitime import TimeSeries
# from mne.stats import fdr_correction


# %% Permutation test.

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
    np.load("labels_ts_hyp_press_post_mean-flip_zscore_resample_crop_ba.npy")
labelTsNormalCrop =\
    np.load("labels_ts_normal_press_post_mean-flip_zscore_resample_crop_BA.npy")

# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
labels = mne.read_labels_from_annot('subject_1', parc='PALS_B12_Brodmann',
                                    regexp="Brodmann",
                                    subjects_dir=subjects_dir)

# labels = mne.read_labels_from_annot('subject_1', parc='aparc.DKTatlas40',
#                                     subjects_dir=subjects_dir)

labels_name = []
for label in labels:
    labels_name += [label.name]


# %%
cohListNormal = []
cohListHyp = []

for j in range(len(labelTsNormalCrop)):
    nits = TimeSeries(labelTsNormalCrop[j],
                      sampling_rate=250)  # epochs_normal.info["sfreq"])
    nits.metadata["roi"] = labels_name

    cohListNormal += [MTCoherenceAnalyzer(nits)]

for j in range(len(labelTsHypCrop)):
    nits = TimeSeries(labelTsHypCrop[j],
                      sampling_rate=250)  # epochs_normal.info["sfreq"])
    nits.metadata["roi"] = labels_name

    cohListHyp += [MTCoherenceAnalyzer(nits)]

# Compute a source estimate per frequency band
bands = dict(theta=[4, 8],
             alpha=[8, 12],
             beta=[13, 25],
             gamma_low=[30, 48],
             gamma_high=[52, 90])


# bands = dict(theta=[4, 8])

results_degree = []
results_CC = []

for band in bands.keys():
    print "\n******************"
    print "\nAnalysing band: %s" % band
    print "\n******************"

    # extract coherence values
    f_lw, f_up = bands[band]  # lower & upper limit for frequencies

    cohMatrixNormal = np.empty([len(labels_name),
                                len(labels_name),
                                len(labelTsNormalCrop)])
    cohMatrixHyp = np.empty([len(labels_name),
                             len(labels_name),
                             len(labelTsHypCrop)])

    # confine analysis to Aplha (8  12 Hz)
    freq_idx = np.where((cohListHyp[0].frequencies >= f_lw) *
                        (cohListHyp[0].frequencies <= f_up))[0]

    print cohListNormal[0].frequencies[freq_idx]

    # compute average coherence &  Averaging on last dimension
    for j in range(cohMatrixNormal.shape[2]):
        cohMatrixNormal[:, :, j] = np.mean(
            cohListNormal[j].coherence[:, :, freq_idx], -1)

    for j in range(cohMatrixHyp.shape[2]):
        cohMatrixHyp[:, :, j] = np.mean(
            cohListHyp[j].coherence[:, :, freq_idx], -1)

    #
    fullMatrix = np.concatenate([cohMatrixNormal, cohMatrixHyp], axis=2)

    threshold = np.median(fullMatrix[np.nonzero(fullMatrix)]) + \
        np.std(fullMatrix[np.nonzero(fullMatrix)])

    binMatrixNormal = cohMatrixNormal > threshold
    binMatrixHyp = cohMatrixHyp > threshold

    #
    nxNormal = []
    for j in range(binMatrixNormal.shape[2]):
        nxNormal += [nx.from_numpy_matrix(binMatrixNormal[:, :, j])]

    nxHyp = []
    for j in range(binMatrixHyp.shape[2]):
        nxHyp += [nx.from_numpy_matrix(binMatrixHyp[:, :, j])]

    #
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

    # Degress
    pvalList = []
    for degreeNumber in range(binMatrixHyp.shape[0]):

        postHyp = np.empty(len(degreesHyp))
        for j in range(len(postHyp)):
            postHyp[j] = degreesHyp[j][degreeNumber]

        postNormal = np.empty(len(degreesNormal))
        for j in range(len(postNormal)):
            postNormal[j] = degreesNormal[j][degreeNumber]

        pval, observed_diff, diffs = \
            permutation_test(postHyp, postNormal, 10000, np.mean)

        pvalList += [{'area': labels_name[degreeNumber],
                      'pval': pval,
                      "obsDiff": observed_diff,
                      "band": band}]

        results_degree.append(pd.DataFrame.from_dict(pvalList))
        pd.concat(results_degree)

    #  for CC
    pvalListCC = []
    for ccNumber in range(binMatrixHyp.shape[0]):

        postHyp = np.empty(len(ccHyp))
        for j in range(len(ccHyp)):
            postHyp[j] = ccHyp[j][ccNumber]

        postNormal = np.empty(len(ccNormal))
        for j in range(len(postNormal)):
            postNormal[j] = ccNormal[j][ccNumber]

        pval, observed_diff, diffs = \
            permutation_test(postHyp, postNormal, 10000, np.mean)

        pvalListCC += [{'area': labels_name[degreeNumber],
                        'pval': pval,
                        "obsDiff": observed_diff,
                        "band": band}]

        results_CC.append(pd.DataFrame.from_dict(pvalList))
        pd.concat(results_CC)
