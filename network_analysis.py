import numpy as np
import networkx as nx
import numpy.random as npr
import os
import socket
import mne

from nitime.analysis import MTCoherenceAnalyzer
from nitime import TimeSeries
from mne.stats import fdr_correction

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


epochs_fnormal = data_path + "tone_task_normal-epo.fif"
epochs_normal = mne.read_epochs(epochs_fnormal)
# load numpy files
labelTsHypZscore = np.load("labelTsHypZscore.npy")
labelTsNormalZscore = np.load("labelTsNormalZscore.npy")

# crop zscored TS
fromTime = np.argmax(epochs_normal.times == 0)
toTime = np.argmax(epochs_normal.times == 0.7)

labelTsNormalZscoreCrop = []
for j in range(len(labelTsNormalZscore)):
    labelTsNormalZscoreCrop += [labelTsNormalZscore[j][:, fromTime:toTime]]

labelTsHypZscoreCrop = []
for j in range(len(labelTsHypZscore)):
    labelTsHypZscoreCrop += [labelTsHypZscore[j][:, fromTime:toTime]]


# %%
# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
labels = mne.read_labels_from_annot('subject_1', parc='PALS_B12_Brodmann',
                                    regexp="Brodmann",
                                    subjects_dir=subjects_dir)

labels_name = []
for label in labels:
    labels_name += [label.name]

# %%
cohListNormal = []
cohListHyp = []

for j in range(len(labelTsNormalZscoreCrop)):
    nits = TimeSeries(labelTsNormalZscoreCrop[j],
                      sampling_rate=epochs_normal.info["sfreq"])
    nits.metadata["roi"] = labels_name

    cohListNormal += [MTCoherenceAnalyzer(nits)]


for j in range(len(labelTsHypZscoreCrop)):
    nits = TimeSeries(labelTsHypZscoreCrop[j],
                      sampling_rate=epochs_normal.info["sfreq"])
    nits.metadata["roi"] = labels_name

    cohListHyp += [MTCoherenceAnalyzer(nits)]


# %% extract coherence values
f_lw, f_up = 52, 90  # lower & upper limit for frequencies

cohMatrixNormal = np.empty([len(labels_name),
                            len(labels_name),
                            len(labelTsNormalZscoreCrop)])
cohMatrixHyp = np.empty([len(labels_name),
                         len(labels_name),
                         len(labelTsHypZscoreCrop)])

# confine analysis to Aplha (8  12 Hz)
freq_idx = np.where((cohListHyp[0].frequencies >= f_lw) *
                    (cohListHyp[0].frequencies <= f_up))[0]

# compute average coherence &  Averaging on last dimension
for j in range(cohMatrixNormal.shape[2]):
    cohMatrixNormal[:, :, j] = np.mean(
        cohListNormal[j].coherence[:, :, freq_idx], -1)

for j in range(cohMatrixHyp.shape[2]):
    cohMatrixHyp[:, :, j] = np.mean(
        cohListHyp[j].coherence[:, :, freq_idx], -1)


# %%
fullMatrix = np.concatenate([cohMatrixNormal, cohMatrixHyp], axis=2)

threshold = np.median(fullMatrix[np.nonzero(fullMatrix)]) + \
    np.std(fullMatrix[np.nonzero(fullMatrix)])

binMatrixNormal = cohMatrixNormal > threshold
binMatrixHyp = cohMatrixHyp > threshold

# %%
nxNormal = []
for j in range(binMatrixNormal.shape[2]):
    nxNormal += [nx.from_numpy_matrix(binMatrixNormal[:, :, j])]

nxHyp = []
for j in range(binMatrixHyp.shape[2]):
    nxHyp += [nx.from_numpy_matrix(binMatrixHyp[:, :, j])]


# %%
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


# %% Permutation test
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


# %% Degress
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

    pvalList += [{'pval': pval, "obsDiff": observed_diff, "diffs": diffs}]

# %% for CC
pvalListCC = []
for ccNumber in range(binMatrixHyp.shape[0]):

    postHyp = np.empty(len(ccHyp))
    for j in range(len(ccHyp)):
        postHyp[j] = ccHyp[j][ccNumber]

    postNormal = np.empty(len(ccNormal))
    for j in range(len(postNormal)):
        postNormal[j] = ccNormal[j][ccNumber]

    pval, observed_diff, diffs = \
        permutation_resampling(postHyp, postNormal,
                               10000, np.mean)

    pvalListCC += [{'pval': pval, "obsDiff": observed_diff, "diffs": diffs}]


# %% Correct for multiple comparisons

pvals = np.empty(len(pvalList))
for j in range(len(pvals)):
    pvals[j] = pvalList[j]["pval"]

rejected, pvals_corrected = fdr_correction(pvals)

corrIndex = pvals_corrected < (0.05)

print "\nSignificient regions for Degrees:"
for i in range(len(labels_name)):
    if corrIndex[i]:
        print labels_name[i], \
            "pval:", pvalList[i]["pval"], \
            "observed differnce:", pvalList[i]["obsDiff"], \
            "mean random difference:", np.asarray(pvalList[i]["diffs"]).mean()

# %% for CC
# pvals = np.empty(len(pvalListCC))
# for j in range(len(pvals)):
#     pvals[j] = pvalListCC[j]["pval"]

# rejected, pvals_corrected = mne.stats.fdr_correction(pvals)

# corrIndex = pvals_corrected < (0.05)

# print "\nSignificient regions for CC:"
# for i in range(len(labels_name)):
#     if corrIndex[i]:
#         print labels_name[i], \
#             "pval:", pvalList[i]["pval"], \
#             "observed differnce:", pvalList[i]["obsDiff"], \
#             "mean random difference:", np.asarray(pvalList[i]["diffs"]).mean()


# rejected, pvals_corrected = mne.stats.fdr_correction(pvals)

# corrIndex = pvals_corrected < 0.05

# for i in range(len(labels_name)):
#     if corrIndex[i]:
#         print labels_name[i], \
#             "pval:", pvals_corrected[i]
