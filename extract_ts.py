
import numpy as np
import numpy.random as npr
import mne
import os
import socket
import networkx as nx

from mne.minimum_norm import (apply_inverse_epochs, read_inverse_operator)
from mne.baseline import rescale
from nitime import TimeSeries
from nitime.analysis import MTCoherenceAnalyzer

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


epochs_fnormal = data_path + "tone_task_normal-epo.fif"
epochs_fhyp = data_path + "tone_task_hyp-epo.fif"
inverse_fnormal = data_path + "tone_task_normal-inv.fif"
inverse_fhyp = data_path + "tone_task_hyp-inv.fif"
# change dir to save files the rigth place
os.chdir(data_path)

reject = dict(grad=4000e-13,  # T / m (gradiometers)
              mag=4e-12,  # T (magnetometers)
              #  eog=250e-6  # uV (EOG channels)
              )

# %%
snr = 1.0  # Standard assumption for average data but using it for single trial
lambda2 = 1.0 / snr ** 2
method = "MNE"

# Load data
inverse_normal = read_inverse_operator(inverse_fnormal)
inverse_hyp = read_inverse_operator(inverse_fhyp)

epochs_normal = mne.read_epochs(epochs_fnormal)
epochs_hyp = mne.read_epochs(epochs_fhyp)

epochs_normal = epochs_normal["press"]
epochs_hyp = epochs_hyp["press"]

# epochs_normal.crop(tmin=0, tmax=0.5)
# epochs_hyp.crop(tmin=0, tmax=0.5)


# %%
stcsNormal = apply_inverse_epochs(epochs_normal, inverse_normal, lambda2,
                                  method, pick_ori="normal",
                                  return_generator=True)
stcsHyp = apply_inverse_epochs(epochs_hyp, inverse_hyp, lambda2,
                               method, pick_ori="normal",
                               return_generator=True)

# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
labels = mne.read_labels_from_annot('subject_1', parc='PALS_B12_Brodmann',
                                    regexp="Brodmann",
                                    subjects_dir=subjects_dir)

# Average the source estimates within eachh label using sign-flips to reduce
# signal cancellations, also here we return a generator
src_normal = inverse_normal['src']
labelTsNormal = mne.extract_label_time_course(stcsNormal, labels,
                                              src_normal,
                                              mode='mean_flip',
                                              return_generator=False)
src_hyp = inverse_hyp['src']
labelTsHyp = mne.extract_label_time_course(stcsHyp, labels, src_hyp,
                                           mode='mean_flip',
                                           return_generator=False)

# standardize TS's
labelTsNormalZscore = []
for j in range(len(labelTsNormal)):
    labelTsNormalZscore += [rescale(labelTsNormal[j], epochs_normal.times,
                                    baseline=(None, -0.7), mode="zscore")]

labelTsHypZscore = []
for j in range(len(labelTsHyp)):
    labelTsHypZscore += [rescale(labelTsHyp[j], epochs_hyp.times,
                                 baseline=(None, -0.7), mode="zscore")]

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
labels_name = []
for label in labels:
    labels_name += [label.name]

cohListNormal = []
cohListHyp = []

for j in range(len(labelTsNormal)):
    nits = TimeSeries(labelTsNormalZscoreCrop[j],
                      sampling_rate=epochs_normal.info["sfreq"])
    nits.metadata["roi"] = labels_name

    cohListNormal += [MTCoherenceAnalyzer(nits)]


for j in range(len(labelTsHyp)):
    nits = TimeSeries(labelTsHypZscoreCrop[j],
                      sampling_rate=epochs_hyp.info["sfreq"])
    nits.metadata["roi"] = labels_name

    cohListHyp += [MTCoherenceAnalyzer(nits)]

# %% extract coherence values
f_lw, f_up = 52, 90 # lower & upper limit for frequencies

cohMatrixNormal = np.empty([len(labels_name),
                            len(labels_name),
                            len(labelTsNormalZscore)])
cohMatrixHyp = np.empty([len(labels_name),
                         len(labels_name),
                         len(labelTsHypZscore)])

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

rejected, pvals_corrected = mne.stats.fdr_correction(pvals)

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
