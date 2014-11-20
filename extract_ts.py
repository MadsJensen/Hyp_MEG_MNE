
import numpy as np
import numpy.random as npr
import statsmodels.api as sm
import matplotlib.pyplot as plt
import mne
import os
import socket
from mne.minimum_norm import (apply_inverse_epochs, read_inverse_operator)

# Setup paths and prepare raw data
hostname = socket.gethostname()

if hostname == "wintermute":
    data_path = "/home/mje/mnt/Hyp_meg/scratch/Tone_task_MNE/"
else:
    data_path = "/scratch1/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                "Tone_task_MNE/"
    subjects_dir = "/scratch1/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                    "fs_subjects_dir"



epochs_normal = data_path + "tone_task_normal-epo.fif"
epochs_hyp = data_path + "tone_task_hyp-epo.fif"
inverse_normal = data_path + "tone_task_normal-inv.fif"
inverse_hyp = data_path + "tone_task_hyp-inv.fif"
# change dir to save files the rigth place
os.chdir(data_path)

reject = dict(grad=4000e-13,  # T / m (gradiometers)
              mag=4e-12,  # T (magnetometers)
              #  eog=250e-6  # uV (EOG channels)
              )


# %%
# Using the same inverse operator when inspecting single trials Vs. evoked
snr = 1.0  # Standard assumption for average data but using it for single trial
lambda2 = 1.0 / snr ** 2

method = "MNE"  # use dSPM method (could also be MNE or sLORETA)

# Load data

conditions = ["normal"]

exec("inverse_operator = read_inverse_operator(inverse_%s)" % conditions[0])

exec("epochs = mne.read_epochs(epochs_%s)" %conditions[0])


# %%
stcsNormal = apply_inverse_epochs(epochs, inverse_operator, lambda2,
                                method, pick_ori="normal",
                                return_generator=True)

# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
labels = mne.read_labels_from_annot('subject_1', parc='aparc.DKTatlas40',
                                    subjects_dir=subjects_dir)


# Average the source estimates within each label using sign-flips to reduce
# signal cancellations, also here we return a generator
src = inverse_operator['src']
labelTsNormal = mne.extract_label_time_course(stcsNormal, labels, src,
                                            mode='mean_flip',
                                            return_generator=False)


# %%
from nitime import TimeSeries
from nitime.analysis import MTCoherenceAnalyzer
from nitime.viz import drawmatrix_channels

f_up = 13  # upper limit
f_lw = 8  # lower limit

cohMatrixNormal = np.empty([np.shape(labelTsNormal)[1], np.shape(labelTsNormal)[1],
                          np.shape(labelTsNormal)[0]])

labels_name = []
for label in labels:
    labels_name += [label.name]

for j in range(cohMatrixNormal.shape[2]):
    niTS = TimeSeries(labelTsNormal[j], sampling_rate=epochs.info["sfreq"])
    niTS.metadata["roi"] = labels_name

    C = MTCoherenceAnalyzer(niTS)

    # confine analysis to Aplha (8  12 Hz)
    freq_idx = np.where((C.frequencies > f_lw) * (C.frequencies < f_up))[0]

    # compute average coherence &  Averaging on last dimension
    cohMatrixNormal[:, :, j] = np.mean(C.coherence[:, :, freq_idx], -1)


# %%
drawmatrix_channels(bin.astype(int), labels_name, color_anchor=0,
                    title='MEG coherence')

plt.show()


# %%
thresholdLeft = np.median(cohMatrixLeft[np.nonzero(cohMatrixLeft)]) \
    + np.std(cohMatrixLeft[np.nonzero(cohMatrixLeft)])
binMatrixLeft = cohMatrixLeft > thresholdLeft

thresholdRight = np.median(cohMatrixRight[np.nonzero(cohMatrixRight)]) \
    + np.std(cohMatrixRight[np.nonzero(cohMatrixRight)])
binMatrixRight = cohMatrixRight > thresholdRight


# %%
import networkx as nx

nxLeft = []
for j in range(binMatrixLeft.shape[2]):
    nxLeft += [nx.from_numpy_matrix(binMatrixLeft[:, :, j])]

nxRight = []
for j in range(binMatrixRight.shape[2]):
    nxRight += [nx.from_numpy_matrix(binMatrixRight[:, :, j])]


# %%
degreesRight = []
for j, trial in enumerate(nxRight):
    degreesRight += [trial.degree()]

degreesLeft = []
for j, trial in enumerate(nxLeft):
    degreesLeft += [trial.degree()]

# %%
pvalList = []
for degreeNumber in range(binMatrixLeft.shape[0]):

    postRight = np.empty(len(degreesRight))
    for j in range(len(degreesRight)):
        postRight[j] = degreesRight[j][degreeNumber]

    postLeft = np.empty(len(degreesLeft))
    for j in range(len(postLeft)):
        postLeft[j] = degreesLeft[j][degreeNumber]

    pval, observed_diff, diffs = \
        permutation_resampling(postRight, postLeft,
                               10000, np.mean)

    pvalList += [{'pval': pval, "obsDiff": observed_diff, "diffs": diffs}]


# %% Correct for multiple comparisons

pvals = np.empty(len(pvalList))
for j in range(len(pvals)):
    pvals[j] = pvalList[j]["pval"]

corrIndex = pvals < (0.05)

for i in range(62):
    if corrIndex[i] is True:
        print labels_names[i], \
            "pval:", pvalList[i]["pval"], \
            "observed differnce:", pvalList[i]["obsDiff"], \
            "mean random difference:", np.asarray(pvalList[i]["diffs"]).mean()


# %%
rejected, pvals_corrected = sm.stats.fdrcorrection(pvals)

corrIndex = pvals_corrected < 0.05

for i in range(62):
    if corrIndex[i] is True:
        print RowNames[i], \
            "pval:", pvals_corrected[i]
