import numpy as np
import networkx as nx
# import numpy.random as npr
# import cPickle as pickle
import os
import socket
import mne
# import bct
# import cPickle as pickle
import powerlaw as pl

from mne.minimum_norm import (apply_inverse_epochs, read_inverse_operator)
from mne.baseline import rescale
from nitime.analysis import MTCoherenceAnalyzer
from nitime import TimeSeries
# from mne.stats import fdr_correction


# Setup paths and prepare raw data
hostname = socket.gethostname()
if hostname == "Wintermute":
    data_path = "/home/mje/mnt/Hyp_meg/scratch/Tone_task_MNE/"
    subjects_dir = "/home/mje/mnt/Hyp_meg/scratch/fs_subjects_dir/"
    n_jobs = 1
else:
    data_path = "/scratch1/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                "Tone_task_MNE/"
    subjects_dir = "/scratch1/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                   "fs_subjects_dir"
    n_jobs = 1

result_dir = data_path + "network_connect_res"

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


# %%
stcsNormal = apply_inverse_epochs(epochs_normal, inverse_normal, lambda2,
                                  method, pick_ori="normal",
                                  return_generator=False)
# stcsHyp = apply_inverse_epochs(epochs_hyp, inverse_hyp, lambda2,
#                                method, pick_ori="normal",
#                                return_generator=False)


# resample
[stc.resample(250) for stc in stcsNormal]
# [stc.resample(2-50) for stc in stcsHyp]

# Get labels from FreeSurfer cortical parcellation
# labels = mne.read_labels_from_annot('subject_1', parc='aparc.a2009s',
#                                     regexp="[G|S]",
#                                     subjects_dir=subjects_dir)

# labels = mne.read_labels_from_annot('subject_1', parc='aparc.DKTatlas40',
#                                     subjects_dir=subjects_dir)
labels = mne.read_labels_from_annot('subject_1', parc='PALS_B12_Brodmann',
                                    regexp="Bro",
                                    subjects_dir=subjects_dir)
labels_name = [label.name for label in labels]

# Average the source estimates within eachh label using sign-flips to reduce
# signal cancellations, also here we return a generator
src_normal = inverse_normal['src']
src_hyp = inverse_hyp['src']

ba6_normal = [stc.in_label(labels[-8]).data for stc in stcsNormal]
labelTsNormal = ba6_normal

# standardize TS's
labelTsNormalRescaled = []
for j in range(len(labelTsNormal)):
    labelTsNormalRescaled += [rescale(labelTsNormal[j], epochs_normal.times,
                                      baseline=(None, -0.7), mode="zscore")]

# labelTsHypRescaled = []
# for j in range(len(labelTsHyp)):
#     labelTsHypRescaled += [rescale(labelTsHyp[j], epochs_hyp.times,
#                                    baseline=(None, -0.7), mode="zscore")]


fromTime = np.argmax(stcsNormal[0].times == 0)
toTime = np.argmax(stcsNormal[0].times == 0.5)

labelTsNormalRescaledCrop = []
for j in range(len(labelTsNormal)):
    labelTsNormalRescaledCrop += [labelTsNormalRescaled[j][:, fromTime:toTime]]

# labelTsHypRescaledCrop = []
# for j in range(len(labelTsHyp)):
#     labelTsHypRescaledCrop += [labelTsHypRescaled[j][:, fromTime:toTime]]


# %%
cohListNormal = []
cohListHyp = []

for j in range(len(labelTsNormalRescaledCrop)):
    nits = TimeSeries(labelTsNormalRescaledCrop[j],
                      sampling_rate=250)  # epochs_normal.info["sfreq"])
    nits.metadata["roi"] = labels_name

    cohListNormal += [MTCoherenceAnalyzer(nits)]

# for j in range(len(labelTsHypRescaledCrop)):
#     nits = TimeSeries(labelTsHypRescaledCrop[j],
#                       sampling_rate=250)  # epochs_normal.info["sfreq"])
#     nits.metadata["roi"] = labels_name

#     cohListHyp += [MTCoherenceAnalyzer(nits)]

# Compute a source estimate per frequency band
bands = dict(theta=[4, 8],
             alpha=[8, 12],
             beta=[13, 25],
             gamma_low=[30, 48],
             gamma_high=[52, 90])

bands = dict(alpha=[8, 12])


for band in bands.keys():
    print "\n******************"
    print "\nAnalysing band: %s" % band
    print "\n******************"

    # extract coherence values
    f_lw, f_up = bands[band]  # lower & upper limit for frequencies

    cohMatrixNormal = np.empty([labelTsNormalRescaledCrop[0].shape[0],
                                labelTsNormalRescaledCrop[0].shape[0],
                                len(labelTsNormalRescaledCrop)])
    # cohMatrixHyp = np.empty([len(labels_name),
    #                          len(labels_name),
    #                          len(labelTsHypRescaledCrop)])

    # confine analysis to specific band
    freq_idx = np.where((cohListNormal[0].frequencies >= f_lw) *
                        (cohListNormal[0].frequencies <= f_up))[0]

    print cohListNormal[0].frequencies[freq_idx]

    # compute average coherence &  Averaging on last dimension
    for j in range(cohMatrixNormal.shape[2]):
        cohMatrixNormal[:, :, j] = np.mean(
            cohListNormal[j].coherence[:, :, freq_idx], -1)

    # for j in range(cohMatrixHyp.shape[2]):
    #     cohMatrixHyp[:, :, j] = np.mean(
    #         cohListHyp[j].coherence[:, :, freq_idx], -1)

    #
    fullMatrix = np.concatenate([cohMatrixNormal], axis=2)

    threshold = np.median(fullMatrix[np.nonzero(fullMatrix)]) + \
        np.std(fullMatrix[np.nonzero(fullMatrix)])

    bin_normal = cohMatrixNormal > threshold
    # bin_hyp = cohMatrixHyp > threshold

    nx_normal = []
    for j in range(bin_normal.shape[2]):
        nx_normal += [nx.from_numpy_matrix(bin_normal[:, :, j])]

    # nx_hyp = []
    # for j in range(bin_hyp.shape[2]):
    #     nx_hyp += [nx.from_numpy_matrix(bin_hyp[:, :, j])]

    # alpha_normal = np.asarray([pl.Fit(g.degree().values()).alpha
    #                            for g in nx_normal if nx.is_connected(g)])
    # alpha_hyp = np.asarray([pl.Fit(g.degree().values()).alpha
    #                         for g in nx_hyp if nx.is_connected(g)])

    # #
    # np.savetxt("network_connect_res/degree_dist_press_normal_%s.csv"
    #            % band, alpha_normal)
    # np.savetxt("network_connect_res/degree_dist_press_hyp_%s.csv"
    #            % band, alpha_hyp)
