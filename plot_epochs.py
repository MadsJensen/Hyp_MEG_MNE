# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import os
import numpy as np
import mne
import socket
from mne.minimum_norm import (apply_inverse_epochs, read_inverse_operator)
from mne.baseline import rescale
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
    n_jobs = 4

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

epochs_normal = epochs_normal["Tone"]
epochs_hyp = epochs_hyp["Tone"]


# %%
stcsNormal = apply_inverse_epochs(epochs_normal, inverse_normal, lambda2,
                                  method, pick_ori="normal",
                                  return_generator=False)
stcsHyp = apply_inverse_epochs(epochs_hyp, inverse_hyp, lambda2,
                               method, pick_ori="normal",
                               return_generator=False)


# resample
[stc.resample(250) for stc in stcsNormal]
[stc.resample(250) for stc in stcsHyp]

# Get labels from FreeSurfer cortical parcellation
# labels = mne.read_labels_from_annot('subject_1', parc='aparc.a2009s',
#                                     regexp="[G|S]",
#                                     subjects_dir=subjects_dir)
labels = mne.read_labels_from_annot('subject_1', parc='aparc.DKTatlas40',
                                    regexp="precentral-lh",
                                    subjects_dir=subjects_dir)
labels_name = [label.name for label in labels]

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
labelTsNormalRescaled = []
for j in range(len(labelTsNormal)):
    labelTsNormalRescaled += [rescale(labelTsNormal[j], epochs_normal.times,
                                      baseline=(None, -0.7), mode="zscore")]

labelTsHypRescaled = []
for j in range(len(labelTsHyp)):
    labelTsHypRescaled += [rescale(labelTsHyp[j], epochs_hyp.times,
                                   baseline=(None, -0.7), mode="zscore")]


# %%plot of full epoch
times = stcsNormal[0].times

TS_normal = np.asarray(labelTsNormalRescaled)
TS_normal = TS_normal[:, 0, :]
TS_hyp = np.asarray(labelTsHypRescaled)
TS_hyp = TS_hyp[:, 0, :]

plt.figure()
plt.plot(times, TS_normal.mean(axis=0), "b")
plt.plot(times, TS_hyp.mean(axis=0), "r")

normal_std = TS_normal.std(axis=0)
hyp_std = TS_hyp.std(axis=0)

plt.xlabel('Times (seconds)')
plt.ylabel('Zscore')
plt.title('Mean Time series for left premotor area')

plt.plot(times, TS_hyp.mean(axis=0), 'r',  linewidth=3,
         label="mean activation in hypnosis")
# plt.plot(times, TS_hyp.mean(axis=0) + hyp_std, 'r--', alpha=0.8)
# plt.plot(times, TS_hyp.mean(axis=0) - hyp_std, 'r--', alpha=0.8)

plt.plot(times, TS_normal.mean(axis=0), 'b', linewidth=3,
         label="mean activation in normal")
# plt.plot(times, TS_normal.mean(axis=0) + normal_std, 'b--', alpha=0.8)
# plt.plot(times, TS_normal.mean(axis=0) - normal_std, 'b--', alpha=0.8)

plt.axvline(-0.7, color='lightgrey', label='Baseline period')
plt.axvspan(-1, -0.7, alpha=0.75, color='lightgrey')

plt.axvline(-0, color='darkgrey', label='No stimulus period')
plt.axvspan(-0.5, 0, alpha=0.5, color='darkgrey')


plt.axvline(0, color='k', label='Tone is played', linewidth=3,
            linestyle="dashed")

plt.legend(loc=3)

plt.show()


# %%plot of full epoch
# REMEMBER TO USE THE RIGHT EPOCHS!!!!!!!!!

plt.figure()
plt.plot(times, TS_normal.mean(axis=0), "b")
plt.plot(times, TS_hyp.mean(axis=0), "r")

normal_std = TS_normal.std(axis=0)
hyp_std = TS_hyp.std(axis=0)

plt.xlabel('Times (seconds)')
plt.ylabel('dSPM value')
plt.title('Mean Time series for BA6-lh')

plt.plot(times, TS_hyp.mean(axis=0), 'r',  linewidth=3,
         label="mean activation in hypnosis")
plt.plot(times, TS_hyp.mean(axis=0) + hyp_std, 'r--', alpha=0.8)
plt.plot(times, TS_hyp.mean(axis=0) - hyp_std, 'r--', alpha=0.8)

plt.plot(times, TS_normal.mean(axis=0), 'b', linewidth=3,
         label="mean activation in normal")
plt.plot(times, TS_normal.mean(axis=0) + normal_std, 'b--', alpha=0.8)
plt.plot(times, TS_normal.mean(axis=0) - normal_std, 'b--', alpha=0.8)


plt.axvline(0, color='k', label='Key press', linewidth=3,
            linestyle="dashed")
plt.axvspan(0, 0.3, alpha=0.5, color='g')

plt.axvline(-0.2, color='m', label='prepress period')
plt.axvspan(-0.2, 0, alpha=0.5, color='m')

plt.legend(loc=2)

plt.show()
