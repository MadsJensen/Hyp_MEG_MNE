import numpy as np
import mne
import os
import socket
from mne.time_frequency import tfr_morlet


###############################################################################
# Set parameters
# Setup paths and prepare raw data
hostname = socket.gethostname()

if hostname == "wintermute":
    data_path = "/home/mje/mnt/Hyp_meg/scratch/Tone_task_MNE/"
    subjects_dir = "/home/mje/mnt/Hyp_meg/scratch/fs_subjects_dir/"
    n_jobs = 3
else:
    data_path = "/scratch1/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                "Tone_task_MNE/"
    subjects_dir = "/scratch1/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                   "fs_subjects_dir"
    n_jobs = 8

# change dir to save files the rigth place
os.chdir(data_path)

###############################################################################
epochs_fnormal = data_path + "tone_task_normal-epo.fif"
epochs_fhyp = data_path + "tone_task_hyp-epo.fif"
epochs_normal = mne.read_epochs(epochs_fnormal)
epochs_hyp = mne.read_epochs(epochs_fhyp)
epochs_normal = epochs_normal["press"]
epochs_hyp = epochs_hyp["press"]

###############################################################################
# Calculate power and intertrial coherence

freqs = np.arange(6, 90, 2)  # define frequencies of interest
n_cycles = freqs / 3.  # different number of cycle per frequency

power_normal = []
for j in range(len(epochs_normal)):
    power_normal += [tfr_morlet(epochs_normal[j], freqs=freqs,
                                n_cycles=n_cycles, use_fft=False,
                                return_itc=False, decim=3, n_jobs=1)]
power_hyp = []
for j in range(len(epochs_hyp)):
    power_hyp += [tfr_morlet(epochs_hyp, freqs=freqs,
                             n_cycles=n_cycles, use_fft=False,
                             return_itc=False, decim=3, n_jobs=1)]

# Baseline correction can be applied to power or done in plots
# To illustrate the baseline correction in plots the next line is commented
# power.apply_baseline(baseline=(-0.5, 0), mode='logratio')

# Inspect power
power_normal.plot_topo(baseline=(-0.9, -0.7), mode='zscore',
                       title='Average power')
power_normal.plot([82], baseline=(-0.9, -0.7), mode='zscore')

import matplotlib.pyplot as plt
fig, axis = plt.subplots(1, 2, figsize=(7, 4))
power_normal[0].plot_topomap(ch_type='grad', tmin=0, tmax=1, fmin=52, fmax=90,
                             baseline=(-0.9, -0.7), mode='zscore',
                             axes=axis[0],
                             title='Beta')
power_normal[0].plot_topomap(ch_type='grad', tmin=0, tmax=1, fmin=52, fmax=80,
                             baseline=(-0.9, -0.7), mode='zscore',
                             vmin=0, vmax=9,
                             title='Alpha')
mne.viz.tight_layout()

