import numpy as np
import mne
import os
import socket

from mne.viz import iter_topography
from mne.time_frequency import compute_epochs_psd

import matplotlib.pyplot as plt

# Setup paths and prepare epochs_normal data
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
epochs_normal.crop(0, 0.7)
epochs_hyp.crop(0, 0.7)

picks = mne.pick_types(epochs_normal.info, meg="grad", exclude=[])
fmin, fmax = 2, 90  # look at frequencies between 2 and 20Hz
n_fft = 2048  # the FFT size (n_fft). Ideally a power of 2
psds_normal, freqs_normal = compute_epochs_psd(epochs_normal, picks=picks,
                                               fmin=fmin, fmax=fmax)
# psds_normal = 20 * np.log10(psds_normal)  # scale to dB
psds_avg_normal = np.mean(psds_normal, axis=0)

psds_hyp, freqs_hyp = compute_epochs_psd(epochs_hyp, picks=picks,
                                         fmin=fmin, fmax=fmax)
# psds_hyp = 20 * np.log10(psds_hyp)  # scale to dB
psds_avg_hyp = np.mean(psds_hyp, axis=0)


def my_callback(ax, ch_idx):
    """
    This block of code is executed once you click on one of the channel axes
    in the plot. To work with the viz internals, this function should only take
    two parameters, the axis and the channel or data index.
    """
    ax.plot(freqs_normal, psds_avg_normal[ch_idx], color='lightblue')
    ax.plot(freqs_hyp, psds_avg_hyp[ch_idx], color='red')
    ax.set_xlabel = 'Frequency (Hz)'
    ax.set_ylabel = 'Power'

for ax, idx in iter_topography(epochs_normal.info,
                               fig_facecolor='black',
                               axis_facecolor='black',
                               axis_spinecolor='black',
                               on_pick=my_callback):
    ax.plot(psds_avg_normal[idx], color='lightblue', label="normal")
    ax.plot(psds_avg_hyp[idx], color='red', label="Hyp")

plt.gcf().suptitle('Power spectral densities')
plt.legend()
plt.show()
