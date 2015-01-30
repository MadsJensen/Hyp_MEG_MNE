import cPickle as pickle
import os
import socket

# import matplotlib.pyplot as plt
import mne
# import numpy as np
from mne.minimum_norm import (read_inverse_operator,
                              source_band_induced_power)
# compute_source_psd_epochs)


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
    n_jobs = 6

# change dir to save files the rigth place
os.chdir(data_path)

###############################################################################
epochs_fnormal = data_path + "tone_task_normal-epo.fif"
epochs_fhyp = data_path + "tone_task_hyp-epo.fif"
inverse_fnormal = data_path + "tone_task_normal-inv.fif"
inverse_fhyp = data_path + "tone_task_hyp-inv.fif"

inverse_normal = read_inverse_operator(inverse_fnormal)
inverse_hyp = read_inverse_operator(inverse_fhyp)

epochs_normal = mne.read_epochs(epochs_fnormal)
epochs_hyp = mne.read_epochs(epochs_fhyp)

epochs_normal = epochs_normal["press"]
epochs_hyp = epochs_hyp["press"]

# labels = mne.read_labels_from_annot('subject_1', parc='PALS_B12_Brodmann',
#                                     regexp="Brodmann",
#                                     subjects_dir=subjects_dir)

labels = mne.read_labels_from_annot('subject_1', parc='aparc.DKTatlas40',
                                    subjects_dir=subjects_dir)
snr = 1.0  # Standard assumption for average data but using it for single trial
lambda2 = 1.0 / snr ** 2

# %%
# Compute a source estimate per frequency band
bands = dict(theta=[4, 8],
             alpha=[8, 12],
             beta=[13, 25],
             gamma_low=[30, 48],
             gamma_high=[52, 90])

# for band in bands.keys():
fbands = ["theta", "alpha", "beta", "gamma_low", "gamma_high"]
ffreqs = [[4, 8], [8, 13], [13, 25], [30, 48], [52, 90]]

for j in range(len(fbands)):
    values = ffreqs[j]
    band = {fbands[j]: values}
    band_name = fbands[j]

    stcs_normal = []
    for j in range(len(epochs_normal)):
        print "\n cond: %s, band: %s, trial: %d of %d \n" \
            % ("normal", band_name, j+1, len(epochs_normal))
        stcs_normal += [source_band_induced_power(epochs_normal[j],
                                                  inverse_normal,
                                                  band,
                                                  lambda2=lambda2,
                                                  method="MNE",
                                                  n_cycles=2,
                                                  use_fft=False,
                                                  baseline=(-0.95, -0.7),
                                                  baseline_mode="zscore",
                                                  n_jobs=1)]

    for stc in stcs_normal:
        stc[band_name].crop(0, 0.5)

    pickle.dump(stcs_normal,
                open("stcs_normal_press_source_induced" +
                     "_%s_0-05_DKT.p" % band_name, "wb"))

    stcs_hyp = []
    for j in range(len(epochs_hyp)):
        print "\n cond: %s, band: %s, trial: %d of %d \n" \
            % ("hyp", band_name, j+1, len(epochs_hyp))
        stcs_hyp += [source_band_induced_power(epochs_hyp[j], inverse_hyp,
                                               band,
                                               lambda2=lambda2,
                                               method="MNE",
                                               n_cycles=2, use_fft=False,
                                               baseline=(-0.95, -0.7),
                                               baseline_mode="zscore",
                                               n_jobs=1)]

    for stc in stcs_hyp:
        stc[band_name].crop(0, 0.5)

    pickle.dump(stcs_hyp,
                open("stcs_hyp_press_source_induced_%s_0-05_DKT.p"
                     % band_name, "wb"))
