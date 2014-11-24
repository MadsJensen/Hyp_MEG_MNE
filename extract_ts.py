
import numpy as np
import numpy.random as npr
import mne
import os
import socket

from mne.minimum_norm import (apply_inverse_epochs, read_inverse_operator)
from mne.baseline import rescale

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


np.save("labelTsHypZscore.npy", labelTsHypZscore)
np.save("labeltsnormalzscore", labelTsNormalZscore)
