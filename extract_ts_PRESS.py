import numpy as np
import mne
import os
import socket

from mne.minimum_norm import (apply_inverse_epochs, read_inverse_operator)
from mne.baseline import rescale

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
stcsHyp = apply_inverse_epochs(epochs_hyp, inverse_hyp, lambda2,
                               method, pick_ori="normal",
                               return_generator=False)


# resample
[stc.resample(250) for stc in stcsNormal]
[stc.resample(250) for stc in stcsHyp]

# Get labels from FreeSurfer cortical parcellation
# labels = mne.read_labels_from_annot('subject_1', parc='PALS_B12_Brodmann',
#                                     regexp="Brodmann",
#                                     subjects_dir=subjects_dir)

labels = mne.read_labels_from_annot('subject_1', parc='aparc.DKTatlas40',
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
labelTsNormalRescaled = []
for j in range(len(labelTsNormal)):
    labelTsNormalRescaled += [rescale(labelTsNormal[j], epochs_normal.times,
                                      baseline=(None, -0.7), mode="zscore")]

labelTsHypRescaled = []
for j in range(len(labelTsHyp)):
    labelTsHypRescaled += [rescale(labelTsHyp[j], epochs_hyp.times,
                                   baseline=(None, -0.7), mode="zscore")]


fromTime = np.argmax(stcsNormal[0].times == 0)
toTime = np.argmax(stcsNormal[0].times == 0.5)

labelTsNormalRescaledCrop = []
for j in range(len(labelTsNormal)):
    labelTsNormalRescaledCrop += [labelTsNormalRescaled[j][:, fromTime:toTime]]

labelTsHypRescaledCrop = []
for j in range(len(labelTsHyp)):
    labelTsHypRescaledCrop += [labelTsHypRescaled[j][:, fromTime:toTime]]

np.save("labelTsHypPressMean-flipZscore_resample_crop_DKT.npy",
        labelTsHypRescaledCrop)
np.save("labelTsNormalPressMean-flipZscore_resample_crop_DKT.npy",
        labelTsNormalRescaledCrop)
