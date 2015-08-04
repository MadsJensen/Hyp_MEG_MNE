import mne
import os
import socket
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mne.minimum_norm import read_inverse_operator, apply_inverse_epochs
from mne.baseline import rescale

plt.ion()
# Setup paths and prepare raw data
hostname = socket.gethostname()

if hostname == "Wintermute":
    data_path = "/home/mje/mnt/Hyp_meg/scratch/Tone_task_MNE/"
    script_path = "/home/mje/mnt/Hyp_meg/scripts/MNE_analysis/"
    subjects_dir = "/home/mje/mnt/Hyp_meg/scratch/fs_subjects_dir/"
    n_jobs = 1
else:
    data_path = "/scratch1/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                "Tone_task_MNE/"
    script_path = "/projects/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                  "scripts/MNE_analysis/"
    subjects_dir = "/scratch1/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                   "fs_subjects_dir"
    n_jobs = 1

result_dir = data_path + "/class_result"

os.chdir(data_path)

epochs_fnormal = data_path + "tone_task_normal-epo.fif"
epochs_fhyp = data_path + "tone_task_hyp-epo.fif"
inverse_fnormal = data_path + "tone_task_normal-inv.fif"
inverse_fhyp = data_path + "tone_task_hyp-inv.fif"

epochs_normal = mne.read_epochs(epochs_fnormal)
epochs_hyp = mne.read_epochs(epochs_fhyp)

epochs_normal = epochs_normal["Tone"]
epochs_hyp = epochs_hyp["Tone"]


snr = 1.0  # Standard assumption for average data but using it for single trial
lambda2 = 1.0 / snr ** 2
method = "dSPM"

# Load data
inverse_normal = read_inverse_operator(inverse_fnormal)
inverse_hyp = read_inverse_operator(inverse_fhyp)
src_normal = inverse_normal['src']
src_hyp = inverse_hyp['src']

stcs_normal = apply_inverse_epochs(epochs_normal, inverse_normal,
                                   lambda2, method,
                                   pick_ori="normal",
                                   return_generator=False)

stcs_hyp = apply_inverse_epochs(epochs_hyp, inverse_hyp,
                                lambda2, method,
                                pick_ori="normal",
                                return_generator=False)

# resample
[stc.resample(250) for stc in stcs_normal]
[stc.resample(250) for stc in stcs_hyp]

label_dir = subjects_dir + "/subject_1/label/"

# labels = mne.read_labels_from_annot('subject_1', parc='aparc.DKTatlas40',
#                                     subjects_dir=subjects_dir)
labels = mne.read_labels_from_annot('subject_1', parc='PALS_B12_Brodmann',
                                    regexp="Bro",
                                    subjects_dir=subjects_dir)
labels_name = [label.name for label in labels]

labels_single = [labels[-26]]

# find index for start and stop times
from_time = np.abs(stcs_normal[-1].times - -1).argmin()
to_time = np.abs(stcs_normal[0].times - 1).argmin()

# from_time = np.abs(stcs_normal[0].times - 0).argmin()
# to_time = np.abs(stcs_normal[0].times - 0.5).argmin()

for label in labels_single:
    labelTsNormal = mne.extract_label_time_course(stcs_normal,
                                                  labels=label,
                                                  src=src_normal,
                                                  mode='mean',
                                                  return_generator=False)

    labelTsHyp = mne.extract_label_time_course(stcs_hyp,
                                               labels=label,
                                               src=src_hyp,
                                               mode='mean',
                                               return_generator=False)

    labelTsNormalRescaled = []
    for j in range(len(labelTsNormal)):
        labelTsNormalRescaled += [rescale(labelTsNormal[j],
                                          stcs_normal[0].times,
                                          baseline=(None, -0.7),
                                          mode="zscore")]

    labelTsHypRescaled = []
    for j in range(len(labelTsHyp)):
        labelTsHypRescaled += [rescale(labelTsHyp[j],
                                       stcs_hyp[0].times,
                                       baseline=(None, -0.7),
                                       mode="zscore")]

    labelTsNormalRescaledCrop = []
    for j in range(len(labelTsNormal)):
        labelTsNormalRescaledCrop +=\
            [labelTsNormalRescaled[j][:, from_time:to_time]]

    labelTsHypRescaledCrop = []
    for j in range(len(labelTsHyp)):
        labelTsHypRescaledCrop +=\
            [labelTsHypRescaled[j][:, from_time:to_time]]


results_pd = []
times = stcs_normal[0].times[from_time:to_time]

for j in range(len(labelTsNormal)):
    results_pd.append(pd.DataFrame(dict(condition="normal",
                                        time=times * 1e3,
                                        trial=j,
                                        MNE=labelTsNormal[j]
                                        [:, from_time:to_time].reshape(-1))))

for j in range(len(labelTsHyp)):
    results_pd.append(pd.DataFrame(dict(condition="hyp",
                                        time=times * 1e3,
                                        trial=j,
                                        MNE=labelTsHyp[j]
                                        [:, from_time:to_time].reshape(-1))))

results_pd = pd.concat(results_pd)

color_map = dict(hyp="indianred", normal="steelblue")

plt.figure()
ax = sns.tsplot(results_pd,
                time="time",
                unit="trial",
                condition="condition",
                value="MNE",
                # err_style="boot_traces",
                linewidth=3,
                color=color_map)
ax.set_xlabel("time (miliseconds)")
ax.set_title(labels_single[0].name)
