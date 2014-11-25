import cPickle
import os
import socket

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.minimum_norm import (read_inverse_operator,
                              source_band_induced_power,
                              compute_source_psd_epochs)


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
inverse_fnormal = data_path + "tone_task_normal-inv.fif"
inverse_fhyp = data_path + "tone_task_hyp-inv.fif"

inverse_normal = read_inverse_operator(inverse_fnormal)
inverse_hyp = read_inverse_operator(inverse_fhyp)

epochs_normal = mne.read_epochs(epochs_fnormal)
epochs_hyp = mne.read_epochs(epochs_fhyp)

epochs_normal = epochs_normal["press"]
epochs_hyp = epochs_hyp["press"]

labels = mne.read_labels_from_annot('subject_1', parc='PALS_B12_Brodmann',
                                    regexp="Brodmann",
                                    subjects_dir=subjects_dir)

# Compute a source estimate per frequency band
bands = dict(alpha=[8, 12],
             beta=[13, 20])

#             gamma_low=[30, 48],
#             gamma_high=[52, 90])

snr = 1.0  # Standard assumption for average data but using it for single trial
lambda2 = 1.0 / snr ** 2

stcs_normal = []
for j in range(len(epochs_normal)):
    stcs_normal += [source_band_induced_power(epochs_normal[j], inverse_normal,
                                              bands,
                                              lambda2=lambda2,
                                              method="MNE",
                                              n_cycles=2,
                                              use_fft=False,
                                              baseline=(-0.9, -0.7),
                                              baseline_mode="zscore",
                                              n_jobs=n_jobs)]

stcs_hyp = []
for j in range(len(epochs_hyp)):
    stcs_hyp += [source_band_induced_power(epochs_hyp[j], inverse_hyp,
                                           bands,
                                           lambda2=lambda2,
                                           method="MNE",
                                           n_cycles=2, use_fft=False,
                                           baseline=(-0.9, -0.7),
                                           baseline_mode="zscore",
                                           n_jobs=n_jobs)]


for j, stc in enumerate(stcs_hyp):
    stc["alpha"].crop(-0.1, 0.7)
    stc["beta"].crop(-0.1, 0.7)

cPickle.dump(stcs_normal, open("stcs_normal_souce_induced_A_B.p", "wb"))
cPickle.dump(stcs_hyp, open("stcs_hyp_souce_induced_A_B.p", "wb"))

# for b, stc in stcs.iteritems():
#     stc.save('induced_power_%s' % b)


###############################################################################
# plot mean power

for j, stcs in enumerate(stcs_normal):
    if j == 0:
        stcs_normal_alpha = stcs_normal[0]["alpha"].data.mean(axis=0)
    else:
        stcs_normal_alpha = np.vstack([stcs_normal_alpha,
                                       stcs_normal[j]["alpha"].data
                                       .mean(axis=0)])

for j, stcs in enumerate(stcs_hyp):
    if j == 0:
        stcs_hyp_alpha = stcs_hyp[0]["alpha"].data.mean(axis=0)
    else:
        stcs_hyp_alpha = np.vstack([stcs_hyp_alpha,
                                    stcs_hyp[j]["alpha"].data.mean(axis=0)])


times = stcs_normal[0]["alpha"].times
mean_normal = stcs_normal_alpha.mean(axis=0)
mean_hyp = stcs_hyp_alpha.mean(axis=0)
std_normal = stcs_normal_alpha.std(axis=0)
std_hyp = stcs_hyp_alpha.std(axis=0)
hyp_limits_normal = (mean_normal - std_normal, mean_normal + std_normal)
plt.fill_between(times, hyp_limits_normal[0], y2=hyp_limits_normal[1],
                 color='b', alpha=0.5)
plt.plot(times, mean_normal, label='Normal: Alpha', color='b')
hyp_limits_hyp = (mean_hyp - std_hyp, mean_hyp + std_hyp)
plt.fill_between(times, hyp_limits_hyp[0], y2=hyp_limits_hyp[1],
                 color='b', alpha=0.5)
plt.plot(times, mean_hyp, label='Hyp: Alpha', color='r')
plt.plot(times, mean_normal + std_normal, color='b--')

plt.plot(stcs['beta'].times, stcs['beta'].data.mean(axis=0), label='Beta')
plt.xlabel('Time (ms)')
plt.ylabel('Power')
plt.legend()
plt.title('Mean source induced power')
plt.show()


# %%
# Compute Power Spectral Density of inverse solution from single epochs
# define frequencies of interest
fmin, fmax = 0., 90.
bandwidth = 4.  # bandwidth of the windows in Hz
snr = 1.0  # use smaller SNR for raw data
lambda2 = 1.0 / snr ** 2
method = "MNE"  # use dSPM method (could also be MNE or sLORETA)
# compute source space psd in label

# Note: By using "return_generator=True" stcs will be a generator object
# instead of a list. This allows us so to iterate without having to
# keep everything in memory.

stcs_normal = compute_source_psd_epochs(epochs_normal, inverse_normal,
                                        lambda2=lambda2,
                                        method=method, fmin=fmin, fmax=fmax,
                                        bandwidth=bandwidth,
                                        return_generator=True)

# compute average PSD over the first 10 epochs
# n_epochs = 4
for i, stc in enumerate(stcs_normal):
    # if i >= n_epochs:
    #     break
    if i == 0:
        psd_normal = np.mean(stc.data, axis=0)
    else:
        psd_normal = np.vstack([psd_normal, np.mean(stc.data, axis=0)])

psd_avg_normal = np.mean(psd_normal, axis=0)
psd_std_normal = np.std(psd_normal, axis=0)
freqs = stc.times  # the frequencies are stored here

stcs_hyp = compute_source_psd_epochs(epochs_hyp, inverse_hyp,
                                     lambda2=lambda2,
                                     method=method, fmin=fmin, fmax=fmax,
                                     bandwidth=bandwidth,
                                     return_generator=True)

# compute average PSD over the first 10 epochs
for i, stc in enumerate(stcs_hyp):
    # if i >= n_epochs:
    #     break
    if i == 0:
        psd_hyp = np.mean(stc.data, axis=0)
    else:
        psd_hyp = np.vstack([psd_hyp, np.mean(stc.data, axis=0)])

psd_avg_hyp = np.mean(psd_hyp, axis=0)
psd_std_hyp = np.std(psd_hyp, axis=0)
freqs_hyp = stc.times  # the frequencies are stored here


plt.figure()
plt.plot(freqs, psd_avg_normal, color="b", label="Normal")
plt.plot(freqs, psd_avg_hyp, color="r", label="Hyp")
hyp_limits_normal = (psd_avg_normal - psd_std_normal, 
                     psd_avg_normal + psd_std_normal)
plt.fill_between(freqs, hyp_limits_normal[0], y2=hyp_limits_normal[1],
                 color='b', alpha=0.5)
hyp_limits_hyp = (psd_avg_hyp- psd_std_hyp, 
                  psd_avg_hyp+ psd_std_hyp)
plt.fill_between(freqs, hyp_limits_hyp[0], y2=hyp_limits_hyp[1],
                 color='b', alpha=0.5)

plt.xlabel('Freq (Hz)')
plt.ylabel('Power Spectral Density')
plt.legend()
plt.show()
