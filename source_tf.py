import mne
import socket
import os
from mne.minimum_norm import read_inverse_operator, source_band_induced_power

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


# Compute a source estimate per frequency band
bands = dict(alpha=[8, 12], beta=[13, 20])

stcs_normal = source_band_induced_power(epochs_normal, inverse_normal, bands,
                                        n_cycles=2, use_fft=False, n_jobs=3)
stcs_hyp = source_band_induced_power(epochs_hyp, inverse_hyp, bands,
                                     n_cycles=2, use_fft=False, n_jobs=3)

# for b, stc in stcs.iteritems():
#     stc.save('induced_power_%s' % b)

###############################################################################
# plot mean power
import matplotlib.pyplot as plt
plt.plot(stcs_normal['alpha'].times, stcs_normal['alpha'].data.mean(axis=0),
         label='Alpha', color='b')
plt.plot(stcs_hyp['alpha'].times, stcs_hyp['alpha'].data.mean(axis=0),
         label='Alpha', color='r')

plt.plot(stcs['beta'].times, stcs['beta'].data.mean(axis=0), label='Beta')
plt.xlabel('Time (ms)')
plt.ylabel('Power')
plt.legend()
plt.title('Mean source induced power')
plt.show()
