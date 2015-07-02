import cPickle as pickle
import os
import socket
import numpy as np
import matplotlib.pylab as plt

# import matplotlib.pyplot as plt
import mne
# import numpy as np
from mne.minimum_norm import (read_inverse_operator,
                              source_induced_power)


def plot_tfr(data, times, frequencies, max_freq):
    """
    Parameteres:
    ------------

    data : data to be plotted
    times : times for the data
    frequencies : the frequencies
    max_freq : the upper limit to show
    """

    plt.imshow(20 * data,
               extent=[times[0], times[-1], frequencies[0], frequencies[-1]],
               aspect='auto', origin='lower', vmin=0., vmax=30., cmap='RdBu_r')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    # plt.title('Power (%s)' % title)
    plt.colorbar()


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
label_dir = subjects_dir + "/subject_1/label/"
labels = mne.read_labels_from_annot('subject_1', parc='aparc.a2009s',
                                    regexp="[G|S]",
                                    subjects_dir=subjects_dir)
#
snr = 1.0  # Standard assumption for average data but using it for single trial
lambda2 = 1.0 / snr ** 2
method = "dSPM"

frequencies = np.arange(7, 48, 2)  # define frequencies of interest
label_single = [labels[13]]
n_cycles = frequencies / 3.  # different number of cycle per frequency


tfr_result = np.empty([len(epochs_normal) + len(epochs_hyp),
                       len(frequencies),
                       len(epochs_normal[0].times)])

for j, label in enumerate(epochs_normal):
    print "\n" + "==================="
    print "working on %d of %d" % (j+1, len(epochs_normal))
    print "===================" + "\n"

    sip, phase_lock = source_induced_power(
        epochs_normal[j], inverse_normal, frequencies, label_single[0],
        baseline=(-1, -0.7),
        baseline_mode='percent', n_cycles=n_cycles, n_jobs=1)

    tfr_result[j, :, :] = sip.mean(axis=0)

for j, label in enumerate(epochs_hyp):
    print "\n" + "==================="
    print "working on %d of %d" % (j+1, len(epochs_hyp))
    print "===================" + "\n"

    sip, phase_lock = source_induced_power(
        epochs_hyp[j], inverse_hyp, frequencies, label_single[0],
        baseline=(-1, -0.7),
        baseline_mode='percent', n_cycles=n_cycles, n_jobs=1)

    tfr_result[j + len(epochs_normal), :, :] = sip.mean(axis=0)
