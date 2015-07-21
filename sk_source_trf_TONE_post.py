# import cPickle as pickle
import os
import socket
import numpy as np
import matplotlib.pylab as plt
import mne
import csv
# import numpy as np
from mne.minimum_norm import (read_inverse_operator,
                              source_induced_power)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import (StratifiedShuffleSplit,
                                      permutation_test_score)


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
elif hostname == "isis":
    data_path = "/scratch1/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                "Tone_task_MNE/"
    subjects_dir = "/scratch1/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                   "fs_subjects_dir"
    n_jobs = 2
else:
    raise RuntimeWarning('Unkwon host')

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

epochs_normal = epochs_normal["Tone"]
epochs_hyp = epochs_hyp["Tone"]

epochs_normal.resample(250)
epochs_hyp.resample(250)

label_dir = subjects_dir + "/subject_1/label/"
labels = mne.read_labels_from_annot('subject_1', parc='aparc.a2009s',
                                    regexp="[G|S]",
                                    subjects_dir=subjects_dir)
label_single = [labels[3]]

#
snr = 1.0  # Standard assumption for average data but using it for single trial
lambda2 = 1.0 / snr ** 2
method = "dSPM"

frequencies = np.arange(7, 48, 2)  # define frequencies of interest
label_single = [labels[13]]
n_cycles = frequencies / 3.  # different number of cycle per frequency

# Classification setting
n_splits = 10
LR = LogisticRegression()
gnb = GaussianNB()

classifiers = [gnb]
clf_names = ["gnb"]

for h, clf in enumerate(classifiers):
    p_results = {}
    score_results = {}

    for label in labels:
        print "Working on: %s" % label.name

        tfr_result = np.empty([len(epochs_normal) + len(epochs_hyp),
                               len(frequencies),
                               len(epochs_normal[0].times)])

        for j, slet in enumerate(epochs_normal):
            print "\n" + "======================="
            print "TFR working on %d of %d" % (j+1, len(epochs_normal))
            print "=======================" + "\n"

            sip, phase_lock = source_induced_power(epochs_normal[j],
                                                   inverse_normal,
                                                   frequencies,
                                                   label=label,
                                                   lambda2=lambda2,
                                                   method=method,
                                                   nave=1,
                                                   n_cycles=n_cycles,
                                                   baseline=(-1, -0.7),
                                                   baseline_mode='percent',
                                                   n_jobs=n_jobs)

            tfr_result[j, :, :] = sip.mean(axis=0)

        for j, slet in enumerate(epochs_hyp):
            print "\n" + "======================="
            print "TFR working on %d of %d" % (j+1, len(epochs_hyp))
            print "=======================" + "\n"

            sip, phase_lock = source_induced_power(epochs_hyp[j],
                                                   inverse_hyp,
                                                   frequencies,
                                                   label=label,
                                                   lambda2=lambda2,
                                                   method=method,
                                                   nave=1,
                                                   n_cycles=n_cycles,
                                                   baseline=(-1, -0.7),
                                                   baseline_mode='percent',
                                                   n_jobs=n_jobs)

            tfr_result[j + len(epochs_normal), :, :] = sip.mean(axis=0)

        from_time = np.abs(epochs_normal[0].times - 0).argmin()
        to_time = np.abs(epochs_normal[0].times - 0.2).argmin()

        times = epochs_normal.times[from_time:to_time]

        tfr_crop = tfr_result[:, :, from_time:to_time]
        X = tfr_crop.reshape([tfr_crop.shape[0],
                             tfr_crop.shape[1]*tfr_crop.shape[2]])
        y = np.concatenate([np.zeros(len(epochs_normal)),
                            np.ones(len(epochs_hyp))])

        cv = StratifiedShuffleSplit(y, n_splits, test_size=0.2)

        score, permutation_scores, pvalue =\
            permutation_test_score(
                clf, X, y, scoring="accuracy",
                cv=cv, n_permutations=5000, verbose=True)

        score_results[label.name] = score
        p_results[label.name] = pvalue

        outfile_p_name = "p_results_DA_tone_TFR_" +\
            "dSPM_0-02_%s.csv" % clf_names[h]
        outfile_score_name = "score_results_DA_tone_TFR_" +\
            "dSPM_0-02_%s.csv" % clf_names[h]

        with open(outfile_p_name, "w") as outfile:
            writer = csv.writer(outfile)
            for key, val in p_results.items():
                writer.writerow([key, val])

        with open(outfile_score_name, "w") as outfile:
            writer = csv.writer(outfile)
            for key, val in score_results.items():
                writer.writerow([key, val])
