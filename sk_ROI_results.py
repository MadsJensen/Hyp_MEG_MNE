# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import socket
import mne
import pandas as pd

from mne.minimum_norm import read_inverse_operator, apply_inverse_epochs
# from mne.baseline import rescale


from mne.stats import fdr_correction

plt.ion()


def load_result(fname):
    """
    Keyword Arguments:
    name -- the file to be loaded.
        """

    result_clf = pd.read_csv(
        fname,
        header=None)
    result_clf.columns = ["ROI", "pval"]  # rename columns
    result_clf = result_clf.sort("ROI")

    res_score = pd.read_csv(
        "score_" + fname[2:],
        header=None)

    result_clf["score"] = res_score[1]
    result_clf["rejected"], result_clf["pval_corr"] =\
        fdr_correction(result_clf["pval"])
    result_clf.index = range(0, len(result_clf))

    result_clf["rejected"], result_clf["pval_corr"] =\
        fdr_correction(result_clf["pval"])

    ROIs = [roi[:-3] for roi in result_clf.ROI]
    hemi = [roi[-2:] for roi in result_clf.ROI]
    result_clf["hemi"] = hemi
    result_clf.ROI = ROIs

    return result_clf


# Setup paths and prepare raw data
hostname = socket.gethostname()

if hostname == "Wintermute":
    data_path = "/home/mje/mnt/Hyp_meg/scratch/Tone_task_MNE/"
    subjects_dir = "/home/mje/mnt/Hyp_meg/scratch/fs_subjects_dir/"
    n_jobs = 1
elif hostname == "isis":
    data_path = "/scratch1/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                "Tone_task_MNE/"
    script_path = "/projects/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                  "scripts/MNE_analysis/"
    subjects_dir = "/scratch1/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                   "fs_subjects_dir"
    n_jobs = 6
else:
    print "unknown host"

os.chdir(data_path)

# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
labels = mne.read_labels_from_annot('subject_1', parc='aparc.a2009s',
                                    regexp="[G|S]",
                                    subjects_dir=subjects_dir)
# labels = mne.read_labels_from_annot('subject_1', parc='PALS_B12_Brodmann',
#                                     regexp="Brodmann",
#                                     subjects_dir=subjects_dir)

# load Class csv file
press_post_clf = load_result(
    "p_results_DA_press_surf-normal_dSPM_0-05_LR_std_mean.csv")
press_post_index =\
    press_post_clf[press_post_clf["rejected"] == True].index.get_values()
print "Press POST\n", press_post_clf[press_post_clf["rejected"] == True]


press_pre_clf = load_result(
    "p_results_DA_press_surf-normal_dSPM_-02-0_LR_std_mean.csv")
press_pre_index =\
    press_pre_clf[press_pre_clf["rejected"] == True].index.get_values()
print "Press PRE\n", press_pre_clf[press_pre_clf["rejected"] == True]


tone_post_clf = load_result(
    "p_results_DA_tone_surf-normal_MNE_0-02_LR_std_mean.csv")
tone_post_index =\
    tone_post_clf[tone_post_clf["rejected"] == True].index.get_values()
print "Tone POST\n", tone_post_clf[tone_post_clf["rejected"] == True]


tone_pre_clf = load_result(
    "p_results_DA_tone_surf-normal_MNE_-05-0_LR_std_mean.csv")
tone_pre_index =\
    tone_pre_clf[tone_post_clf["rejected"] == True].index.get_values()
print "Tone PRE\n", tone_pre_clf[tone_pre_clf["rejected"] == True]


# LOAD DATA
epochs_fnormal = data_path + "tone_task_normal-epo.fif"
epochs_fhyp = data_path + "tone_task_hyp-epo.fif"
inverse_fnormal = data_path + "tone_task_normal-inv.fif"
inverse_fhyp = data_path + "tone_task_hyp-inv.fif"

epochs_normal = mne.read_epochs(epochs_fnormal)
epochs_hyp = mne.read_epochs(epochs_fhyp)

epo_normal = epochs_normal.copy()
epo_hyp = epochs_hyp.copy()

epochs_normal = epochs_normal["Tone"]
epochs_hyp = epochs_hyp["Tone"]

evoked_normal = epochs_normal.average()
evoked_hyp = epochs_hyp.average()

snr = 1.0  # Standard assumption for average data but using it for single trial
lambda2 = 1.0 / snr ** 2
method = "dSPM"

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

# [stc.crop(-0.2, 0) for stc in stcs_normal]
# [stc.crop(-0.2, 0) for stc in stcs_hyp]


label_dir = subjects_dir + "/subject_1/label/"

labels = mne.read_labels_from_annot('subject_1', parc='aparc.a2009s',
                                    regexp="[G|S]",
                                    subjects_dir=subjects_dir)
labels_name = [label.name for label in labels]

press_labels = [labels[index] for index in press_post_index]
tone_post_labels = [labels[index] for index in tone_post_index]

# PLOTS
# find index for start and stop times
# labels_single = [labels[-3]]

# from_time = np.abs(stcs_normal[0].times - 0).argmin()
# to_time = np.abs(stcs_normal[0].times - 0.5).argmin()

from_time = stcs_normal[0].times[0]
to_time = stcs_normal[0].times[-1]

for h, label in enumerate(tone_post_labels):
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

    # labelTsNormalRescaled = []
    # for j in range(len(labelTsNormal)):
    #     labelTsNormalRescaled += [rescale(labelTsNormal[j],
    #                                       stcs_normal[0].times,
    #                                       baseline=(None, -0.7),
    #                                       mode="zscore")]

    # labelTsHypRescaled = []
    # for j in range(len(labelTsHyp)):
    #     labelTsHypRescaled += [rescale(labelTsHyp[j],
    #                                    stcs_hyp[0].times,
    #                                    baseline=(None, -0.7),
    #                                    mode="zscore")]

    # labelTsNormalRescaledCrop = []
    # for j in range(len(labelTsNormal)):
    #     labelTsNormalRescaledCrop +=\
    #         [labelTsNormalRescaled[j][:, from_time:to_time]]

    # labelTsHypRescaledCrop = []
    # for j in range(len(labelTsHyp)):
    #     labelTsHypRescaledCrop +=\
    #         [labelTsHypRescaled[j][:, from_time:to_time]]

    results_pd = []
    times = stcs_normal[0].times

    for j in range(len(labelTsNormal)):
        results_pd.append(pd.DataFrame(dict(condition="normal",
                                            time=times * 1e3,
                                            trial=j,
                                            Value=labelTsNormal[j].reshape(-1))))

    for j in range(len(labelTsHyp)):
        results_pd.append(pd.DataFrame(dict(condition="hyp",
                                            time=times * 1e3,
                                            trial=j,
                                            Value=labelTsHyp[j].reshape(-1))))

    results_pd = pd.concat(results_pd)

    color_map = dict(hyp="indianred", normal="steelblue")

    title = "Label: %s, score: %0.4f" % (label.name,
                                         tone_post_clf.ix[tone_post_index[h]].score)

    plt.figure()
    ax = sns.tsplot(results_pd,
                    time="time",
                    unit="trial",
                    condition="condition",
                    value="Value",
                    # err_style="boot_traces",
                    linewidth=3,
                    color=color_map)
    ax.set_xlabel("time (miliseconds)")
    ax.set_ylabel("dSPM")
    ax.set_title(title)

    # ax.set_title(labels_single[0].name)


# Boxplot

# data_normal = np.asarray([t.mean() for t in labelTsNormalRescaledCrop])
# data_hyp = np.asarray([t.mean() for t in labelTsHypRescaledCrop])
# sns.boxplot([data_normal, data_hyp])

for j, label in enumerate(tone_post_labels):
    print label.name
    print "score: %f3" % (tone_post_clf.ix[tone_post_index[0]].score)
