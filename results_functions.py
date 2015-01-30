import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import os
import socket
import csv
import mne
import pandas as pd

from mne.stats import fdr_correction


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
# labels = mne.read_labels_from_annot('subject_1', parc='PALS_B12_Brodmann',
#                                    regexp="Brodmann",
#                                    subjects_dir=subjects_dir)

labels = mne.read_labels_from_annot('subject_1', parc='aparc.DKTatlas40',
                                    subjects_dir=subjects_dir)

# %%
# bands = ["theta", "alpha", "beta", "gamma_low", "gamma_high"]
conditions = ["press", "tone"]
for condition in conditions:
    if condition is "press":
        tmp = pickle.load(open(
            "MI_press_zscore_DKT_0-05_resample_crop_deg.p", "rb"))
    elif condition is "tone":
        tmp = pickle.load(open(
            "MI_tone_zscore_DKT_-05-0_resample_crop_deg.p", "rb"))

    filter_keys = ['pval', 'area', 'obsDiff']
    filtered_dict = []
    for d in tmp:
        filtered_dict += [{key: d[key] for key in filter_keys if key in d}]

    result = pd.DataFrame(columns=filter_keys)
    result = result.append(filtered_dict, ignore_index=True)
    result["condition"] = condition

    result["rejected"], result["pval_corr"] =\
        mne.stats.fdr_correction(result["pval"])

    exec("result_%s=%s" % (condition, "result"))

for condition in conditions:
        exec("%s=result_%s" % ("result", condition))
        print "\nCondition: %s" % (condition)
        print result[(result["obsDiff"] != 0)
                     & (result["rejected"] == True)]
# %%

bands=["theta", "alpha", "beta", "gamma_low", "gamma_high"]
# bands = ["beta"]
conditions = ["degrees"]

for band in bands:
    result = pickle.load(open(
        "power_press_MI_%s_0-05_fdr_DKT.p" % band, "rb"))
    result = result[:][:]
    for i, condition in enumerate(conditions):
        for j in range(len(result[i])):
            tmp = result[i][j]["obs_diff"]
            if tmp != 0:
                print "condition: % s; band: %s" % (condition, band)
                print result[i][j]["label"]
                print result[i][j]

with open('mycsvfile.csv', 'wb') as f:  # Just use 'w' mode in 3.x
    w = csv.DictWriter(f, result[0][0][0].keys())
    w.writeheader()

    for j in range(2):
        w.writerow(result[j][0][0])


# %%pd
bands=["theta", "alpha", "beta"]
conditions=["press", "tone"]
# bands=["beta", "gamma_low"]
for condition in conditions:
    for band in bands:
        if condition is "press":
            tmp=pickle.load(open(
                "network_press_MTCOH_DKT_%s_0-05_deg.p"
                % band, "rb"))
        elif condition is "tone":
            tmp=pickle.load(open(
                "network_tone_MTCOH_DKT_%s_-05-0_deg.p"
                % band, "rb"))

        filter_keys = ['pval', 'area', 'obsDiff']
        filtered_dict = []
        for d in tmp:
            filtered_dict += [{key: d[key] for key in filter_keys if key in d}]

        result = pd.DataFrame(columns=filter_keys)
        result = result.append(filtered_dict, ignore_index=True)
        result["band"] = band
        result["condition"] = condition

        result["rejected"], result["pval_corr"] =\
            mne.stats.fdr_correction(result["pval"])

        exec("result_%s_%s=%s" % (condition, band, "result"))

#bands = ["theta", "alpha", "beta"]
for band in bands:
    for condition in conditions:
            exec("%s=result_%s_%s" % ("result", condition, band))
            print "\nCondition: %s, band: %s" % (condition, band)
            print result[(result["obsDiff"] != 0)
                         & (result["rejected"] == True)]
            
# %%

    # plot functions

stcs_normal = pickle.load(
    open("stcs_normal_tone_source_induced_beta_-05-0.p", "rb"))

stcs_hyp = pickle.load(
    open("stcs_hyp_tone_source_induced_beta_-05-0.p", "rb"))

times = stcs_normal[0].values()[0].times


data_norm = np.empty([len(stcs_normal), 501])
for j, stc in enumerate(stcs_normal):
    data_norm[j, :] = stc.values()[0].in_label(labels[6]).data.mean(axis=0)

data_hyp = np.empty([len(stcs_hyp), 501])
for j, stc in enumerate(stcs_hyp):
    data_hyp[j, :] = stc.values()[0].in_label(labels[6]).data.mean(axis=0)

# normal_std = data_norm.std(axis=0)
# hyp_std = data_hyp.std(axis=0)

# for i in range(len(data_norm)):
#     plt.plot(times, np.mean(data_norm, axis=0), 'b')

# for i in range(len(data_hyp)):
#     plt.plot(times, np.mean(data_hyp, axis=0), 'r')

#     plt.xlabel('Times (seconds)')
#     plt.ylabel('dSPM value')

# plt.show()


def plot_TS_from_ROI(TS_1, TS_2, name, times, y_label="MI value", show=True):

    plt.plot(times, TS_1.mean(axis=0), "b")
    plt.plot(times, TS_2.mean(axis=0), "r")

    normal_std = TS_1.std(axis=0)
    hyp_std = TS_2.std(axis=0)

    plt.xlabel('Times (seconds)')
    plt.ylabel(y_label)
    plt.title('Mean Time series for %s' % (name))

    plt.plot(times, TS_2.mean(axis=0), 'r',  linewidth=3,
             label="mean activation in hypnosis")
    plt.plot(times, TS_2.mean(axis=0) + hyp_std, 'r--', alpha=0.8)
    plt.plot(times, TS_2.mean(axis=0) - hyp_std, 'r--', alpha=0.8)

    plt.plot(times, TS_1.mean(axis=0), 'b', linewidth=3,
             label="mean activation in normal")
    plt.plot(times, TS_1.mean(axis=0) + normal_std, 'b--', alpha=0.8)
    plt.plot(times, TS_1.mean(axis=0) - normal_std, 'b--', alpha=0.8)

    plt.legend()
    if show:
        plt.show()

    return plt.figure()

# %% load Class csv file
classf_press = pd.read_csv(
    "p_results_DKT_press_surf-normal_MNE_zscore_-02-0_LR_no-std.csv",
    header=None)
classf_press.columns = ["area", "pval"]  # rename columns
classf_press = classf_press.sort("area")

res_score = pd.read_csv(
    "score_results_DKT_press_surf-normal_MNE_zscore_-02-0_LR_no-std.csv",
    header=None)

classf_press["score"] = res_score[1]
classf_press["rejected"], classf_press["pval_corr"] =\
    fdr_correction(classf_press["pval"])
classf_press.index = range(0, len(classf_press))
    

classf_press["rejected"], classf_press["pval_corr"] =\
    fdr_correction(classf_press["pval"])

print classf_press[classf_press["rejected"]  == True]

classf_tone = pd.read_csv(
    "p_results_DKT_tone_surf-normal_MNE_zscore_-05-0_LR_no-std.csv",
    header=None)
classf_tone.columns = ["area", "pval"]  # rename columns
classf_tone = classf_tone.sort("area")

res_score = pd.read_csv(
    "score_results_DKT_tone_surf-normal_MNE_zscore_-05-0_LR_no-std.csv",
    header=None)

classf_tone["score"] = res_score[1]


classf_tone["rejected"], classf_tone["pval_corr"] =\
    fdr_correction(classf_tone["pval"])
classf_tone.index = range(0, len(classf_tone))

print classf_tone[classf_tone["rejected"]  == True]
