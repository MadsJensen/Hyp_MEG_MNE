# -*- coding: utf-8 -*-

import os
import cPickle as pickle
from surfer import Brain
import mne
import pandas as pd
import socket
from mayavi import mlab

# Setup paths and prepare raw data
hostname = socket.gethostname()

if hostname == "Wintermute":
    data_path = "/home/mje/mnt/Hyp_meg/scratch/Tone_task_MNE/"
    subjects_dir = "/home/mje/mnt/Hyp_meg/scratch/fs_subjects_dir/"
    n_jobs = 1
    os.environ["SUBJECTS_DIR"] = "/home/mje/mnt/Hyp_meg/" +\
                                 "scratch/fs_subjects_dir/"
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
labels = mne.read_labels_from_annot('subject_1', parc='PALS_B12_Brodmann',
                                    regexp="Brodmann",
                                    subjects_dir=subjects_dir)

# labels = mne.read_labels_from_annot('subject_1', parc='aparc.DKTatlas40',
#                                    subjects_dir=subjects_dir)


# make pd dataframe
results_press = pd.read_csv(
    "p_results_BA_press_surf-normal_MNE_zscore_0-05_LR_std.csv",
    header=None)
results_press.columns = ["area", "pval"]  # rename columns
results_press = results_press.sort("area")
# score_press = pd.read_csv(
#     "score_results_BAs_press_surf-normal_MNE_zscore_0-05_LR_no-std.csv",
#     header=None)
# results_press["score"] = score_press[1]
results_press["rejected"], results_press["pval_corr"] =\
    mne.stats.fdr_correction(results_press["pval"])
results_press.index = range(0, len(results_press))


results_tone = pd.read_csv(
    "p_results_BA_tone_surf-normal_MNE_zscore_-05-0_LR_std.csv",
    header=None)
results_tone.columns = ["area", "pval"]  # rename columns
results_tone = results_tone.sort("area")
# score_tone = pd.read_csv(
#     "score_results_BAs_tone_surf-normal_MNE_zscore_-05-0_LR_no-std.csv",
#     header=None)
# results_tone["score"] = score_tone[1]
results_tone["rejected"], results_tone["pval_corr"] =\
    mne.stats.fdr_correction(results_tone["pval"])
results_tone.index = range(0, len(results_tone))


# areas_to_plot_list =\
#     results_classf[results_classf["Rejected"] == True].Area.index

def areas_to_plot(dataframe):
        """
        """
        return dataframe[dataframe["rejected"] == True].ROI.index

list_press = areas_to_plot(classf_press)
list_tone = areas_to_plot(tone_post_clf)
list_commen = list_press.intersection(list_tone)


# setup "brain"
subject_id = "subject_1"
hemi = "both"
surf = "inflated"
brain = Brain(subject_id, hemi, surf, subjects_dir=subjects_dir)
labels_press = []
labels_tone, labels_common = [], []

list_commen = []
for area in list_press:
    if area not in list_commen:
        brain.add_label(labels[area],
                        color="blue",
                        alpha=0.7,
                        hemi=labels[area].hemi)
        brain.add_label(labels[area],
                        color="darkblue",
                        borders=True,
                        hemi=labels[area].hemi)

        labels_press += [labels[area]]

for area in list_tone:
    if area not in list_commen:
        brain.add_label(labels[area],
                        color="red",
                        alpha=0.7,
                        hemi=labels[area].hemi)
        brain.add_label(labels[area],
                        color="darkred",
                        borders=True,
                        hemi=labels[area].hemi)

        labels_tone += [labels[area]]

for area in list_commen:
        brain.add_label(labels[area],
                        color="white",
                        alpha=0.7,
                        hemi=labels[area].hemi)
        brain.add_label(labels[area],
                        color="black",
                        borders=True,
                        hemi=labels[area].hemi)

        labels_common += [labels[area]]

print "press:", [label.name for label in labels_press]
print "Tone:", [label.name for label in labels_tone]
print "Common:", [label.name for label in labels_common]


# plot network analysis

#bands = ["theta", "alpha", "beta", "gamma_low", "gamma_high"]
## bands = ["alpha"]
#conditions = ["press", "tone"]
#for condition in conditions:
#    for band in bands:
#        if condition is "press":
#            tmp = pickle.load(open(
#                "power_press_MI_DKT_%s_0-05_deg.p" % band,
#                "rb"))
#        elif condition is "tone":
#            tmp = pickle.load(open(
#                "power_tone_MI_DKT_%s_-05-0_deg.p" % band,
#                "rb"))
#
#        filter_keys = ['pval', 'area', 'obsDiff']
#        filtered_dict = []
#        for d in tmp:
#            filtered_dict += [{key: d[key] for key in filter_keys if key in d}]
#
#        result = pd.DataFrame(columns=filter_keys)
#        result = result.append(filtered_dict, ignore_index=True)
#        result["band"] = band
#        result["condition"] = condition
#
#        result["rejected"], result["pval_corr"] =\
#            mne.stats.fdr_correction(result["pval"])
#
#        exec("result_%s_%s=%s" % (condition, band, "result"))
#
#bands = ["beta"]
#for band in bands:
#    """
#    """
#
#    exec("tmp_press=result_press_%s" % (band))
#    exec("tmp_tone=result_tone_%s" % (band))
#
#    list_press = areas_to_plot(tmp_press)
#    list_tone = areas_to_plot(tmp_tone)
#    list_commen = list_press.intersection(list_tone)
#
#    # setup "brain"
#    subject_id = "subject_1"
#    hemi = "both"
#    surf = "inflated"
#    curv = True
#    brain = Brain(subject_id, hemi, surf, curv)
#    mlab.title(band)
#
#    labels_press, labels_tone, labels_common = [], [], []
#
#    for area in list_press:
#        if area not in list_commen:
#            brain.add_label(labels[area],
#                            color="blue",
#                            alpha=0.7,
#                            hemi=labels[area].hemi)
#            brain.add_label(labels[area],
#                            color="darkblue",
#                            borders=True,
#                            hemi=labels[area].hemi)
#
#            labels_press += [labels[area]]
#
#    for area in list_tone:
#        if area not in list_commen:
#            brain.add_label(labels[area],
#                            color="red",
#                            alpha=0.7,
#                            hemi=labels[area].hemi)
#            brain.add_label(labels[area],
#                            color="darkred",
#                            borders=True,
#                            hemi=labels[area].hemi)
#
#            labels_tone += [labels[area]]
#
#    for area in list_commen:
#            brain.add_label(labels[area],
#                            color="white",
#                            alpha=0.7,
#                            hemi=labels[area].hemi)
#            brain.add_label(labels[area],
#                            color="black",
#                            borders=True,
#                            hemi=labels[area].hemi)
#
#            labels_common += [labels[area]]
#
#    print "press:", [label.name for label in labels_press]
#    print "Tone:", [label.name for label in labels_tone]
#    print "Common:", [label.name for label in labels_common]
