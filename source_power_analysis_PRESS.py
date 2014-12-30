import cPickle as Pickle
import os
import socket
import numpy as np
import networkx as nx

# import matplotlib.pyplot as plt
import mne
from mne.minimum_norm import read_inverse_operator
from mne import extract_label_time_course as extract_tc
from mne.stats import fdr_correction, bonferroni_correction


# Setup paths and prepare raw data
hostname = socket.gethostname()

if hostname == "wintermute":
    data_path = "/home/mje/mnt/Hyp_meg/scratch/Tone_task_MNE/"
    script_path = "/home/mje/mnt/Hyp_meg/scripts/MNE_analysis/"
    subjects_dir = "/home/mje/mnt/Hyp_meg/scratch/fs_subjects_dir/"
    n_jobs = 3
else:
    data_path = "/scratch1/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                "Tone_task_MNE/"
    script_path = "/projects/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                  "scripts/MNE_analysis/"
    subjects_dir = "/scratch1/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                   "fs_subjects_dir"
    n_jobs = 6

# Import MI functions
os.chdir(script_path)
from MI_functions import calc_MI, FDbinSize
from permTest import permutation_resampling

# change dir to save files the rigth place
os.chdir(data_path)

# Load files
inverse_fnormal = data_path + "tone_task_normal-inv.fif"
inverse_fhyp = data_path + "tone_task_hyp-inv.fif"

inverse_normal = read_inverse_operator(inverse_fnormal)
inverse_hyp = read_inverse_operator(inverse_fhyp)

src_normal = inverse_normal["src"]
src_hyp = inverse_hyp["src"]
# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
labels = mne.read_labels_from_annot('subject_1', parc='PALS_B12_Brodmann',
                                    regexp="Brodmann",
                                    subjects_dir=subjects_dir)

labels_name = []
for label in labels:
    labels_name += [label.name]

bands = ["theta", "alpha", "beta", "gamma_low", "gamma_high"]

for band in bands:
    # load source power files
    print "load normal"
    stcs_normal =\
        Pickle.load(open("stcs_normal_press_source_induced_%s_0-05.p" % band,
                         "rb"))

    print "load hyp"
    stcs_hyp =\
        Pickle.load(open("stcs_hyp_press_source_induced_%s_0-05.p" % band,
                         "rb"))

    # Extract time
    print "\n************* \nextracting TS\n************"
    label_ts_normal = []
    for j in range(len(stcs_normal)):
        label_ts_normal += [extract_tc(stcs_normal[j][band],
                                       labels,
                                       src_normal)]

    label_ts_hyp = []
    for j in range(len(stcs_hyp)):
        label_ts_hyp += [extract_tc(stcs_hyp[j][band],
                                    labels,
                                    src_hyp)]

    # Calculate Bin size
    n_trials_normal = 80  # labelTsNormalCrop.shape[0]
    n_labels_normal = 82  # labelTsNormalCrop.shape[1]

    n_trials_hyp = 74  # labelTsHypCrop.shape[0]
    n_labels_hyp = 82  # labelTsHypCrop.shape[1]

    # calculate the number of bins
    bins = np.empty(0)
    for t in range(80):
        for l in range(82):
            bins = np.append(bins, FDbinSize(label_ts_normal[t][l, :]))
    for t in range(74):
        for l in range(82):
            bins = np.append(bins, FDbinSize(label_ts_hyp[t][l, :]))

    bestBinsize = np.ceil(np.mean(bins))

    # calc MI for normal
    print "\n************* \ncalculating MI for normal\n*************"
    MI_normal = np.empty([n_labels_normal, n_labels_normal,
                          n_trials_normal])

    for h in range(n_trials_normal):
        counter = 0
        print "Normal #: ", h
        tmpResult = np.empty([n_labels_normal * n_labels_normal])
        for j in range(n_labels_normal):
            for k in range(n_labels_normal):
                tmpResult[counter] = (calc_MI(label_ts_normal[h][j, :],
                                              label_ts_normal[h][k, :],
                                              bestBinsize))
                counter += 1
        MI_normal[:, :, h] = np.reshape(tmpResult, [n_labels_normal,
                                                    n_labels_normal])

    # calc MI for Hyp
    print "\n************* \ncalculating MI for hyp\n*************"
    MI_hyp = np.empty([n_labels_hyp, n_labels_hyp, n_trials_hyp])

    for h in range(n_trials_hyp):
        counter = 0
        print "Hyp #: ", h
        tmpResult = np.empty([n_labels_hyp * n_labels_hyp])
        for j in range(n_labels_hyp):
            for k in range(n_labels_hyp):
                tmpResult[counter] = (calc_MI(label_ts_hyp[h][j, :],
                                              label_ts_hyp[h][k, :],
                                              bestBinsize))
                counter += 1
        MI_hyp[:, :, h] = np.reshape(tmpResult, [n_labels_hyp,
                                                 n_labels_hyp])

    fullMatrix = np.concatenate([MI_normal, MI_hyp], axis=2)

    threshold = np.median(fullMatrix[np.nonzero(fullMatrix)]) + \
        (np.std(fullMatrix[np.nonzero(fullMatrix)]))

    binMatrixNormal = MI_normal > threshold
    binMatrixHyp = MI_hyp > threshold

    # %%
    print "\n************* \nMaking network classes\n*************"
    nxNormal = []
    for j in range(binMatrixNormal.shape[2]):
        nxNormal += [nx.from_numpy_matrix(binMatrixNormal[:, :, j])]

    nxHyp = []
    for j in range(binMatrixHyp.shape[2]):
        nxHyp += [nx.from_numpy_matrix(binMatrixHyp[:, :, j])]

    degreesNormal = []
    for j, trial in enumerate(nxNormal):
        degreesNormal += [trial.degree()]

    degreesHyp = []
    for j, trial in enumerate(nxHyp):
        degreesHyp += [trial.degree()]

    ccNormal = []
    for j, trial in enumerate(nxNormal):
        ccNormal += [nx.cluster.clustering(trial)]
    ccHyp = []
    for j, trial in enumerate(nxHyp):
        ccHyp += [nx.cluster.clustering(trial)]

    # %% Degress
    print "\n************* \nTesting degrees\n*************"
    pvalList = []
    for degreeNumber in range(binMatrixHyp.shape[0]):

        postHyp = np.empty(len(degreesHyp))
        for j in range(len(postHyp)):
            postHyp[j] = degreesHyp[j][degreeNumber]

        postNormal = np.empty(len(degreesNormal))
        for j in range(len(postNormal)):
            postNormal[j] = degreesNormal[j][degreeNumber]

        pval, observed_diff, diffs = \
            permutation_resampling(postHyp, postNormal,
                                   10000, np.mean)

        pvalList += [{'pval': pval, "obsDiff": observed_diff, "diffs": diffs}]

    # %% for CC
    print "\n************* \nTesting cluster-coefficient\n*************"
    pvalListCC = []
    for ccNumber in range(binMatrixHyp.shape[0]):

        postHyp = np.empty(len(ccHyp))
        for j in range(len(ccHyp)):
            postHyp[j] = ccHyp[j][ccNumber]

        postNormal = np.empty(len(ccNormal))
        for j in range(len(postNormal)):
            postNormal[j] = ccNormal[j][ccNumber]

        pval, observed_diff, diffs = \
            permutation_resampling(postNormal, postHyp,
                                   10000, np.mean)

        pvalListCC += [{'pval': pval, "obsDiff": observed_diff,
                        "diffs": diffs}]

    # %% Correct for multiple comparisons

    pvals = np.empty(len(pvalList))
    for j in range(len(pvals)):
        pvals[j] = pvalList[j]["pval"]

    rejected, pvals_corrected = bonferroni_correction(pvals)

    print "\nSignificient regions for Degrees:"
    for i in range(len(labels_name)):
        if rejected[i] and pvalList[i]["obsDiff"] != 0:
            print "\n", labels_name[i], \
                "pval:", pvals_corrected[i], \
                "observed differnce:", pvalList[i]["obsDiff"], \

    results_degrees = []
    for i in range(len(labels_name)):
        if rejected[i] and pvalList[i]["obsDiff"] != 0:
            results_degrees += [{"label": labels_name[i],
                                "pval_corr": pvals_corrected[i],
                                 "obs_diff":
                                 pvalListCC[i]["obsDiff"],
                                 "mean_random_diff":
                                 np.asarray(pvalListCC[i]["diffs"]).mean()}]

    # %% for Cluster coefficient (CC)
    pvalsCC = np.empty(len(pvalListCC))
    for j in range(len(pvalsCC)):
        pvalsCC[j] = pvalListCC[j]["pval"]

    rejectedCC, pvals_correctedCC = bonferroni_correction(pvalsCC)

    print "\nSignificient regions for CC:"
    for i in range(len(labels_name)):
        if rejectedCC[i] and pvalListCC[i]["obsDiff"] != 0:
            print "\n", labels_name[i], \
                "pval:", pvals_correctedCC[i], \
                "observed differnce:", pvalListCC[i]["obsDiff"]

    results_CC = []
    for i in range(len(labels_name)):
        if rejectedCC[i] and pvalListCC[i]["obsDiff"] != 0:
            results_CC += [{"label": labels_name[i],
                            "pval_corr": pvals_correctedCC[i],
                            "obs_diff": pvalListCC[i]["obsDiff"],
                            "mean_random_diff:":
                            np.asarray(pvalListCC[i]["diffs"]).mean()}]

    results_all = [[results_degrees], [results_CC]]
    Pickle.dump(results_all,
                open("power_press_MI_%s_0-05_bonf.p" % band, "wb"))
