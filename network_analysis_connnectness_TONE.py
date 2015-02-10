import numpy as np
import networkx as nx
import numpy.random as npr
import cPickle as pickle
import os
import socket
import mne
import bct
# import cPickle as pickle

from mne.minimum_norm import (apply_inverse_epochs, read_inverse_operator)
from mne.baseline import rescale
from nitime.analysis import MTCoherenceAnalyzer
from nitime import TimeSeries
# from mne.stats import fdr_correction


# %% Permutation test
def permutation_test(a, b, num_samples, statistic):
    """Returns p-value that statistic for a is different
    from statistc for b."""

    observed_diff = abs(statistic(b) - statistic(a))
    num_a = len(a)

    combined = np.concatenate([a, b])
    diffs = []
    for i in range(num_samples):
        xs = npr.permutation(combined)
        diff = np.mean(xs[:num_a]) - np.mean(xs[num_a:])
        diffs.append(diff)

    pval = np.sum(np.abs(diffs) >= np.abs(observed_diff)) / float(num_samples)
    return pval, observed_diff, diffs


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

result_dir = data_path + "network_connect_res"

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

epochs_normal = epochs_normal["Tone"]
epochs_hyp = epochs_hyp["Tone"]


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
labels = mne.read_labels_from_annot('subject_1', parc='aparc.a2009s',
                                   regexp="[G|S]",
                                   subjects_dir=subjects_dir)

labels = mne.read_labels_from_annot('subject_1', parc='aparc.DKTatlas40',
                                    subjects_dir=subjects_dir)
labels_name = [label.name for label in labels]

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


fromTime = np.argmax(stcsNormal[0].times == -0.5)
toTime = np.argmax(stcsNormal[0].times == 0)

labelTsNormalRescaledCrop = []
for j in range(len(labelTsNormal)):
    labelTsNormalRescaledCrop += [labelTsNormalRescaled[j][:, fromTime:toTime]]

labelTsHypRescaledCrop = []
for j in range(len(labelTsHyp)):
    labelTsHypRescaledCrop += [labelTsHypRescaled[j][:, fromTime:toTime]]


# %%
cohListNormal = []
cohListHyp = []

for j in range(len(labelTsNormalRescaledCrop)):
    nits = TimeSeries(labelTsNormalRescaledCrop[j],
                      sampling_rate=250)  # epochs_normal.info["sfreq"])
    nits.metadata["roi"] = labels_name

    cohListNormal += [MTCoherenceAnalyzer(nits)]

for j in range(len(labelTsHypRescaledCrop)):
    nits = TimeSeries(labelTsHypRescaledCrop[j],
                      sampling_rate=250)  # epochs_normal.info["sfreq"])
    nits.metadata["roi"] = labels_name

    cohListHyp += [MTCoherenceAnalyzer(nits)]

# Compute a source estimate per frequency band
bands = dict(theta=[4, 8],
             alpha=[8, 12],
             beta=[13, 25],
             gamma_low=[30, 48],
             gamma_high=[52, 90])

# bands = dict(alpha=[8, 12])

pvals = []

for band in bands.keys():
    print "\n******************"
    print "\nAnalysing band: %s" % band
    print "\n******************"

    # extract coherence values
    f_lw, f_up = bands[band]  # lower & upper limit for frequencies

    cohMatrixNormal = np.empty([len(labels_name),
                                len(labels_name),
                                len(labelTsNormalRescaledCrop)])
    cohMatrixHyp = np.empty([len(labels_name),
                             len(labels_name),
                             len(labelTsHypRescaledCrop)])

    # confine analysis to Aplha (8  12 Hz)
    freq_idx = np.where((cohListHyp[0].frequencies >= f_lw) *
                        (cohListHyp[0].frequencies <= f_up))[0]

    print cohListNormal[0].frequencies[freq_idx]

    # compute average coherence &  Averaging on last dimension
    for j in range(cohMatrixNormal.shape[2]):
        cohMatrixNormal[:, :, j] = np.mean(
            cohListNormal[j].coherence[:, :, freq_idx], -1)

    for j in range(cohMatrixHyp.shape[2]):
        cohMatrixHyp[:, :, j] = np.mean(
            cohListHyp[j].coherence[:, :, freq_idx], -1)

    #
    fullMatrix = np.concatenate([cohMatrixNormal, cohMatrixHyp], axis=2)

    threshold = np.median(fullMatrix[np.nonzero(fullMatrix)]) + \
        np.std(fullMatrix[np.nonzero(fullMatrix)])

    binMatrixNormal = cohMatrixNormal > threshold
    binMatrixHyp = cohMatrixHyp > threshold

    #
    nxNormal = []
    for j in range(binMatrixNormal.shape[2]):
        nxNormal += [nx.from_numpy_matrix(binMatrixNormal[:, :, j])]

    nxHyp = []
    for j in range(binMatrixHyp.shape[2]):
        nxHyp += [nx.from_numpy_matrix(binMatrixHyp[:, :, j])]

    eff_normal = np.empty(binMatrixNormal.shape[2])
    for j, graph in enumerate(nxNormal):
        eff_normal[j] = bct.efficiency_bin(nx.to_numpy_matrix(graph))

    eff_hyp = np.empty(binMatrixHyp.shape[2])
    for j, graph in enumerate(nxHyp):
        eff_hyp[j] = bct.efficiency_bin(nx.to_numpy_matrix(graph))

    cc_hyp = np.asarray([np.mean(nx.cluster.clustering(g).values())
                        for g in nxHyp])

    cc_normal = np.asarray([np.mean(nx.cluster.clustering(g).values())
                            for g in nxNormal])

    deg_hyp= np.asarray([np.mean(g.degree().values())
                            for g in nxHyp])
                                
    deg_normal = np.asarray([np.mean(g.degree().values())
                            for g in nxNormal])
    
    trans_hyp= np.asarray([nx.cluster.transitivity(g)
                            for g in nxHyp])
                                
    trans_normal = np.asarray([nx.cluster.transitivity(g)
                            for g in nxNormal])
    
    np.savetxt("network_connect_res/deg_tone_normal_%s.csv"
        % band, deg_normal)
    np.savetxt("network_connect_res/deg_tone_hyp_%s.csv"
        % band, deg_hyp)        

    np.savetxt("network_connect_res/cc_tone_normal_%s.csv"
        % band, cc_normal)
    np.savetxt("network_connect_res/cc_tone_hyp_%s.csv"
        % band, cc_hyp)  
    
    np.savetxt("network_connect_res/trans_tone_normal_%s.csv"
        % band, trans_normal)
    np.savetxt("network_connect_res/trans_tone_hyp_%s.csv"
        % band, trans_hyp)   
        
    np.savetxt("network_connect_res/eff_tone_normal_%s.csv"
        % band, eff_normal)
    np.savetxt("network_connect_res/eff_tone_hyp_%s.csv"
        % band, eff_hyp)   

#
#    
#    pval, obs_diff, diffs =\
#        permutation_test(cc_normal, cc_hyp, 10000, np.mean)
#    print band, pval, obs_diff, np.mean(diffs)
#
#    results_cc = {"pval": pval, "obs_diff": obs_diff, "diffs": diffs, 
#               "real_diff": np.mean(cc_normal.mean() - cc_hyp.mean())}
#    
#    pickle.dump(results_cc,
#            open(result_dir + \
#            "/network_connect_tone_zscore_DKT_%s_0-05_resample_crop_CC.p"
#                 % band, "wb"))
#
#    pval, obs_diff, diffs =\
#        permutation_test(deg_normal, deg_hyp, 10000, np.mean)
#    print band, pval, obs_diff, np.mean(diffs)
#
#    results_deg = {"pval": pval, "obs_diff": obs_diff, "diffs": diffs, 
#               "real_diff": np.mean(deg_normal.mean() - deg_hyp.mean())}
#    
#    pickle.dump(results_deg,
#            open(result_dir + \
#            "/network_connect_tone_zscore_DKT_%s_0-05_resample_crop_deg.p"
#                 % band, "wb"))
#
#    pval, obs_diff, diffs =\
#        permutation_test(trans_normal, trans_hyp, 10000, np.mean)
#    print band, pval, obs_diff, np.mean(diffs)
#
#    results_trans = {"pval": pval, "obs_diff": obs_diff, "diffs": diffs, 
#               "real_diff": np.mean(trans_normal.mean() - trans_hyp.mean())}
#    
#    pickle.dump(results_trans,
#            open(result_dir + \
#            "/network_connect_tone_zscore_DKT_%s_0-05_resample_crop_trans.p"
#                 % band, "wb"))
