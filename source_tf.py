import cPickle as Pickle
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
    n_jobs = 6

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

labels = mne.read_labels_from_annot('subject_1', parc='PALS_B12_Brodmann',
                                    regexp="Brodmann",
                                    subjects_dir=subjects_dir)

snr = 1.0  # Standard assumption for average data but using it for single trial
lambda2 = 1.0 / snr ** 2

# %%
# Compute a source estimate per frequency band
bands = dict(alpha=[8, 12],
             beta=[13, 25],
             gamma_low=[30, 48],
             gamma_high=[52, 90])


# stcs_normal = []
# for j in range(len(epochs_normal)):
#     stcs_normal += [source_band_induced_power(epochs_normal[j],
#                                               inverse_normal,
#                                               bands,
#                                               lambda2=lambda2,
#                                               method="MNE",
#                                           n_cycles=2,
#                                           use_fft=False,
#                                           baseline=(-0.95, -0.7),
#                                           baseline_mode="zscore",
#                                           n_jobs=1)]

stcs_hyp = []
for j in range(len(epochs_hyp)):
    stcs_hyp += [source_band_induced_power(epochs_hyp[j], inverse_hyp,
                                           bands,
                                           lambda2=lambda2,
                                           method="MNE",
                                           n_cycles=2, use_fft=False,
                                           baseline=(-0.95, -0.7),
                                           baseline_mode="zscore",
                                           n_jobs=1)]


# for j, stc in enumerate(stcs_normal):
#     stc["alpha"].crop(-0.5, 0)
#     stc["beta"].crop(-0.5, 0)
#     stc["gamma_low"].crop(-0.5, 0)
#     stc["gamma_high"].crop(-0.5, 0)

bands = ["alpha", "beta", "gamma_low", "gamma_high"]    
for band in bands:
    print band
    for stc in stcs_hyp:
        stc["%s" % band].crop(-0.5, 0)

for band in bands:
    print band    
    stcs_normal_alpha = []
    for stc in stcs_hyp:
        stcs_normal_alpha += [stc["%s" %(band)]] 

    Pickle.dump(stcs_normal_alpha,
                open("stcs_hyp_tone_souce_induced_%s_-05-0.p" % band, "wb" ))


# Pickle.dump(stcs_hyp, open("stcs_hyp_source_tone_induced_A_B_Gl_Gu-05-0.p", "wb"))

# for b, stc in stcs.iteritems():
#     stc.save('induced_power_%s' % b)


# %% extract labels
src_normal = inverse_normal['src']

stcs_normal_alpha = []
stcs_normal_beta = []
for stc in stcs_normal:
    stcs_normal_alpha += [stc["alpha"]]
    stcs_normal_beta += [stc["beta"]]

labelTsNormal = mne.extract_label_time_course(stcs_normal_alpha, labels,
                                              src_normal,
                                              mode='mean_flip',
                                              return_generator=False)

labelTsNormalBeta = mne.extract_label_time_course(stcs_normal_beta, labels,
                                                  src_normal,
                                                  mode='mean_flip',
                                                  return_generator=False)

src_hyp = inverse_hyp['src']
stcs_hyp_alpha = []
stcs_hyp_beta = []
for stc in stcs_hyp:
    stcs_hyp_alpha += [stc["alpha"]]
    stcs_hyp_beta += [stc["beta"]]
    
labelTsHyp = mne.extract_label_time_course(stcs_hyp_alpha, labels,
                                              src_hyp,
                                              mode='mean_flip',
                                              return_generator=False)

labelTsHypBeta = mne.extract_label_time_course(stcs_hyp_beta, labels,
                                           src_hyp,
                                           mode='mean_flip',
                                           return_generator=False)

###############################################################################
# plot mean power

# for j, stcs in enumerate(stcs_normal):
#     if j == 0:
#         stcs_normal_alpha = stcs_normal[0]["alpha"].data.mean(axis=0)
#     else:
#         stcs_normal_alpha = np.vstack([stcs_normal_alpha,
#                                        stcs_normal[j]["alpha"].data
#                                        .mean(axis=0)])

# for j, stcs in enumerate(stcs_hyp):
#     if j == 0:
#         stcs_hyp_alpha = stcs_hyp[0]["alpha"].data.mean(axis=0)
#     else:
#         stcs_hyp_alpha = np.vstack([stcs_hyp_alpha,
#                                     stcs_hyp[j]["alpha"].data.mean(axis=0)])


# times = stcs_normal[0]["alpha"].times
# mean_normal = stcs_normal_alpha.mean(axis=0)
# mean_hyp = stcs_hyp_alpha.mean(axis=0)
# std_normal = stcs_normal_alpha.std(axis=0)
# std_hyp = stcs_hyp_alpha.std(axis=0)
# hyp_limits_normal = (mean_normal - std_normal, mean_normal + std_normal)
# plt.fill_between(times, hyp_limits_normal[0], y2=hyp_limits_normal[1],
#                  color='b', alpha=0.5)
# plt.plot(times, mean_normal, label='Normal: Alpha', color='b')
# hyp_limits_hyp = (mean_hyp - std_hyp, mean_hyp + std_hyp)
# plt.fill_between(times, hyp_limits_hyp[0], y2=hyp_limits_hyp[1],
#                  color='b', alpha=0.5)
# plt.plot(times, mean_hyp, label='Hyp: Alpha', color='r')
# plt.plot(times, mean_normal + std_normal, color='b--')

# plt.plot(stcs['beta'].times, stcs['beta'].data.mean(axis=0), label='Beta')
# plt.xlabel('Time (ms)')
# plt.ylabel('Power')
# plt.legend()
# plt.title('Mean source induced power')
# plt.show()


# %%
###############################################################################
# Compute Power Spectral Density of inverse solution from single epochs
###############################################################################
# define frequencies of interest
fmin, fmax = 4., 90.
bandwidth = 2.  # bandwidth of the windows in Hz
snr = 1.0  # use smaller SNR for raw data
lambda2 = 1.0 / snr ** 2
method = "MNE"  # use dSPM method (could also be MNE or sLORETA)
# compute source space psd in label
epochs_normal.crop(-0.5, 0)
epochs_hyp.crop(-0.5, 0)


# Note: By using "return_generator=True" stcs will be a generator object
# instead of a list. This allows us so to iterate without having to
# keep everything in memory.

stcs_normal = compute_source_psd_epochs(epochs_normal,
                                        inverse_normal,
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


# plt.figure()
# plt.plot(freqs, psd_avg_normal, color="b", label="Normal")
# plt.plot(freqs, psd_avg_hyp, color="r", label="Hyp")
# # hyp_limits_normal = (psd_avg_normal - psd_std_normal,
# #                      psd_avg_normal + psd_std_normal)
# # plt.fill_between(freqs, hyp_limits_normal[0], y2=hyp_limits_normal[1],
# #                  color='b', alpha=0.5)
# # hyp_limits_hyp = (psd_avg_hyp - psd_std_hyp,
# #                   psd_avg_hyp + psd_std_hyp)
# # plt.fill_between(freqs, hyp_limits_hyp[0], y2=hyp_limits_hyp[1],
# #                  color='b', alpha=0.5)

# plt.xlabel('Freq (Hz)')
# plt.ylabel('Power Spectral Density')
# plt.legend()
# plt.show()


# %% for each label
psds_normal_labels = np.empty([len(freqs), len(labels), len(epochs_normal)])

for e, epoch in enumerate(epochs_normal):
    for l, label in enumerate(labels):
        stcs_normal = compute_source_psd_epochs(epochs_normal[e],
                                                inverse_normal,
                                                lambda2=lambda2,
                                                label=label,
                                                method=method,
                                                fmin=fmin, fmax=fmax,
                                                bandwidth=bandwidth,
                                                return_generator=False)
        psds_normal_labels[:, l, e] = np.mean(stcs_normal[0].data, axis=0)


psds_hyp_labels = np.empty([len(freqs), len(labels), len(epochs_hyp)])

for e, epoch in enumerate(epochs_hyp):
    for l, label in enumerate(labels):
        stcs_hyp = compute_source_psd_epochs(epochs_hyp[e],
                                             inverse_hyp,
                                             lambda2=lambda2,
                                             label=label,
                                             method=method,
                                             fmin=fmin, fmax=fmax,
                                             bandwidth=bandwidth,
                                             return_generator=False)
        psds_hyp_labels[:, l, e] = np.mean(stcs_hyp[0].data, axis=0)


# reshape results arrays
psds_normal_labels_X = np.empty([psds_normal_labels.shape[2],
                                psds_normal_labels.shape[0] *
                                psds_normal_labels.shape[1]])

psds_hyp_labels_X = np.empty([psds_hyp_labels.shape[2],
                              psds_hyp_labels.shape[0] *
                              psds_hyp_labels.shape[1]])

for j in range(psds_normal_labels.shape[2]):
    psds_normal_labels_X[j, :] = psds_normal_labels[:, :, j].reshape(-1)

for j in range(psds_hyp_labels.shape[2]):
    psds_hyp_labels_X[j, :] = psds_hyp_labels[:, :, j].reshape(-1)


# %% stats
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import scale
from sklearn.cross_validation import (cross_val_score, StratifiedKFold,
                                      permutation_test_score, LeaveOneOut)
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

X = np.vstack([psds_normal_labels_X, psds_hyp_labels_X])
X_scl = scale(X)
y = np.concatenate([np.zeros(psds_normal_labels_X.shape[0]),
                    np.ones(psds_hyp_labels_X.shape[0])])


X2 = X * 1e20
X2_scl = scale(X2)
# ### CV ####
cv = StratifiedKFold(y, 10)
loo = LeaveOneOut(len(y))

logistic = LogisticRegression()
pipe = Pipeline(steps=[('logistic', logistic)])
Cs = np.logspace(-5, 5, 25)
estimator = GridSearchCV(pipe, dict(logistic__C=Cs))
estimator.fit(X2, y)

C = estimator.best_params_.values()[0]


# ### Log Reg ####
logReg = LogisticRegression()
mnb = MultinomialNB()

cross_score_LR = cross_val_score(logReg, X2, y, scoring="accuracy",
                                 cv=cv, n_jobs=n_jobs, verbose=True)

print "Cross val score: ", cross_score_LR.mean()
print "The different cross_scores: ", cross_score_LR


score, perm_scores, pval = permutation_test_score(logReg, X, y,
                                                  scoring="accuracy",
                                                  cv=cv, n_permutations=2000,
                                                  n_jobs=n_jobs,
                                                  verbose=True)
print "score: ", score, "pval: ", pval

from sklearn.linear_model import RandomizedLogisticRegression
randomized_logistic = RandomizedLogisticRegression()
randomized_logistic.fit(X2, y)

# from sklearn.feature_selection import SelectPercentile
# from sklearn.feature_selection import chi2
# X_new = SelectFdr(chi2).fit_transform(X_scl, y)
# X_new.shape
