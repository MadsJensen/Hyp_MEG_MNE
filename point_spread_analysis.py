import mne
import os
import socket
from mne.minimum_norm import (read_inverse_operator, point_spread_function,
                              cross_talk_function)


hostname = socket.gethostname()

if hostname == "wintermute":
    data_path = "/home/mje/mnt/Hyp_meg/scratch/Tone_task_MNE/"
    subjects_dir = "/home/mje/mnt/Hyp_meg/scratch/fs_subjects_dir/"
else:
    data_path = "/scratch1/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                "Tone_task_MNE/"
    subjects_dir = "/scratch1/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                   "fs_subjects_dir"


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
# Load data
inverse_normal = read_inverse_operator(inverse_fnormal)
inverse_hyp = read_inverse_operator(inverse_fhyp)

epochs_normal = mne.read_epochs(epochs_fnormal)
epochs_hyp = mne.read_epochs(epochs_fhyp)

epochs_normal = epochs_normal["press"]
epochs_hyp = epochs_hyp["press"]

condition = "normal"
fname_fwd = data_path + 'tone_task_%s-fwd.fif' % condition
fname_cov = data_path + 'tone_task_%s-cov.fif' % condition
fname_evoked = data_path + 'tone_task_%s-ave.fif' % condition

# read forward solution (sources in surface-based coordinates)
forward = mne.read_forward_solution(fname_fwd, force_fixed=True,
                                    surf_ori=True)

# read inverse operators
inverse_operator_meg = read_inverse_operator(inverse_fnormal)

# read label(s)
labels = mne.read_labels_from_annot('subject_1', parc='aparc.a2009s',
                                    regexp="pariet",
                                    subjects_dir=subjects_dir)


# regularisation parameter
snr = 3.0
lambda2 = 1.0 / snr ** 2
method = 'MNE'  # can be 'MNE' or 'sLORETA'
mode = 'svd'
n_svd_comp = 1

stc_psf_meg, _ = point_spread_function(inverse_operator_meg,
                                       forward, method=method,
                                       labels=labels,
                                       lambda2=lambda2,
                                       pick_ori='normal',
                                       mode=mode,
                                       n_svd_comp=n_svd_comp)

# save for viewing in mne_analyze in order of labels in 'labels'
# last sample is average across PSFs
# stc_psf_eegmeg.save('psf_eegmeg')
# stc_psf_meg.save('psf_meg')

from mayavi import mlab
fmin = 0.
time_label = "MEG %d"
fmax = stc_psf_meg.data[:, 0].max()
fmid = fmax / 2.
brain_meg = stc_psf_meg.plot(surface='inflated', hemi='both',
                             subjects_dir=subjects_dir,
                             time_label=time_label, fmin=fmin,
                             fmid=fmid, fmax=fmax,
                             figure=mlab.figure(size=(500, 500)))

brain_meg.add_label(labels[0], hemi="lh", borders=True)
brain_meg.add_label(labels[1], hemi="rh", borders=True)

# The PSF is centred around the right auditory cortex label,
# but clearly extends beyond it.
# It also contains "sidelobes" or "ghost sources"
# in middle/superior temporal lobe.
# For the Aud-RH example, MEG and EEGMEG do not seem to differ a lot,
# but the addition of EEG still decreases point-spread to distant areas
# (e.g. to ATL and IFG).
# The chosen labels are quite far apart from each other, so their PSFs
# do not overlap (check in mne_analyze)


# %% CROSS-TALK FUNCTION
# regularisation parameter
snr = 3.0
lambda2 = 1.0 / snr ** 2
mode = 'svd'
n_svd_comp = 1

method = 'MNE'  # can be 'MNE', 'dSPM', or 'sLORETA'
stc_ctf_mne = cross_talk_function(inverse_operator_meg, forward, labels[0:1],
                                  method=method, lambda2=lambda2,
                                  signed=False, mode=mode,
                                  n_svd_comp=n_svd_comp)

from mayavi import mlab
fmin = 0.
time_label = "MNE %d"
fmax = stc_ctf_mne.data[:, 0].max()
fmid = fmax / 2.
brain_mne = stc_ctf_mne.plot(surface='inflated', hemi='both',
                             subjects_dir=subjects_dir,
                             time_label=time_label, fmin=fmin,
                             fmid=fmid, fmax=fmax,
                             figure=mlab.figure(size=(500, 500)))

brain_meg.add_label(labels[0], hemi="lh", borders=True)
brain_meg.add_label(labels[1], hemi="rh", borders=True)
# Cross-talk functions for MNE and dSPM (and sLORETA) have the same shapes
# (they may still differ in overall amplitude).
# Point-spread functions (PSfs) usually differ significantly.
