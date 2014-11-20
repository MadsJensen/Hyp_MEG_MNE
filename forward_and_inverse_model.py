# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 22:50:30 2014

@author: mje
"""

import mne
import socket
import os
import matplotlib.pyplot as plt
from mne.minimum_norm import make_inverse_operator, write_inverse_operator

# Setup paths and prepare raw data
hostname = socket.gethostname()

if hostname == "wintermute":
    data_path = "/home/mje/mnt/Hyp_meg/scratch/Tone_task_MNE"
    mri_path = '/home/mje/mnt/Hyp_meg/scratch/fs_subjects_dir/subject_1'
    subjects_dir = '/home/mje/mnt/Hyp_meg/scratch/fs_subjects_dir/'
    raw_fnormal = data_path + "tone_task-normal-tsss-mc-autobad-ica_raw.fif"
    raw_fhyp = data_path + "tone_task-hyp-tsss-mc-autobad-ica_raw.fif"
    mri_normal = data_path + '/subject_1-normal-trans.fif'
    mri_hyp = data_path + '/subject_1-hyp-trans.fif'
else:
    data_path = "/projects/" + \
                "MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                "scratch/Tone_task_MNE/"
    mri_path = "/projects/" + \
               "MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
               "scratch/fs_subjects_dir/subject_1"
    raw_fnormal = data_path + "tone_task-normal-tsss-mc-autobad-ica_raw.fif"
    raw_fhyp = data_path + "tone_task-hyp-tsss-mc-autobad-ica_raw.fif"
    mri_normal = data_path + '/subject_1-normal-trans.fif'
    mri_hyp = data_path + '/subject_1-hyp-trans.fif'

# change dir to save files the rigth place
os.chdir(data_path)

src = mri_path + "/bem/subject_1-oct-6-src.fif"
bem = mri_path + "/bem/subject_1-5120-bem-sol.fif"

conditions = ["normal", "hyp"]

# ## Do normal condition

fwd = mne.make_forward_solution(raw_fhyp, mri=mri_hyp, src=src, bem=bem,
                                fname=None, meg=True, eeg=False, mindist=5.0,
                                n_jobs=3, overwrite=True)

mne.write_forward_solution("Tone_task_hyp-fwd.fif", fwd,
                           overwrite=True)

# convert to surface orientation for better visualization
fwd = mne.convert_forward_solution(fwd, surf_ori=True)
leadfield = fwd['sol']['data']

print("Leadfield size : %d x %d" % leadfield.shape)

grad_map = mne.sensitivity_map(fwd, ch_type='grad', mode='fixed')
mag_map = mne.sensitivity_map(fwd, ch_type='mag', mode='fixed')

###############################################################################
# Show gain matrix a.k.a. leadfield matrix with sensitivity map

# import matplotlib.pyplot as plt
picks_meg = mne.pick_types(fwd['info'], meg=True, eeg=False)

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle('Lead field matrix (500 dipoles only)', fontsize=14)
for ax, picks, ch_type in zip(axes, [picks_meg], ['meg']):
    im = ax.imshow(leadfield[picks, :500], origin='lower', aspect='auto')
    ax.set_title(ch_type.upper())
    ax.set_xlabel('sources')
    ax.set_ylabel('sensors')
    plt.colorbar(im, ax=ax, cmap='hot')
plt.show()

plt.figure()
plt.hist([grad_map.data.ravel(), mag_map.data.ravel()],
         bins=20, label=['Gradiometers', 'Magnetometers'],
         color=['c', 'b'])
plt.legend()
plt.title('Normal orientation sensitivity')
plt.xlabel('sensitivity')
plt.ylabel('count')
plt.show()

# plot sensitivity maps
grad_map.plot(subject='subject_1', hemi="both",
              time_label='Gradiometer sensitivity',
              subjects_dir=subjects_dir, fmin=0.1, fmid=0.5, fmax=0.9,
              smoothing_steps=7)
mag_map.plot(subject='subject_1', hemi="both",
             time_label='Gradiometer sensitivity',
             subjects_dir=subjects_dir, fmin=0.1, fmid=0.5, fmax=0.9,
             smoothing_steps=7)

###############################################################################
# make noise coveriance
fname = data_path + '/tone_task_normal-epo.fif'
epochs = mne.read_epochs(fname)

# Compute the covariance from the raw data
cov = mne.compute_covariance(epochs, tmin=None, tmax=-0.7)
mne.write_cov("tone_task_normal-cov.fif", cov)
print(cov)

###############################################################################
# Show covariance
# fig_cov, fig_svd = mne.viz.plot_cov(cov,
#                                     epochs.info, colorbar=True, proj=True)

###############################################################################
# make inverse model
for condition in conditions:
    fname_fwd = data_path + 'tone_task_%s-fwd.fif' % condition
    fname_cov = data_path + 'tone_task_%s-cov.fif' % condition
    fname_evoked = data_path + 'tone_task_%s-ave.fif' % condition
    snr = 1.0
    lambda2 = 1.0 / snr ** 2

    # Load data
    evoked = mne.read_evokeds(fname_evoked, condition=0, baseline=(None, 0))
    forward = mne.read_forward_solution(fname_fwd, surf_ori=True)
    noise_cov = mne.read_cov(fname_cov)
    info = evoked.info

    # regularize noise covariance
    noise_cov = mne.cov.regularize(noise_cov, info,
                                   mag=0.05, grad=0.05, proj=False)

    info = evoked.info
    inverse_operator = make_inverse_operator(info, forward, noise_cov,
                                             loose=0.2, depth=0.8)

    write_inverse_operator('tone_task_%s-inv.fif' % (condition),
                           inverse_operator)
