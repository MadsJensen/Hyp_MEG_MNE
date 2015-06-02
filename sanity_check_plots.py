import mne
from mne.viz import plot_trans
import socket
import os


# Setup paths and prepare raw data
hostname = socket.gethostname()
if hostname == "Wintermute":
    data_path = "/home/mje/mnt/Hyp_meg/scratch/Tone_task_MNE/"
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

condition = "normal"
fname_evoked = data_path + 'tone_task_%s-ave.fif' % condition
evoked = mne.read_evokeds(fname_evoked, condition=0, baseline=(None, -0.7))
plot_trans(evoked.info, trans_fname=mri_normal, subject='subject_1',
           subjects_dir=subjects_dir)

condition = "hyp"
fname_evoked = data_path + 'tone_task_%s-ave.fif' % condition
evoked = mne.read_evokeds(fname_evoked, condition=0, baseline=(None, -0.7))
plot_trans(evoked.info, trans_fname=mri_hyp, subject='subject_1',
           subjects_dir=subjects_dir)
