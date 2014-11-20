# This script is heavily based on the forward model script from cam meg wiki @:
# http://imaging.mrc-cbu.cam.ac.uk/meg/AnalyzingData/MNE_ForwardSolution

# variables
datapath=/home/mje/mnt/Hyp_meg/scratch/Tone_task_MNE
SUBJECTS_DIR=/home/mje/mnt/Hyp_meg/scratch/fs_subjects_dir

# The subjects and sessions to be used
subject='subject_1'

echo " "
echo " Making BEM solution for SUBJECT:  $subject"
echo " "

mne_watershed_bem --subject ${subject} --overwrite
  
ln -sf $SUBJECTS_DIR/${subject}/bem/watershed/${subject}_inner_skull_surface $SUBJECTS_DIR/${subject}/bem/${subject}_inner_skull.surf
ln -sf $SUBJECTS_DIR/${subject}/bem/watershed/${subject}_outer_skull_surface $SUBJECTS_DIR/${subject}/bem/${subject}_outer_skull.surf
ln -sf $SUBJECTS_DIR/${subject}/bem/watershed/${subject}_outer_skin_surface  $SUBJECTS_DIR/${subject}/bem/${subject}_outer_skin.surf
ln -sf $SUBJECTS_DIR/${subject}/bem/watershed/${subject}_brain_surface       $SUBJECTS_DIR/${subject}/bem/${subject}_brain_surface.surf

echo " "
echo " Setting up forward model for  SUBJECT:  $subject"
echo " "
  
