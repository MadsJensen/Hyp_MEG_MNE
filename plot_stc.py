import os
import numpy as np

from surfer import Brain
from surfer.io import read_stc

"""
define subject, surface and hemisphere(s) to plot
"""
subject_id, surface = 'subject_1', 'inflated'
hemi = 'split'

"""
create Brain object for visualization
"""
brain = Brain(subject_id, hemi, surface,
              config_opts=dict(width=800, height=400))

"""
read MNE dSPM inverse solution
"""
for hemi in ['lh', 'rh']:
    stc_fname = ("slet-" + hemi + '.stc')
    stc = read_stc(stc_fname)

    """
    data and vertices for which the data is defined
    """
    data = stc['data']
    vertices = stc['vertices']

    """
    time points in milliseconds
    """
    time = 1e3 * np.linspace(stc['tmin'],
                             stc['tmin'] + data.shape[1] * stc['tstep'],
                             data.shape[1])
    """
    colormap to use
    """
    colormap = 'hot'

    """
    label for time annotation
    """
    time_label = 'time=%0.2f ms'

    brain.add_data(data, colormap=colormap, vertices=vertices,
                   smoothing_steps=10, time=time, time_label=time_label,
                   hemi=hemi)

"""
scale colormap and set time (index) to display
"""
brain.scale_data_colormap(fmin=0, fmid=35, fmax=70, transparent=True)
brain.set_data_time_index(1400)

"""
uncomment these lines to use the interactive TimeViewer GUI
"""
# from surfer import TimeViewer
# viewer = TimeViewer(brain)
