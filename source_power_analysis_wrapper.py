# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 22:12:08 2014

@author: mje
"""

import os
path = "/projects/MINDLAB2013_18-MEG-HypnosisAnarchicHand/" + \
                  "scripts/MNE_analysis/"
os.chdir(path)
%run source_power_analysis_PRESS

os.chdir(path)
%run source_power_analysis_TONE

