# Driver for making L1 files from NEON data
# L0 is data directly from NEON, and L1 is full timeseries to some
# averaging period. This script requires an already created
# set of L1 files that includes turbulence information. To create
# initial L1 files, use the appropriate script in turb_tw.
#
# There is a different file for each site.

# ------------------------------------- #
#              IMPORT                   #
# ------------------------------------- #
import numpy as np
import os
from subprocess import run

# ------------------------------------- #
#             USER INPUT                #
# ------------------------------------- #
# user needs to specify variables that they would like to add
# OPTIONS:
l1dir = '' # L1 directory
varlist = [] # list of variables to add i.e. ['ANI_YB','ANI_XB','H']
scale = '' # averaging period in minutes

######################## DIRECTORIES #############################
# directories of other data needed for certain variables.
# directories should contain a folder for each site, with data within
adir = ''

# ------------------------------------- #
#              SETUP                    #
# ------------------------------------- #



