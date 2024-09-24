import numpy as np
import os
from subprocess import run
import h5py
from datetime import datetime
from mpi4py import MPI
from datetime import timedelta
from calendar import monthrange
import math
import warnings
import tracemalloc
import gc

tracemalloc.start()

warnings.filterwarnings('ignore',category=RuntimeWarning)

# MPI4PY Stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tsw35/soteria/software/miniconda/envs/eddy4R/lib/R/modules

#### USER INPUT ####
raw_dir='/home/tsw35/xSot_shared/NEON_raw_data/'
dp4_dir='/home/tsw35/soteria/data/NEON/dp04ex/'
out_dir='/home/tsw35/soteria/data/NEON/aniso_1mv2/'
sites=['ALL']
years=['ALL']
dp4type=['expanded','basic'][0]
overwrite=False # if true, will overwrite already created output files
testing=True # if testing, will output more information

# timesteps per day for 1m and 30m intervals
n1m=1440
n30m=48


#### UNITS and DESCRIPTION ####
units={}
units['CO2FX_1m']='umolCo2 m-2 s-1'
desc={}



##########################
#### HELPER FUNCTIONS ####
##########################



####################################
####################################


#### SETUP ####

# Load in R packages
# Setup the list of sites and years
if 'ALL' in sites:
    sites=[]
    for file in os.listdir(raw_dir):
        if len(file)!=4:
            continue
        else:
            sites.append(file)
if 'ALL' in years:
    years=[2017,2018,2019,2020,2021,2022,2023]

if rank==0:
    print('Processing the following sites: ')
    print(sites)
    print('Processing the following years: ')
    print(years)

sites.sort()
years.sort()

# compile lists of NEON domains and sonic levels
domain={} #i.e. 16 for ABBY, as in D16
s_level={} #i.e. 050 for ABBY, as in 000_050_01m
for site in sites:
    file=os.listdir(dp4_dir+site)[1]
    domain[site]=file[6:8]
    fp=h5py.File(dp4_dir+site+'/'+file,'r')
    lvls=list(fp[site+'/dp01/data/amrs/'].keys())
    s_level[site]=lvls[0][4:7]
    fp.close()

# Make an array of site years to loop through
st_yr=[]
for site in sites:
    for year in years:
        st_yr.append(site+'_'+str(year))

print('STARTING')
for site in sites:
    ax=[]
    ay=[]
    pf=[]
    for file in os.listdir(dp4_dir+site+'/'):
        fp4=h5py.File(dp4_dir+site+'/'+file,'r')
        angEnuXaxs=float(fp4[site].attrs['Pf$AngEnuXaxs'][:][0])
        angEnuYaxs=float(fp4[site].attrs['Pf$AngEnuYaxs'][:][0])
        pf_ofst=float(fp4[site].attrs['Pf$Ofst'][:][0])
        ax.append(angEnuXaxs)
        ay.append(angEnuYaxs)
        pf.append(pf_ofst)
    print(site)
    print('    AX: '+str(np.unique(ax)))
    print('    AY: '+str(np.unique(ax)))
    print('    PF: '+str(np.unique(pf)),flush=True)
