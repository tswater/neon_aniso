# Driver for making L1 files from NEON data
# L0 is data directly from NEON, and L1 is full timeseries to some
# averaging period.
#
# There is a different file for each site.

#### IMPORTANT NOTES
# Must link neonutil folder in same folder as driver to successfully run

# ------------------------------------- #
#              IMPORT                   #
# ------------------------------------- #
import numpy as np
import os
import time
from subprocess import run
from neonutil.nutil import SITES
from neonutil.l1tools import make_base, add_turb, add_stationarity,\
        add_derived, add_core_attrs, add_profile_tqc, add_radiation,\
        add_ghflx, add_precip, add_qaqc, add_pheno, l1_2_l1, remove_var,\
        update_var, add_dp04
start_time=time.time()

# ------------------------------------- #
#             USER INPUT                #
# ------------------------------------- #
# user needs to specify variables that they would like to add

#### BASIC OPTIONS
scale   = 10 # averaging period in minutes
stscl   = 2 # averaging period for stationarity test; approx 1/5th of scale

#### PLACES and THINGS TO PROCESS
sites    = None # if None or [] will use all
l1dir_   = '/home/tswater/Documents/tyche/data/neon/L1/' # L1 directory
varlist  = ['U','V','W','Us','Vs','UsUs','WW','WTHETA','WQ','WC',\
            'VsVs','UsVs','UsW','VsW','THETA','Q',\
            'T','ANI_XBs','ANI_YBs','ANID_YBs','ANID_XBs','G','H','LE','USTAR','RHO','L_MOST',\
            'NETRAD','GROWING','SOLAR_ALTITUDE','P','RH','TD','DMOLAIRDRY'] # [] variables to add. None is all. see readme.md for options
vstat    = ['UsUs','VsVs'] # [] variables to compute stationarity stats for;
qlist    = ['Q','C','THETA','USTAR','WC','LE','H','UVW'] # [] variables for core qaqc (see l1tools.add_qaqc)
dp4vlist = ['PA'] # [] variables to get from dp4 files; to add non-standard
                  # values you must also specify basepath dictionary

#### PROCESSING OPTIONS
replace = False # overwrite old data if it exists
timeout = True # print times as each function progresses
debug   = False # print debug information when available

#### QUALITY FLAGS to INCLUDE
qc_prof = True # add quality flags for profiles of tqc
qc_wind = True # add quality flags for profiles of wind
qc_rad  = True # add quality flags for radiation
qc_g    = True # add quality flags for ground heat flux
qc_sci  = False # add quality flags from added science review

#### DIRECTORIES
# directories should contain a folder for each site, with data within
l1dir_    = '/home/tswater/Documents/tyche/data/neon/L1/' # L1 directory
bd1       = '/run/media/tswater/Elements/NEON/downloads/' # base directory
bd2       = '/home/tswater/Documents/tyche/data/neon/'
turb_dir  = bd2+'multiscale/' # turbulence directory
l1_30_dir = l1dir_+'neon_30m/' # L1 directory for 30 minute averaging period
dp4_dir   = bd1+'dp4/' # directory for dp4 files from NEON
rad_dir   = bd1+'netrad/' # directory for incoming radiation files
ghflx_dir = bd1+'soil/' # directory with ground heat flux data
p1_dir    = bd1+'precip_primary/' # primary precipitation directory
p2_dir    = bd1+'precip_secondary/' # secondary precipitation directory
pheno_dir = bd1+'phenocam_data/' # phenocam directory
lai_dir   = bd1+'lai/' # leaf area index directory
wind_dir  = bd1+'wind2d/' # wind profile directory

# ------------------------------------- #
#               SETUP                   #
# ------------------------------------- #

# compute dlt; this line may need to be changed
if scale>30:
    dlt=30
else:
    dlt=scale

# get sitelist and, if mpi is used, divide sites
if sites in [[],None]:
    sites=SITES
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except:
    rank=0
    size=1

sites=sites[rank::size]

# ------------------------------------- #
#                RUN                    #
# ------------------------------------- #
# Contains blocks of functions; comment out functions/processes you do not
# wish to complete.
# As of 24/06/25 Contains all working functions

prefix=str(rank)+':: ' # print statement progress prefix

#######
nm='neon_'+str(scale)+'m'
l1dir=l1dir_+nm+'/'
if (nm not in os.listdir(l1dir_)):
    run('mkdir '+l1dir,shell=True)
#######


#### Remove Var
# # removes all delvar variables from sites in sites
# delvar = []
#
# remove_var(scale,l1dir,delvar,sites=sites)


#### Change Var
# # changes a given variable based on inputs
# var = ''
# newname = '' # new name for var
# desc = '' # new description for var
# us = '' # new units for var
# ats = {} # dictionary of new attributes for var {NAME:VALUE}
# fac = 1 # multiply all values of var by this factor
# update_var(scale,l1dir,var,rename=newname,desc=desc,units=us,\
#        attr=ats,factor=fac,sites=sites)


#### Make Base

'''
start_time=time.time()
print(prefix+'Making base file',flush=True)
make_base(scale,l1dir,dlt=dlt,overwrite=replace,sites=sites)
if timeout:
    print(prefix+"Making Base Done; took %s seconds to run" % (np.round(time.time() - start_time)))

# #### L1 to L1
# scale2=5 # scale to pull from
# dlt2=5 # dlt of scale to pull from
# l1_5_dir=l1dir_+'neon_5m/' # directory of scale to pull from
# l12l1list = []
# start_time=time.time()
# print(prefix+'Adding turbulence information',flush=True)
# l1_2_l1(scale2,l1_5_dir,scale2,l1dir,l12l1list,dlt2=dlt2,overwrite=replace,sites=sites)
# if timeout:
#     print(prefix+"Adding Turb Done; took %s seconds to run" % (np.round(time.time() - start_time)))

#### Add Turbulence Information
start_time=time.time()
print(prefix+'Adding turbulence information',flush=True)
add_turb(scale,l1dir,turb_dir,ivars=varlist,dlt=dlt,overwrite=replace,sites=sites,debug=debug)
if timeout:
    print(prefix+"Adding Turb Done; took %s seconds to run" % (np.round(time.time() - start_time)))

#### Add DP04 (Pressure) Information
# recommend to run before derived variables which need pressure
start_time=time.time()
print(prefix+'Adding dp04 characteristics',flush=True)
add_dp04(scale,l1dir,dp4_dir,dlt=dlt,ivars=dp4vlist,overwrite=True,sites=sites)
if timeout:
    print(prefix+"Adding Dp04 variables; took %s seconds to run" % (np.round(time.time() - start_time)))

#### Computing derived variables
start_time=time.time()
print(prefix+'Adding derived turublence characteristics',flush=True)
add_derived(scale,l1dir,ivars=varlist,overwrite=replace,sites=sites)
if timeout:
    print(prefix+"Adding Derived Done; took %s seconds to run" % (np.round(time.time() - start_time)))

'''
#### Add static attributes via copying
start_time=time.time()
print(prefix+'Copy static site characteristics',flush=True)
add_core_attrs(scale,l1dir,nbdir=l1_30_dir,bscl=30,ivars=None,sites=sites)
if timeout:
    print(prefix+"Static attrs Done; took %s seconds to run" % (np.round(time.time() - start_time)))

'''
#### Add Stationarity
start_time=time.time()
print(prefix+'Adding stationarity; this will take a long time',flush=True)
add_stationarity(l1dir,l1dir_+'neon_'+str(stscl)+'m/',scale,stscl,ivars=vstat,overwrite=False,sites=SITES,debug=True)
if timeout:
    print(prefix+"Adding stationarity; took %s seconds to run" % (np.round(time.time() - start_time)))
'''

print(prefix+'COMPLETE!!!!!!!', flush=True)

print()
print('L1 Creation complete as specified')
print()
