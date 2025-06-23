# Driver for making L1 files from NEON data
# L0 is data directly from NEON, and L1 is full timeseries to some
# averaging period.
#
# There is a different file for each site.

# ------------------------------------- #
#              IMPORT                   #
# ------------------------------------- #
import numpy as np
import os
import time
from subprocess import run
from neonutil.nutil import SITES
from neonutil.l1tools import make_base, add_turb, add_stationarity_zahn23,\
        add_derived, add_core_attrs, add_profile_tqc, add_radiation,\
        add_ghflx, add_precip, add_qaqc, add_pheno

start_time=time.time()

# ------------------------------------- #
#             USER INPUT                #
# ------------------------------------- #
# user needs to specify variables that they would like to add
# OPTIONS:
sites   = ['ABBY','GUAN'] # if None or [] will use all
l1dir_  = '/home/tswater/Documents/tyche/data/neon/L1/' # L1 directory
varlist = None # [] variables to add. None is all. see readme.md for options
vstat   = None # [] variables to compute stationarity stats for;
               #     ONLY USE ABOVE IF SCALE = 30
qlist   = None # [] variables for core qaqc (see l1tools.add_qaqc)

replace = True # overwrite old data if it exists
timeout = True # print times

qc_prof = True # add quality flags for profiles of tqc
qc_wind = True # add quality flags for profiles of wind
qc_rad  = True # add quality flags for radiation
qc_g    = True # add quality flags for ground heat flux
qc_sci  = True # add quality flags from added science review

scale   = 15 # averaging period in minutes


########## DIRECTORIES ###################
# directories should contain a folder for each site, with data within
bd        = '/run/media/tswater/Elements/NEON/' # base directory
turb_dir  = bd+'multiscale/' # turbulence directory
l1_30_dir = bd+'neon_30m/' # L1 directory for 30 minute averaging period
dp4_dir   = bd+'downloads/dp4/' # directory for dp4 files from NEON
rad_dir   = bd+'downloads/netrad/' # directory for incoming radiation files
ghflx_dir = bd+'downloads/soil/' # directory with ground heat flux data
p1_dir    = bd+'downloads/precip_primary/' # primary precipitation directory
p2_dir    = bd+'downloads/precip_secondary/' # secondary precipitation directory
pheno_dir = bd+'downloads/phenocam_data/' # phenocam directory
lai_dir   = bd+'downloads/lai/' # leaf area index directory
wind_dir  = bd+'downloads/wind2d/' # wind profile directory

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
for site in sites:
    prefix=site+':'+str(rank)+':: '

    #######
    nm='neon_'+str(scale)+'m'
    l1dir=l1dir_+nm+'/'
    if nm not in os.listdir(l1dir_):
        run('mkdir '+l1dir,shell=True)
    #######

    # Make Base
    start_time=time.time()
    print(prefix+'Making base file',flush=True)
    make_base(scale,l1dir,dlt=dlt,overwrite=replace,sites=sites)
    if timeout:
        print(prefix+"Making Base Done; took %s seconds to run" % (np.round(time.time() - start_time)))

    # Add Turbulence Information
    start_time=time.time()
    print(prefix+'Adding turbulence information',flush=True)
    add_turb(scale,l1dir,turb_dir,ivars=varlist,dlt=dlt,overwrite=replace,sites=sites)
    if timeout:
        print(prefix+"Adding Turb Done; took %s seconds to run" % (np.round(time.time() - start_time)))

    # Computing derived variables
    start_time=time.time()
    print(prefix+'Adding derived turublence characteristics',flush=True)
    add_derived(scale,l1dir,ivars=varlist,overwrite=replace,sites=sites)
    if timeout:
        print(prefix+"Adding Derived Done; took %s seconds to run" % (np.round(time.time() - start_time)))

    # Add static attributes via copying
    start_time=time.time()
    print(prefix+'Copy static site characteristics',flush=True)
    add_core_attrs(scale,l1dir,nbdir=l1_30_dir,bscl=30,ivars=None,sites=sites)
    if timeout:
        print(prefix+"Static attrs Done; took %s seconds to run" % (np.round(time.time() - start_time)))

    # Add profile information for T,Q,C
    start_time=time.time()
    print(prefix+'Adding profiles of temp, water and carbon',flush=True)
    add_profile_tqc(scale,l1dir,dp4_dir,addqaqc=qc_prof,ivars=varlist,\
            overwrite=replace,sites=sites)
    if timeout:
        print(prefix+"TQC Profiles Done; took %s seconds to run" % (np.round(time.time() - start_time)))

    # Add Radiation information
    start_time=time.time()
    print(prefix+'Adding incomming/outgoing radiation',flush=True)
    add_radiation(scale,l1dir,rad_dir,addqaqc=qc_rad,ivars=varlist,\
            overwrite=replace,sites=sites)
    if timeout:
        print(prefix+"Radiation Done; took %s seconds to run" % (np.round(time.time() - start_time)))

    # Add Ground Heat Flux
    start_time=time.time()
    print(prefix+'Adding ground heat flux',flush=True)
    add_ghflx(scale,l1dir,ghflx_dir,addqaqc=qc_rad,ivars=varlist,\
            overwrite=replace,sites=sites)
    if timeout:
        print(prefix+"Ground Heat Flux Done; took %s seconds to run" % (np.round(time.time() - start_time)))

    # Add Precipitation
    start_time=time.time()
    print(prefix+'Adding Precipitation',flush=True)
    add_precip(scale,l1dir,p1_dir,p2_dir,ivars=varlist,\
            overwrite=replace,sites=sites)
    if timeout:
        print(prefix+"Precipitation Done; took %s seconds to run" % (np.round(time.time() - start_time)))

    # Add qaqc
    start_time=time.time()
    print(prefix+'Adding QAQC',flush=True)
    add_qaqc(scale,l1dir,dp4_dir,ivars=qlist,qsci=qc_sci,\
            overwrite=replace,sites=sites)
    if timeout:
        print(prefix+"Add QAQC Done; took %s seconds to run" % (np.round(time.time() - start_time)))

    # Add pheno
    start_time=time.time()
    print(prefix+'Adding Phenocam data',flush=True)
    add_pheno(scale,l1dir,pheno_dir,ivars=varlist,overwrite=replace,sites=sites)
    if timeout:
        print(prefix+"Phenocam Done; took %s seconds to run" % (np.round(time.time() - start_time)))

    print(prefix+'COMPLETE!!!!!!!', flush=True)

print()
print('L1 Creation complete as specified')
print()
