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
import h5py
import time
from subprocess import run
from neonutil.nutil import SITES,out_to_h5
from neonutil.l1tools import make_base, add_turb,\
        add_derived, add_core_attrs, add_profile_tqc, add_radiation,\
        add_ghflx, add_precip, add_qaqc, add_pheno, l1_2_l1, remove_var,\
        update_var, add_dp04, _aniso, _bij
start_time=time.time()

# ------------------------------------- #
#             USER INPUT                #
# ------------------------------------- #
# user needs to specify variables that they would like to add

l1dir_    = '/home/tswater/Documents/tyche/data/neon/L1/' # L1 directory
bd1       = '/run/media/tswater/Elements/NEON/downloads/' # base directory
dp4_dir   = bd1+'dp4/'
debug=False

qlist=['THETA','USTAR','LE','H','UVW']
'''
start_time=time.time()
prefix='2 minute: '
print(prefix+'Adding QAQC',flush=True)
add_qaqc(2,l1dir_+'neon_2m/',dp4_dir,dlt=30,ivars=qlist,qsci=False,\
        overwrite=False,debug=debug)
print(prefix+"Add QAQC Done; took %s seconds to run" % (np.round(time.time() - start_time)))

start_time=time.time()
prefix='10 minute: '
print(prefix+'Adding QAQC',flush=True)
add_qaqc(10,l1dir_+'neon_10m/',dp4_dir,dlt=30,ivars=qlist,qsci=False,\
        overwrite=False,debug=debug)
print(prefix+"Add QAQC Done; took %s seconds to run" % (np.round(time.time() - start_time)))

start_time=time.time()
prefix='60 minute: '
print(prefix+'Adding QAQC',flush=True)
add_qaqc(60,l1dir_+'neon_60m/',dp4_dir,dlt=60,ivars=qlist,qsci=False,\
        overwrite=False,debug=debug)
print(prefix+"Add QAQC Done; took %s seconds to run" % (np.round(time.time() - start_time)))
'''
start_time=time.time()
prefix='1 minute: '
print(prefix+'Adding ANID',flush=True)

for site in SITES:
    fp=h5py.File(l1dir_+'neon_1m/'+site+'_1m.h5','r+')
    cero=np.zeros((len(fp['UsUs'][:]),))
    rstd=_bij(fp['UsUs'][:],fp['VsVs'][:],fp['WW'][:],cero,cero,cero)
    yb,xb=_aniso(rstd)
    ov={'ANID_XBs':xb,'ANID_YBs':yb}
    out_to_h5(fp,ov,False)
print(prefix+"Add ANID Done; took %s seconds to run" % (np.round(time.time() - start_time)))


#ivars=['ST_UsUs_1','ST_VsVs_1']

#l1_2_l1(5,l1dir_+'neon_5m/',1,l1dir_+'neon_1m/',ivars,overwrite=False)
#l1_2_l1(5,l1dir_+'neon_5m/',2,l1dir_+'neon_2m/',ivars,overwrite=False)

print()
print('L1 Creation complete as specified')
print()
