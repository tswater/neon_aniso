# ---------------------------------------------------------------- #
# ----------- CODE DESCRIPTION FOR ADD_DP04.py ------------------- #
# ---------------------------------------------------------------- #
# Add the important contents of the main product files to the base #
# h5 including stuff

import os
import netCDF4 as nc
import numpy as np
import h5py
import datetime
import ephem
import rasterio
import csv
import subprocess
import sys
from mpi4py import MPI
# MPI4PY Stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

is1m=bool(int(sys.argv[1]))
if is1m:
    dt_=1
    ds_='01m'
else:
    dt_=30
    ds_='30m'
base_dir='/home/tsw35/tyche/neon_'+str(dt_)+'m/'


# ------------------------- #
# USER INPUTS AND CONSTANTS #
# ------------------------- #
neon_dir = '/home/tsw35/soteria/data/NEON/aniso_1m/'

outvar = {'WTHETA':[]}
units={}

desc = {}


######### ANISOTROPY ###############
def aniso(uu,vv,ww,uv,uw,vw):
    n=len(uu)
    m=uu<10000
    m=m&(vv<10000)
    m=m&(ww<10000)
    m=m&(uv>-9999)
    m=m&(uw>-9999)
    m=m&(vw>-9999)
    for i in [uu,vv,ww,uv,uw,vw]:
        m=m&(~np.isnan(i))

    k=uu[m]+vv[m]+ww[m]
    ani=np.ones((n,3,3))*-9999
    ani[m,0,0]=uu[m]/k-1/3
    ani[m,1,1]=vv[m]/k-1/3
    ani[m,2,2]=ww[m]/k-1/3
    ani[m,0,1]=uv[m]/k
    ani[m,1,0]=uv[m]/k
    ani[m,2,0]=uw[m]/k
    ani[m,0,2]=uw[m]/k
    ani[m,1,2]=vw[m]/k
    ani[m,2,1]=vw[m]/k
    return ani





# -------------- #
# MAIN CODE LOOP #
# -------------- #
sites=os.listdir(neon_dir)
#sites=['ABBY']
for site in sites[rank::size]:
    if len(site)>4:
        continue
    print(site+': ',end='',flush=True)
    
    # Load in the base file
    fp_out=h5py.File(base_dir+site+'_L'+str(dt_)+'.h5','r+')
    time=fp_out['TIME'][:] 
    
    ovar=outvar.copy()
    for k in ovar.keys():
        ovar[k]=np.ones((len(time),))*-9999


    # ----------------- #
    # LOOP THROUGH TIME #
    # ----------------- #
    flist=os.listdir(neon_dir+site)
    flist.sort()
    for file in flist:
        print('.',end='',flush=True)
        fp_in=h5py.File(neon_dir+site+'/'+file,'r')
        stdt=datetime.datetime(int(file[8:12]),int(file[13:15]),1,0,tzinfo=datetime.timezone.utc)
        try:
            a=np.where(time==stdt.timestamp())[0][0]
        except Exception as e:
            print(e)
            continue
        if is1m:
            add='_1m'
            N=len(fp_in['UW_1m'][:])
            if (stdt.year==2023) and (stdt.month==12):
                N=44611
        else:
            add=''
            N=len(fp_in['UW'][:])
        
        for var in ovar.keys():
            ovar[var][a:a+N]=fp_in[var+add][0:N]

    # -------------------- #
    # CREATE HDF5 DATASETS #
    # -------------------- #
    for key in ovar.keys():
        try:
            fp_out.create_dataset(key,data=np.array(ovar[key][:]))
        except:
            fp_out[key][:]=np.array(ovar[key][:])
        fp_out[key].attrs['missing_value']=-9999
        fp_out[key].attrs['source']='NEON_dp04'
        if key in units.keys():
            fp_out[key].attrs['units']=units[key]
        if key in desc.keys():
             fp_out[key].attrs['description']=desc[key]
    fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())
    print('*',flush=True)
