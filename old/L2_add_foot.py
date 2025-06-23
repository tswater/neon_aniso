import numpy as np
import h5py
import os

fdir='/home/tswater/tyche/data/neon/foot_stats/'

ulist=[]
slist=[]
for file in os.listdir(fdir):
    if '_S.h5' in file:
        slist.append(file)
    elif '_U.h5' in file:
        ulist.append(file)

errorsites=['OSBS','SRER','STER','UNDE','YELL']

fp=h5py.File('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data_v2/NEON_TW_U_UVWT.h5','r+')
fp0=h5py.File('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data/NEON_TW_U_UVWT.h5','r')

fpsite=fp['SITE'][:]
fp0site=fp0['SITE'][:]

def getff(fp,fp0,ff,site):
    siteb=bytes(site,'utf-8')
    tt=fp['TIME'][:][fp['SITE'][:]==siteb]
    t0=fp0['TIME'][:][fp0['SITE'][:]==siteb]
    out=[]
    it=0
    for i0 in range(len(t0)):
        if tt[it]!=t0[i0]:
            out.append(0)
        else:
            out.append(1)
            it=it+1
    return out

ff_u={}
ulist.sort()
start=True
for file in ulist:
    ff=h5py.File(fdir+file,'r')
    if file[0:4] in errorsites:
        m=np.array(getff(fp,fp0,ff,file[0:4]),dtype=bool)
    for var in ff.keys():
        if len(ff[var][:])==0:
            continue
        if start:
            ff_u[var]=[]
        if file[0:4] in errorsites:
            ff_u[var].extend(ff[var][:][m])
        else:
            ff_u[var].extend(ff[var][:])
    start=False

print(len(ff_u[var]))
print(len(fp['TIME'][:]))

if len(ff_u[var])==len(fp['TIME'][:]):
    for var in ff_u.keys():
        fp.create_dataset(var,data=np.array(ff_u[var][:]))


fp=h5py.File('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data_v2/NEON_TW_S_UVWT.h5','r+')
fp0=h5py.File('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data/NEON_TW_S_UVWT.h5','r')
ff_s={}
slist.sort()
start=True
fpsite=fp['SITE'][:]
fp0site=fp0['SITE'][:]
for file in slist:
    ff=h5py.File(fdir+file,'r')
    if file[0:4] in errorsites:
        m=np.array(getff(fp,fp0,ff,file[0:4]),dtype=bool)
        if file[0:4]=='YELL':
            m=m[0:-1]
    for var in ff.keys():
        if var=='std_slope':
            continue
        if len(ff[var][:])==0:
            continue
        if start:
            ff_s[var]=[]
        if file[0:4] in errorsites:
            ff_s[var].extend(ff[var][:][m])
            if file[0:4]=='YELL':
                ff_s[var].extend([0])
        else:
            ff_s[var].extend(ff[var][:])
    start=False

## EXPORT
if len(ff_s[var])==len(fp['TIME'][:]):
    for var in ff_s.keys():
        fp.create_dataset(var,data=np.array(ff_s[var][:]))

print(len(ff_s[var][:]))
print(len(fp['TIME'][:]))

