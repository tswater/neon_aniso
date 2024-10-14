import os
import numpy as np
import h5py

L2dir='/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data_v2/'
old_L2dir='/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data/'

add_vars=['T_SONIC_SIGMA','H2O_SIGMA','CO2_SIGMA','RHO','CO2']

for file in os.listdir(L2dir):
    out={}
    for var in add_vars:
        out[var]=[]

    fp=h5py.File(L2dir+file,'r+')
    fp0=h5py.File(old_L2dir+file,'r')
    fpsite=fp['SITE'][:]
    fp0site=fp0['SITE'][:]
    fpt=fp['TIME'][:]
    fp0t=fp0['TIME'][:]
    for site in np.unique(fp0site):
        print(site)
        m_0=fp0site==site
        m_=fpsite==site
        if np.sum(m_0)==np.sum(m_):
            m_0out=m_0
        elif np.sum(m_0)>np.sum(m_):
            print(np.sum(m_0))
            print(np.sum(m_))
            m_0out=np.zeros((len(m_0),)).astype(bool)
            idxs=np.where(m_0)[0][0]
            t=0
            fpts=fpt[m_]
            fp0ts=fp0t[m_0]
            for t0 in range(len(fp0ts)):
                if fpts[t]==fp0ts[t0]:
                    t=t+1
                    m_0out[idxs+t0]=True
                else:
                    pass
        elif np.sum(m_0)<np.sum(m_):
            print('ERROR ----- NEED TO WRITE THIS CODE',flush=True)

        for var in add_vars:
            out[var].extend(fp0[var][m_0out])
    for var in add_vars:
        try:
            del fp[var]
        except:
            pass
        fp.create_dataset(var,data=np.array(out[var][:]))
    fp.close()

