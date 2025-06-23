import os
import numpy as np
import h5py

L2dir='/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data_v2/'
old_L2dir='/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data/'
def getzzd(fp,fp0):
    fpsite=fp['SITE'][:]
    fp0site=fp0['SITE'][:]
    zzd=np.zeros((len(fpsite),))
    for site in np.unique(fpsite):
        m_fp=fpsite==site
        m_fp0=fp0site==site
        zzd[m_fp]=fp0['tow_height'][m_fp0][0]-fp0['zd'][m_fp0][0]
    return zzd

for file in os.listdir(L2dir):
    fp=h5py.File(L2dir+file,'r+')
    fp0=h5py.File(old_L2dir+file,'r')
    zzd=getzzd(fp,fp0)
    fp.create_dataset('zzd',data=np.array(zzd[:]))
    fp.close()

