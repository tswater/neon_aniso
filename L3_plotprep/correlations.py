import numpy as np
import h5py
import pickle
import os

sites=[]
for file in os.listdir('/home/tswater/Documents/Elements_Temp/NEON/neon_1m'):
    if 'L1' in file:
        sites.append(file[0:4])
sites.sort()

fpu=
fps=

xvars=['ANI_XB','ANI_YB','AD_MOST_U','AD_MOST_V','AD_MOST_W','AD_SC23_U',\
       'AD_SC23_V','AD_SC23_W','AD_TSW_U','AD_TSW_V','AD_TSW_W']
yvars=['f_aspect_dtm','f_mean_chm','f_range_chm','f_range_dtm','f_slope_dtm',\
       'f_std_chm','f_std_dsm','f_std_dtm','f_treecover_lidar',\
       'ff_cov_chm','ff_cov_dsmchm',\
       'tke','z_zd','zL']

for k in fpu.keys():
    if 'nlcd' in k:
        pass
    elif k in ['UU','VV','WW','UV','VW','UW']:
        pass
    elif k in ['U_SIGMA','V_SIGMA','W_SIGMA']:
        pass
    elif k=='SITE':
        pass


Nx=len(xvars)
Ny=len(yvars)

d_unst={'xvars':xvars,'yvars':yvars,'spearmanr':np.ones((Nx,Ny)*float('nan')),'pearsonr':np.ones((Nx,Ny)*float('nan')),'sitelevel':{}}
d_stbl={'xvars':xvars,'yvars':yvars,'spearmanr':np.ones((Nx,Ny)*float('nan')),'pearsonr':np.ones((Nx,Ny)*float('nan')),'sitelevel':{}}

for site in sites:
    d_unst['sitelevel'][site]={'spearmanr':np.ones((Nx,Ny)*float('nan')),'pearsonr':np.ones((Nx,Ny)*float('nan'))}
    d_stbl['sitelevel'][site]={'spearmanr':np.ones((Nx,Ny)*float('nan')),'pearsonr':np.ones((Nx,Ny)*float('nan'))}


# need to compute 'ff_cov_chm','ff_cov_dsmchm','tke','z_zd','zL'
for i in range(2):
    if i==0:
        d_=d_unst
        fp=fpu
    else:
        d_=d_stbl
        fp=fps
        for j in range(Nx):
            varx=xvar[j]

            for k in range(Ny):
                vary=yvar[k]
                if vary=='ff_cov_chm':

                elif vary=='ff_cov_dsmchm':

                elif 'f_' in vary:





####################################
######## PICKLE ####################
####################################
pickle.dump(d_unst,open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_corr_v0.p','wb'))
pickle.dump(d_stbl,open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_corr_v0.p','wb'))

