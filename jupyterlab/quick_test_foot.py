# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import netCDF4 as nc
import pickle
import h5py
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as pe
import matplotlib
import scipy as sci
from IPython.display import HTML
from sklearn.metrics import mean_squared_error
from matplotlib.gridspec import GridSpec
import cmasher as cmr
from numpy.polynomial import polynomial as P
import matplotlib.ticker as tck
import rasterio
from pyproj import Transformer
from datetime import datetime,timedelta
mpl.rcParams['figure.dpi'] = 100
sns.set_theme()
plt.rcParams.update({'figure.max_open_warning': 0})

# %%
fpdsm=rasterio.open('/home/tswater/Documents/tyche/data/neon/dsm/STEI/dsm_STEI.tif')
stei=fpdsm.read(1)
transformer=Transformer.from_crs('EPSG:4326',fpdsm.crs,always_xy=True)
lat=45.50895
lon=-89.58636
xx_,yy_=transformer.transform(lon,lat)
xx,yy=fpdsm.index(xx_,yy_)

# %%
plt.imshow(stei,cmap='terrain',vmin=0)
plt.scatter(xx,yy)
plt.colorbar()

# %%
ffu=h5py.File('/home/tswater/tyche/data/neon/foot_stats/ABBY_U.h5','r')
ffs=h5py.File('/home/tswater/tyche/data/neon/foot_stats/ABBY_S.h5','r')

# %%
idir='/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data/'
fpu=h5py.File(idir+'/NEON_TW_U_UVWT.h5','r')
fps=h5py.File(idir+'/NEON_TW_S_UVWT.h5','r')
d_u=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_unst_v3.p','rb'))
d_s=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_stbl_v3.p','rb'))
fpst=h5py.File('/home/tswater/tyche/data/neon/static_data.h5','r')

# %%
ms=fps['SITE'][:]==b'ABBY'
mu=fpu['SITE'][:]==b'ABBY'

# %%
fpst.keys()

# %%
ffu.keys()

# %%
plt.hist(ffu['mean_chm'][:]-fpst['mean_chm'][0])

# %%
plt.hist(ffs['mean_chm'][:]-fpst['mean_chm'][0])

# %%
phi=np.sqrt(fpu['UU'][:][mu])/fpu['USTAR'][:][mu]
zL=(fpu['tow_height'][:][mu]-fpu['zd'][:][mu])/fpu['L_MOST'][:][mu]
ani=fpu['ANI_YB'][:][mu]
fpsite=fpu['SITE'][:][mu]
a=.784-2.582*np.log10(ani)
phi_stp=a*(1-3*zL)**(1/3)
phi_old=2.55*(1-3*zL)**(1/3)
anix=fpu['ANI_XB'][:][mu]

# %%
ffu.keys()

# %%
ad_most=np.abs(phi-phi_old)
ad_sc23=np.abs(phi-phi_stp)
plt.hexbin(ffu['range_chm'][:],ad_most,mincnt=1,cmap='terrain',gridsize=100,extent=(8,55,0,5),vmin=0,vmax=50)


# %%
def getbins(A,n):
    A2=A[~np.isnan(A)]
    B=np.sort(A2)
    bins=[]
    for i in np.linspace(0,len(A2)-1,n):
        i=int(i)
        bins.append(B[i])
    return np.array(bins)


# %%
def binmean(a,b,bins=50):
    bine=getbins(a,bins+1)
    binc=(bine[1:]+bine[0:-1])/2
    mean=[]
    q1=[]
    q3=[]
    for i in range(bins):
        q1.append(np.nanpercentile(b[(a<bine[i+1])&(a>bine[i])],25))
        q3.append(np.nanpercentile(b[(a<bine[i+1])&(a>bine[i])],75))
        mean.append(np.nanmean(b[(a<bine[i+1])&(a>bine[i])]))
    return binc,mean,q1,q3
    


# %%
ffu=h5py.File('/home/tswater/tyche/data/neon/foot_stats/STEI_U.h5','r')
mu=fpu['SITE'][:]==b'STEI'
phi=np.sqrt(fpu['UU'][:][mu])/fpu['USTAR'][:][mu]
zL=(fpu['tow_height'][:][mu]-fpu['zd'][:][mu])/fpu['L_MOST'][:][mu]
ani=fpu['ANI_YB'][:][mu]
fpsite=fpu['SITE'][:][mu]
a=.784-2.582*np.log10(ani)
phi_stp=a*(1-3*zL)**(1/3)
phi_old=2.55*(1-3*zL)**(1/3)
anix=fpu['ANI_XB'][:][mu]
ad_most=np.abs(phi-phi_old)
ad_sc23=np.abs(phi-phi_stp)

# %%
np.sum(~np.isnan(ffu['std_dtm'][:]))

# %%
for k in ffu.keys():
    print(k)
    if 'nlcd' in k:
        continue
    if 'std_slope' in k:
        continue
    if 'aspect' in k:
        continue
    x,mean,q1,q3=binmean(ffu[k][:],ad_most,20)
    plt.figure()
    plt.plot(x,mean)
    plt.plot(x,q1,'--')
    plt.plot(x,q3,'--')
    plt.title(k)

# %%
ffu=h5py.File('/home/tswater/tyche/data/neon/foot_stats/STEI_U.h5','r')
print(ffu.keys())

# %%
sites=[]
for f in os.listdir('/home/tswater/tyche/data/neon/foot_stats/'):
    ss=f[0:4]
    if ss=='test':
        continue
    elif ss in sites:
        continue
    else:
        sites.append(ss)

# %%
from scipy import stats

# %%
varlist=['aspect_dtm', 'mean_chm', 'nlcd_2', 'nlcd_3', 'nlcd_dom', 'nlcd_iqv', 'range_chm', 'range_dtm', 'slope_dtm', 'std_aspect', 'std_chm', 'std_dsm', 'std_dtm', 'std_slope', 'treecover_lidar']
scl_vars=['U_stbl','U_unst','V_stbl','V_unst','W_stbl','W_unst']
mad=np.zeros((len(scl_vars),len(sites)))
xdata=np.zeros((2,len(varlist),len(sites)))
cors=np.zeros((6,len(varlist)))


for i in range(len(sites)):
    site=sites[i]
    ffu=h5py.File('/home/tswater/tyche/data/neon/foot_stats/'+site+'_U.h5','r')
    ffs=h5py.File('/home/tswater/tyche/data/neon/foot_stats/'+site+'_U.h5','r')
    j=0
    for svar in ['U','V','W']:
        mad[j+1,i]=d_u[svar[0:1]]['MAD_OLD_s'][site]
        mad[j,i]=d_s[svar[0:1]]['MAD_OLD_s'][site]
        j=j+2
    j=0
    for var in varlist:
        if 'nlcd' in var:
            xdata[0,j,i]=stats.mode(ffu[var][:])[0]
            xdata[1,j,i]=stats.mode(ffu[var][:])[0]
        else:
            xdata[0,j,i]=np.nanmean(ffs[var][:])
            xdata[1,j,i]=np.nanmean(ffs[var][:])
        j=j+1
    
    
    

# %%
def fix(d1,d2):
    new1=[]
    new2=[]
    for i in range(len(d1)):
        if (np.isnan(d2[i])) or (d2[i]==float('Inf')):
            continue
        else:
            new1.append(d1[i])
            new2.append(d2[i])
    return np.array(new1),np.array(new2)


# %%
mad[i,:]
xdata[k,j,:]

# %%
for i in range(len(scl_vars)):
    for j in range(len(varlist)):
        plt.figure()
        if i in [0,2,4]:
            k=1
        plt.scatter(xdata[k,j,:],mad[i,:])
        xx,yy=fix(mad[i,:],xdata[k,j,:])
        try:
            cors[i,j]=stats.pearsonr(xx,yy)[0]
        except:
            pass
        tstr=scl_vars[i]+': '+str(varlist[j])+' vs MAD '+str(cors[i,j])[0:5]
        plt.title(tstr)

# %%
