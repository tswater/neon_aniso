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
import rasterio
from pyproj import Transformer
from datetime import datetime,timedelta
mpl.rcParams['figure.dpi'] = 100
sns.set_theme()
plt.rcParams.update({'figure.max_open_warning': 0})

# %%
fpu=h5py.File('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data_v2/NEON_TW_U_UVWT.h5','r')

# %%
fpu=h5py.File('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data_v2/NEON_TW_U_UVWT.h5','r')
fps=h5py.File('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data_v2/NEON_TW_S_UVWT.h5','r')


# %%
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


# %%
uu=fpu['UU'][:]
vv=fpu['VV'][:]
ww=fpu['WW'][:]
uv=np.zeros((len(ww),))
vw=np.zeros((len(ww),))
uw=np.zeros((len(ww),))
N=len(ww)
bij=aniso(uu,vv,ww,uv,uw,vw)
ybdu=np.ones((N,))*-9999
xbdu=np.ones((N,))*-9999
for t in range(N):
    if np.sum(bij[t,:,:]==-9999)>0:
        continue
    lams=np.linalg.eig(bij[t,:,:])[0]
    lams.sort()
    lams=lams[::-1]
    xbdu[t]=lams[0]-lams[1]+.5*(3*lams[2]+1)
    ybdu[t]=np.sqrt(3)/2*(3*lams[2]+1)


# %%
uu=fps['UU'][:]
vv=fps['VV'][:]
ww=fps['WW'][:]
uv=np.zeros((len(ww),))
vw=np.zeros((len(ww),))
uw=np.zeros((len(ww),))
N=len(ww)
bij=aniso(uu,vv,ww,uv,uw,vw)
ybds=np.ones((N,))*-9999
xbds=np.ones((N,))*-9999
for t in range(N):
    if np.sum(bij[t,:,:]==-9999)>0:
        continue
    lams=np.linalg.eig(bij[t,:,:])[0]
    lams.sort()
    lams=lams[::-1]
    xbds[t]=lams[0]-lams[1]+.5*(3*lams[2]+1)
    ybds[t]=np.sqrt(3)/2*(3*lams[2]+1)

# %%
fps.keys()

# %%
xbu=fpu['ANI_XB'][:]
ybu=fpu['ANI_YB'][:]
xbs=fps['ANI_XB'][:]
ybs=fps['ANI_YB'][:]
zLu=(fpu['zzd'][:])/fpu['L_MOST'][:]
zLs=(fps['zzd'][:])/fps['L_MOST'][:]

# %%
m=fpu['SITE'][:]==b'SJER'
plt.hist(np.sqrt(fpu['UU'][m]),bins=np.linspace(0,2),alpha=.3)
plt.hist(np.sqrt(fpu['VV'][m]),bins=np.linspace(0,2),alpha=.3)
plt.hist(np.sqrt(fpu['WW'][m]),bins=np.linspace(0,2),alpha=.3)

# %%
plt.hist(rt,bins=np.linspace(.4,1))


# %%
def getbins(A,n):
     B=np.sort(A)
     bins=[]
     for i in np.linspace(0,len(A)-1,n):
         i=int(i)
         bins.append(B[i])
     return bins


# %%
import matplotlib as pl

# %%
zLbins=np.array(getbins(zLu,51))
zL_norm=(zLbins+zLbins)/2
zL_norm=np.log10(-zL_norm)
zL_norm=(zL_norm-np.min(zL_norm))/(np.max(zL_norm)-np.min(zL_norm))
cc=pl.cm.terrain(zL_norm)
fig,ax=plt.subplots(1,1,figsize=(4,3),dpi=200)
for i in range(50):
    m=(zLu>zLbins[i])&(zLu<zLbins[i+1])
    y,binEdges=np.histogram((ybu/ybdu)[m],bins=50,density=True)
    bincenters=.5*(binEdges[1:]+binEdges[:-1])
    ax.plot(bincenters,y,c=cc[i],linewidth=2,alpha=.75)
    i=i+1

# %%
zLbins=np.array(getbins(zLs,51))
zL_norm=(zLbins+zLbins)/2
zL_norm=np.log10(zL_norm)
zL_norm=(zL_norm-np.min(zL_norm))/(np.max(zL_norm)-np.min(zL_norm))
cc=pl.cm.terrain(zL_norm)
fig,ax=plt.subplots(1,1,figsize=(4,3),dpi=200)
fpsite=fps['SITE'][:]
for i in range(50):
    m=(zLs>zLbins[i])&(zLs<zLbins[i+1])&(fpsite==b'ONAQ')
    y,binEdges=np.histogram((ybs/ybds)[m],bins=50,density=True)
    bincenters=.5*(binEdges[1:]+binEdges[:-1])
    ax.plot(bincenters,y,c=cc[i],linewidth=2,alpha=.75)
    i=i+1

# %%
zLbins=np.array(getbins(zLu,51))
data=[]
for i in range(50):
    m=(zLu>zLbins[i])&(zLu<zLbins[i+1])
    data.append(np.median(xbu[m]/xbdu[m]))
plt.semilogx(-(zLbins[1:]+zLbins[0:-1])/2,data,'-o')
plt.gca().invert_xaxis()

# %%
zLbins=np.array(getbins(zLu,51))
data=[]
for i in range(50):
    m=(zLu>zLbins[i])&(zLu<zLbins[i+1])
    data.append(np.median(ybu[m]/ybdu[m]))
plt.semilogx(-(zLbins[1:]+zLbins[0:-1])/2,data,'-o')
plt.gca().invert_xaxis()

# %%
zLbins=np.array(getbins(zLs,51))
data=[]
for i in range(50):
    m=(zLs>zLbins[i])&(zLs<zLbins[i+1])
    data.append(np.median(ybs[m]/ybds[m]))
plt.semilogx((zLbins[1:]+zLbins[0:-1])/2,data,'-o')

# %%
zLbins=np.array(getbins(zLs,51))
data=[]
for i in range(50):
    m=(zLs>zLbins[i])&(zLs<zLbins[i+1])
    data.append(np.median(xbs[m]/xbds[m]))
plt.semilogx((zLbins[1:]+zLbins[0:-1])/2,data,'-o')

# %%
fpsite=fpu['SITE'][:]
m=(np.abs(zLu)>-10**(-3))&(fpsite==b'NOGP')
plt.hist((ybu/ybdu)[m],bins=50)

# %%
fpu

# %%
sites=[]
out={'sites':[],'site_std_dsm':[],'yb_site':np.zeros((47,100)),'ratio_site':np.zeros((47,100)),'ratio_all':np.zeros((100,)),'yb_all':np.zeros((100,)),'ybd_site':np.zeros((47,100)),'ybd_all':np.zeros((100,))}

zLu=fpu['zzd'][:]/fpu['L_MOST'][:]
zLs=fps['zzd'][:]/fps['L_MOST'][:]

zLbinsu=np.array(getbins(zLu,51))
zL_norm=(zLbinsu+zLbinsu)/2
zL_norm=np.log10(-zL_norm)
zL_normu=(zL_norm-np.min(zL_norm))/(np.max(zL_norm)-np.min(zL_norm))

zLbinss=np.array(getbins(zLs,51))
zL_norm=(zLbinss+zLbinss)/2
zL_norm=np.log10(zL_norm)
zL_norms=(zL_norm-np.min(zL_norm))/(np.max(zL_norm)-np.min(zL_norm))

out['by_site']=np.zeros((47,100))
s=0

ybds=fps['ANID_YB'][:]
ybdu=fpu['ANID_YB'][:]
ybs=fps['ANI_YB'][:]
ybu=fpu['ANI_YB'][:]

for site in np.unique(fpu['SITE'][:]):
    print(site,flush=True,end=',')
    m0=fpu['SITE'][:]==site
    out['sites'].append(str(site)[2:-1])
    out['site_std_dsm'].append(np.nanmedian(fpu['std_dsm'][m0]))
    for i in range(50):
        m=m0&(zLu>zLbinsu[i])&(zLu<zLbinsu[i+1])
        out['ybd_site'][s,i]=np.nanmedian(ybdu[m])
        out['yb_site'][s,i]=np.nanmedian(ybu[m])
        out['ratio_site'][s,i]=np.nanmedian(ybu[m]/ybdu[m])
    m0=fps['SITE'][:]==site
    for i in range(50):
        m=m0&(zLs>zLbinss[i])&(zLs<zLbinss[i+1])
        out['ybd_site'][s,i+50]=np.nanmedian(ybds[m])
        out['yb_site'][s,i+50]=np.nanmedian(ybs[m])
        out['ratio_site'][s,i+50]=np.nanmedian(ybs[m]/ybds[m])
    s=s+1
    

# %%
for i in range(50):
    m=(zLu>zLbinsu[i])&(zLu<zLbinsu[i+1])
    out['ybd_all'][i]=np.nanmedian(ybdu[m])
    out['yb_all'][i]=np.nanmedian(ybu[m])
    out['ratio_all'][i]=np.nanmedian(ybu[m]/ybdu[m])
for i in range(50):
    m=(zLs>zLbinss[i])&(zLs<zLbinss[i+1])
    out['ybd_all'][i+50]=np.nanmedian(ybds[m])
    out['yb_all'][i+50]=np.nanmedian(ybs[m])
    out['ratio_all'][i+50]=np.nanmedian(ybs[m]/ybds[m])

# %%
zLout=list((zLbinsu[1:]+zLbinsu[:-1])/2)
zLout.extend(list((zLbinss[1:]+zLbinss[:-1])/2))

# %%
out['zL']=np.array(zLout)

# %%
plt.scatter(out['site_std_dsm'][:],np.mean(out['ratio_site'][:],axis=1))

# %%
plt.semilogx(out['zL'][:],out['ratio_all'][:])

# %%
plt.semilogx(-out['zL'][:],out['ratio_all'][:])

# %%
pickle.dump(out,open('/home/tswater/Documents/data4ben.p','wb'))

# %%

# %%
