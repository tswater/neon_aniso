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
def getbins(A,n):
     B=np.sort(A)
     bins=[]
     for i in np.linspace(0,len(A)-1,n):
         i=int(i)
         bins.append(B[i])
     return bins

def getbins2D(A,B,n,n2):
    bina=getbins(A,n)
    binb=np.zeros((n-1,n2))
    for i in range(n-1):
        m=(A>bina[i])&(A<bina[i+1])
        binb[i,:]=getbins(B[m],n2)
    return bina,binb


# %%
import matplotlib as pl

# %%
N=20
N2=25
m=~(fpu['SITE'][:]==7)
zL_=(fpu['zzd'][m])/fpu['L_MOST'][m]#(fpu['zzd'][:])/fpu['L_MOST'][:]
ybine,zbine=getbins2D(fpu['ANI_YB'][m],zL_,N,N2)
#phi_=np.sqrt(fpu['UU'][m]/fpu['WW'][m])
phi_=np.abs(fpu['T_SONIC_SIGMA'][:]/(fpu['WTHETA'][:]/np.sqrt(fpu['WW'][:])))

xb_=fpu['ANI_XB'][m]
yb_=fpu['ANI_YB'][m]

zbtrue=np.ones((len(ybine)-1,len(zbine[0])-1))*float('nan')
ybtrue=np.ones((len(ybine)-1,len(zbine[0])-1))*float('nan')

for i in range(len(ybine)-1):
    for j in range(len(zbine[0])-1):
        m=(zL_<zbine[i,j+1])&(zL_>zbine[i,j])&(yb_<ybine[i+1])&(yb_>ybine[i])
        zbtrue[i,j]=np.nanmean(zL_[m])
        ybtrue[i,j]=np.nanmean(yb_[m])

mad_norm=(ybtrue[:,0]-.05)/(.6-.05)
cc=pl.cm.terrain(mad_norm)
sm = plt.cm.ScalarMappable(cmap=pl.cm.terrain, norm=plt.Normalize(vmin=.05, vmax=.6))

fig,ax=plt.subplots(1,1,figsize=(4,3),dpi=200)

phim=np.zeros((N-1,N2-1))
phima=np.zeros((N-1,N2-1))
for i in range(len(ybine)-1):
    for j in range(len(zbine[0])-1):
        m=(zL_<zbine[i,j+1])&(zL_>zbine[i,j])&(yb_<ybine[i+1])&(yb_>ybine[i])
        zbtrue[i,j]=np.nanmean(zL_[m])
        ybtrue[i,j]=np.nanmean(yb_[m])
        phim[i,j]=np.nanmedian(phi_[m])

for i in range(len(ybine)-1):
    plt.semilogx(-zbtrue[i,:],phim[i,:],c=cc[i])


plt.gca().invert_xaxis()
plt.ylabel('$\Phi_v$')
fig.colorbar(sm,cax=ax.inset_axes([0.95, 0.1, 0.05, 0.8]),label='$y_B$')

# %%

# %%
N=20
N2=25

m=~(fpu['SITE'][:]==7)
yb_=np.sqrt(fpu['WW'][:])/fpu['USTAR'][:]#fpu['ANID_YB'][:]
zL_=(fpu['zzd'][m])/fpu['L_MOST'][m]#(fpu['zzd'][:])/fpu['L_MOST'][:]
ybine,zbine=getbins2D(yb_,zL_,N,N2)
phi_=np.sqrt(fpu['UU'][m])/fpu['USTAR'][m]
#phi_=np.abs(fpu['T_SONIC_SIGMA'][:]/(fpu['WTHETA'][:]/fpu['USTAR'][:]))
xb_=fpu['ANI_XB'][m]

zbtrue=np.ones((len(ybine)-1,len(zbine[0])-1))*float('nan')
ybtrue=np.ones((len(ybine)-1,len(zbine[0])-1))*float('nan')

for i in range(len(ybine)-1):
    for j in range(len(zbine[0])-1):
        m=(zL_<zbine[i,j+1])&(zL_>zbine[i,j])&(yb_<ybine[i+1])&(yb_>ybine[i])
        zbtrue[i,j]=np.nanmedian(zL_[m])
        ybtrue[i,j]=np.nanmedian(yb_[m])

mad_norm=(ybtrue[:,0]-.05)/(3-.05)
cc=pl.cm.terrain(mad_norm)
sm = plt.cm.ScalarMappable(cmap=pl.cm.terrain, norm=plt.Normalize(vmin=.05, vmax=3))

fig,ax=plt.subplots(1,1,figsize=(4,3),dpi=200)

phim=np.zeros((N-1,N2-1))
phima=np.zeros((N-1,N2-1))
for i in range(len(ybine)-1):
    for j in range(len(zbine[0])-1):
        m=(zL_<zbine[i,j+1])&(zL_>zbine[i,j])&(yb_<ybine[i+1])&(yb_>ybine[i])
        phim[i,j]=np.nanmedian(phi_[m])

for i in range(len(ybine)-1):
    ax.semilogx(-zbtrue[i,:],phim[i,:],c=cc[i])
    
plt.ylabel('$\Phi_v$')
plt.gca().invert_xaxis()
fig.colorbar(sm,cax=ax.inset_axes([0.95, 0.1, 0.05, 0.8]),label='$\sigma_W/\overline{U}$')

# %%
N=20
N2=25
ybine,zbine=getbins2D(fps['ANI_YB'][:],fps['zzd'][:]/fps['L_MOST'][:],N,N2)

phi_=np.sqrt(fps['VV'][:])/fps['USTAR'][:]
zL_=(fps['zzd'][:])/fps['L_MOST'][:]
xb_=fps['ANI_XB'][:]
yb_=fps['ANI_YB'][:]

zbtrue=np.ones((len(ybine)-1,len(zbine[0])-1))*float('nan')
ybtrue=np.ones((len(ybine)-1,len(zbine[0])-1))*float('nan')

for i in range(len(ybine)-1):
    for j in range(len(zbine[0])-1):
        m=(zL_<zbine[i,j+1])&(zL_>zbine[i,j])&(yb_<ybine[i+1])&(yb_>ybine[i])
        zbtrue[i,j]=np.nanmean(zL_[m])
        ybtrue[i,j]=np.nanmean(yb_[m])

# %%

mad_norm=(ybtrue[:,0]-np.nanmin(ybtrue[:,0]))/(np.nanmax(ybtrue[:,0])-np.nanmin(ybtrue[:,0]))
cc=pl.cm.terrain(mad_norm)
sm = plt.cm.ScalarMappable(cmap=pl.cm.terrain, norm=plt.Normalize(vmin=np.nanmin(ybtrue[:,0]), vmax=np.nanmax(ybtrue[:,0])))

fig,ax=plt.subplots(1,1,figsize=(4,3),dpi=200)

phim=np.zeros((N-1,N2-1))
phima=np.zeros((N-1,N2-1))
for i in range(len(ybine)-1):
    for j in range(len(zbine[0])-1):
        m=(zL_<zbine[i,j+1])&(zL_>zbine[i,j])&(yb_<ybine[i+1])&(yb_>ybine[i])
        zbtrue[i,j]=np.nanmean(zL_[m])
        ybtrue[i,j]=np.nanmean(yb_[m])
        phim[i,j]=np.nanmedian(phi_[m])

for i in range(len(ybine)-1):
    plt.semilogx(zbtrue[i,:],phim[i,:],c=cc[i])

fig.colorbar(sm,cax=ax.inset_axes([0.95, 0.1, 0.05, 0.8]),label='$y_B$')

# %%
N=20
N2=25
yb_=fps['ANID_YB'][:]#np.sqrt(fps['WW'][:])/np.abs(fps['Ustr'][:])
ybine,zbine=getbins2D(yb_,fps['zzd'][:]/fps['L_MOST'][:],N,N2)

phi_=np.sqrt(fps['VV'][:])/fps['USTAR'][:]
zL_=(fps['zzd'][:])/fps['L_MOST'][:]
xb_=fps['ANI_XB'][:]

zbtrue=np.ones((len(ybine)-1,len(zbine[0])-1))*float('nan')
ybtrue=np.ones((len(ybine)-1,len(zbine[0])-1))*float('nan')

for i in range(len(ybine)-1):
    for j in range(len(zbine[0])-1):
        m=(zL_<zbine[i,j+1])&(zL_>zbine[i,j])&(yb_<ybine[i+1])&(yb_>ybine[i])
        zbtrue[i,j]=np.nanmedian(zL_[m])
        ybtrue[i,j]=np.nanmedian(yb_[m])

# %%
mad_norm=(ybtrue[:,0]-np.nanmin(ybtrue[:,0]))/(np.nanmax(ybtrue[:,0])-np.nanmin(ybtrue[:,0]))
cc=pl.cm.terrain(mad_norm)
sm = plt.cm.ScalarMappable(cmap=pl.cm.terrain, norm=plt.Normalize(vmin=np.nanmin(ybtrue[:,0]), vmax=np.nanmax(ybtrue[:,0])))

fig,ax=plt.subplots(1,1,figsize=(4,3),dpi=200)

phim=np.zeros((N-1,N2-1))
phima=np.zeros((N-1,N2-1))
for i in range(len(ybine)-1):
    for j in range(len(zbine[0])-1):
        m=(zL_<zbine[i,j+1])&(zL_>zbine[i,j])&(yb_<ybine[i+1])&(yb_>ybine[i])
        phim[i,j]=np.nanmedian(phi_[m])

for i in range(len(ybine)-1):
    ax.semilogx(zbtrue[i,:],phim[i,:],c=cc[i])
    
fig.colorbar(sm,cax=ax.inset_axes([0.95, 0.1, 0.05, 0.8]),label='$\sigma_W/\overline{U}$')

# %%
# ls /home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/

# %%
d_u=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_corr_ust_v1.p','rb'))
d_s=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_corr_stb_v1.p','rb'))

# %%

# %%
data=np.zeros((47,nx,ny))
i=0
for site in d_u['sitelevel'].keys():
    data[i,:,:]=d_u['sitelevel'][site]['spearmanr'][:]
    i=i+1

# %%
data=np.nanmedian(data,axis=0)
xvars=d_u['xvars'][:]
yvars=d_u['yvars'][:]
nx=len(xvars)
ny=len(yvars)
plt.figure(figsize=(16,16))
plt.imshow(data.T,cmap='coolwarm',vmin=-0.5,vmax=.5,interpolation=None)
plt.xticks(np.linspace(.5,nx-.5,nx),xvars,rotation=45)
plt.yticks(np.linspace(.5,ny-.5,ny),yvars,rotation=45)
ax=plt.gca()
ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
#plt.title()
plt.colorbar(orientation='horizontal',shrink=.3)

# %%
for site in ['ABBY','ONAQ','NOGP','TOOL','WREF','SOAP','BART','SJER','JORN','KONA']:
    data=d_s['sitelevel'][site]['spearmanr'][:]
    xvars=d_s['xvars'][:]
    yvars=d_s['yvars'][:]
    nx=len(xvars)
    ny=len(yvars)
    plt.figure(figsize=(16,16))
    plt.imshow(data.T,cmap='coolwarm',vmin=-0.5,vmax=.5,interpolation=None)
    plt.xticks(np.linspace(.5,nx-.5,nx),xvars,rotation=45)
    plt.yticks(np.linspace(.5,ny-.5,ny),yvars,rotation=45)
    ax=plt.gca()
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    #plt.title()
    plt.colorbar(orientation='horizontal',shrink=.3)

# %%
data=d_u['spearmanr'][:]
xvars=d_u['xvars'][:]
yvars=d_u['yvars'][:]
nx=len(xvars)
ny=len(yvars)
plt.figure(figsize=(16,16))
plt.imshow(data.T,cmap='coolwarm',vmin=-0.5,vmax=.5,interpolation=None)
plt.xticks(np.linspace(.5,nx-.5,nx),xvars,rotation=45)
plt.yticks(np.linspace(.5,ny-.5,ny),yvars,rotation=45)
ax=plt.gca()
ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
#plt.title()
plt.colorbar(orientation='horizontal',shrink=.3)

# %%
data=d_s['spearmanr'][:]
xvars=d_s['xvars'][:]
yvars=d_s['yvars'][:]
nx=len(xvars)
ny=len(yvars)
plt.figure(figsize=(16,16))
plt.imshow(data.T,cmap='coolwarm',vmin=-0.5,vmax=.5,interpolation=None)
plt.xticks(np.linspace(.5,nx-.5,nx),xvars,rotation=45)
plt.yticks(np.linspace(.5,ny-.5,ny),yvars,rotation=45)
ax=plt.gca()
ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
#plt.title()
plt.colorbar(orientation='horizontal',shrink=.3)

# %%
from scipy import stats
from scipy import optimize

# %%
phi=np.sqrt(fps['UU'][:])/fps['USTAR'][:]
zL=fps['zzd'][:]/fps['L_MOST'][:]
phi_old=2.06#phi_old=2.05*(1-3*zL)**(1/3)
ad=np.abs(phi-phi_old)

# %%
a=stats.spearmanr(fps['ANI_XB'][:],ad)
print(a)

# %%
corr1=[]
corr2=[]
corr3=[]
sites=[]
phi=np.sqrt(fpu['UU'][:])/fpu['USTAR'][:]
zL=fpu['zzd'][:]/fpu['L_MOST'][:]
phi_old=2.05*(1-3*zL)**(1/3)
ad=np.abs(phi-phi_old)
for site in np.unique(fpu['SITE'][:]):
    m=fpu['SITE'][:]==site
    corr1.append(stats.spearmanr(fpu['UU'][m],fpu['VV'][m])[0])
    corr2.append(stats.spearmanr(fpu['VV'][m],fpu['WW'][m])[0])
    corr3.append(stats.spearmanr(fpu['WW'][m],fpu['VV'][m])[0])
    sites.append(str(site)[2:-1])


# %%
plt.figure(figsize=(14,8))
plt.subplot(3,1,1)
plt.bar(sites,np.abs(corr1))
plt.xticks(rotation=45)

plt.subplot(3,1,2)
plt.bar(sites,np.abs(corr2))
plt.xticks(rotation=45)

plt.subplot(3,1,3)
plt.bar(sites,np.abs(corr3))
plt.xticks(rotation=45)
plt.title('')


# %%

# %%

# %%
# CORRELATIONS (UNSTABLE)
# YB - AD_U : .17
# W/WS - AD_U: .34
# XB - AD_U: .24
# W/Ust - AD_U: .26

# YB - AD_V: -.13
# W/WS - AD_V: .11
# XB - AD_V: .26
# W/Ust - AD_U: .19

# XB - AD_W: .25
# YB - AD_W: -.12

# %%
# CORRELATIONS (Stable)
# YB - AD_U : -.09
# W/WS - AD_U: .06
# XB - AD_U: .23
# W/Ust - AD_U: .28

# YB - AD_V: -.13
# W/WS - AD_V: .11
# XB - AD_V: .26
# W/Ust - AD_U: .19

# XB - AD_W: .25
# YB - AD_W: -.12

# %%

# %%

# %% [markdown]
# # TEST SELF SIMILARITY

# %%
def getlines(zL_,phi_):
    zLbins=np.logspace(-3,2)
    zLbc=(zLbins[1:]+zLbins[0:-1])/2
    phiout=[]
    for i in range(len(zLbc)):
        phiout.append(np.nanmedian(phi_[(zL_>zLbins[i])&(zL_<zLbins[i+1])]))
    return zLbc,np.array(phiout)


# %%
xb=fpu['ANI_XB'][:]
m=xb>0
uu=fpu['VV'][:][m]
ust=fpu['USTAR'][:][m]
wT=fpu['WTHETA'][:][m]
gt=9.8/fpu['T_SONIC'][:][m]
zzd=fpu['zzd'][:][m]

# %%
uur=np.random.choice(uu,size=(30000,))
ustr=np.random.choice(ust,size=(30000,))
wTr=np.random.choice(wT,size=(30000,))
gtr=np.random.choice(gt,size=(30000,))
zzdr=np.random.choice(zzd,size=(30000,))

# %%
Lr=-ustr**3*(1/gtr)/.4/wTr
L=-ust**3*(1/gt)/.4/wT
zLr=zzdr/Lr
zL=zzd/L

# %%
m2=np.random.randint(low=0,high=len(uu),size=(30000,))

# %%
plt.semilogx(-zL[m2],np.sqrt(uu[m2])/ust[m2],'o',markersize=1,alpha=.1)
x,y=getlines(-zL[m2],np.sqrt(uu[m2])/ust[m2])
plt.plot(x,y)
plt.xlim(10**(-2),10**(4))
plt.ylim(0,7.5)

# %%
plt.semilogx(-zLr,np.sqrt(uur)/ustr,'o',markersize=1,alpha=.1)
x,y=getlines(-zLr,np.sqrt(uur)/ustr)
plt.plot(x,y)
plt.xlim(10**(-2),10**(4))
plt.ylim(0,7.5)


# %%

# %% [markdown]
# # Self Similarity yb

# %%
def getlines(ani_,phi_,anibins):
    anibc=(anibins[1:]+anibins[0:-1])/2
    phiout=[]
    for i in range(len(anibc)):
        phiout.append(np.nanmedian(phi_[(ani_>anibins[i])&(ani_<anibins[i+1])]))
    return anibc,np.array(phiout)


# %%

# %%
phi_=np.sqrt(fpu['WW'][:])/fpu['USTAR'][:]
m=np.random.randint(low=0,high=len(phi_),size=(30000,))
zL_=fpu['zzd'][:]/fpu['L_MOST'][:]
ani_=fpu['ANI_YB'][:]
plt.plot(ani_[m],phi_[m],'o',markersize=1,alpha=.1)
x,y=getlines(ani_,phi_,np.linspace(0,.8))
plt.plot(x,y)
plt.ylim(1,6)
plt.xlim(0,.6)

# %%
m=fpu['SITE'][:]==b'NOGP'
plt.semilogx(-zL_[m],phi_[m],'o',markersize=1,alpha=.1)
plt.ylim(1,6)
plt.xlim(10**-3,10**1)

# %%
uur=np.random.choice(fpu['UU'][:],size=(30000,))
vvr=np.random.choice(fpu['VV'][:],size=(30000,))
wwr=np.random.choice(fpu['WW'][:],size=(30000,))
uvr=np.random.choice(fpu['UV'][:],size=(30000,))
uwr=np.random.choice(fpu['UW'][:],size=(30000,))
vwr=np.random.choice(fpu['VW'][:],size=(30000,))
ustr=(uwr**2+vwr**2)**(1/4)

# %%
bij=aniso(uur,vvr,wwr,uvr,uwr,vwr)
N=len(uur)
ybr=np.ones((N,))*-9999
xbr=np.ones((N,))*-9999
for t in range(N):
    if np.sum(bij[t,:,:]==-9999)>0:
        continue
    lams=np.linalg.eig(bij[t,:,:])[0]
    lams.sort()
    lams=lams[::-1]
    xbr[t]=lams[0]-lams[1]+.5*(3*lams[2]+1)
    ybr[t]=np.sqrt(3)/2*(3*lams[2]+1)


# %%
def ymx(x,m,b):
    return m*x+b
from scipy import optimize

# %%
m=ybr>0
plt.plot(ybr[m],np.sqrt(wwr[m])/ustr[m],'o',markersize=1,alpha=.1)
phir=np.sqrt(wwr)/ustr
x1,y1=getlines(ybr[m],phir[m],np.linspace(0,.8))
plt.plot(x1,y1)
params,pcov=optimize.curve_fit(ymx,ybr[m],phir[m],[1,1],bounds=([-10,-2],[10,2]),loss='cauchy')
plt.plot(x1,ymx(x1,params[0],params[1]))
plt.ylim(1,6)
plt.xlim(-.1,.8)

# %%

# %%
m=np.random.randint(low=0,high=len(phi_),size=(30000,))
phi_=np.sqrt(fpu['WW'][:])/fpu['USTAR'][:]
zL_=fpu['zzd'][:]/fpu['L_MOST'][:]
ani_=fpu['ANI_YB'][:]
plt.plot(ani_[m],phi_[m],'o',markersize=1,alpha=.1)
x,y=getlines(ani_,phi_,np.linspace(0,.8))
plt.plot(x,y)
plt.ylim(1,6)
plt.xlim(-.1,.8)

# %%
sz=1

# %%
import sys
sys.version


# %%
def get_random(var,num=10000):
    match var:
        case 'Uu': fp=fpu
        case 'Us': fp=fps
        case 'Wu': fp=fpu
        case 'Ws': fp=fps
    uur=np.random.choice(fp['UU'][:],size=(num,))
    wwr=np.random.choice(fp['WW'][:],size=(num,))
    uwr=np.random.choice(fp['UW'][:],size=(num,))
    vwr=np.random.choice(fp['VW'][:],size=(num,))
    uvr=np.random.choice(fp['UV'][:],size=(num,))
    vvr=np.random.choice(fp['VV'][:],size=(num,))
    ustr=(uwr**2+vwr**2)**(1/4)
    match var:
        case 'Uu': phir=np.sqrt(uur)/ustr; print(var)
        case 'Us': phir=np.sqrt(uur)/ustr; print(var)
        case 'Wu': phir=np.sqrt(wwr)/ustr; print(var)
        case 'Ws': phir=np.sqrt(wwr)/ustr; print(var)
    bij=aniso(uur,vvr,wwr,uvr,uwr,vwr)
    N1=len(phir)
    ybr=np.ones((N1,))*-9999
    xbr=np.ones((N1,))*-9999
    for t in range(N1):
        if np.sum(bij[t,:,:]==-9999)>0:
            continue
        lams=np.linalg.eig(bij[t,:,:])[0]
        lams.sort()
        lams=lams[::-1]
        xbr[t]=lams[0]-lams[1]+.5*(3*lams[2]+1)
        ybr[t]=np.sqrt(3)/2*(3*lams[2]+1)
    m=(ybr>0)
    print(var+': '+str(np.sum(m)))
    return phir[m],ybr[m]

def getlines(ani_,phi_,anibins):
    anibc=(anibins[1:]+anibins[0:-1])/2
    phiout=[]
    for i in range(len(anibc)):
        phiout.append(np.nanmedian(phi_[(ani_>anibins[i])&(ani_<anibins[i+1])]))
    return anibc,np.array(phiout)


# %%
sz=1
fig=plt.figure(figsize=(9*sz,5*sz))
sbf = fig.subfigures(2, 2, hspace=0,wspace=0,frameon=False)

titles={'Uu':r'U: $\zeta <0$', 
        'Us':r'U: $\zeta >0$',
        'Wu':r'W: $\zeta <0$',
        'Ws':r'W: $\zeta >0$'}

for var in ['Uu','Us','Wu','Ws']:
    # plotting setup
    match var:
        case 'Uu': i=0; j=0
        case 'Us': i=0; j=1
        case 'Wu': i=1; j=0
        case 'Ws': i=1; j=1
    gs = GridSpec(2, 2, figure=sbf[i,j],width_ratios=[1,1.7])
    ax00=sbf[i,j].add_subplot(gs[0,0])
    ax01=sbf[i,j].add_subplot(gs[1,0])
    ax10=sbf[i,j].add_subplot(gs[:,1])
    
    # data loading
    if 's' in var:
        phir,ybr=get_random(var,num=15000)
    elif 'u' in var:
        phir,ybr=get_random(var)
    N=len(phir)
    x1,y1=getlines(ybr,phir,np.linspace(0.05,.7,25))
    match var:
        case 'Uu': 
            fp=fpu
            m=np.random.randint(low=0,high=len(fp['TIME'][:]),size=(N,))
            m.sort()
            phi=np.sqrt(fp['UU'][:][m])/fp['USTAR'][:][m]
        case 'Us': 
            fp=fps
            m=np.random.randint(low=0,high=len(fp['TIME'][:]),size=(N,))
            m.sort()
            phi=np.sqrt(fp['UU'][:][m])/fp['USTAR'][:][m]
        case 'Wu': 
            fp=fpu
            m=np.random.randint(low=0,high=len(fp['TIME'][:]),size=(N,))
            m.sort()
            phi=np.sqrt(fp['WW'][:][m])/fp['USTAR'][:][m]
        case 'Ws': 
            fp=fps
            m=np.random.randint(low=0,high=len(fp['TIME'][:]),size=(N,))
            m.sort()
            phi=np.sqrt(fp['WW'][:][m])/fp['USTAR'][:][m]
    yb=fp['ANI_YB'][:][m]
    x,y=getlines(yb,phi,np.linspace(0.05,.7,25))
    
    # plot
    ax00.scatter(yb,phi,s=1,alpha=.1,color='dimgrey')
    ax00.set_xlim(0,.8)
    ax00.set_ylim(0,5)
    ax00.plot(x,y,color='k')
    ax00.set_xticks([0,np.sqrt(3)/6,np.sqrt(3)/3],[])
    ax00.set_yticks([0,2,4])

    ### MAD ####
    ml=(yb>np.min(x))&(yb<np.max(x))
    xx=yb[ml]
    yy=phi[ml]
    y2=np.interp(xx,x,y)
    print(np.median(np.abs(y2-yy)))
    print('    '+str(stats.spearmanr(yy,y2)[0]))
    
    ax01.scatter(ybr,phir,s=1,alpha=.1,color='darkgrey')
    ax01.set_xlim(0,.8)
    ax01.set_ylim(0,5)
    params,pcov=optimize.curve_fit(ymx,ybr,phir,[1,1],bounds=([-10,-2],[10,2]),loss='cauchy')
    y1=ymx(x1,params[0],params[1])
    ax01.plot(x1,y1,color='k')

    ml=(ybr>np.min(x1))&(ybr<np.max(x1))
    xx=ybr[ml]
    yy=phir[ml]
    y2=np.interp(xx,x1,y1)
    print(np.median(np.abs(y2-yy)))
    print('    '+str(stats.spearmanr(yy,y2)[0]))
    print()
    
    #['0',r'$\sqrt{3}/6$',r'$\sqrt{3}/3$']
    if 'U' in var:
        ax01.set_xticks([0,np.sqrt(3)/6,np.sqrt(3)/3],[])
        ax01.set_ylabel(r'                 $\Phi_u$')
    else:
        ax01.set_xticks([0,np.sqrt(3)/6,np.sqrt(3)/3],['0',r'$\frac{\sqrt{3}}{6}$',r'$\frac{\sqrt{3}}{3}$'])
        ax01.set_ylabel(r'                 $\Phi_v$')
    ax01.set_yticks([0,2,4]) 

    ax10.set_title(titles[var])
    ax10.plot(x,y-y1,color='k')
    ax10.plot([-1,1],[0,0],color='w',zorder=0,linewidth=3)
    ax10.scatter(yb,phi-ymx(yb,params[0],params[1]),s=1,alpha=.1,color='slategrey')
    ax10.set_xlim(0,.8)
    ax10.set_ylim(-2,2)
    if 'U' in var:
        ax10.set_xticks([0,np.sqrt(3)/6,np.sqrt(3)/3],[])
    else:
        ax10.set_xticks([0,np.sqrt(3)/6,np.sqrt(3)/3],['0',r'$\frac{\sqrt{3}}{6}$',r'$\frac{\sqrt{3}}{3}$'])
    
    plt.subplots_adjust(wspace=0.3,hspace=0.1)


# %%
m

# %%
3/6

# %%
