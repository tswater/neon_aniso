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
def sort_together(X,Y):
    # X is an N length array to sort based on. Y is an M x N array of things that will sort
    X=X.copy()
    dic={}
    for i in range(len(X)):
        dic[X[i]]=[]
    for i in range(len(Y)):
        for j in range(len(X)):
            dic[X[j]].append(Y[i][j])
    X=np.array(X)
    X.sort()
    Yout=[]
    for i in range(len(Y)):
        Yout.append([])
    for i in range(len(Y)):
        for j in range(len(X)):
            Yout[i].append(dic[X[j]][i])
    return X,Yout

class_names={41:'Deciduous',42:'Evergreen',43:'Mix Forest',51:'Dwarf Scrub',52:'Shrub',\
             71:'Grassland',72:'AK:Sedge',81:'Pasture',82:'Crops',90:'Wetland'}
class_colors={11:'royalblue',12:'whitesmoke',21:'pink',22:'lightcoral',23:'red',24:'maroon',31:'silver',\
              41:'limegreen',42:'darkgreen',43:'yellowgreen',51:'darkgoldenrod',52:'tan',\
              71:'wheat',72:'khaki',73:'darkkhaki',74:'darkseagreen',81:'gold',82:'sandybrown',90:'skyblue',95:'cadetblue',0:'white'}

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

# %% [markdown]
# # Check Traditional MOST scaling yb diag

# %%
zLu=-np.logspace(-3,4)
zLs=np.logspace(-3,4)
phiu_u=2.55*(1-3*zLu)**(1/3)
phiu_s=2.06*np.ones(zLs.shape)
phiv_u=2.05*(1-3*zLu)**(1/3)
phiv_s=2.06*np.ones(zLs.shape)
phiw_u=1.35*(1-3*zLu)**(1/3)
phiw_s=1.6*np.ones(zLs.shape)

# %%
ustr=.00027*(10-np.log(-zLu))**3
uu=(phiu_u*ustr)**2
vv=(phiv_u*ustr)**2
ww=(phiw_u*ustr)**2
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
uu=(phiu_s*ustr)**2
vv=(phiv_s*ustr)**2
ww=(phiw_s*ustr)**2
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
plt.semilogx(zLs,ybds)

# %%
plt.semilogx(zLs,ybds)

# %%
print(np.mean(ybdu))
print(np.mean(xbdu))
print(np.mean(ybds))
print(np.mean(xbds))

# %%
np.sqrt(3)/4

# %%
plt.hexbin(np.log(-fpu['zzd'][:]/fpu['L_MOST'][:]),fpu['USTAR'][:],mincnt=1,cmap='terrain')
plt.gca().invert_xaxis()
plt.plot(np.log(-zLu),.00027*(10-np.log(-zLu))**3,'k--')
plt.xlim(5,-5)
plt.ylim(-.5,3)

# %% [markdown]
# # Check SC23 scaling yb diag

# %%
anibins=np.linspace(0.05,np.sqrt(3)/2-.05)
anilvl=(anibins[0:-1]+anibins[1:])/2

zL_u=-np.logspace(-4,2,40)
zL=zL_u.copy()
zL=zL.reshape(1,40).repeat(49,0)

ani=anilvl.copy()
ani=ani.reshape(49,1).repeat(40,1)

# U Unstable
a=.784-2.582*np.log10(ani)
atw=0.5-2.96*np.log10(ani)
u_u_stp=a*(1-3*zL)**(1/3)
u_u_tsw=atw*(1-3*zL)**(1/3)
u_u_old=2.55*(1-3*zL)**(1/3)

# V Unstable
a=.725-2.702*np.log10(ani)
atw=.397-2.721*np.log10(ani)
v_u_stp=a*(1-3*zL)**(1/3)
v_u_tsw=atw*(1-3*zL)**(1/3)
v_u_old=2.05*(1-3*zL)**(1/3)

# W Unstable 
a=1.119-0.019*ani-.065*ani**2+0.028*ani**3
atw=.878-.790*np.log10(ani)-.688*np.log10(ani)**2
w_u_tsw=atw*(1-3*zL)**(1/3)
w_u_stp=a*(1-3*zL)**(1/3)
w_u_old=1.35*(1-3*zL)**(1/3)
#print(a)


############### STABLE ##################
zL_s=np.logspace(-4,2,40)
zL=zL_s.copy()
zL=zL.reshape(1,40).repeat(49,0)

# U Stable
a_=np.array([2.332,-2.047,2.672])
c_=np.array([.255,-1.76,5.6,-6.8,2.65])
a=0
c=0
for i in range(3):
    a=a+a_[i]*ani**i
for i in range(5):
    c=c+c_[i]*ani**i
u_s_stp=a*(1+3*zL)**(c)
u_s_old=2.06*np.ones(zL.shape)

# V Stable
a_=np.array([2.385,-2.781,3.771])
c_=np.array([.654,-6.282,21.975,-31.634,16.251])
a=0
c=0
for i in range(3):
    a=a+a_[i]*ani**i
for i in range(5):
    c=c+c_[i]*ani**i
v_s_stp=a*(1+3*zL)**(c)
v_s_old=2.06*np.ones(zL.shape)


# W Stable
a_=np.array([.953,.188,2.243])
c_=np.array([.208,-1.935,6.183,-7.485,3.077])
a=0
c=0
for i in range(3):
    a=a+a_[i]*ani**i
for i in range(5):
    c=c+c_[i]*ani**i
w_s_stp=a*(1+3*zL)**(c)
w_s_old=1.6*np.ones(zL.shape)

ust_stp=[u_u_stp,v_u_stp,w_u_stp]
stb_stp=[u_s_stp,v_s_stp,w_s_stp]
ust_old=[u_u_old,v_u_old,w_u_old]
stb_old=[u_s_old,v_s_old,w_s_old]

# %%
uu=np.zeros((v_s_stp.shape[0]*v_s_stp.shape[1],))
vv=np.zeros((uu.shape))
ww=np.zeros((uu.shape))
uv=np.zeros((uu.shape))
uw=np.zeros((uu.shape))
vw=np.zeros((uu.shape))
anilong=np.zeros((uu.shape))
zLlong=np.zeros((uu.shape))

k=0
for i in range(ani.shape[0]):
    for j in range(ani.shape[1]):
        uu[k]=u_u_stp[i,j]**2
        vv[k]=v_u_stp[i,j]**2
        ww[k]=w_u_stp[i,j]**2
        anilong[k]=ani[i,j]
        zLlong[k]=-zL[i,j]
        k=k+1

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
abin=np.linspace(.05,.8)
alvl=(abin[1:]+abin[0:-1])/2
med_ybd=[]
med_yb=[]
yb=fpu['ANI_YB'][:]
for i in range(len(abin)-1):
    m=(yb>abin[i])&(yb<abin[i+1])
    med_ybd.append(np.median(fpu['ANID_YB'][m]))
    med_yb.append(np.median(fpu['ANI_YB'][m]))

# %%
plt.hexbin(fpu['ANI_YB'][:],fpu['ANID_YB'][:],mincnt=1,cmap='terrain')
plt.scatter(anilong,ybdu,color='white')
plt.scatter(med_yb,med_ybd,color='black')
plt.xlabel('$y_b$')
plt.ylabel('$y_{b\ diag}$')
plt.savefig('../../plot_output/trash1a.png', bbox_inches = "tight")

# %%
plt.hexbin(fpu['ANI_YB'][:],fpu['ANID_YB'][:],mincnt=1,cmap='terrain')

# %%
plt.hexbin(fpu['ANI_YB'][:],fpu['ANI_YB'][:]/fpu['ANID_YB'][:],mincnt=1,cmap='terrain')
plt.scatter(anilong,anilong/ybdu,c='white')
plt.scatter(anilong,anilong/.378,c='lightgrey')
plt.scatter(med_yb,np.array(med_yb)/np.array(med_ybd),c='black')
plt.ylim(.5,1.05)
plt.xlabel('$y_b$')
plt.ylabel('$y_b/y_{b\ diag}$')
plt.savefig('../../plot_output/trash2b.png', bbox_inches = "tight")

# %%
zlb=-np.logspace(-4,2)
med_ybd=[]
med_yb=[]
med_zl=[]
med_rtio=[]
yb=fpu['ANI_YB'][:]
zl=fpu['zzd'][:]/fpu['L_MOST'][:]
for i in range(len(zlb)-1):
    m=(zl<zlb[i])&(zl>zlb[i+1])#&(yb>.3)
    med_ybd.append(np.median(fpu['ANID_YB'][m]))
    med_yb.append(np.median(fpu['ANI_YB'][m]))
    med_zl.append(np.median(zl[m]))
    med_rtio.append(np.median(fpu['ANI_YB'][m]/fpu['ANID_YB'][m]))
    

# %%

# %%
med_zl=np.array(med_zl)
plt.semilogx(-np.array(med_zl),med_rtio)
plt.semilogx(-med_zl,.13*(1-2*med_zl)**(1/3)+.705)
#plt.semilogx(-np.array(med_zl),med_ybd)
plt.ylim(.83,1)
plt.gca().invert_xaxis()

# %%
np.min(.13*(1-2*med_zl)**(1/3)+.705)

# %%
m2=(fpu['ANI_YB'][:]>.3)
plt.hexbin(np.log10(-zl[m2]),fpu['ANI_YB'][m2]/fpu['ANID_YB'][m2],mincnt=1,cmap='terrain')
plt.plot(np.log10(-np.array(med_zl)),med_rtio,'w--')
plt.plot([-4,2],[.86,.86])
plt.gca().invert_xaxis()
plt.xlim(-4,2)
plt.ylim(.2,1.05)

# %%
.1/.35

# %%
med_yb=np.array(med_yb)
plt.plot(med_yb,np.array(med_yb)/np.array(med_ybd))
#plt.plot(med_yb,fxn_fit(med_yb))
plt.plot(med_yb,1.03-.7*med_yb)
plt.plot(med_yb,.735+.25*med_yb)
plt.ylim(.8,.95)

# %%
pcov=np.polyfit(med_yb,np.array(med_yb)/np.array(med_ybd),2)
def fxn_fit(x):
    return pcov[5]+pcov[4]*x+pcov[3]*x**2+pcov[2]*x**3+pcov[1]*x**4+pcov[0]*x**5
def fxn_fit(x):
    return pcov[4]+pcov[3]*x**1+pcov[2]*x**2+pcov[1]*x**3+pcov[0]*x**4
def fxn_fit(x):
    return pcov[3]+pcov[2]*x**1+pcov[1]*x**2+pcov[0]*x**3
def fxn_fit(x):
    return pcov[2]+pcov[1]*x**1+pcov[0]*x**2
print(pcov)

# %% [markdown]
# # Using the adjustment based on the peaks

# %%
fz=.13*(1-2*med_zl)**(1/3)+.705
fz[fz>.975]=.975
yb1=(1.03-fz)/.7
yb2=(fz-.735)/.25
yb1[yb1>.28]=float('nan')
yb2[yb2<.4]=float('nan')
plt.semilogx(-med_zl,yb1)
plt.semilogx(-med_zl,yb2)

# %%
plt.semilogx(-zL_u,u_u_old[0,:])
plt.semilog
plt.gca().invert_xaxis()
plt.ylim(2,10)

# %%
plt.plot(med_yb,med_ybd)


# %%
fz[35]

# %%
med_zl[35]

# %%
x0=.4
x0p=np.interp(x0,med_yb,med_ybd)
x1=fz[35]*x0p
x1p=np.interp(x1,med_yb,med_ybd)
x2=fz[35]*x1p
x2p=np.interp(x2,med_yb,med_ybd)


# %%
def iterate(y0,z):
    fz_=np.interp(z,med_zl,fz)
    y=[y0]
    yp=[]
    for i in range(50):
        yp.append(np.interp(y[-1],med_yb,med_ybd))
        y.append(fz_*yp[-1])
    return y


# %%
out=iterate(.35,-1)
plt.plot(out)

# %%
final=[]
for zzz in med_zl:
    out=iterate(.01,zzz)
    final.append(out[-1])
plt.semilogx(-med_zl,final)

# %%
plt.semilogx(-med_zl,fz)

# %% [markdown]
# # other Stuff

# %%
m=(fpu['SITE'][:]==b'WREF')&(sday<30000)
plt.hexbin(fpu['Ustr'][m],fpu['ANI_YB'][m],mincnt=1,cmap='terrain')

# %%
75000/3600

# %%
m=(fpu['SITE'][:]==b'BONA')#&(sday<30000)
plt.hexbin(fpu['Ustr'][m],fpu['ANI_YB'][m],mincnt=1,cmap='terrain')

# %%
x=sday
xbins=np.linspace(np.nanpercentile(x,1),np.nanpercentile(x,99),47)
xtrue=[]
ytrue=[]
for i in range(46):
    m=(x>xbins[i])&(x<xbins[i+1])
    xtrue.append(np.nanmedian(x[m]))
    ytrue.append(np.sum(m))
plt.plot(xtrue,ytrue,color='black',linewidth=2)

# %%
m=fpu['SITE'][:]!='hello'
m=fpu['Ustr'][:]>2
m=m&(fpu['SITE'][:]==b'SERC')
x=sday[m] #fpu['SW_IN'][m] #sday[m]
y=fpu['H'][m]
xbins=np.linspace(np.nanpercentile(x,1),np.nanpercentile(x,99))
xtrue=[]
ytrue=[]
for i in range(49):
    m=(x>xbins[i])&(x<xbins[i+1])
    xtrue.append(np.nanmedian(x[m]))
    ytrue.append(np.nanmedian(y[m]))
    
plt.hexbin(x,y,mincnt=1,cmap='terrain',extent=(0,80000,0,500))
plt.plot(xtrue,ytrue,color='white',linewidth=2)

# %%
ffu.keys()

# %%
from scipy import stats
fpsite=fpu['SITE'][:]
nlcds=[]
treecvr=[]
for i in range(47):
    site=np.unique(fpsite)[i]
    ss=str(site)[2:-1]
    ffu=h5py.File('/home/tswater/tyche/data/neon/foot_stats/'+ss+'_U.h5','r')
    if site==b'PUUM':
        nlcds.append(42)
    else:
        nlcds.append(stats.mode(ffu['nlcd_dom'])[0])
    if site==b'MOAB':
        treecvr.append(0)
    elif site in [b'ORNL',b'GRSM']:
        treecvr.append(5)
    else:
        treecvr.append(np.nanmedian(ffu['mean_chm'][:]))

# %%
for i in range(47):
    site=np.unique(fpsite)[i]
    try:
        print(str(site)+' '+str(treecvr[i])[0:5]+' '+str(class_names[nlcds[i]]))
    except:
        print(str(site)+' '+str(treecvr[i])[0:5])

# %%
np.unique(fpu['zzd'][:])

# %%
j=0
for site in np.unique(fpu['SITE'][:]):
    m=fpu['SITE'][:]==site
    zzd=fpu['zzd'][m][0]
    #if zzd<8:
    #    j=j+1
    #    continue
    #if site in [b'JORN',b'OAES',b'ONAQ',b'SRER',b'MOAB',b'CPER',b'TOOL',b'HEAL']:
    #    j=j+1
    #    continue
    #if site in
    #if treecvr[j]<1:
    #    j=j+1
    #    continue
    m=m&(fpu['Ustr'][:]>4)
    x=sday[m]
    y=fpu['ANI_YB'][m]
    xbins=np.linspace(np.nanpercentile(x,1),np.nanpercentile(x,99),6)
    xtrue=[]
    ytrue=[]

    try:
        color=class_colors[nlcds[j]]
    except:
        color='darkgreen'
        
    
    for i in range(5):
        m=(x>xbins[i])&(x<xbins[i+1])
        xtrue.append(np.nanmedian(x[m]))
        ytrue.append(np.nanmedian(y[m]))
    plt.plot(xtrue,ytrue,color=color)
    j=j+1
plt.ylim(0,.6)

# %%
site=b'SERC' #BONA, SJER, #DEJU SERC JERC
# coastal virginia SERC, southern georgia JERC, Alaska Taiga BONA/DEJU, oak savannah (SJER)
m=fpu['SITE'][:]==site
m=m&(fpu['Ustr'][:]>2)
x=sday[m]
y=fpu['ANI_YB'][m]
xbins=np.linspace(np.nanpercentile(x,1),np.nanpercentile(x,99),24)
xtrue=[]
ytrue=[]

try:
    idx=np.where(fpst['site'][:]==site)[0][0]
    #print(idx)
    nlcd=fpst['nlcd_dom'][idx]
    nlcd=int(nlcd)
    #print(str(site)+str(nlcd))
    if nlcd==0:
        color='darkgreen'
    else:
        color=class_colors[nlcd]
except Exception as e:
    print(e)
    color='darkgreen'
    

for i in range(23):
    m=(x>xbins[i])&(x<xbins[i+1])
    xtrue.append(np.nanmedian(x[m]))
    ytrue.append(np.nanmedian(y[m]))
plt.plot(xtrue,ytrue,color=color)
plt.ylim(0,.6)

# %%
class_colors[42]

# %%
delta12=[]
delta13=[]
varis=[]
for v in fpu.keys():
    if v=='SITE':
        continue
    m1=(fpu['ANI_YB'][:]>.4)&(fpu['Ustr'][:]<1.5)
    m2=((fpu['ANI_YB'][:]<.4)&(fpu['ANI_YB'][:]>.2))&(fpu['Ustr'][:]<1.5)
    m3=(fpu['ANI_YB'][:]<.2)&(fpu['Ustr'][:]<1.5)
    delta12.append((np.median(fpu[v][m1])-np.median(fpu[v][m2]))/(np.nanmedian(np.abs(fpu[v][:]))+.0001))
    delta13.append((np.median(fpu[v][m1])-np.median(fpu[v][m3]))/(np.nanmedian(np.abs(fpu[v][:]))+.0001))
    varis.append(v)

# %%
plt.figure(figsize=(16,3))
delta12=np.array(delta12)
m=np.abs(delta12)>.05
varis=np.array(varis)
plt.bar(varis[m],delta12[m])
plt.xticks(rotation=45)
plt.ylim(-1,1)

# %%
fpst=h5py.File('/home/tswater/tyche/data/neon/static_data.h5','r')


# %%
fpst.keys()

# %%
fpst['site'][:]

# %%
np.unique(fpu['SITE'][:])

# %%
fpst['nlcd_dom'][:]

# %%
truehour=[]
truedoy=[]
oldsite=b'SAFD'
s=-1
for t in range(len(fpu['TIME'][:])):
    site=fpu['SITE'][t]
    if site != oldsite:
        print(site)
        s=s+1
        try:
            offset=float(fpst['utc_off'][s])
        except:
            if site==b'WREF':
                offset=-9
            elif site==b'YELL':
                offset=-8
        oldsite=site
    tt=fpu['TIME'][t]
    dt=datetime.utcfromtimestamp(tt)+timedelta(hours=offset)
    truehour.append(dt)
    truedoy.append(dt.timetuple().tm_yday)

# %%
hour=[]
sday=[]
for t in truehour:
    hour.append(t.hour)
    sday.append(t.hour*3600+t.minute*60+t.second)
sday=np.array(sday)
hour=np.array(hour)
truedoy=np.array(truedoy)

# %%
3600*24

# %%

# %%
