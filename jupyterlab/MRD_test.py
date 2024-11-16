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
def MRD(u):
    testNan = np.isnan(u)
    ind = np.where(testNan == True)
    u[ind] = 0

    Number_of_Nans = np.size(ind) #Determines how many NaNs have been found.
    # Next, we define a couple parameters related to the signal.print(f"The total number of NaNs found is of {Number_of_Nans} points")
    print(np.size(ind)/len(u))

    #Make the Signal a power of 2.
    M = np.int64(np.floor(np.log2(len(u)))) #Maximum Power of 2 within the length of the signal measure at 20Hz.
    u_short = u[0:int(2**M)]

    #-----------------------------------------------------------------------------------
    var1 = u_short
    var2 = u_short
    
    a = np.array(var1)
    b = np.array(var2)
    
    D = np.zeros(M+1)
    Mx = 0
    for ims in range(0,M-Mx+1):
        ms = M-ims  # Scale
        l = 2**ms    # Number of points (width) of the averaging segments "nw" at a given scale "m".
        nw = np.int64((2**M)/l)  # Number of segments, each with "l" number of points.
        
        sumab = 0
        
        
        for i in range(1,nw+1):  #Loop through the different averaging segments "nw" 
            k = (i-1)*l
            za = a[k]
            zb = b[k]
        
            for j in range(k+1,k+l):  #Loop within the datapoints inside one specific [i] segment (ot of the total "nw").
                za = za + a[j]  #Cumulative sum of subsegment "i" in time series "a"
                zb = zb + b[j]  #Cumulative sum of subsegment "i" in time series "b"
            
            za = za/l
            zb = zb/l
            sumab = sumab + (za*zb)
                
            for j in range(k,i*l): #Subtract the mean from the time series to form the residual to be reused in next iteration. 
                tmpa = a[j] - za
                tmpb = b[j] - zb
                a[j] = tmpa
                b[j] = tmpb
            
        
        if nw>1: #Computing the MR spectra at a given scale[m]. For scale ms = M is the largest scale.
            D[ms] = (sumab/nw)

    var = np.var(u_short)
    MRDvar = np.sum(D)
    print(var,MRDvar)  
    
    return D,M

# %%
rawdir='/run/media/tswater/Elements/NEON/neon_raw/'
fp=h5py.File(rawdir+'UKFS/NEON.D06.UKFS.IP0.00200.001.ecte.2023-04-16.l0p.h5','r')
#rawdir='/run/media/tswater/Elements/NEON/neon_raw/'
#fp=h5py.File(rawdir+'ONAQ/NEON.D15.ONAQ.IP0.00200.001.ecte.2023-04-16.l0p.h5','r')

# %%
fp['UKFS/dp0p/data/soni/000_060/veloXaxs'][:]

# %%
fp['UKFS/dp0p/qfqm/soni/000_060'].keys()

# %%
data2=fp['UKFS/dp0p/data/soni/000_060/veloYaxs'][:]

# %%
plt.plot(data2)

# %%
for site in ['ONAQ','SOAP','NOGP','BONA','UKFS','MLBS']:
    fp=h5py.File('/home/tswater/tyche/data/neon/raw_streamwise/'+site+'/NEON_TW_2023-07.h5','r')
    u=fp['Wstr'][:]
    d_e=np.linspace(0,len(u),32)
    out=[]
    for i in range(31):
        out.append(np.sum(np.isnan(u[int(d_e[i]):int(d_e[i+1])])/(d_e[i+1]-d_e[i])))
    plt.plot(out)
    plt.ylim(-.01,.1)
plt.legend(['ONAQ','SOAP','NOGP','BONA','UKFS','MLBS'])

# %%
fp=h5py.File('/home/tswater/tyche/data/neon/raw_streamwise/SOAP/NEON_TW_2023-08.h5','r')
u=fp['Vstr'][:]
plt.plot(u)

# %%
#3-16
d_e=np.linspace(0,len(u),32)
out=[]
for i in range(31):
    out.append(np.sum(np.isnan(u[int(d_e[i]):int(d_e[i+1])])/(d_e[i+1]-d_e[i])))
plt.plot(out)
plt.ylim(-.01,.1)

# %%
E,M=MRD(u[int(d_e[13]):int(d_e[15])])

# %%
E.shape

# %%
float('nan')/float('nan')

# %%
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.patches as mpatches

Mx=0
dt = 1/20  #Frequency of measurements.
t = 2**(np.arange(Mx,M+1))*dt
f = 1/t


fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.semilogx(f[1:-1],E[1:-1],color='black',linewidth=8,alpha=0.4)
ax2.loglog(f[1:-1],E[1:-1],color='black',linewidth=8,alpha=0.4)

ax1.set(ylabel='$f\,S_{TKE}(f)$'); ax2.set(ylabel='$f\,S_{TKE}(f)$')
ax1.set(xlabel='$f\,\,[Hz]$'); ax2.set(xlabel='$f\,\,[Hz]$')

ax2.set_ylim((1e-5, 1e0))

ylim_ax1 = 1.2; ax1.set_ylim(0,ylim_ax1)
ymin = 1e-3
ylim_ax2 = 2; ax2.set_ylim(ymin,ylim_ax2)

day = 1/(24*3600)
twelveHr = 1/(12*3600)
sixhour = 1/(6*3600)
halfhour = 1/(1800)
fivemin = 1/(5*60)

ax2.plot([day, day], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
ax2.plot([twelveHr, twelveHr], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
ax2.plot([sixhour, sixhour], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
ax2.plot([halfhour, halfhour], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
ax2.plot([fivemin, fivemin], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)

angle = 90
ycorner = 1
l1 = np.array((1e-5, 1e-2))
th1 = ax2.text(l1[0], l1[1], '$T_p = 1 \,day$', fontsize=10,
           rotation=angle, rotation_mode='anchor')
l2 = np.array((2e-5, 1e-2))
th2 = ax2.text(l2[0], l2[1], '$T_p = 12\,h $', fontsize=10,
           rotation=angle, rotation_mode='anchor')
l3 = np.array((8e-5, 1e-2))
th2 = ax2.text(l3[0], l3[1], '$T_p = 6\,h $', fontsize=10,
           rotation=angle, rotation_mode='anchor')
l4 = np.array((4e-4, 1e-2))
th2 = ax2.text(l4[0], l4[1], '$T_p = 30\,min$', fontsize=10,
           rotation=angle, rotation_mode='anchor')
l5 = np.array((2e-3, 1e-2))
th2 = ax2.text(l5[0], l5[1], '$T_p = 5\,min$', fontsize=10,
           rotation=angle, rotation_mode='anchor')

plt.tight_layout()
plt.show()

# %%

# %%
# June Comparison


fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

clrs={'ONAQ':'khaki',
      'NOGP':'greenyellow',
      'UKFS':'yellowgreen',
      'MLBS':'forestgreen',
      'BONA':'mediumaquamarine'}
domain={'ONAQ':'D15',
        'NOGP':'D09',
        'UKFS':'D06',
        'MLBS':'D07',
        'BONA':'D19'}
lvl={'ONAQ':'040',
     'UKFS':'060',
     'NOGP':'040',
     'MLBS':'060',
     'BONA':'050'}

for site in ['ONAQ','NOGP','UKFS','MLBS','BONA']:
    print(site)
    fp=h5py.File('/home/tswater/tyche/data/neon/raw_streamwise/'+site+'/NEON_TW_2023-07.h5','r')
    u=fp['Wstr'][:]
    d_e=np.linspace(0,len(u),32)
    E,M=MRD(u[int(d_e[15]):int(d_e[17])])

    Mx=0
    dt = 1/20  #Frequency of measurements.
    t = 2**(np.arange(Mx,M+1))*dt
    f = 1/t
    
    ax1.semilogx(f[1:-1],E[1:-1],color=clrs[site],linewidth=2,alpha=0.75)
    ax2.loglog(f[1:-1],E[1:-1],color=clrs[site],linewidth=2,alpha=0.75)

ax1.set(ylabel='$f\,S_{TKE}(f)$'); ax2.set(ylabel='$f\,S_{TKE}(f)$')
ax1.set(xlabel='$f\,\,[Hz]$'); ax2.set(xlabel='$f\,\,[Hz]$')

ax2.set_ylim((1e-5, 1e0))

ylim_ax1 = .2; ax1.set_ylim(0,ylim_ax1)
ymin = 1e-4
ylim_ax2 = .2; ax2.set_ylim(ymin,ylim_ax2)

day = 1/(24*3600)
twelveHr = 1/(12*3600)
sixhour = 1/(6*3600)
halfhour = 1/(1800)
fivemin = 1/(5*60)

ax2.plot([day, day], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
ax2.plot([twelveHr, twelveHr], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
ax2.plot([sixhour, sixhour], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
ax2.plot([halfhour, halfhour], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
ax2.plot([fivemin, fivemin], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)

angle = 90
ycorner = 1
l1 = np.array((1e-5, 1e-2))
th1 = ax2.text(l1[0], l1[1], '$T_p = 1 \,day$', fontsize=10,
           rotation=angle, rotation_mode='anchor')
l2 = np.array((2e-5, 1e-2))
th2 = ax2.text(l2[0], l2[1], '$T_p = 12\,h $', fontsize=10,
           rotation=angle, rotation_mode='anchor')
l3 = np.array((8e-5, 1e-2))
th2 = ax2.text(l3[0], l3[1], '$T_p = 6\,h $', fontsize=10,
           rotation=angle, rotation_mode='anchor')
l4 = np.array((4e-4, 1e-2))
th2 = ax2.text(l4[0], l4[1], '$T_p = 30\,min$', fontsize=10,
           rotation=angle, rotation_mode='anchor')
l5 = np.array((2e-3, 1e-2))
th2 = ax2.text(l5[0], l5[1], '$T_p = 5\,min$', fontsize=10,
           rotation=angle, rotation_mode='anchor')

plt.tight_layout()
plt.show()

# %%
# June Comparison


fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

clrs={'ONAQ':'khaki',
      'NOGP':'greenyellow',
      'UKFS':'yellowgreen',
      'MLBS':'forestgreen',
      'BONA':'mediumaquamarine'}
domain={'ONAQ':'D15',
        'NOGP':'D09',
        'UKFS':'D06',
        'MLBS':'D07',
        'BONA':'D19'}
lvl={'ONAQ':'040',
     'UKFS':'060',
     'NOGP':'040',
     'MLBS':'060',
     'BONA':'050'}

for site in ['ONAQ','NOGP','UKFS','MLBS','BONA']:
    print(site)
    fp=h5py.File('/home/tswater/tyche/data/neon/raw_streamwise/'+site+'/NEON_TW_2023-07.h5','r')
    u=fp['Vstr'][:]
    d_e=np.linspace(0,len(u),32)
    E,M=MRD(u[int(d_e[15]):int(d_e[17])])

    Mx=0
    dt = 1/20  #Frequency of measurements.
    t = 2**(np.arange(Mx,M+1))*dt
    f = 1/t
    
    ax1.semilogx(f[1:-1],E[1:-1],color=clrs[site],linewidth=2,alpha=0.75)
    ax2.loglog(f[1:-1],E[1:-1],color=clrs[site],linewidth=2,alpha=0.75)

ax1.set(ylabel='$f\,S_{TKE}(f)$'); ax2.set(ylabel='$f\,S_{TKE}(f)$')
ax1.set(xlabel='$f\,\,[Hz]$'); ax2.set(xlabel='$f\,\,[Hz]$')

ax2.set_ylim((1e-5, 1e0))

ylim_ax1 = 1.2; ax1.set_ylim(0,ylim_ax1)
ymin = 1e-3
ylim_ax2 = 2; ax2.set_ylim(ymin,ylim_ax2)

day = 1/(24*3600)
twelveHr = 1/(12*3600)
sixhour = 1/(6*3600)
halfhour = 1/(1800)
fivemin = 1/(5*60)

ax2.plot([day, day], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
ax2.plot([twelveHr, twelveHr], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
ax2.plot([sixhour, sixhour], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
ax2.plot([halfhour, halfhour], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
ax2.plot([fivemin, fivemin], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)

angle = 90
ycorner = 1
l1 = np.array((1e-5, 1e-2))
th1 = ax2.text(l1[0], l1[1], '$T_p = 1 \,day$', fontsize=10,
           rotation=angle, rotation_mode='anchor')
l2 = np.array((2e-5, 1e-2))
th2 = ax2.text(l2[0], l2[1], '$T_p = 12\,h $', fontsize=10,
           rotation=angle, rotation_mode='anchor')
l3 = np.array((8e-5, 1e-2))
th2 = ax2.text(l3[0], l3[1], '$T_p = 6\,h $', fontsize=10,
           rotation=angle, rotation_mode='anchor')
l4 = np.array((4e-4, 1e-2))
th2 = ax2.text(l4[0], l4[1], '$T_p = 30\,min$', fontsize=10,
           rotation=angle, rotation_mode='anchor')
l5 = np.array((2e-3, 1e-2))
th2 = ax2.text(l5[0], l5[1], '$T_p = 5\,min$', fontsize=10,
           rotation=angle, rotation_mode='anchor')

plt.tight_layout()
plt.show()

# %%
# June Comparison


fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

clrs={'ONAQ':'khaki',
      'NOGP':'greenyellow',
      'UKFS':'yellowgreen',
      'MLBS':'forestgreen',
      'BONA':'mediumaquamarine'}
domain={'ONAQ':'D15',
        'NOGP':'D09',
        'UKFS':'D06',
        'MLBS':'D07',
        'BONA':'D19'}
lvl={'ONAQ':'040',
     'UKFS':'060',
     'NOGP':'040',
     'MLBS':'060',
     'BONA':'050'}

for site in ['ONAQ','NOGP','UKFS','MLBS','BONA']:
    print(site)
    fp=h5py.File('/home/tswater/tyche/data/neon/raw_streamwise/'+site+'/NEON_TW_2023-07.h5','r')
    u=fp['Ustr'][:]
    d_e=np.linspace(0,len(u),32)
    E,M=MRD(u[int(d_e[15]):int(d_e[17])])

    Mx=0
    dt = 1/20  #Frequency of measurements.
    t = 2**(np.arange(Mx,M+1))*dt
    f = 1/t
    
    ax1.semilogx(f[1:-1],E[1:-1],color=clrs[site],linewidth=2,alpha=0.75)
    ax2.loglog(f[1:-1],E[1:-1],color=clrs[site],linewidth=2,alpha=0.75)

ax1.set(ylabel='$f\,S_{TKE}(f)$'); ax2.set(ylabel='$f\,S_{TKE}(f)$')
ax1.set(xlabel='$f\,\,[Hz]$'); ax2.set(xlabel='$f\,\,[Hz]$')

ax2.set_ylim((1e-5, 1e0))

ylim_ax1 = .2; ax1.set_ylim(0,ylim_ax1)
ymin = 1e-4
ylim_ax2 = .2; ax2.set_ylim(ymin,ylim_ax2)

day = 1/(24*3600)
twelveHr = 1/(12*3600)
sixhour = 1/(6*3600)
halfhour = 1/(1800)
fivemin = 1/(5*60)

ax2.plot([day, day], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
ax2.plot([twelveHr, twelveHr], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
ax2.plot([sixhour, sixhour], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
ax2.plot([halfhour, halfhour], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)
ax2.plot([fivemin, fivemin], [ymin, ylim_ax2],linestyle=':',color='k',linewidth=1)

angle = 90
ycorner = 1
l1 = np.array((1e-5, 1e-2))
th1 = ax2.text(l1[0], l1[1], '$T_p = 1 \,day$', fontsize=10,
           rotation=angle, rotation_mode='anchor')
l2 = np.array((2e-5, 1e-2))
th2 = ax2.text(l2[0], l2[1], '$T_p = 12\,h $', fontsize=10,
           rotation=angle, rotation_mode='anchor')
l3 = np.array((8e-5, 1e-2))
th2 = ax2.text(l3[0], l3[1], '$T_p = 6\,h $', fontsize=10,
           rotation=angle, rotation_mode='anchor')
l4 = np.array((4e-4, 1e-2))
th2 = ax2.text(l4[0], l4[1], '$T_p = 30\,min$', fontsize=10,
           rotation=angle, rotation_mode='anchor')
l5 = np.array((2e-3, 1e-2))
th2 = ax2.text(l5[0], l5[1], '$T_p = 5\,min$', fontsize=10,
           rotation=angle, rotation_mode='anchor')

plt.tight_layout()
plt.show()

# %% [markdown]
# # MRD from Python

# %%
mrd_dir='/home/tswater/tyche/data/neon/mrd_30/'

# %%
onaq_mrd=pickle.load(open(mrd_dir+'ONAQ_mrd_v2.p','rb'))
nogp_mrd=pickle.load(open(mrd_dir+'NOGP_mrd_v2.p','rb'))
mlbs_mrd=pickle.load(open(mrd_dir+'MLBS_mrd_v2.p','rb'))
ukfs_mrd=pickle.load(open(mrd_dir+'UKFS_mrd_v2.p','rb'))
soap_mrd=pickle.load(open(mrd_dir+'SOAP_mrd_v2.p','rb'))
bona_mrd=pickle.load(open(mrd_dir+'BONA_mrd_v2.p','rb'))

# %%
soap_mrd['unstable']['Mu']

# %%

# %%
mrd=soap_mrd
Mx=0
dt = 1/20  #Frequency of measurements.
for i in range(len(mrd['unstable']['Mu'][:])):
    t = 2**(np.arange(Mx,mrd['unstable']['Mu'][i]+1))*dt
    f = t
    plt.loglog(f[1:-1],mrd['unstable']['Du'][i][1:-1],alpha=.5)

day = 1/(24*3600)
twelveHr = 1/(12*3600)
sixhour = 1/(6*3600)
halfhour = 1/(1800)
fivemin = 1/(5*60)

ymin=1e-4
ylim_ax2=.2

# %%
mx_t=[]
for i in range(len(mrd['stable']['Mu'][:])):
    D=mrd['stable']['Dv'][i][1:-1]
    idx=np.where(D==np.max(D))[0]
    t = 2**(np.arange(Mx,mrd['stable']['Mv'][i]+1))*dt
    f = 1/t
    mx_t.append(f[1:-1][idx])

# %%
1/np.mean(mx_t)

# %%
60*30

# %%
Mx=0
dt = 1/20
mrd=soap_mrd
Dm=np.zeros((22,))
count=np.zeros((22,))
for i in range(len(mrd['unstable']['Mu'][:])):
    Di=mrd['unstable']['Du'][i]
    n=len(Di)
    Dm[0:n]=Dm[0:n]+Di[:]
    count[0:n]=count[0:n]+1

Dm=Dm/count
t = 2**(np.arange(Mx,21+1))*dt
f = t
plt.loglog(f[1:-1],Dm[1:-1]/np.nanmax(Dm),alpha=.5,c='blue')

Dm=np.zeros((22,))
count=np.zeros((22,))
for i in range(len(mrd['stable']['Mu'][:])):
    Di=mrd['stable']['Du'][i]
    n=len(Di)
    Dm[0:n]=Dm[0:n]+Di[:]
    count[0:n]=count[0:n]+1

Dm=Dm/count
t = 2**(np.arange(Mx,21+1))*dt
f = t
plt.loglog(f[1:-1],Dm[1:-1]/np.nanmax(Dm),alpha=.5,c='blue',linestyle='--')

mrd=nogp_mrd
Dm=np.zeros((22,))
count=np.zeros((22,))
for i in range(len(mrd['unstable']['Mu'][:])):
    Di=mrd['unstable']['Du'][i]
    n=len(Di)
    Dm[0:n]=Dm[0:n]+Di[:]
    count[0:n]=count[0:n]+1

Dm=Dm/count
t = 2**(np.arange(Mx,21+1))*dt
f = t
plt.loglog(f[1:-1],Dm[1:-1]/np.nanmax(Dm),alpha=.5,c='red')

Dm=np.zeros((22,))
count=np.zeros((22,))
for i in range(len(mrd['stable']['Mu'][:])):
    Di=mrd['stable']['Du'][i]
    n=len(Di)
    Dm[0:n]=Dm[0:n]+Di[:]
    count[0:n]=count[0:n]+1

Dm=Dm/count
t = 2**(np.arange(Mx,21+1))*dt
f = t
plt.loglog(f[1:-1],Dm[1:-1]/np.nanmax(Dm),alpha=.5,c='red',linestyle='--')

plt.ylim(10e-3,1.1)
plt.legend(['unstable','stable','unstable','stable'])
plt.xlabel('seconds')

# %%
t[-1]

# %%
500/60

# %%

# %%

# %% [markdown]
# # TEST Different Scale Aniso

# %%

# %%
site='UKFS'
idir='/home/tswater/tyche/data/neon/mrd_30/'
data=pickle.load(open(idir+site+'_aniscale.p','rb'))
cc=['tan','rosybrown','lightcoral','indianred','brown','green','maroon','darkred','black']
i=0
for s in [1,2,5,10,15,30,60,120,180]:
    yb=data['yb'+str(s)][:]
    yb2=[]
    for j in range(len(yb)):
        if np.isnan(yb[j]):
            pass
        else:
            yb2.append(yb[j])
    y,binEdges=np.histogram(yb2,bins=50,density=True)
    bincenters=.5*(binEdges[1:]+binEdges[:-1])
    plt.plot(bincenters,y,c=cc[i],linewidth=2,alpha=.75)
    i=i+1
plt.legend([1,2,5,10,15,30,60,120,180])
plt.figure()
plt.hist(np.array(data['yb30'][:])-np.array(data['yb2'][:]),bins=np.linspace(-.4,.4))

# %%
site='SOAP'
idir='/home/tswater/tyche/data/neon/mrd_30/'
data=pickle.load(open(idir+site+'_aniscale.p','rb'))
#cc=['rosybrown','lightcoral','indianred','brown','firebrick','maroon','darkred']
i=0
for s in [1,2,5,10,15,30,60,120,180]:
    yb=data['xb'+str(s)][:]
    yb2=[]
    for j in range(len(yb)):
        if np.isnan(yb[j]):
            pass
        else:
            yb2.append(yb[j])
    y,binEdges=np.histogram(yb2,bins=50,density=True)
    bincenters=.5*(binEdges[1:]+binEdges[:-1])
    plt.plot(bincenters,y,c=cc[i],linewidth=2,alpha=.75)
    i=i+1
plt.legend([1,2,5,10,15,30])
plt.figure()
plt.hist(np.array(data['yb30'][:])-np.array(data['yb2'][:]),bins=np.linspace(-.4,.4))

# %%
for site in ['NOGP','ONAQ','BONA','UKFS','MLBS','SOAP']:
    data=pickle.load(open(idir+site+'_aniscale.p','rb'))
    sd=[]
    for s in [1,2,5,10,15,30,60,120,180]:
        sd.append(np.nanmedian(data['xb'+str(s)]))
    plt.plot([1,2,5,10,15,30,60,120,180],sd,'-o')
plt.legend(['NOGP','ONAQ','BONA','UKFS','MLBS','SOAP'],loc='upper right')
plt.xlabel('Averaging Time')
plt.ylabel('Median $y_b$')

# %%

# %%

# %%

# %%
