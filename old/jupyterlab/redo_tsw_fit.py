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
import inspect
from scipy import optimize
from numpy.polynomial import polynomial as P

mpl.rcParams['figure.dpi'] = 100
sns.set_theme()
plt.rcParams.update({'figure.max_open_warning': 0})


# %%
def binit(ani,binsize=float('nan'),n=100,vmx_a=.7,vmn_a=.1):
    mmm=(ani>=vmn_a)&(ani<=vmx_a)
    N=np.sum(mmm)
    if np.isnan(binsize):
        pass
    else:
        n=np.floor(N/binsize)
    anibins=np.zeros((n+1,))
    for i in range(n+1):
        anibins[i]=np.nanpercentile(ani[mmm],i/n*100)
    return anibins,mmm


# %%
fpu=h5py.File('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data_v2/NEON_TW_U_UVWT.h5','r')
fps=h5py.File('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data_v2/NEON_TW_S_UVWT.h5','r')

# %%
phiu_=np.sqrt(fps['UU'][:])/fps['USTAR'][:]
phiv_=np.sqrt(fps['VV'][:])/fps['USTAR'][:]
phiw_=np.sqrt(fps['WW'][:])/fps['USTAR'][:]
zL_=fps['zzd'][:]/fps['L_MOST'][:]
ani_=fps['ANI_YB'][:]
anibins,m=binit(ani_,vmn_a=.1,vmx_a=.7)


# %%
#phi_=np.sqrt(fpu['WW'][:])/fpu['USTAR'][:]
#zL_=fpu['zzd'][:]/fpu['L_MOST'][:]
#ani_=fpu['ANI_YB'][:]
#anibins,m=binit(ani_,vmn_a=.1,vmx_a=.7)

# %%
def U_ust(zL,a):
    return a*(1-3*zL)**(1/3)
def U_ust2(zL,a):
    return a*(1-3*zL)**(1/3)
def U_ust3(zL,a,b):
    return a*(1-3*zL)**(b)
def U_stb(zL,a,b):
    return a*(1+3*zL)**(b)
def U_stb2(zL,a):
    return U_stb(zL,a,.06)
def U_stb3(zL,a,b,c):
    return a*(1+b*zL)**(c)
def U_stb4(zL,a,b):
    return a*(1+b*zL)**(.6)
fxns_={'Uu':U_ust,'Uu2':U_ust2,'Uu3':U_ust3,'Us':U_stb,'Us2':U_stb2,'Us3':U_stb3,'Us4':U_stb4}
bounds={'Uu':([0],[10]),'Uu2':([.5],[5]),'Uu3':([.5,0.1],[5,.6]),'Us3':([1,.1,0],[5,5,1/3]),'Us':([0,0],[5,1]),'Us2':([0],[5]),'Us4':([0,0],[5,300])}
p0s={'Uu':[2.5],'Uu2':[2.5],'Uu3':[2.5,0.4],'Us':[2,.08],'Us2':[2],'Us3':[2,1,.06],'Us4':[2,3]}


# %%
.4**(.1)

# %%

# %%
var='Us4'
phi_=phiw_
Np=len(inspect.signature(fxns_[var]).parameters)-1
params=np.ones((len(anibins)-1,Np))*float('nan')
p_vars=np.ones((len(anibins)-1,Np))*float('nan')
for i in range(len(anibins)-1):
    print('.',end='',flush=True)
    m=(ani_>anibins[i])&(ani_<anibins[i+1])&(~np.isnan(phi_))
    try:
        if var in bounds.keys():
            params[i,:],pcov=optimize.curve_fit(fxns_[var],zL_[m],phi_[m],p0s[var],bounds=bounds[var], loss='cauchy')
        else:
            params[i,:],pcov=optimize.curve_fit(fxns_[var],zL_[m],phi_[m],p0s[var])
        for p in range(Np):
            p_vars[i,p]=pcov[p,p]
    except Exception as e:
        print(e)


# %%
anic=(anibins[0:-1]+anibins[1:])/2

# %%
letters=['a','b','c','d']
degs={'Uu':[4],'Uu2':[3],'Uu3':[1,2],'Vu':[2],'Wu':[3],
      'Us':[2,2],'Vs':[2,3],'Ws':[2,3],'Us2':[2],'Us3':[2,2,3],'Us4':[3,3]}
lg={'Uu':[False],'Uu2':[True],'Uu3':[True,False],
    'Us':[False,False],'Us2':[False],'Us3':[False, False, False],'Us4':[False,True]}
maxs={}
mins={}
for var in [var]:
    param=params
    anibins=anibins
    anilvls=(anibins[1:]+anibins[0:-1])/2
    pall=np.nanmean(param,axis=0)
    m=~np.isnan(param[:,0])
    if var in maxs.keys():
        for mx in range(len(maxs[var])):
            m=m&(param[:,mx]<maxs[var][mx])
    absz=anibins[1:]-anibins[0:-1]
    absz=absz[m]
    plt.figure(figsize=(4,3*len(pall)),dpi=100)
    print("'"+str(var)+"':{",end='')
    out=[]
    for i in range(len(pall)):
        if lg[var][i]:
            x=np.log10(anilvls[m])
        else:
            x=anilvls[m]
        y=param[:,i][m]
        c = P.polyfit(x,y,degs[var][i])
        out.append(c)
        plt.subplot(len(pall),1,i+1)
        plt.scatter(x,param[:,i][m])
        if i>0:
            print(',',end='')
        print("'"+letters[i]+"':[",end='')
        for j in range(len(c)):
            print(str(c[j]),end='')
            if j<(len(c)-1):
                print(',',end='')
            else:
                print(']',end='')
        y2=c[0]+c[1]*x
        if degs[var][i]>1:
            y2=y2+c[2]*x**2
        if degs[var][i]>2:
            y2=y2+c[3]*x**3
        if degs[var][i]>3:
            y2=y2+c[4]*x**4
        plt.plot(x,y2,'k--')
        #plt.ylim(np.nanpercentile(param[:,i],1)*.9,np.nanpercentile(param[:,i],99))
        plt.title(var)
    print('},')

# %%
out

# %%
phi_oldwu=1.35*(1-3*zL_)**(1/3)
phi_oldus=1.6

phi_old=phi_oldus
#phi_old=2.06
#a=out[0][0]+out[0][1]*np.log10(ani_[:])
#a=out[0][0]+out[0][1]*np.log10(ani_[:])+out[0][2]*np.log10(ani_[:])**2
#a=out[0][0]+out[0][1]*np.log10(ani_[:])+out[0][2]*np.log10(ani_[:])**2+out[0][3]*np.log10(ani_[:])**3
#a=.725-2.702*np.log10(ani_[:])
#a=out[0][0]+out[0][1]*(ani_[:])+out[0][2]*ani_[:]**2
#a=2.332-2.781*(ani_[:])+2.672*ani_[:]**2
a=out[0][0]+out[0][1]*(ani_[:])+out[0][2]*ani_[:]**2+out[0][3]*ani_[:]**3
b=out[1][0]+out[1][1]*np.log10(ani_[:])+out[1][2]*np.log10(ani_[:])**2+out[1][3]*np.log10(ani_[:])**3
#b=out[1][0]+out[1][1]*(ani_[:])+out[1][2]*ani_[:]**2
#a=.953+.188*(ani_[:])+2.253*ani_[:]**2
#b=.208-1.935*(ani_[:])+6.183*ani_[:]**2-7.485*ani_[:]**3+3.077*ani_[:]**4
#a=out[0][0]+out[0][1]*(ani_[:])#+out[0][2]*ani_[:]**2#+out[0][3]*ani_[:]**3
#b=out[1][0]+out[1][1]*np.log10(ani_[:])+out[1][2]*np.log10(ani_[:])**2+out[1][3]*np.log10(ani_[:])**3#+out[1][3]*ani_[:]**3
#b=out[1][0]+out[1][1]*(ani_[:])+out[1][2]*ani_[:]**2#+out[1][3]*ani_[:]**3
#c=out[2][0]+out[2][1]*(ani_[:])+out[2][2]*ani_[:]**2+out[2][3]*ani_[:]**3
#a=1.119-.019*(ani_[:])-.065*ani_[:]**2+.028*ani_[:]**3
#b=out[1][0]+out[1][1]*(ani_[:])+out[1][2]*ani_[:]**2
#b=0.255-1.76*(ani_[:])+5.6*ani_[:]**2-6.8*ani_[:]**3+2.65*ani_[:]**4
#b=out[1][0]+out[1][1]*np.log10(ani_[:])+out[1][2]*np.log10(ani_[:])**2
phi_new=fxns_[var](zL_,a,b)
#m=(zL_>0)
madn=np.nanmedian(np.abs(phi_-phi_new))
mado=np.nanmedian(np.abs(phi_-phi_old))
SS=1-(madn/mado)
print(SS)
SSs=[]
for site in np.unique(fpu['SITE'][:]):
    m=fps['SITE'][:]==site
    madn=np.nanmedian(np.abs(phi_[m]-phi_new[m]))
    mado=np.nanmedian(np.abs(phi_[m]-phi_old))
    SS=1-(madn/mado)
    SSs.append(SS)
plt.boxplot(SSs)

# %%
print(np.sum(np.array(SSs)<0))

# %%
# Us (deg 2,2): (old: -.07 ) (new: -0.0018) (-.075,-0.025,.025,.05) (19<0)
'Us':{'a':[2.48999325402708,-2.9309894919097794,3.9665235021601855],'b':[-0.014091947587046363,0.4413024489123778,-0.4734026436270551]},
# Us (deg 2): (old: -.07) (new: -.003) (-.07,-.02,.02,.05) (20<0)
'Us':{'a':[2.402924304108439,-2.4723645199047186,3.609442171163428]},
# Us (deg 2,1) log, b is b, c=.06. (old -.07) (new: .07) (-.07,-.015,.03,.08) (14<0)
'Us':{'a':[2.5040234841099136,-2.9135680594318836,3.879724240951841],'b':[9.633916512522173,11.208076650244074]},

# Vs (deg 1,3) log, b is b, c=.06. (old .001) (new: .253) (.1,.2,.275,.35) (0<0)
'Vs':{'a':[1.1215876915839131,1.6584671522267522],'b':[-113.6176737221579,-938.9228856874983,-2312.8838627003097,-1859.8624983507027]},
# Vs (deg 2,2): (old: .001) (new:.2586 ) (.125,.2,.29,.37) (0<0) old (10<0
'Vs':{'a':[1.3330870632847518,0.5841790794590472,1.3070551206541443],'b':[0.2225823294120512,-0.42859347002903014,0.32840958194612313]},

# Ws (deg 2,2): old:.615 (new: .5824)
'Ws':{'a':[0.8568270381820401,0.38500921564736157,2.399008378516547],'b':[0.09569689610086621,-0.004254222387943103,-0.03760808711271964]},

# %%
# Wu log (deg 2): (old: .5720) (new: .577) (.4,.52,.64,.74) (0<0) old: 
'Wu':{'a':[0.8781963394412855,-0.7903441230331336,-0.6884177296285137]},
# Wu non-log (deg 3): (old: .5720) (new: .5722) (.4,.52,.64,.74) (0<0) old: 
'Wu':{'a':[0.8322249105998253,2.2982074088860984,-5.924591113145409,4.359928264469673]},

# %%
# Uu log: (old: .4124) (new: .4215) (-.15,.2,.56,.7) (3<0) old: 3<0
'Uu':{'a':[0.5003755881285298,-2.959804623542144]},
# Vu log: (old: .1129) (new: .2913) (-.01,.18,.35,.5) (3<0) (old: 16<0)
'Vu':{'a':[0.3972669466422134,-2.72146993605693]},

# %%

# %%
x=np.linspace(0,.8)
plt.plot(x,4-9*x+8*x**2)


# %%
def get_phis(var,fp):
    


# %%
x=np.logspace(-3,1)
plt.semilogx(x,2*(1+1*x)**(.1))
plt.semilogx(x,x**(0)*2.8)

# %%

# %%
