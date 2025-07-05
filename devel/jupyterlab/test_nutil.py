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
from neonutil.nutil import nscale
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100
sns.set_theme()
plt.rcParams.update({'figure.max_open_warning': 0})

# %% [markdown]
# # Test NSCALE

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### TIN is COARSER (ninterp)

# %%
tout=np.linspace(0,1000,101)

# %%
# test basic tin at coarser resolution
tin=np.linspace(0,1000,21)
din=np.sin(tin/(10*np.pi))
plt.figure(figsize=(12,3))
plt.plot(tin,din,'-o')
plt.plot(tout,nscale(tout,tin,din),"-o",markersize=4)

# %%
# test tin at coarser resolution with a gap
tin=np.array([ 0.,   50.,  100.,  150.,  200.,  250.,  300.,  350.,  400, 600.,  650.,  700.,900.,  950., 1000])
din=np.sin(tin/(10*np.pi))
plt.figure(figsize=(12,3))
plt.plot(tin,din,'-o')
plt.plot(tout,nscale(tout,tin,din,maxdelta=3),"-o",markersize=4)

# %%
# test tin at coarser resolution with a gap using linear 
tin=np.array([ 0.,   50.,  100.,  150.,  200.,  250.,  300.,  350.,  400, 600.,  650.,  700.,900.,  950., 1000])
din=np.sin(tin/(10*np.pi))
plt.figure(figsize=(12,3))
plt.plot(tin,din,'-o')
plt.plot(tout,nscale(tout,tin,din,maxdelta=3,nearest=False),"-o",markersize=4)

# %%
# test tin at coarser resolution with tin resolution is NOT a multiple of tout resolution
tin=np.linspace(0,1000,41)
din=np.sin(tin/(10*np.pi))
plt.figure(figsize=(12,3))
plt.plot(tin,din,'-o')
plt.plot(tout,nscale(tout,tin,din,maxdelta=3),"-o",markersize=4)

# %%
# tin covers wider extent than tout
# test tin at coarser resolution with tin resolution is NOT a multiple of tout resolution
tin=np.linspace(-125,1125,51)
din=np.sin(tin/(10*np.pi))
plt.figure(figsize=(12,3))
plt.plot(tin,din,'-o')
plt.plot(tout,nscale(tout,tin,din,maxdelta=3),"-o",markersize=4)

# %%
# tin covers smaller extent than tout
# test tin at coarser resolution with tin resolution is NOT a multiple of tout resolution
tin=np.linspace(125,875,27)
din=np.sin(tin/(10*np.pi))
plt.figure(figsize=(12,3))
plt.plot(tin,din,'-o')
plt.plot(tout,nscale(tout,tin,din,maxdelta=3),"-o",markersize=4)

# %%
# tin has variable resolution
# test tin at coarser resolution with tin resolution is NOT a multiple of tout resolution
tin=np.linspace(0,1000,26)
tin=tin+17*np.sin(tin/(6*np.pi))
din=np.sin(tin/(10*np.pi))
plt.figure(figsize=(12,3))
plt.plot(tin,din,'-o')
plt.plot(tout,nscale(tout,tin,din,maxdelta=3),"-o",markersize=4)

# %%
# tin has variable resolution and does not cover the full timeseries and is linear
# test tin at coarser resolution with tin resolution is NOT a multiple of tout resolution
tin=np.linspace(80,920,21)
tin=tin+17*np.sin(tin/(6*np.pi))
din=np.sin(tin/(10*np.pi))
plt.figure(figsize=(12,3))
plt.plot(tin,din,'-o')
plt.plot(tout,nscale(tout,tin,din,maxdelta=3,nearest=False),"-o",markersize=4)

# %% [markdown]
# ### TIN is FINER (nupscale)

# %%
tout=np.linspace(0,36000,21) # 60 minutes

# %%
# tin is shorter than tout i
tin=np.linspace(0,15000,51)
din=np.sin(tin/(.05*np.pi))
plt.figure(figsize=(12,3))
plt.plot(tin,din,'-o')
plt.plot(tout,nscale(tout,tin,din),"-o",markersize=4)

# %%
# tin is longer than tout and exactly twice as small (30 min)
tin=np.linspace(-7200,36000+7200,29)+15*60
din=np.sin(tin/(.1*np.pi))
plt.figure(figsize=(12,3))
plt.plot(tin,din,'-o')
out=nscale(tout,tin,din,debug=True)
plt.plot(tout,out,"-o",markersize=4)

# %%
tout[1]-tout[0]

# %%
# tin is longer than tout and exactly twice as small (30 min) with gaps
tin=np.array([-6300., -4500., -2700.,  -900.,   900.,  2700.,  4500.,  6300.,
        8100.,  9900., 11700., 20700.,
       22500., 24300.,31500., 33300., 35100.,
       36900., 38700., 40500., 44100.])
din=np.sin(tin/(.1*np.pi))
plt.figure(figsize=(12,3))
plt.plot(tin,din,'-o')
plt.plot(tout,nscale(tout,tin,din,nearest=True),"-o",markersize=4)

# %%

# %%

# %%
# nscale but with different scale and dlt
tout=np.linspace(0,18000,6) # 60 minutes
tin=np.linspace(-7200,18000+7200,21)+15*75
din=tin/np.max(tin)#np.sin(tin/(.1*np.pi))+tin/np.max(tin)
plt.figure(figsize=(12,3))
plt.plot(tin,din,'-o')
plt.plot(tout,nscale(tout,tin,din,scl=120),"-o",markersize=4)
plt.ylim(0,.2)
plt.xlim(0,5000)

# %%
(tout[1]-tout[0])/60

# %%
scl=5
dlt=3
a=np.linspace(0,15,6)+.5
t=np.linspace(a[0]-scl/2,a[-1]+scl/2,(len(a)-1)*dlt+scl+1)

# %%
out=np.zeros(a.shape)
for i in range(scl):
    if i == (scl-1):
        jf=None
    else:
        jf=i-scl
    out=out+t[i:jf:dlt]

# %%
t

# %%
t[0:-scl:dlt]

# %%
a[-1]+scl/2-(a[0]-scl/2)

# %%
(len(a)+1)*dlt+scl-1

# %%
a

# %%
outscl=120
dlt=60
tmid=np.linspace(tout[0]-outscl*30,tout[-1]+outscl*30,dlt*(len(tout)-1)+outscl+1)


# %%
tmid-np.min(tmid)

# %%
