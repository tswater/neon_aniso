# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
from neonutil.nutil import nscale,get_phi,get_phio
from neonutil.curvefit import Cfit,load_fit,compare_fits
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import h5py
from scipy.optimize import curve_fit
mpl.rcParams['figure.dpi'] = 100
sns.set_theme()
plt.rcParams.update({'figure.max_open_warning': 0})

# %%
fp=h5py.File('/home/tswater/tyche/data/neon/L2/L2_revision/l2test.h5','r+')

# %%
zL=fp['main/data']['zL'][:]
ani=fp['main/data']['ANI_YB'][:]
phi=get_phi(fp['main/data'],'UU')

# %% [markdown]
# ## Testing Cfit 2 Stage

# %%
p0=[2.55]
params=['a0','a1']
const={'a':None,'b':1,'c':-3,'d':1/3,'e':0}
bnds=([.1],[10])
loss='cauchy'
cf=Cfit('test','UU',False,'2_stage','cbase',p0=p0,params=params,const=const,bnds=bnds,loss=loss,deg=[2],typ='log')

# %%
cf.cfit(zL,ani,phi)

# %%
cf.eval_cfit(zL,ani,phi,fp['main/data']['SITE'][:])

# %% [markdown]
# ## Testing Cfit Full

# %%
# a*(b+c*zL)**(d)+e
p0=[2.55,0]
params=['a0','a1']
const={'a':None,'b':1,'c':-3,'d':1/3,'e':0}
bnds=([.1,-5],[10,5])
loss='cauchy'
cf=Cfit('test','UU',False,'full','cbase',p0=p0,params=params,const=const,bnds=bnds,loss=loss)

# %%
cf.cfit(zL,ani,phi)

# %%
cf.eval_cfit(zL,ani,phi,fp['main/data']['SITE'][:])

# %%
#cf.save_fit(fp['main'])

# %%
cf2=load_fit(fp,'main','test','UU',False)

# %%
cf2.stats

# %%
compare_fits([cf,cf2])

# %%
cf2.cfit(zL,ani,phi)
cf2.eval_cfit(zL,ani,phi,fp['main/data']['SITE'][:])

# %%
