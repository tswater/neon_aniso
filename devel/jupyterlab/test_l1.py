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
import h5py
mpl.rcParams['figure.dpi'] = 100
sns.set_theme()
plt.rcParams.update({'figure.max_open_warning': 0})

# %% [markdown]
# # Test 15 Minute Files

# %% [markdown]
# ### Basic

# %%
fa15=h5py.File('/home/tswater/Documents/tyche/data/neon/L1/neon_15m/ABBY_15m.h5','r')
fg15=h5py.File('/home/tswater/Documents/tyche/data/neon/L1/neon_15m/GUAN_15m.h5','r')

# %%
# Check for Consistent Variable Length
for f in [fa15,fg15]:
    print('SITE TEST')
    n=len(f['TIME'][:])
    for var in f.keys():
        if ('profile' in var) and ('qprofile' not in var):
            pass
        elif len(f[var][:]) != n:
            print(var+' issue; length '+str(len(f[var][:]))+' but time length '+str(n))


# %% [markdown]
# ### Comparisons with 30m

# %%
fa30=h5py.File('/home/tswater/Documents/tyche/data/neon/L1/neon_30m/ABBY_30m.h5','r')
fg30=h5py.File('/home/tswater/Documents/tyche/data/neon/L1/neon_30m/GUAN_30m.h5','r')
abby=[fa15,fa30]
guan=[fg15,fg30]


# %%
# check nan frequency similarity
def getnan(data):
    return np.sum(np.isnan(data)|(data==-9999))/len(data)

convert={'THETA':'T_SONIC','Q':'H2O','C':'CO2'}
i=0
for pair in [abby,guan]:
    print(['ABBY','GUAN'][i])
    i=i+1
    f15=pair[0]
    f30=pair[1]
    for var in f15.keys():
        if 'q' in var:
            continue
        if 'profile' in var:
            continue
        if var in convert.keys():
            var3=convert[var]
        fq5=getnan(f15[var])
    
