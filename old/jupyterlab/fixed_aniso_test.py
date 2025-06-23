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

# %%
fpu1=h5py.File('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data/NEON_TW_U_UVWT.h5','r')
fps1=h5py.File('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data/NEON_TW_S_UVWT.h5','r')
fpu2=h5py.File('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data_v2/NEON_TW_U_UVWT.h5','r')
fps2=h5py.File('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data_v2/NEON_TW_S_UVWT.h5','r')

# %%
timefixU={'OSBS':[1617229800.0, 1617278400.0], 'SRER':[1548975600.0, 1548986400.0], 'STER':[1661983200.0, 1661990400.0], 'UNDE':[1667251800.0, 1667313000.0], 'YELL':[1606775400.0, 1606838400.0]}

# %%
errorsites=['OSBS','SRER','STER','UNDE','YELL']
fpsite1=fpu1['SITE'][:]
fpsite2=fpu2['SITE'][:]
for site in errorsites:
    m1=fpsite1==bytes(site,'utf-8')
    m2=fpsite2==bytes(site,'utf-8')
    time1=fpu1['TIME'][m1]
    time2=fpu2['TIME'][m2]
    for i in range(len(time2)):
        if time1[i]==time2[i]:
            continue
        else:
            print("'"+site+"':["+str(time1[i-1])+', '+str(time1[i])+'], ',end='')
            break

# %%
for site in errorsites:
    plt.figure(dpi=400)
    m1=fpsite1==bytes(site,'utf-8')
    m2=fpsite2==bytes(site,'utf-8')
    time1=fpu1['TIME'][m1]
    time2=fpu2['TIME'][m2]

# %%
errorsites=['OSBS','SRER','STER','UNDE','YELL']
fpsite1=fpu1['SITE'][:]
fpsite2=fpu2['SITE'][:]
for site in errorsites:
    plt.figure(dpi=400)
    m1=fpsite1==bytes(site,'utf-8')
    m2=fpsite2==bytes(site,'utf-8')
    time1=fpu1['TIME'][m1]
    time2=fpu2['TIME'][m2]
    m22=(time2<=timefixU[site][0])|(time2>=timefixU[site][1])
    plt.plot(time1)
    plt.plot(time2[m22])
    print(np.sum(m22)-len(time2))

# %%
d_fit=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_fit_v2.p','rb'))

# %%
d_fit.keys()

# %%
d_fit['Uu']['param']

# %%

# %%
