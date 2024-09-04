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
ms=fps['SITE']==b'ABBY'
mu=fpu['SITE']==b'ABBY'

# %%
fpst.keys()

# %%
ffu.keys()

# %%
plt.hist(ffu['mean_chm'][:]-fpst['mean_chm'][0])

# %%
plt.hist(ffs['mean_chm'][:]-fpst['mean_chm'][0])

# %%
