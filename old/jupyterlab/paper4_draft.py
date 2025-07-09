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
import pickle
import h5py
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib as mpl
from matplotlib.lines import Line2D
from scipy import optimize
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import matplotlib as pl
import scipy as sci
from IPython.display import HTML
from sklearn.metrics import mean_squared_error
from matplotlib.gridspec import GridSpec
import cmasher as cmr
import matplotlib.ticker as tck
from scipy import stats
import matplotlib
from datetime import datetime,timedelta
mpl.rcParams['figure.dpi'] = 300
sns.set_theme()
plt.rcParams.update({'figure.max_open_warning': 0})

# %% [markdown]
# # Common Files/Setup

# %%
fpu=h5py.File('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data_v2/NEON_TW_U_UVWT.h5','r')
fps=h5py.File('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data_v2/NEON_TW_S_UVWT.h5','r')
d_u=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_unst_v4.p','rb'))
d_s=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_stbl_v4.p','rb'))
d_uabs=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_unst_v4_noabs.p','rb'))
d_sabs=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_stbl_v4_noabs.p','rb'))
d_utw=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_unst_tw_v5.p','rb'))
d_stw=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_stbl_tw_v5.p','rb'))
dc_u=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_corr_ust_v2.p','rb'))
dc_s=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_corr_stb_v2.p','rb'))
dc_u2=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_corr_ust_zL.p','rb'))
dc_s2=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_corr_stb_zL.p','rb'))
d_uxb=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_unst_xbbins_v3.p','rb'))
d_sxb=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_stbl_xbbins_v3.p','rb'))
fpst=h5py.File('/home/tswater/tyche/data/neon/static_data.h5','r')
ft_dir='/home/tswater/tyche/data/neon/foot_stats'

# %%
vmx_a=.7
vmn_a=.1
iva_colors_HEX = ["#410d00","#831901","#983e00","#b56601","#ab8437",
              "#b29f74","#7f816b","#587571","#596c72","#454f51"]
#Transform the HEX colors to RGB.
from PIL import ImageColor
iva_colors_RGB = np.zeros((np.size(iva_colors_HEX),3),dtype='int')
for i in range(0,np.size(iva_colors_HEX)):
    iva_colors_RGB[i,:] = ImageColor.getcolor(iva_colors_HEX[i], "RGB")
iva_colors_RGB = iva_colors_RGB[:,:]/(256)
colors = iva_colors_RGB.tolist()
#----------------------------------------------------
from matplotlib.colors import LinearSegmentedColormap,ListedColormap
inbetween_color_amount = 10
newcolvals = np.zeros(shape=(10 * (inbetween_color_amount) - (inbetween_color_amount - 1), 3))
newcolvals[0] = colors[0]
for i, (rgba1, rgba2) in enumerate(zip(colors[:-1], np.roll(colors, -1, axis=0)[:-1])):
    for j, (p1, p2) in enumerate(zip(rgba1, rgba2)):
        flow = np.linspace(p1, p2, (inbetween_color_amount + 1))
        # discard first 1 since we already have it from previous iteration
        flow = flow[1:]
        newcolvals[ i * (inbetween_color_amount) + 1 : (i + 1) * (inbetween_color_amount) + 1, j] = flow
newcolvals
cmap_ani = ListedColormap(newcolvals, name='from_list', N=None)

def cani_norm(x):
    try:
        x_=x.copy()
        x_[x_<vmn_a]=vmn_a
        x_[x_>vmx_a]=vmx_a
    except:
        x_=x.copy()
        if x_>vmx_a:
            x_=vmx_a
        elif x_<vmn_a:
            x_=vmn_a
    
    x_=(x_-vmn_a)/(vmx_a-vmn_a)
    return cmap_ani(x_)


# %%
def cani_norm(x):
    try:
        x_=x.copy()
        x_[x_<vmn_a]=vmn_a
        x_[x_>vmx_a]=vmx_a
    except:
        x_=x.copy()
        if x_>vmx_a:
            x_=vmx_a
        elif x_<vmn_a:
            x_=vmn_a
    
    x_=(x_-vmn_a)/(vmx_a-vmn_a)
    return cmap_ani(x_)


# %%

# %%

# %% [markdown]
# # FIGURE: Anisotropy Distribution

# %% [markdown]
# ### Setup Figure/Data

# %%
lmost=fps['L_MOST'][:]
zzd=fps['zzd'][:]
zL_s=zzd/lmost

lmost=fpu['L_MOST'][:]
zzd=fpu['zzd'][:]
zL_u=zzd/lmost

xb_u=fpu['ANI_XB'][:]
xb_s=fps['ANI_XB'][:]

yb_u=fpu['ANI_YB'][:]
yb_s=fps['ANI_YB'][:]

# %%
cmp='terrain'
mp=5
szf=1.5
fig=plt.figure(figsize=(3.5*szf,4.5*szf))
grid=ImageGrid(fig, 111,  # similar to subplot(111)
                nrows_ncols=(2, 2),
                axes_pad=0.05,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=.02,
                cbar_size="5%")

### YB ####
im=grid[0].hexbin(np.log10(-zL_u),np.array(yb_u)*mp,mincnt=1,cmap=cmp,gridsize=150,extent=(-4.5,2,0,np.sqrt(3)/2*mp),vmin=0,vmax=600)
im=grid[1].hexbin(np.log10(zL_s),np.array(yb_s*mp),mincnt=1,cmap=cmp,gridsize=700,extent=(-4.5,2,0,np.sqrt(3)/2*mp),vmin=0,vmax=600)
#extent=(-4.5,2,0,.8)
#,extent=(10**(-4.5),10**2,0,np.sqrt(3)/2)

grid[0].set_yticks(np.array([.2,.4,.6,.8])*mp,[.2,.4,.6,.8])
grid[1].set_yticks(np.array([.2,.4,.6,.8])*mp,[.2,.4,.6,.8])
grid[1].set_xticks([-3,-1,1],[r'$10^{-3}$',r'$10^{-1}$',r'$10^{1}$'])
grid[0].set_xticks([-3,-1,1],[r'$-10^{-3}$',r'$-10^{-1}$',r'$-10^{1}$'])

grid[0].set_ylabel(r'$y_b$')

grid[0].set_xlabel(r'$\zeta$')
grid[1].set_xlabel(r'$\zeta$')
grid[0].invert_xaxis()

cb=grid.cbar_axes[0].colorbar(im,label='Valid Points')

### XB ####
im=grid[2].hexbin(np.log10(-zL_u),np.array(xb_u)*mp,mincnt=1,cmap=cmp,gridsize=150,extent=(-4.5,2,0,1*mp),vmin=0,vmax=600)
im=grid[3].hexbin(np.log10(zL_s),np.array(xb_s*mp),mincnt=1,cmap=cmp,gridsize=700,extent=(-4.5,2,0,1*mp),vmin=0,vmax=600)
#extent=(-4.5,2,0,.8)
#,extent=(10**(-4.5),10**2,0,np.sqrt(3)/2)

grid[2].set_yticks(np.array([.2,.4,.6,.8])*mp,[.2,.4,.6,.8])
grid[3].set_yticks(np.array([.2,.4,.6,.8])*mp,[.2,.4,.6,.8])
grid[3].set_xticks([-3,-1,1],[r'$10^{-3}$',r'$10^{-1}$',r'$10^{1}$'])
grid[2].set_xticks([-3,-1,1],[r'$-10^{-3}$',r'$-10^{-1}$',r'$-10^{1}$'])

grid[2].set_ylabel(r'$y_b$')

grid[2].set_xlabel(r'$\zeta$')
grid[3].set_xlabel(r'$\zeta$')
grid[2].invert_xaxis()

# %%

# %%

# %% [markdown]
# ### ALT: Contour

# %%
mp=.33
counts1,xbins1,ybins1,image = plt.hist2d(np.log10(-zL_u)*mp,yb_u,bins=75,density=True)
counts2,xbins2,ybins2,image = plt.hist2d(np.log10(zL_s)*mp,yb_s,bins=75,density=True)
counts3,xbins3,ybins3,image = plt.hist2d(np.log10(-zL_u)*mp,xb_u,bins=75,density=True)
counts4,xbins4,ybins4,image = plt.hist2d(np.log10(zL_s)*mp,xb_s,bins=75,density=True)

counts1=counts1/np.max(counts1)
counts2=counts2/np.max(counts2)
counts3=counts3/np.max(counts3)
counts4=counts4/np.max(counts4)
plt.clf()
#levels=np.array([2,10,50,100,200,300,500,750,1000,2000,3000,4000])
levels=[.005,.15,.3,.45,.6,.75,.9]
cmp='terrain'
szf=1.5
fig=plt.figure(figsize=(3.5*szf,4.5*szf))
grid=ImageGrid(fig, 111,  # similar to subplot(111)
                nrows_ncols=(2, 2),
                axes_pad=0.05,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=.02,
                cbar_size="5%")

### YB ####
im=grid[0].contour(np.transpose(counts1),extent=[xbins1.min(),xbins1.max(),ybins1.min(),ybins1.max()],levels=levels,linewidths=1,cmap=cmp)
im=grid[1].contour(np.transpose(counts2),extent=[xbins2.min(),xbins2.max(),ybins2.min(),ybins2.max()],levels=levels,linewidths=1,cmap=cmp)
#extent=(-4.5,2,0,.8)
#,extent=(10**(-4.5),10**2,0,np.sqrt(3)/2)

grid[0].set_yticks(np.array([.2,.4,.6,.8]),[.2,.4,.6,.8])
grid[1].set_yticks(np.array([.2,.4,.6,.8]),[.2,.4,.6,.8])
grid[1].set_xticks(np.array([-3,-1,1])*mp,[r'$10^{-3}$',r'$10^{-1}$',r'$10^{1}$'])
grid[0].set_xticks(np.array([-3,-1,1])*mp,[r'$-10^{-3}$',r'$-10^{-1}$',r'$-10^{1}$'])

grid[0].set_ylabel(r'$y_b$')

grid[0].set_xlabel(r'$\zeta$')
grid[1].set_xlabel(r'$\zeta$')
grid[0].set_xlim(-4*mp,2*mp)
grid[1].set_xlim(-4*mp,2*mp)

grid[0].invert_xaxis()

#cb=grid.cbar_axes[0].colorbar(im,label='Valid Points')

### XB ####
im=grid[2].contour(np.transpose(counts3),extent=[xbins3.min(),xbins3.max(),ybins3.min(),ybins3.max()],levels=levels,linewidths=1,cmap=cmp)
im=grid[3].contour(np.transpose(counts4),extent=[xbins4.min(),xbins4.max(),ybins4.min(),ybins4.max()],levels=levels,linewidths=1,cmap=cmp)
#extent=(-4.5,2,0,.8)
#,extent=(10**(-4.5),10**2,0,np.sqrt(3)/2)

norm= mpl.colors.Normalize(vmin=0, vmax=1)

grid[2].set_yticks(np.array([.2,.4,.6,.8]),[.2,.4,.6,.8])
grid[3].set_yticks(np.array([.2,.4,.6,.8]),[.2,.4,.6,.8])
grid[3].set_xticks(np.array([-3,-1,1])*mp,[r'$10^{-3}$',r'$10^{-1}$',r'$10^{1}$'])
grid[2].set_xticks(np.array([-3,-1,1])*mp,[r'$-10^{-3}$',r'$-10^{-1}$',r'$-10^{1}$'])
grid[2].set_xlim(-4*mp,2*mp)
grid[3].set_xlim(-4*mp,2*mp)
grid[2].set_ylabel(r'$x_b$')
grid[2].invert_xaxis()

grid[2].set_xlabel(r'$\zeta$')
grid[3].set_xlabel(r'$\zeta$')

sm = plt.cm.ScalarMappable(norm=norm, cmap = im.cmap)
sm.set_array([])
cb=grid.cbar_axes[0].colorbar(sm,ticks=levels,label='Density')
#cb=grid.cbar_axes[0].colorbar(im,label='Valid Points')
plt.savefig('../../plot_output/a1_img_1.png', bbox_inches = "tight")

# %%

# %%

# %% [markdown]
# # FIGURE: Anisotropy Binplots

# %%
anibins=np.linspace(vmn_a,vmx_a,11)
anilvl=(anibins[0:-1]+anibins[1:])/2
anic=cani_norm(anilvl)

# %%

# %%
sz=1.4
fig,axs=plt.subplots(3,3,figsize=(5*sz,4.5*sz),gridspec_kw={'width_ratios': [1,1,.06]},dpi=400)
ss=.1
alph=.2
minpct=1e-04

ylabels=[r'$\Phi_{u}$',r'$\Phi_{v}$',r'$\Phi_{w}$',r'$\Phi_{\theta}$',r'$\Phi_{q}$',r'$\Phi_{C}$']

j=0
for v in ['U','V','W']:
    # SETUP
    cc=cani_norm(d_u[v]['ani'][:])
    cc_s=cani_norm(d_s[v]['ani'][:])
    phi=d_u[v]['phi']
    phi_s=d_s[v]['phi']
    
    ymin=min(np.nanpercentile(phi,.1),np.nanpercentile(phi_s,.1))
    ymax=max(np.nanpercentile(phi,99),np.nanpercentile(phi_s,99))
    if v in ['H2O','CO2']:
        ymax=10
    
    ##### UNSTABLE #####
    # SCATTER UNSTABLE
    axs[j,0].scatter(-d_u[v]['zL'][:],phi,color=cc,s=ss,alpha=alph,marker=".")
    
    # LINES UNSTABLE
    xplt=d_u[v]['p_zL']
    yplt=d_u[v]['p_phi']
    cnt=d_u[v]['p_cnt']
    tot=np.sum(cnt)
    yplt[cnt/tot<minpct]=float('nan')
    for i in range(yplt.shape[0]):
        if v in ['U','V','W','H2O','CO2']:
            axs[j,0].semilogx(-xplt,yplt[i,:],color=anic[i],linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
        else:
            axs[j,0].loglog(-xplt,yplt[i,:],color=anic[i],linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])

    yplt=ust_old[j][0]
    if v in ['U','V','W']:
        axs[j,0].semilogx(-zL_u,yplt,'--',color='k',linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()],zorder=5)
    else:
        axs[j,0].loglog(-zL_u,yplt,'--',color='k',linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()],zorder=5)

    
    # LABELING
    if j==2:
        axs[j,0].tick_params(which="both", bottom=True)
        locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )) 
        axs[j,0].xaxis.set_minor_locator(locmin)
        axs[j,0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axs[j,0].set_xticks([10**-3,10**-2,10**-1,1,10],[r'$-10^{-3}$','',r'$-10^{-1}$','',r'$-10^{1}$'])
        axs[j,0].set_xlim(10**(-3.5),10**(1.1))
        axs[j,0].set_xlabel(r'$\zeta$')
    else:
        axs[j,0].set_xticks([10**-3,10**-2,10**-1,1,10],[])
        axs[j,0].set_xlim(10**(-3.5),10**(1.1))
    #axs[j,0].xaxis.set_minor_locator(tck.AutoMinorLocator())
    axs[j,0].set_ylabel(ylabels[j])
    axs[j,0].set_ylim(ymin,ymax)
    axs[j,0].invert_xaxis()
    
    ##### STABLE #####
    # SCATTER STABLE
    axs[j,1].scatter(d_s[v]['zL'][:],phi_s,color=cc_s,s=ss,alpha=alph,marker=".")
    
    # LINES STABLE
    xplt=d_s[v]['p_zL']
    yplt=d_s[v]['p_phi']
    cnt=d_s[v]['p_cnt']
    tot=np.sum(cnt)
    yplt[cnt/tot<minpct]=float('nan')
    for i in range(yplt.shape[0]):
        if v in ['U','V','W','H2O','CO2']:
            axs[j,1].semilogx(xplt,yplt[i,:],color=anic[i],linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
        else:
            axs[j,1].loglog(xplt,yplt[i,:],color=anic[i],linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
    yplt=stb_old[j][0]
    if v in ['U','V','W']:
        axs[j,1].semilogx(zL_s,yplt[:],'--',color='k',linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()],zorder=5)
    else:
        axs[j,1].loglog(zL_s,yplt[:],'--',color='k',linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()],zorder=5)

    # LABELING
    if j==2:
        axs[j,1].tick_params(which="both", bottom=True)
        locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )) 
        axs[j,1].xaxis.set_minor_locator(locmin)
        axs[j,1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axs[j,1].set_xticks([10**-3,10**-2,10**-1,1,10],[r'$10^{-3}$','',r'$10^{-1}$','',r'$10^{1}$'])
        axs[j,1].set_xlim(10**(-3.5),10**(1.1))
        axs[j,1].set_xlabel(r'$\zeta$')
    else:
        locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )) 
        axs[j,1].xaxis.set_minor_locator(locmin)
        axs[j,1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axs[j,1].set_xticks([10**-3,10**-2,10**-1,1,10],[])
        axs[j,1].set_xlim(10**(-3.5),10**(1.1))
    axs[j,1].tick_params(labelleft=False)
    #axs[j,1].grid(False)
    #axs[j,1].xaxis.grid(True, which='minor')
    axs[j,1].set_ylim(ymin,ymax)
    
    
    #### COLORBAR ####
    axs[j,2].imshow(anic.reshape(10,1,4),origin='lower',interpolation=None)
    axs[j,2].set_xticks([],[])
    inta=[anilvl[0]-(anilvl[1]-anilvl[0])]
    inta.extend(anilvl)
    inta.extend([anilvl[-1]+(anilvl[1]-anilvl[0])])
    xtc=np.interp([.2,.3,.4,.5,.6,.7],inta,np.linspace(-1,10,12))
    axs[j,2].set_yticks(xtc,[.2,.3,.4,.5,.6,.7])
    axs[j,2].grid(False)
    #plt.yticks([.2,.3,.4,.5,.6,.7]),[.2,.3,.4,.5,.6,.7])
    axs[j,2].yaxis.tick_right()
    
    j=j+1

plt.subplots_adjust(hspace=.08,wspace=.02)
plt.savefig('../../plot_output/a1_img_2.png', bbox_inches = "tight")

# %%

# %%

# %% [markdown]
# # FIGURE: SC23 LINES

# %% [markdown]
# ### Data Prep

# %%
zL_u=-np.logspace(-4,2,40)
zL=zL_u.copy()
zL=zL.reshape(1,40).repeat(10,0)

ani=anilvl.copy()
ani=ani.reshape(10,1).repeat(40,1)

# U Unstable
a=.784-2.582*np.log10(ani)
u_u_stp=a*(1-3*zL)**(1/3)
u_u_old=2.55*(1-3*zL)**(1/3)

# V Unstable
a=.725-2.702*np.log10(ani)
v_u_stp=a*(1-3*zL)**(1/3)
v_u_old=2.05*(1-3*zL)**(1/3)

# W Unstable 
a=1.119-0.019*ani-.065*ani**2+0.028*ani**3
w_u_stp=a*(1-3*zL)**(1/3)
w_u_old=1.35*(1-3*zL)**(1/3)
#print(a)


############### STABLE ##################
zL_s=np.logspace(-4,2,40)
zL=zL_s.copy()
zL=zL.reshape(1,40).repeat(10,0)

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
sz=1.4
fig,axs=plt.subplots(3,3,figsize=(5*sz,4.5*sz),gridspec_kw={'width_ratios': [1,1,.06]})
ss=.03
alph=.1
minpct=1e-05

anibins=np.linspace(vmn_a,vmx_a,11)
anilvl=(anibins[0:-1]+anibins[1:])/2
anic=cani_norm(anilvl)
ylabels=[r'$\Phi_{u}$',r'$\Phi_{v}$',r'$\Phi_{w}$',r'$\Phi_{\theta}$',r'$\Phi_{q}$',r'$\Phi_{C}$']

j=0
for v in ['U','V','W']:
    # SETUP
    cc=cani_norm(d_u[v]['ani'][:])
    cc_s=cani_norm(d_s[v]['ani'][:])
    phi=d_u[v]['phi']
    phi_s=d_s[v]['phi']
    
    ymin=min(np.nanpercentile(phi,.1),np.nanpercentile(phi_s,.1))
    ymax=max(np.nanpercentile(phi,99),np.nanpercentile(phi_s,99))
    
    ##### UNSTABLE #####
    # SCATTER UNSTABLE
    axs[j,0].scatter(-d_u[v]['zL'][:],phi,color=cc,s=ss,alpha=alph)
    
    # LINES UNSTABLE
    xplt=zL_u
    yplt=ust_stp[j]
    for i in range(yplt.shape[0]):
        if v in ['U','V','W']:
            axs[j,0].semilogx(-xplt,yplt[i,:],color=anic[i],linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
        else:
            axs[j,0].loglog(-xplt,yplt[i,:],color=anic[i],linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
    yplt=ust_old[j][0]
    if v in ['U','V','W']:
        axs[j,0].semilogx(-zL_u,yplt,'--',color='k',linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()],zorder=5)
    else:
        axs[j,0].loglog(-zL_u,yplt,'--',color='k',linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()],zorder=5)

    
    # LABELING
    if j==2:
        axs[j,0].tick_params(which="both", bottom=True)
        axs[j,0].set_xlabel(r'$\zeta$')
        locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )) 
        axs[j,0].xaxis.set_minor_locator(locmin)
        axs[j,0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axs[j,0].set_xticks([10**-3,10**-2,10**-1,1,10],[r'$-10^{-3}$','',r'$-10^{-1}$','',r'$-10^{1}$'])
        axs[j,0].set_xlim(10**(-3.5),10**(1.1))
    else:
        axs[j,0].set_xticks([10**-3,10**-2,10**-1,1,10],[])
        axs[j,0].set_xlim(10**(-3.5),10**(1.1))
    #axs[j,0].xaxis.set_minor_locator(tck.AutoMinorLocator())
    axs[j,0].set_ylabel(ylabels[j])
    axs[j,0].set_ylim(ymin,ymax)
    axs[j,0].invert_xaxis()
    
    ##### STABLE #####
    # SCATTER STABLE
    axs[j,1].scatter(d_s[v]['zL'][:],phi_s,color=cc_s,s=ss,alpha=alph)
    
    # LINES STABLE
    xplt=zL_s
    yplt=stb_stp[j]
    for i in range(yplt.shape[0]):
        if v in ['U','V','W']:
            axs[j,1].semilogx(xplt,yplt[i,:],color=anic[i],linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
        else:
            axs[j,1].loglog(xplt,yplt[i,:],color=anic[i],linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
    yplt=stb_old[j][0]
    if v in ['U','V','W']:
        axs[j,1].semilogx(zL_s,yplt[:],'--',color='k',linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()],zorder=5)
    else:
        axs[j,1].loglog(zL_s,yplt[:],'--',color='k',linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()],zorder=5)

    
    # LABELING
    if j==2:
        axs[j,1].tick_params(which="both", bottom=True)
        axs[j,1].set_xlabel(r'$\zeta$')
        axs[j,1].set_xticks([10**-3,10**-2,10**-1,1,10],[r'$10^{-3}$','',r'$10^{-1}$','',r'$10^{1}$'])
        locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )) 
        axs[j,1].xaxis.set_minor_locator(locmin)
        axs[j,1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axs[j,1].set_xlim(10**(-3.5),10**(1.1))
    else:
        locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )) 
        axs[j,1].xaxis.set_minor_locator(locmin)
        axs[j,1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axs[j,1].set_xticks([10**-3,10**-2,10**-1,1,10],[])
        axs[j,1].set_xlim(10**(-3.5),10**(1.1))
    axs[j,1].tick_params(labelleft=False)
    #axs[j,1].grid(False)
    #axs[j,1].xaxis.grid(True, which='minor')
    axs[j,1].set_ylim(ymin,ymax)
    
    
    #### COLORBAR ####
    axs[j,2].imshow(anic.reshape(10,1,4),origin='lower',interpolation=None)
    axs[j,2].set_xticks([],[])
    inta=[anilvl[0]-(anilvl[1]-anilvl[0])]
    inta.extend(anilvl)
    inta.extend([anilvl[-1]+(anilvl[1]-anilvl[0])])
    xtc=np.interp([.2,.3,.4,.5,.6,.7],inta,np.linspace(-1,10,12))
    axs[j,2].set_yticks(xtc,[.2,.3,.4,.5,.6,.7])
    axs[j,2].grid(False)
    #plt.yticks([.2,.3,.4,.5,.6,.7]),[.2,.3,.4,.5,.6,.7])
    axs[j,2].yaxis.tick_right()
    
    j=j+1
    
#fig.text(0.5, 0.04, r'$\zeta$', ha='center')
plt.subplots_adjust(hspace=.08,wspace=.02)
plt.savefig('../../plot_output/a1_img_3.png', bbox_inches = "tight")

# %%

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # FIGURE: SC23 Performance

# %%
sz=1
j=0
fig,axs=plt.subplots(3,1,figsize=(6*sz,9/4*3*sz))
names=[r'$|\zeta|>.1$','all',r'$|\zeta|<.1$',r'$|\zeta|<.1$','all',r'$|\zeta|>.1$']
ylabels=[r'SS $\Phi_{u}$',r'SS $\Phi_{v}$',r'SS $\Phi_{w}$',r'SS $\Phi_{\theta}$',r'$\Phi_{q}$',r'$\Phi_{C}$']

for v in list(d_u.keys())[0:3]:
    ss=[]
    ss.append(list(d_u[v]['SShi_s'].values()))
    ss.append(list(d_u[v]['SS_s'].values()))
    ss.append(list(d_u[v]['SSlo_s'].values()))
    
    ss.append(list(d_s[v]['SSlo_s'].values()))
    ss.append(list(d_s[v]['SS_s'].values()))
    ss.append(list(d_s[v]['SShi_s'].values()))
    
    ss=np.array(ss)
    axs[j].plot([0,7],[0,0],color='w',linewidth=3)
    axs[j].boxplot(ss.T,labels=names)
    axs[j].plot([3.5,3.5],[-1,1],'k--')
    if j<2:
        axs[j].set_xticks([1,2,3,4,5,6],[])
    else:
        axs[j].set_xlabel(r'$\zeta<0$                               $\zeta>0$',fontsize=14)
    axs[j].set_ylabel(ylabels[j])
    axs[j].set_ylim(-.8,.8)
    axs[j].set_xlim(.5,6.5)
    #axs[j].tick_params(axis='x',labelrotation=45)
    j=j+1
    
plt.subplots_adjust(hspace=.08,wspace=.02)
plt.savefig('../../plot_output/a1_img_4.png', bbox_inches = "tight")

# %%

# %%

# %% [markdown]
# # FIGURE: Refit Lines

# %%
sz=1.4
fig,axs=plt.subplots(3,3,figsize=(5*sz,4.5*sz),gridspec_kw={'width_ratios': [1,1,.06]})
ss=.03
alph=.1
minpct=1e-05

anibins=np.linspace(vmn_a,vmx_a,11)
anilvl=(anibins[0:-1]+anibins[1:])/2
anic=cani_norm(anilvl)
ylabels=[r'$\Phi_{u}$',r'$\Phi_{v}$',r'$\Phi_{w}$',r'$\Phi_{\theta}$',r'$\Phi_{q}$',r'$\Phi_{C}$']

j=0
for v in ['U','V','W']:
    # SETUP
    cc=cani_norm(d_u[v]['ani'][:])
    cc_s=cani_norm(d_s[v]['ani'][:])
    phi=d_u[v]['phi']
    phi_s=d_s[v]['phi']
    
    ymin=min(np.nanpercentile(phi,.1),np.nanpercentile(phi_s,.1))
    ymax=max(np.nanpercentile(phi,99),np.nanpercentile(phi_s,99))
    if v in ['H2O','CO2']:
        ymax=10
    ##### UNSTABLE #####
    # SCATTER UNSTABLE
    axs[j,0].scatter(-d_u[v]['zL'][:],phi,color=cc,s=ss,alpha=alph)
    
    # LINES UNSTABLE
    xplt=d_utw[v]['p_zL'][:]
    yplt=d_utw[v]['p_phi'][:]
    for i in range(yplt.shape[0]):
        if v in ['U','V','W','H2O','CO2']:
            axs[j,0].semilogx(-xplt,yplt[i,:],color=anic[i],linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
        else:
            axs[j,0].loglog(-xplt,yplt[i,:],color=anic[i],linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
    yplt=ust_old[j][0]
    if v in ['U','V','W','H2O','CO2']:
        axs[j,0].semilogx(-zL_u,yplt,'--',color='k',linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()],zorder=5)
    else:
        axs[j,0].loglog(-zL_u,yplt,'--',color='k',linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()],zorder=5)

    
    # LABELING
    if j==2:
        axs[j,0].tick_params(which="both", bottom=True)
        axs[j,0].set_xlabel(r'$\zeta$')
        locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )) 
        axs[j,0].xaxis.set_minor_locator(locmin)
        axs[j,0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axs[j,0].set_xticks([10**-3,10**-2,10**-1,1,10],[r'$-10^{-3}$','',r'$-10^{-1}$','',r'$-10^{1}$'])
        axs[j,0].set_xlim(10**(-3.5),10**(1.1))
    else:
        axs[j,0].set_xticks([10**-3,10**-2,10**-1,1,10],[])
        axs[j,0].set_xlim(10**(-3.5),10**(1.1))
    #axs[j,0].xaxis.set_minor_locator(tck.AutoMinorLocator())
    axs[j,0].set_ylabel(ylabels[j])
    axs[j,0].set_ylim(ymin,ymax)
    axs[j,0].invert_xaxis()
    
    ##### STABLE #####
    # SCATTER STABLE
    axs[j,1].scatter(d_s[v]['zL'][:],phi_s,color=cc_s,s=ss,alpha=alph)
    
    # LINES STABLE
    xplt=d_stw[v]['p_zL'][:]
    yplt=d_stw[v]['p_phi'][:]
    for i in range(yplt.shape[0]):
        if v in ['U','V','W','H2O','CO2']:
            axs[j,1].semilogx(xplt,yplt[i,:],color=anic[i],linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
        else:
            axs[j,1].loglog(xplt,yplt[i,:],color=anic[i],linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
    yplt=stb_old[j][0]
    if v in ['U','V','W','H2O','CO2']:
        axs[j,1].semilogx(zL_s,yplt[:],'--',color='k',linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()],zorder=5)
    else:
        axs[j,1].loglog(zL_s,yplt[:],'--',color='k',linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()],zorder=5)

    
    # LABELING
    if j==2:
        axs[j,1].tick_params(which="both", bottom=True)
        axs[j,1].set_xlabel(r'$\zeta$')
        axs[j,1].set_xticks([10**-3,10**-2,10**-1,1,10],[r'$10^{-3}$','',r'$10^{-1}$','',r'$10^{1}$'])
        locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )) 
        axs[j,1].xaxis.set_minor_locator(locmin)
        axs[j,1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axs[j,1].set_xlim(10**(-3.5),10**(1.1))
    else:
        locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )) 
        axs[j,1].xaxis.set_minor_locator(locmin)
        axs[j,1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axs[j,1].set_xticks([10**-3,10**-2,10**-1,1,10],[])
        axs[j,1].set_xlim(10**(-3.5),10**(1.1))
    axs[j,1].tick_params(labelleft=False)
    #axs[j,1].grid(False)
    #axs[j,1].xaxis.grid(True, which='minor')
    axs[j,1].set_ylim(ymin,ymax)
    
    
    #### COLORBAR ####
    axs[j,2].imshow(anic.reshape(10,1,4),origin='lower',interpolation=None)
    axs[j,2].set_xticks([],[])
    inta=[anilvl[0]-(anilvl[1]-anilvl[0])]
    inta.extend(anilvl)
    inta.extend([anilvl[-1]+(anilvl[1]-anilvl[0])])
    xtc=np.interp([.2,.3,.4,.5,.6,.7],inta,np.linspace(-1,10,12))
    axs[j,2].set_yticks(xtc,[.2,.3,.4,.5,.6,.7])
    axs[j,2].grid(False)
    #plt.yticks([.2,.3,.4,.5,.6,.7]),[.2,.3,.4,.5,.6,.7])
    axs[j,2].yaxis.tick_right()
    
    j=j+1
    
#fig.text(0.5, 0.04, r'$\zeta$', ha='center')
plt.subplots_adjust(hspace=.08,wspace=.02)
plt.savefig('../../plot_output/a1_img_5.png', bbox_inches = "tight")

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # FIGURE: Refit Performance

# %%
sz=1
j=0

fig,axs=plt.subplots(3,1,figsize=(6*sz,6*sz))
names=[r'$|\zeta|>.1$','all',r'$|\zeta|<.1$',r'$|\zeta|<.1$','all',r'$|\zeta|>.1$']
ylabels=[r'SS $\Phi_{u}$',r'SS $\Phi_{v}$',r'SS $\Phi_{w}$',r'SS $\Phi_{\theta}$',r'SS $\Phi_{q}$',r'SS $\Phi_{C}$']

for v in list(d_utw.keys())[0:3]:
    ss=[]
    ss.append(list(d_utw[v]['SShi_s'].values()))
    ss.append(list(d_utw[v]['SS_s'].values()))
    ss.append(list(d_utw[v]['SSlo_s'].values()))
    
    ss.append(list(d_stw[v]['SSlo_s'].values()))
    ss.append(list(d_stw[v]['SS_s'].values()))
    ss.append(list(d_stw[v]['SShi_s'].values()))
    
    ss=np.array(ss)
    axs[j].plot([0,7],[0,0],color='w',linewidth=3)
    axs[j].boxplot(ss.T,labels=names)
    axs[j].plot([3.5,3.5],[-1,1],'k--')
    if j<2:
        axs[j].set_xticks([1,2,3,4,5,6],[])
    else:
        axs[j].set_xlabel(r'$\zeta<0$                               $\zeta>0$',fontsize=14)
    axs[j].set_ylabel(ylabels[j])
    axs[j].set_ylim(-.8,.8)
    axs[j].set_xlim(.5,6.5)
    #axs[j].tick_params(axis='x',labelrotation=45)
    j=j+1
    
plt.subplots_adjust(hspace=.08,wspace=.02)
plt.savefig('../../plot_output/a1_img_6.png', bbox_inches = "tight")

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # FIGURE: Combined Boxplots

# %%
sz=1
fig,axs=plt.subplots(3,1,figsize=(6*sz,6*sz))
names=['',r'$|\zeta|>.1$','','all','',r'$|\zeta|<.1$','',r'$|\zeta|<.1$','','all','',r'$|\zeta|>.1$']
ylabels=[r'SS $\Phi_{u}$',r'SS $\Phi_{v}$',r'SS $\Phi_{w}$',r'SS $\Phi_{\theta}$',r'SS $\Phi_{q}$',r'SS $\Phi_{C}$']
j=0
pos=[0,.5,1.5,2,3,3.5,4.5,5,6,6.5,7.5,8]
bcolor=['mediumpurple','goldenrod','mediumpurple','goldenrod','mediumpurple','goldenrod','mediumpurple','goldenrod','mediumpurple','goldenrod','mediumpurple','goldenrod']
for v in list(d_utw.keys())[0:3]:
    ss=[]

    ss.append(list(d_u[v]['SShi_s'].values()))
    ss.append(list(d_utw[v]['SShi_s'].values()))
    
    ss.append(list(d_u[v]['SS_s'].values()))
    ss.append(list(d_utw[v]['SS_s'].values()))
    
    ss.append(list(d_u[v]['SSlo_s'].values()))
    ss.append(list(d_utw[v]['SSlo_s'].values()))

    ss.append(list(d_s[v]['SSlo_s'].values()))
    ss.append(list(d_stw[v]['SSlo_s'].values()))

    ss.append(list(d_s[v]['SS_s'].values()))
    ss.append(list(d_stw[v]['SS_s'].values()))

    ss.append(list(d_s[v]['SShi_s'].values()))
    ss.append(list(d_stw[v]['SShi_s'].values()))
    
    ss=np.array(ss)
    axs[j].plot([-.5,8.5],[0,0],color='w',linewidth=3)
    print(len(ss))
    print(len(names))
    bplot=axs[j].boxplot(ss.T,labels=names,positions=pos,patch_artist=True)
    # fill with colors
    for patch, color in zip(bplot['boxes'], bcolor):
        patch.set_facecolor(color)
        patch.set_alpha(.35)

    ptch1=mpatches.Patch(facecolor=bcolor[0],alpha=.35,label='SC23 Fit',edgecolor='k')
    ptch2=mpatches.Patch(facecolor=bcolor[1],alpha=.35,label='NEON Fit',edgecolor='k')

    leg=[ptch1,ptch2]
    if j==0:
        axs[j].legend(handles=leg)
    
    axs[j].plot([4,4],[-1,1],'k--')
    if j<2:
        axs[j].set_xticks(pos,[])
    else:
        axs[j].set_xlabel(r'$\zeta<0$                               $\zeta>0$',fontsize=14)
    axs[j].set_ylabel(ylabels[j])
    axs[j].set_ylim(-.8,.8)
    axs[j].set_xlim(-.5,8.5)
    #axs[j].tick_params(axis='x',labelrotation=45)
    j=j+1
    
plt.subplots_adjust(hspace=.08,wspace=.02)
plt.savefig('../../plot_output/a1_img_572.png', bbox_inches = "tight")

# %%
sz=1
fig,axs=plt.subplots(3,1,figsize=(6*sz,6*sz))
names=['',r'$|\zeta|>.1$','','all','',r'$|\zeta|<.1$','',r'$|\zeta|<.1$','','all','',r'$|\zeta|>.1$']
ylabels=[r'SS $\Phi_{u}$',r'SS $\Phi_{v}$',r'SS $\Phi_{w}$',r'SS $\Phi_{\theta}$',r'SS $\Phi_{q}$',r'SS $\Phi_{C}$']
j=0
pos=[0,.5,1.5,2,3,3.5,4.5,5,6,6.5,7.5,8]
bcolor=['mediumpurple','goldenrod','mediumpurple','goldenrod','mediumpurple','goldenrod','mediumpurple','goldenrod','mediumpurple','goldenrod','mediumpurple','goldenrod']
for v in list(d_utw.keys())[0:3]:
    ss=[]

    ss.append(list(d_u[v]['SShi_s'].values()))
    ss.append(list(d_utw[v]['SShi_s'].values()))
    
    ss.append(list(d_u[v]['SS_s'].values()))
    ss.append(list(d_utw[v]['SS_s'].values()))
    
    ss.append(list(d_u[v]['SSlo_s'].values()))
    ss.append(list(d_utw[v]['SSlo_s'].values()))

    ss.append(list(d_s[v]['SSlo_s'].values()))
    ss.append(list(d_stw[v]['SSlo_s'].values()))

    ss.append(list(d_s[v]['SS_s'].values()))
    ss.append(list(d_stw[v]['SS_s'].values()))

    ss.append(list(d_s[v]['SShi_s'].values()))
    ss.append(list(d_stw[v]['SShi_s'].values()))
    
    ss=np.array(ss)
    axs[j].plot([-.5,8.5],[0,0],color='w',linewidth=3)
    print(len(ss))
    print(len(names))
    bplot=axs[j].boxplot(ss.T[:,0::2],labels=names[0::2],positions=pos[0::2],patch_artist=True)
    # fill with colors
    for patch, color in zip(bplot['boxes'], bcolor[0::2]):
        patch.set_facecolor(color)
        patch.set_alpha(.35)
    axs[j].plot([4,4],[-1,1],'k--')
    if j<2:
        axs[j].set_xticks(pos,[])
    else:
        axs[j].set_xlabel(r'$\zeta<0$                               $\zeta>0$',fontsize=14)
        axs[j].set_xticks(pos,names)
    axs[j].set_ylabel(ylabels[j])
    axs[j].set_ylim(-.8,.8)
    axs[j].set_xlim(-.5,8.5)
    #axs[j].tick_params(axis='x',labelrotation=45)
    j=j+1
    
plt.subplots_adjust(hspace=.08,wspace=.02)
plt.savefig('../../plot_output/a1_img_57_SC23only.png', bbox_inches = "tight")


# %%

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # FIGURE: Site Level MAD (U unstable only)

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
var='V'
mad_most=[]
other=[[],[],[],[]]
fpsite=fpu['SITE'][:]
for i in range(47):
    site=np.unique(fpsite)[i]
    other[0].append(str(site)[2:-1])
    ss=str(site)[2:-1]
    ffu=h5py.File('/home/tswater/tyche/data/neon/foot_stats/'+ss+'_U.h5','r')
    std=np.nanmedian(ffu['std_dsm'][:])
    mean=np.nanmedian(ffu['mean_chm'][:])
    if (std>20):
        other[1].append(1)
    elif (std>10):
        other[1].append(.5)
    else:
        other[1].append(0)
    other[2].append(stats.mode(ffu['nlcd_dom'])[0])
    #other[3].append(d_u[var]['MAD_SC23_s'][ss])
    #mad_most.append(d_u[var]['MAD_OLD_s'][ss])
    other[3].append(-d_uabs[var]['MD_SC23_s'][ss])
    mad_most.append(-d_uabs[var]['MD_OLD_s'][ss])

# %%
d_u['U'].keys()

# %%
X,Y2=sort_together(mad_most,other)
plt.figure(figsize=(13,3.5))
colors=[]
hatch=[]
for i in range(len(X)):
    try:
        colors.append(class_colors[Y2[2][i]])
    except:
        colors.append('darkgreen')
    if Y2[1][i]==1:
        hatch.append('OO')
    elif Y2[1][i]==.5:
        hatch.append('..')
    else:
        hatch.append('')
yerr=[np.zeros((len(X),)),X[:]-Y2[3]]
yerr=np.array(yerr)
yerr[0,:][yerr[1,:]<0]=yerr[1,:][yerr[1,:]<0]*(-1)
yerr[1,:][yerr[1,:]<0]=0
a=plt.bar(Y2[0],Y2[3],color=colors,hatch=hatch,edgecolor='black',yerr=yerr,capsize=4)
plt.xticks(rotation=45)
plt.xlim(-.5,47)
#plt.ylim(.2,1.65)
plt.ylabel(r'$Bias$')
leg=[]
for clas in class_names.keys():
    ptch=mpatches.Patch(facecolor=class_colors[clas],label=class_names[clas],edgecolor='k')
    leg.append(ptch)
ptch=mpatches.Patch(label='Complex',hatch='..',edgecolor='k',facecolor='white')
leg.append(ptch)
ptch=mpatches.Patch(label='Very Complex',hatch='OO',edgecolor='k',facecolor='white')
leg.append(ptch)
#plt.legend(handles=leg,ncol=2,loc='upper left',title='Dominant Landcover')
plt.savefig('../../plot_output/a1_img_7vu_bias.png', bbox_inches = "tight")

# %%

# %%

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # FIGURE Error Correlation

# %%

# %%
data=dc_u['spearmanr'][:]
datas=dc_s['spearmanr'][:]
xvars=dc_u['xvars'][:]
yvars=dc_u['yvars'][:]
yvarnames={'ANI_XB':r'$x_b$','ANI_YB':r'$y_b$','USTAR':r'$u*$','UU':"$\overline{u'u'}$",'VV':"$\overline{v'v'}$",'WW':"$\overline{w'w'}$",'Ustr':'$\overline{U}$','Wstr':r'$\overline{w}$',
           'mean_chm':r'$h_c$','std_chm':r'$\sigma_{h_c}$','std_dsm':r'$\sigma_{DSM}$','NETRAD':r'$R_{net}$','H':r'$H$','RH':r'$RH$',\
           'TA':r'$T$','PA':r'$P$'}
ynmlist=[]
mark=['+','s','^','+','s','^']
color=['blue','red','green','cornflowerblue','lightcoral','limegreen']

plt.figure(figsize=(8,5),dpi=300)
#reduce data
datar=np.zeros((6,len(yvarnames.keys())))
i=0
for yv in yvarnames.keys():
    ynmlist.append(yvarnames[yv])
    print(yv)
    idx=np.where(np.array(yvars)==yv)[0][0]
    datar[0,i]=data[2,idx]
    datar[1,i]=data[3,idx]
    datar[2,i]=data[4,idx]
    datar[3,i]=datas[2,idx]
    datar[4,i]=datas[3,idx]
    datar[5,i]=datas[4,idx]
    i=i+1

#datar=np.abs(datar)
plt.subplot(2,1,1)
N=len(datar[0,:])
datab=np.mean(datar,axis=0)

for i in range(6):
    plt.scatter(np.linspace(0,N,N),datar[i,:],marker=mark[i],c=color[i],zorder=i+2)
    plt.xticks(np.linspace(0,N,N),[])
plt.legend([r'$\sigma_U$ unstable',r'$\sigma_V$ unstable',r'$\sigma_W$ unstable',r'$\sigma_U$ stable',r'$\sigma_V$ stable',r'$\sigma_W$ stable'],ncols=2,labelspacing=.2,borderpad=.2,loc='lower right')

plt.plot(np.linspace(-1,N+1,N),np.zeros((N,)),color='white',linewidth=3,zorder=2)
plt.bar(np.linspace(0,N,N),datab,color='black',alpha=.3,zorder=1)

plt.title('')
plt.xlim(-.5,N+.5)
plt.ylim(-.7,.7)
plt.ylabel('Corr. Bias (MOST)')


#reduce data
ynmlist=[]
datar=np.zeros((6,len(yvarnames.keys())))
i=0
for yv in yvarnames.keys():
    ynmlist.append(yvarnames[yv])
    print(yv)
    idx=np.where(np.array(yvars)==yv)[0][0]
    datar[0,i]=data[5,idx]
    datar[1,i]=data[6,idx]
    datar[2,i]=data[7,idx]
    datar[3,i]=datas[5,idx]
    datar[4,i]=datas[6,idx]
    datar[5,i]=datas[7,idx]
    i=i+1

#datar=np.abs(datar)
plt.subplot(2,1,2)
N=len(datar[0,:])
datab=np.mean(datar,axis=0)
plt.plot(np.linspace(-1,N+1,N),np.zeros((N,)),color='white',linewidth=3,zorder=2)
plt.bar(np.linspace(0,N,N),datab,color='black',alpha=.3,zorder=1)

for i in range(6):
    plt.scatter(np.linspace(0,N,N),datar[i,:],marker=mark[i],c=color[i],zorder=i+2)
    plt.xticks(np.linspace(0,N,N),ynmlist)
plt.title('')
plt.xlim(-.5,N+.5)
plt.ylim(-.7,.7)
plt.ylabel('Corr. Bias (SC23)')
plt.subplots_adjust(hspace=.05)
plt.savefig('../../plot_output/a1_img_9bias.png', bbox_inches = "tight")

# %%
yvars

# %%
np.nanmedian(datas_,axis=0).shape

# %%
# assemble data and 
data_=np.zeros((47,8,44))
datas_=np.zeros((47,8,44))

i=0
for site in dc_s['sitelevel'].keys():
    data_[i,:,:]=dc_u['sitelevel'][site]['spearmanr']
    datas_[i,:,:]=dc_s['sitelevel'][site]['spearmanr']
    i=i+1
datas=np.nanmedian(datas_,axis=0)
data=np.nanmedian(data_,axis=0)

datas=dc_s['spearmanr'][:]
xvars=dc_u['xvars'][:]
yvars=dc_u['yvars'][:]
yvarnames={'ANI_XB':r'$x_b$','ANI_YB':r'$y_b$','USTAR':r'$u*$','Ustr':'$\overline{U}$','Wstr':r'$\overline{w}$',
           'mean_chm':r'$h_c$','std_chm':r'$\sigma_{h_c}$','std_dsm':r'$\sigma_{DSM}$','NETRAD':r'$R_{net}$','H':r'$H$','RH':r'$RH$',\
           'TA':r'$T$','PA':r'$P$'}
ynmlist=[]
mark=['+','s','^','+','s','^']
color=['blue','red','green','cornflowerblue','lightcoral','limegreen']

plt.figure(figsize=(7,5),dpi=300)
#reduce data
datar=np.zeros((6,len(yvarnames.keys())))
i=0
for yv in yvarnames.keys():
    ynmlist.append(yvarnames[yv])
    print(yv)
    idx=np.where(np.array(yvars)==yv)[0][0]
    datar[0,i]=data[2,idx]
    datar[1,i]=data[3,idx]
    datar[2,i]=data[4,idx]
    datar[3,i]=datas[2,idx]
    datar[4,i]=datas[3,idx]
    datar[5,i]=datas[4,idx]
    i=i+1

#datar=np.abs(datar)
plt.subplot(2,1,1)
N=len(datar[0,:])
datab=np.mean(datar,axis=0)

for i in range(6):
    plt.scatter(np.linspace(0,N,N),datar[i,:],marker=mark[i],c=color[i],zorder=i+2)
    plt.xticks(np.linspace(0,N,N),[])
plt.legend([r'$\sigma_U$ unstable',r'$\sigma_V$ unstable',r'$\sigma_W$ unstable',r'$\sigma_U$ stable',r'$\sigma_V$ stable',r'$\sigma_W$ stable'],ncols=2,labelspacing=.2,borderpad=.2,loc='lower right')

plt.plot(np.linspace(-1,N+1,N),np.zeros((N,)),color='white',linewidth=3,zorder=2)
plt.bar(np.linspace(0,N,N),datab,color='black',alpha=.3,zorder=1)

plt.title('')
plt.xlim(-.5,N+.5)
plt.ylim(-.55,.55)
plt.ylabel('Corr. Error (MOST)')


#reduce data
ynmlist=[]
datar=np.zeros((6,len(yvarnames.keys())))
i=0
for yv in yvarnames.keys():
    ynmlist.append(yvarnames[yv])
    print(yv)
    idx=np.where(np.array(yvars)==yv)[0][0]
    datar[0,i]=data[5,idx]
    datar[1,i]=data[6,idx]
    datar[2,i]=data[7,idx]
    datar[3,i]=datas[5,idx]
    datar[4,i]=datas[6,idx]
    datar[5,i]=datas[7,idx]
    i=i+1

#datar=np.abs(datar)
plt.subplot(2,1,2)
N=len(datar[0,:])
datab=np.mean(datar,axis=0)
plt.plot(np.linspace(-1,N+1,N),np.zeros((N,)),color='white',linewidth=3,zorder=2)
plt.bar(np.linspace(0,N,N),datab,color='black',alpha=.3,zorder=1)

for i in range(6):
    plt.scatter(np.linspace(0,N,N),datar[i,:],marker=mark[i],c=color[i],zorder=i+2)
    plt.xticks(np.linspace(0,N,N),ynmlist)
plt.title('')
plt.xlim(-.5,N+.5)
plt.ylim(-.55,.55)
plt.ylabel('Corr. Error (SC23)')
plt.subplots_adjust(hspace=.05)

# %%

# %%
#### TESTING STABILITY ####
stb=1
data=dc_u2['spearmanr'][stb,:]
datas=dc_s2['spearmanr'][stb,:]
xvars=dc_u2['xvars'][:]
yvars=dc_u2['yvars'][:]
yvarnames={'ANI_XB':r'$x_b$','ANI_YB':r'$y_b$','USTAR':r'$u*$','Ustr':'$\overline{U}$','Wstr':r'$\overline{w}$',
           'mean_chm':r'$h_c$','std_chm':r'$\sigma_{h_c}$','std_dsm':r'$\sigma_{DSM}$','NETRAD':r'$R_{net}$','H':r'$H$','RH':r'$RH$',\
           'TA':r'$T$','PA':r'$P$'}
ynmlist=[]
mark=['+','s','^','+','s','^']
color=['blue','red','green','cornflowerblue','lightcoral','limegreen']

plt.figure(figsize=(7,5),dpi=300)
#reduce data
datar=np.zeros((6,len(yvarnames.keys())))
i=0
for yv in yvarnames.keys():
    ynmlist.append(yvarnames[yv])
    print(yv)
    idx=np.where(np.array(yvars)==yv)[0][0]
    datar[0,i]=data[2,idx]
    datar[1,i]=data[3,idx]
    datar[2,i]=data[4,idx]
    datar[3,i]=datas[2,idx]
    datar[4,i]=datas[3,idx]
    datar[5,i]=datas[4,idx]
    i=i+1

#datar=np.abs(datar)
plt.subplot(2,1,1)
N=len(datar[0,:])
datab=np.mean(datar,axis=0)

for i in range(6):
    plt.scatter(np.linspace(0,N,N),datar[i,:],marker=mark[i],c=color[i],zorder=i+2)
    plt.xticks(np.linspace(0,N,N),[])
plt.legend([r'$\sigma_U$ unstable',r'$\sigma_V$ unstable',r'$\sigma_W$ unstable',r'$\sigma_U$ stable',r'$\sigma_V$ stable',r'$\sigma_W$ stable'],ncols=2,labelspacing=.2,borderpad=.2,loc='lower right')

plt.plot(np.linspace(-1,N+1,N),np.zeros((N,)),color='white',linewidth=3,zorder=2)
plt.bar(np.linspace(0,N,N),datab,color='black',alpha=.3,zorder=1)

plt.title('')
plt.xlim(-.5,N+.5)
plt.ylim(-.65,.65)
plt.ylabel('Corr. Error (MOST)')


#reduce data
ynmlist=[]
datar=np.zeros((6,len(yvarnames.keys())))
i=0
for yv in yvarnames.keys():
    ynmlist.append(yvarnames[yv])
    print(yv)
    idx=np.where(np.array(yvars)==yv)[0][0]
    datar[0,i]=data[5,idx]
    datar[1,i]=data[6,idx]
    datar[2,i]=data[7,idx]
    datar[3,i]=datas[5,idx]
    datar[4,i]=datas[6,idx]
    datar[5,i]=datas[7,idx]
    i=i+1

#datar=np.abs(datar)
plt.subplot(2,1,2)
N=len(datar[0,:])
datab=np.mean(datar,axis=0)
plt.plot(np.linspace(-1,N+1,N),np.zeros((N,)),color='white',linewidth=3,zorder=2)
plt.bar(np.linspace(0,N,N),datab,color='black',alpha=.3,zorder=1)

for i in range(6):
    plt.scatter(np.linspace(0,N,N),datar[i,:],marker=mark[i],c=color[i],zorder=i+2)
    plt.xticks(np.linspace(0,N,N),ynmlist)
plt.title('')
plt.xlim(-.5,N+.5)
plt.ylim(-.65,.65)
plt.ylabel('Corr. Error (SC23)')
plt.subplots_adjust(hspace=.05)


# %%

# %% [markdown]
# # Figure: Self Similarity YB

# %%
def get_random(var,num=100000):
    match var:
        case 'Uu': fp=fpu
        case 'Us': fp=fps
        case 'Wu': fp=fpu
        case 'Ws': fp=fps
        case 'Vu': fp=fpu
        case 'Vs': fp=fps
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
        case 'Vu': phir=np.sqrt(vvr)/ustr; print(var)
        case 'Vs': phir=np.sqrt(vvr)/ustr; print(var)
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

def get_random_xb(var,num=10000):
    match var:
        case 'Uu': fp=fpu
        case 'Us': fp=fps
        case 'Wu': fp=fpu
        case 'Ws': fp=fps
        case 'Vu': fp=fpu
        case 'Vs': fp=fps
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
        case 'Vu': phir=np.sqrt(vvr)/ustr; print(var)
        case 'Vs': phir=np.sqrt(vvr)/ustr; print(var)
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
    return phir[m],xbr[m]

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

def getlines(ani_,phi_,anibins):
    anibc=(anibins[1:]+anibins[0:-1])/2
    phiout=[]
    for i in range(len(anibc)):
        phiout.append(np.nanmedian(phi_[(ani_>anibins[i])&(ani_<anibins[i+1])]))
    return anibc,np.array(phiout)

def ymx(x,m,b):
    return m*x+b


# %%
sz=1
alpha=.15
fig=plt.figure(figsize=(9*sz,5*sz),dpi=500)
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
        phir,ybr=get_random(var,num=150000)
    elif 'u' in var:
        phir,ybr=get_random(var)
    N=len(phir)
    x1,y1=getlines(ybr,phir,np.linspace(0.01,.7,25))
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
    x,y=getlines(yb,phi,np.linspace(0.01,.7,25))
    
    # plot
    ax00.scatter(yb,phi,s=.025,alpha=alpha,color='dimgrey')
    ax00.set_xlim(0,.8)
    ax00.set_ylim(0,5)
    ax00.plot(x,y,color='k')
    ax00.set_xticks([0,np.sqrt(3)/6,np.sqrt(3)/3],[])
    ax00.set_yticks([0,2,4])
    
    ax01.scatter(ybr,phir,s=.025,alpha=alpha,color='darkgrey')
    ax01.set_xlim(0,.8)
    ax01.set_ylim(0,5)
    #ax01.plot(x1,y1,color='dimgrey')
    params,pcov=optimize.curve_fit(ymx,ybr,phir,[1,1],bounds=([-10,-2],[10,2]),loss='cauchy')
    y1=ymx(x1,params[0],params[1])
    ax01.plot(x1,y1,color='k')

    ### MAD ####
    ml=(yb>np.min(x))&(yb<np.max(x))
    xx=yb[ml]
    yy=phi[ml]
    y2=np.interp(xx,x,y)
    print(np.median(np.abs(y2-yy)))
    print('    '+str(stats.pearsonr(yy,y2)[0]))

    ml=(ybr>np.min(x1))&(ybr<np.max(x1))
    xx=ybr[ml]
    yy=phir[ml]
    y2=np.interp(xx,x1,y1)
    print(np.median(np.abs(y2-yy)))
    print('    '+str(stats.pearsonr(yy,y2)[0]))
    print()
    
    #['0',r'$\sqrt{3}/6$',r'$\sqrt{3}/3$']
    if 'U' in var:
        ax01.set_xticks([0,np.sqrt(3)/6,np.sqrt(3)/3],[])
        ax01.set_ylabel(r'                 $\Phi_u$')
    else:
        ax01.set_xticks([0,np.sqrt(3)/6,np.sqrt(3)/3],['0',r'$\frac{\sqrt{3}}{6}$',r'$\frac{\sqrt{3}}{3}$'])
        ax01.set_xlabel('$y_b$')
        ax01.set_ylabel(r'                 $\Phi_w$')
    ax01.set_yticks([0,2,4]) 

    ax10.set_title(titles[var])
    
    ax10.plot(x,y-y1,color='k')
    ax10.plot([-1,1],[0,0],color='w',zorder=0,linewidth=3)
    ax10.scatter(yb,phi-ymx(yb,params[0],params[1]),s=.025,alpha=.25,color='slategrey')
    ax10.set_xlim(0,.8)
    ax10.set_ylim(-2,2)
    if 'U' in var:
        ax10.set_xticks([0,np.sqrt(3)/6,np.sqrt(3)/3],[])
    else:
        ax10.set_xticks([0,np.sqrt(3)/6,np.sqrt(3)/3],['0',r'$\frac{\sqrt{3}}{6}$',r'$\frac{\sqrt{3}}{3}$'])
    if var=='Us':
        point1 = Line2D([0], [0], label='$\Phi_{true}$', markersize=4, 
         markeredgecolor='dimgrey', markerfacecolor='dimgrey', linestyle='',marker='o',alpha=.8)
        point2 = Line2D([0], [0], label='$\Phi_{random}$', markersize=4, 
         markeredgecolor='darkgrey', markerfacecolor='darkgrey', linestyle='',marker='o',alpha=.8)
        point3 = Line2D([0], [0], label='$\Phi_{true}-fit_{random}$', markersize=4, 
         markeredgecolor='slategrey', markerfacecolor='slategrey', linestyle='',marker='o',alpha=.8)
        line = Line2D([0], [0], label='fit line', color='k')
        legs=[point1,point2,point3,line]
        ax10.legend(handles=legs,labelspacing=.2,borderpad=.2)
    
    plt.subplots_adjust(wspace=0.3,hspace=0.1)
#plt.savefig('../../plot_output/a1_img_selfsim.png', bbox_inches = "tight")

# %%

# %%
sz=1
alpha=.05
fig=plt.figure(figsize=(9*sz,5*sz))
sbf = fig.subfigures(2, 2, hspace=0,wspace=0,frameon=False)

titles={'Uu':r'U: $\zeta <0$', 
        'Us':r'U: $\zeta >0$',
        'Wu':r'W: $\zeta <0$',
        'Ws':r'W: $\zeta >0$',
        'Vu':r'V: $\zeta <0$',
        'Vs':r'V: $\zeta >0$'}

for var in ['Uu','Us','Wu','Ws']:
    # plotting setup
    match var:
        case 'Uu': i=0; j=0
        case 'Us': i=0; j=1
        case 'Wu': i=1; j=0
        case 'Ws': i=1; j=1
        case 'Vu': i=1; j=0
        case 'Vs': i=1; j=1
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
    x1,y1=getlines(ybr,phir,np.linspace(0.05,.7,30))
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
        case 'Vu': 
            fp=fpu
            m=np.random.randint(low=0,high=len(fp['TIME'][:]),size=(N,))
            m.sort()
            phi=np.sqrt(fp['VV'][:][m])/fp['USTAR'][:][m]
        case 'Vs': 
            fp=fps
            m=np.random.randint(low=0,high=len(fp['TIME'][:]),size=(N,))
            m.sort()
            phi=np.sqrt(fp['VV'][:][m])/fp['USTAR'][:][m]
    yb=fp['ANI_YB'][:][m]
    x,y=getlines(yb,phi,np.linspace(0.05,.7,30))
    
    # plot
    ax00.scatter(yb,phi,s=1,alpha=alpha,color='slategrey')
    ax00.set_xlim(0,.8)
    ax00.set_ylim(0,5)
    ax00.plot(x,y,color='k')
    ax00.set_xticks([0,np.sqrt(3)/6,np.sqrt(3)/3],[])
    ax00.set_yticks([0,2,4])
    
    ax01.scatter(ybr,phir,s=1,alpha=alpha,color='darkgrey')
    ax01.set_xlim(0,.8)
    ax01.set_ylim(0,5)
    #ax01.plot(x1,y1,color='dimgrey')
    params,pcov=optimize.curve_fit(ymx,ybr,phir,[1,1],bounds=([-10,-2],[10,2]),loss='cauchy')
    y1=ymx(x1,params[0],params[1])
    ax01.plot(x1,y1,'--',color='k')
    #['0',r'$\sqrt{3}/6$',r'$\sqrt{3}/3$']
    if 'U' in var:
        ax01.set_xticks([0,np.sqrt(3)/6,np.sqrt(3)/3],[])
        ax01.set_ylabel(r'                 $\Phi_u$')
    else:
        ax01.set_xticks([0,np.sqrt(3)/6,np.sqrt(3)/3],['0',r'$\frac{\sqrt{3}}{6}$',r'$\frac{\sqrt{3}}{3}$'])
        ax01.set_xlabel('$y_b$')
        ax01.set_ylabel(r'                 $\Phi_w$')
    ax01.set_yticks([0,2,4]) 

    ax10.set_title(titles[var])
    
    ax10.plot(x,y,color='k')
    ax10.plot(x1,y1,'--',color='k')
    ax10.scatter(yb,phi,s=1,alpha=.1,color='slategrey')
    ax10.set_xlim(0,.8)
    ax10.set_ylim(0,5)
    
    if 'U' in var:
        ax10.set_xticks([0,np.sqrt(3)/6,np.sqrt(3)/3],[])
    else:
        ax10.set_xticks([0,np.sqrt(3)/6,np.sqrt(3)/3],['0',r'$\frac{\sqrt{3}}{6}$',r'$\frac{\sqrt{3}}{3}$'])
    if var=='Us':
        point1 = Line2D([0], [0], label='$\Phi_{true}$', markersize=4, 
         markeredgecolor='slategrey', markerfacecolor='slategrey', linestyle='',marker='o',alpha=.8)
        point2 = Line2D([0], [0], label='$\Phi_{random}$', markersize=4, 
         markeredgecolor='darkgrey', markerfacecolor='darkgrey', linestyle='',marker='o',alpha=.8)
        line = Line2D([0], [0], label='$fit_{true}$', color='k')
        line2 = Line2D([0], [0], label='$fit_{random}$', color='k',linestyle='--')
        legs=[point1,point2,line,line2]
        ax10.legend(handles=legs,labelspacing=.2,borderpad=.2,loc='upper left')
    
    plt.subplots_adjust(wspace=0.3,hspace=0.1)


# %%
sz=1
alpha=.05
vx=150
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
        phir,ybr=get_random(var,num=150000)
    elif 'u' in var:
        phir,ybr=get_random(var,num=100000)
    N=len(phir)
    x1,y1=getlines(ybr,phir,np.linspace(0.01,.7,50))
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
    x,y=getlines(yb,phi,np.linspace(0.01,.7,50))
    
    # plot
    #ax00.scatter(yb,phi,s=1,alpha=alpha,color='dimgrey')
    cmap='nipy_spectral'
    ax00.hexbin(yb,phi,gridsize=100,mincnt=5,cmap=cmap,extent=(0,.8,0,6),vmax=vx,vmin=5)
    ax00.set_xlim(0,.8)
    ax00.set_ylim(0,6)
    ax00.plot(x,y,color='k')
    ax00.set_xticks([0,np.sqrt(3)/6,np.sqrt(3)/3],[])
    ax00.set_yticks([0,2,4])
    
    ax01.hexbin(ybr,phir,gridsize=100,mincnt=5,cmap=cmap,extent=(0,.8,0,6),vmax=vx,vmin=5)
    ax01.set_xlim(0,.8)
    ax01.set_ylim(0,6)
    ax01.plot(x1,y1,color='dimgrey')
    params,pcov=optimize.curve_fit(ymx,ybr,phir,[1,1],bounds=([-10,-2],[10,2]),loss='cauchy')
    #y1=ymx(x1,params[0],params[1])
    #ax01.plot(x1,y1,color='k')
    #['0',r'$\sqrt{3}/6$',r'$\sqrt{3}/3$']
    if 'U' in var:
        ax01.set_xticks([0,np.sqrt(3)/6,np.sqrt(3)/3],[])
        ax01.set_ylabel(r'                 $\Phi_u$')
    else:
        ax01.set_xticks([0,np.sqrt(3)/6,np.sqrt(3)/3],['0',r'$\frac{\sqrt{3}}{6}$',r'$\frac{\sqrt{3}}{3}$'])
        ax01.set_xlabel('$y_b$')
        ax01.set_ylabel(r'                 $\Phi_w$')
    ax01.set_yticks([0,2,4]) 

    ax10.set_title(titles[var])
    
    ax10.plot(x,y-y1,color='k')
    ax10.plot([-1,1],[0,0],color='w',zorder=0,linewidth=3)
    ax10.hexbin(yb,phi-ymx(yb,params[0],params[1]),gridsize=100,mincnt=2,cmap=cmap,extent=(0,.8,-2,2))
    #ax10.scatter(yb,phi-ymx(yb,params[0],params[1]),s=1,alpha=.1,color='slategrey')
    ax10.set_xlim(0,.8)
    ax10.set_ylim(-2,2)
    if 'U' in var:
        ax10.set_xticks([0,np.sqrt(3)/6,np.sqrt(3)/3],[])
    else:
        ax10.set_xticks([0,np.sqrt(3)/6,np.sqrt(3)/3],['0',r'$\frac{\sqrt{3}}{6}$',r'$\frac{\sqrt{3}}{3}$'])
    if var=='Us':
        point1 = Line2D([0], [0], label='$\Phi_{true}$', markersize=4, 
         markeredgecolor='dimgrey', markerfacecolor='dimgrey', linestyle='',marker='o',alpha=.8)
        point2 = Line2D([0], [0], label='$\Phi_{random}$', markersize=4, 
         markeredgecolor='darkgrey', markerfacecolor='darkgrey', linestyle='',marker='o',alpha=.8)
        point3 = Line2D([0], [0], label='$\Phi_{true}-fit_{random}$', markersize=4, 
         markeredgecolor='slategrey', markerfacecolor='slategrey', linestyle='',marker='o',alpha=.8)
        line = Line2D([0], [0], label='fit line', color='k')
        legs=[point1,point2,point3,line]
        ax10.legend(handles=legs,labelspacing=.2,borderpad=.2)
    
    plt.subplots_adjust(wspace=0.3,hspace=0.1)
plt.savefig('../../plot_output/a1_img_selfsim2.png', bbox_inches = "tight")

# %%

# %%
sz=1
alpha=.05
vx=100
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
        phir,ybr=get_random_xb(var,num=150000)
    elif 'u' in var:
        phir,ybr=get_random_xb(var,num=100000)
    N=len(phir)
    x1,y1=getlines(ybr,phir,np.linspace(0.1,.9,50))
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
    yb=fp['ANI_XB'][:][m]
    x,y=getlines(yb,phi,np.linspace(0.1,.9,50))
    
    # plot
    #ax00.scatter(yb,phi,s=1,alpha=alpha,color='dimgrey')
    ax00.hexbin(yb,phi,gridsize=100,mincnt=5,cmap='terrain',extent=(0,1,0,6),vmax=vx,vmin=5)
    ax00.set_xlim(0,1)
    ax00.set_ylim(0,6)
    ax00.plot(x,y,color='k')
    ax00.set_xticks([0,np.sqrt(3)/6,np.sqrt(3)/3],[])
    ax00.set_yticks([0,2,4])
    
    ax01.hexbin(ybr,phir,gridsize=100,mincnt=5,cmap='terrain',extent=(0,1,0,6),vmax=vx,vmin=5)
    ax01.set_xlim(0,1)
    ax01.set_ylim(0,6)
    ax01.plot(x1,y1,color='dimgrey')
    params,pcov=optimize.curve_fit(ymx,ybr,phir,[1,1],bounds=([-10,-2],[10,2]),loss='cauchy')
    #y1=ymx(x1,params[0],params[1])
    #ax01.plot(x1,y1,color='k')
    #['0',r'$\sqrt{3}/6$',r'$\sqrt{3}/3$']
    if 'U' in var:
        ax01.set_xticks([0,np.sqrt(3)/6,np.sqrt(3)/3],[])
        ax01.set_ylabel(r'                 $\Phi_u$')
    else:
        ax01.set_xticks([0,np.sqrt(3)/6,np.sqrt(3)/3],['0',r'$\frac{\sqrt{3}}{6}$',r'$\frac{\sqrt{3}}{3}$'])
        ax01.set_xlabel('$y_b$')
        ax01.set_ylabel(r'                 $\Phi_w$')
    ax01.set_yticks([0,2,4]) 

    ax10.set_title(titles[var])
    
    ax10.plot(x,y-y1,color='k')
    ax10.plot([-1,1],[0,0],color='w',zorder=0,linewidth=3)
    ax10.hexbin(yb,phi-ymx(yb,params[0],params[1]),gridsize=100,mincnt=2,cmap='terrain',extent=(0,.8,-2,2))
    #ax10.scatter(yb,phi-ymx(yb,params[0],params[1]),s=1,alpha=.1,color='slategrey')
    ax10.set_xlim(0,1)
    ax10.set_ylim(-2,2)
    if 'U' in var:
        ax10.set_xticks([0,np.sqrt(3)/6,np.sqrt(3)/3],[])
    else:
        ax10.set_xticks([0,np.sqrt(3)/6,np.sqrt(3)/3],['0',r'$\frac{\sqrt{3}}{6}$',r'$\frac{\sqrt{3}}{3}$'])
    if var=='Us':
        point1 = Line2D([0], [0], label='$\Phi_{true}$', markersize=4, 
         markeredgecolor='dimgrey', markerfacecolor='dimgrey', linestyle='',marker='o',alpha=.8)
        point2 = Line2D([0], [0], label='$\Phi_{random}$', markersize=4, 
         markeredgecolor='darkgrey', markerfacecolor='darkgrey', linestyle='',marker='o',alpha=.8)
        point3 = Line2D([0], [0], label='$\Phi_{true}-fit_{random}$', markersize=4, 
         markeredgecolor='slategrey', markerfacecolor='slategrey', linestyle='',marker='o',alpha=.8)
        line = Line2D([0], [0], label='fit line', color='k')
        legs=[point1,point2,point3,line]
        ax10.legend(handles=legs,labelspacing=.2,borderpad=.2)
    
    plt.subplots_adjust(wspace=0.3,hspace=0.1)


# %%

# %% [markdown]
# # XB SCALING

# %%
def plotit(xplt,yplt,aplt,xplts,yplts,aplts,old,olds,ylim=[1.5,6]):
    zL_u=-np.logspace(-4,2,40)
    zL=zL_u.copy()
    plt.figure(figsize=(8,2.5))
    plt.subplot(1,2,1)
    for i in range(yplt.shape[0]):
        plt.semilogx(-xplt,yplt[i,:],color=cani_norm(aplt[i]),linewidth=2,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])

    plt.semilogx(-zL,old,'k--')
    ax=plt.gca()
    ax.tick_params(which="both", bottom=True)
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )) 
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([10**-3,10**-2,10**-1,1,10],[r'$-10^{-3}$','',r'$-10^{-1}$','',r'$-10^{1}$'])
    ax.set_xlim(10**(-3.5),10**(1.6))
    plt.ylim(ylim[0],ylim[1])
    plt.gca().invert_xaxis()
    
    plt.subplot(1,2,2)
    zL_s=np.logspace(-4,2,40)
    zL=zL_s.copy()
    for i in range(yplts.shape[0]):
        plt.semilogx(xplts,yplts[i,:],color=cani_norm(aplts[i]),linewidth=2,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
    plt.semilogx(zL,olds,'k--')
    
    ax=plt.gca()
    ax.tick_params(which="both", bottom=True)
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )) 
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([10**-3,10**-2,10**-1,1,10],[r'$10^{-3}$','',r'$10^{-1}$','',r'$10^{1}$'])
    ax.set_xlim(10**(-3.5),10**(1.6))
    plt.ylim(ylim[0],ylim[1])
anibins=np.linspace(vmn_a,vmx_a,11)
zLbins=np.logspace(-4,2,40)[-1:0:-1]
def binplot1d(xx,yy,ani,stb,xbins=-zLbins,anibins=anibins):
     if stb:
         xbins=-xbins[-1:0:-1]
     xplot=(xbins[0:-1]+xbins[1:])/2
     yplot=np.zeros((len(anibins)-1,len(xplot)))
     count=np.zeros((len(anibins)-1,len(xplot)))
     aniplot=(anibins[0:-1]+anibins[1:])/2
     for i in range(len(anibins)-1):
         #print('ANI: '+str(anibins[i]))
         for j in range(len(xbins)-1):
             #print('  ZL: '+str(xbins[j]))
             pnts=yy[(ani>=anibins[i])&(ani<anibins[i+1])&(xx>=xbins[j])&(xx<xbins[j+1])]
             count[i,j]=len(pnts)
             yplot[i,j]=np.nanmedian(pnts)
     return xplot,yplot,aniplot,count



# %%
xblims=[0,.3,.45,.6,1]
minpct=1e-4
phi=np.sqrt(fpu['UU'][:])/fpu['USTAR'][:]
zL=(fpu['zzd'][:])/fpu['L_MOST'][:]
ani=fpu['ANI_YB'][:]
xb=fpu['ANI_XB'][:]

phis=np.sqrt(fps['UU'][:])/fps['USTAR'][:]
zLs=(fps['zzd'][:])/fps['L_MOST'][:]
anis=fps['ANI_YB'][:]
xbs=fps['ANI_XB'][:]

xplt={'U':[],'V':[],'W':[]}
yplt={'U':[],'V':[],'W':[]}
aplt={'U':[],'V':[],'W':[]}
xplts={'U':[],'V':[],'W':[]}
yplts={'U':[],'V':[],'W':[]}
aplts={'U':[],'V':[],'W':[]}

for i in range(4):
    for v in ['U','V','W']:
        match v:
            case 'U': phi=np.sqrt(fpu['UU'][:])/fpu['USTAR'][:]; phis=np.sqrt(fps['UU'][:])/fps['USTAR'][:]
            case 'V': phi=np.sqrt(fpu['VV'][:])/fpu['USTAR'][:]; phis=np.sqrt(fps['VV'][:])/fps['USTAR'][:]
            case 'W': phi=np.sqrt(fpu['WW'][:])/fpu['USTAR'][:]; phis=np.sqrt(fps['WW'][:])/fps['USTAR'][:]
        print(v)
        m=(xb>xblims[i])&(xb<xblims[i+1])
        
        xplt_,yplt_,aplt_,cnt=binplot1d(zL[m],phi[m],ani[m],False)
        tot=np.sum(cnt)
        yplt_[cnt/tot<minpct]=float('nan')
    
        xplt[v].append(xplt_)
        yplt[v].append(yplt_)
        aplt[v].append(aplt_)
    
        m=(xbs>xblims[i])&(xbs<xblims[i+1])
        
        xplts_,yplts_,aplts_,cnts=binplot1d(zLs[m],phis[m],anis[m],True)
        tot=np.sum(cnts)
        yplts_[cnts/tot<minpct]=float('nan')
    
        xplts[v].append(xplts_)
        yplts[v].append(yplts_)
        aplts[v].append(aplts_)

# %%
sz=1.25
fig=plt.figure(figsize=(9*sz,5*sz))
sbf = fig.subfigures(1, 3, hspace=0,wspace=0,frameon=False)
axsu=sbf[0].subplots(4,2)
plt.subplots_adjust(wspace=0.02)

axsv=sbf[1].subplots(4,2)
plt.subplots_adjust(wspace=0.02)

axsw=sbf[2].subplots(4,2)
plt.subplots_adjust(wspace=0.02)

sbf[0].suptitle('$\Phi_u$')
sbf[1].suptitle('$\Phi_v$')
sbf[2].suptitle('$\Phi_w$')

axs=[axsu,axsv,axsw]
for i in range(3):
    v=['U','V','W'][i]
    for j in range(4):
        ax1=axs[i][j,0]
        ax2=axs[i][j,1]

        zL_u=-np.logspace(-4,2,40)
        zL=zL_u.copy()

        match v:
            case 'U': oldu=ust_old[0][0,:]; olds=stb_old[0][0,:];ylim=[1.5,6]
            case 'V': oldu=ust_old[1][0,:]; olds=stb_old[1][0,:];ylim=[1.25,6]
            case 'W': oldu=ust_old[2][0,:]; olds=stb_old[2][0,:];ylim=[.8,3]
        
        for k in range(yplt[v][j].shape[0]):
            ax1.semilogx(-xplt[v][j][:],yplt[v][j][k,:],color=cani_norm(aplt[v][j][k]),linewidth=1)
        
        ax1.semilogx(-zL,oldu,'k--')
        ax1.tick_params(which="both", bottom=True)
        #locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )) 
        #ax1.xaxis.set_minor_locator(locmin)
        #ax1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax1.set_xticks([10**-3,10**-2,10**-1,1,10],[])#,[r'$-10^{-3}$','',r'$-10^{-1}$','',r'$-10^{1}$'])
        ax1.set_xlim(10**(-3.5),10**(1.6))
        if v=='W':
            ax1.set_yticks([1,2],[1,2])
            ax1.set_ylim(ylim[0],ylim[1])
        else:
            ax1.set_yticks([2.5,5],[2.5,5])
            ax1.set_ylim(ylim[0],ylim[1])
        ax1.invert_xaxis()

        for k in range(yplts[v][j].shape[0]):
            ax2.semilogx(xplts[v][j][:],yplts[v][j][k,:],color=cani_norm(aplts[v][j][k]),linewidth=1)
        
        ax2.semilogx(-zL,olds,'k--')
        ax2.tick_params(which="both", bottom=True)
        #locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )) 
        #ax1.xaxis.set_minor_locator(locmin)
        #ax1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax2.set_xticks([10**-3,10**-2,10**-1,1,10],[])#,[r'$-10^{-3}$','',r'$-10^{-1}$','',r'$-10^{1}$'])
        ax2.set_xlim(10**(-3.5),10**(1.6))
        if v=='W':
            ax2.set_yticks([1,2],[])
            ax2.set_ylim(ylim[0],ylim[1])
        else:
            ax2.set_yticks([2.5,5],[])
            ax2.set_ylim(ylim[0],ylim[1])

        if j==3:
            ax1.set_xlabel(r'                               $\zeta$')
            ax2.set_xticks([10**-3,10**-2,10**-1,1,10],['',r'$10^{-2}$','','',r'$10^{1}$'])
            ax1.set_xticks([10**-3,10**-2,10**-1,1,10],['',r'$-10^{-2}$','','',r'$-10^{1}$'])
        if i==0:
            ax1.set_ylabel('$\Phi_x$')
plt.savefig('../../plot_output/a1_img_10.png', bbox_inches = "tight",transparent=True)

# %%

# %% [markdown]
# # FIGURE XB Distribution Plot

# %%
mads=[]
fpsites=fps['SITE'][:]
for site in np.unique(fpsites):
    mads.append(d_s['U']['MAD_OLD_s'][str(site)[2:-1]])
mads=np.array(mads)
mad_norm=(mads-.25)/(.52-.3)
cc=pl.cm.coolwarm(mad_norm)
sm = plt.cm.ScalarMappable(cmap=pl.cm.coolwarm, norm=plt.Normalize(vmin=.25, vmax=.52))
i=0
fig=plt.figure(figsize=(3,4))
sbf = fig.subfigures(2, 1, hspace=0,wspace=0,frameon=False)
ax=sbf[1].add_subplot(111)
fpsites=fps['SITE'][:]
for site in np.unique(fpsites):
    m=fpsites==site
    y,binEdges=np.histogram(fps['ANI_XB'][m],bins=np.linspace(0,1),density=True)
    bincenters=.5*(binEdges[1:]+binEdges[:-1])
    ax.plot(bincenters,y,c=cc[i],linewidth=1,alpha=.75)
    i=i+1
ax.set_xlabel(r'$x_B$')
ax.set_ylabel('Frequency')
sbf[1].colorbar(sm,cax=ax.inset_axes([0.95, 0, 0.05, 1]),label='$MAD$')

mads=[]
for site in np.unique(fpsites):
    mads.append(d_u['U']['MAD_OLD_s'][str(site)[2:-1]])
mads=np.array(mads)
fpsites=fpu['SITE'][:]
mad_norm=(mads-.3)/(1-.3) #UNSTABLE
#mad_norm=(mads-.35)/(.6-.35)
cc=pl.cm.coolwarm(mad_norm)
sm = plt.cm.ScalarMappable(cmap=pl.cm.coolwarm, norm=plt.Normalize(vmin=.3, vmax=1))
i=0
ax=sbf[0].add_subplot(111)
for site in np.unique(fpsites):
    m=fpsites==site
    y,binEdges=np.histogram(fpu['ANI_XB'][m],bins=np.linspace(0,1,20),density=True)
    bincenters=.5*(binEdges[1:]+binEdges[:-1])
    ax.plot(bincenters,y,c=cc[i],linewidth=1,alpha=.75)
    i=i+1
    #plt.hist(fpu['ANI_XB'][m],bins=np.linspace(0,1))
    #ss=str(site)[2:-1]
    #plt.title(str(site)+': '+str(d_u['U']['MAD_OLD_s'][ss])[0:5])
#ax.set_xlabel(r'$x_B$')
ax.set_xticks([0,.25,.5,.75,1],[])
ax.set_ylabel('Frequency')
sbf[0].colorbar(sm,cax=ax.inset_axes([0.95, 0, 0.05, 1]),label='$MAD$')

plt.subplots_adjust(hspace=0)
#fig.savefig('trash.png',bbox_inches = "tight")

#fig.savefig('trash.png',bbox_inches = "tight")

# %% [markdown]
# # FIGURE YB Distribution Plot

# %%
i=0
fig=plt.figure(figsize=(3,4))
sbf = fig.subfigures(2, 1, hspace=0,wspace=0,frameon=False)
ax=sbf[1].add_subplot(111)
fpsites=fps['SITE'][:]
for site in np.unique(fpsites):
    m=fpsites==site
    ss=str(site)[2:-1]
    ffu=h5py.File('/home/tswater/tyche/data/neon/foot_stats/'+ss+'_U.h5','r')
    c=stats.mode(ffu['nlcd_dom'])[0]
    try:
        colors=class_colors[c]
    except Exception as e:
        colors='darkgreen'
        print(e)
    y,binEdges=np.histogram(fps['ANI_YB'][m],bins=np.linspace(0,.87),density=True)
    bincenters=.5*(binEdges[1:]+binEdges[:-1])
    ax.plot(bincenters,y,c=colors,linewidth=1,alpha=.75)
    i=i+1
ax.set_xlabel(r'$y_B$')
ax.set_ylabel('Frequency')
ax.set_xticks([0,.25,.5,.75])
ax.set_xlim(-.05,.8)

i=0
ax=sbf[0].add_subplot(111)
fpsites=fpu['SITE'][:]
for site in np.unique(fpsites):
    m=fpsites==site
    ss=str(site)[2:-1]
    ffu=h5py.File('/home/tswater/tyche/data/neon/foot_stats/'+ss+'_U.h5','r')
    c=stats.mode(ffu['nlcd_dom'])[0]
    try:
        colors=class_colors[c]
    except Exception as e:
        colors='darkgreen'
        print(e)
    y,binEdges=np.histogram(fpu['ANI_YB'][m],bins=np.linspace(0,.87,20),density=True)
    bincenters=.5*(binEdges[1:]+binEdges[:-1])
    ax.plot(bincenters,y,c=colors,linewidth=1,alpha=.75)
    i=i+1
    #plt.hist(fpu['ANI_XB'][m],bins=np.linspace(0,1))
    #ss=str(site)[2:-1]
    #plt.title(str(site)+': '+str(d_u['U']['MAD_OLD_s'][ss])[0:5])
#ax.set_xlabel(r'$x_B$')
ax.set_xticks([0,.25,.5,.75],[])
ax.set_xlim(-.05,.8)
ax.set_ylabel('Frequency')
#sbf[0].colorbar(sm,cax=ax.inset_axes([0.95, 0, 0.05, 1]),label='$MAD$')
plt.savefig('../../plot_output/a1/a1_yb_site_distrib.png', bbox_inches = "tight")


# %%

# %% [markdown]
# # FIGURE Lumley Error

# %%
def getbins(A,n):
     B=np.sort(A)
     bins=[]
     for i in np.linspace(0,len(A)-1,n):
         i=int(i)
         bins.append(B[i])
     return bins
def getbins2D(A,B,n):
    bina=getbins(A,n)
    binb=np.zeros((n-1,n))
    for i in range(n-1):
        m=(A>bina[i])&(A<bina[i+1])
        binb[i,:]=getbins(B[m],n)
    return bina,binb
def old_phi(zLL,a,b):
    return a*(1-3*zLL)**(b)


# %%
Nbine=31
vari=['U','V','W']
xbtrue=np.ones((3,2,Nbine-1,Nbine-1))*float('nan')
ybtrue=np.ones((3,2,Nbine-1,Nbine-1))*float('nan')
error=np.ones((3,2,Nbine-1,Nbine-1))*float('nan')

for i in range(len(vari)):
    v=vari[i]
    print(v)
    for j in range(2):
        print(j)
        if j==0:
            fp=fpu
        elif j==1:
            fp=fps
        ybine,xbine=getbins2D(fp['ANI_YB'][:],fp['ANI_XB'][:],Nbine)
        phi_=np.sqrt(fp[v+v][:])/fp['USTAR'][:]
        zL_=(fp['zzd'][:])/fp['L_MOST'][:]
        xb_=fp['ANI_XB'][:]
        yb_=fp['ANI_YB'][:]
        match v:
            case 'U': oldu=old_phi(zL_,2.55,1/3); olds=old_phi(zL_,2.06,0);ylim=[1.5,6]
            case 'V': oldu=old_phi(zL_,2.05,1/3); olds=old_phi(zL_,2.06,0);ylim=[1.25,6]
            case 'W': oldu=old_phi(zL_,1.35,1/3); olds=old_phi(zL_,1.6,0);ylim=[.8,3]
        if j==0:
            old=oldu
        elif j==1:
            old=olds
        for ii in range(len(ybine)-1):
            for jj in range(len(xbine[0])-1):
                #print(str(ii)+','+str(jj)+'   ',end='',flush=True)
                m=(xb_<xbine[ii,jj+1])&(xb_>xbine[ii,jj])&(yb_<ybine[ii+1])&(yb_>ybine[ii])
                xbtrue[i,j,ii,jj]=np.nanmean(xb_[m])
                ybtrue[i,j,ii,jj]=np.nanmean(yb_[m])
                error[i,j,ii,jj]=-np.nanmedian(phi_[m]-old[m])

# %%

# %%
sz=1.6
fig,axs=plt.subplots(3,2,figsize=(4*sz,4.5*sz))
vari=['U','V','W']
stbs=[-1,1]
for i in range(len(vari)):
    for j in range(2):
        vs=vari[i]+str(j)
        v=vari[i]
        #match vs:
        #    case :vmin=;vmax=
        ax=axs[i,j]
        xc = np.array([0, 1, 0.5])
        yc = np.array([0, 0, np.sqrt(3)*0.5])
        ccc=['k','r','b']
        for k in np.arange(3):
            ip1 = (k+1)%3
            ax.plot([xc[k], xc[ip1]], [yc[k], yc[ip1]], 'k', linewidth=2)
            ax.fill_between([xc[k], xc[ip1]], [yc[k], yc[ip1]],y2=[0,0],color=(0.9176470588235294, 0.9176470588235294, 0.9490196078431372, 1.0),zorder=0)
            
        xbtick=[.2,.4,.6,.8]
        ybtick=[0,np.sqrt(3)/8,np.sqrt(3)/4,3*np.sqrt(3)/8,np.sqrt(3)/2]
        for ii in range(5):
            if ii in[0,2,4]: nm=.05
            else: nm=0
            ax.plot([ybtick[ii]/(np.sqrt(3)/2)/2-nm,1-ybtick[ii]/(np.sqrt(3)/2)/2],[ybtick[ii],ybtick[ii]],'--k',linewidth=.5)
        for ii in range(4):
            if ii<2:
                ulim=xbtick[ii]
            else:
                ulim=1-xbtick[ii]
            ax.plot([xbtick[ii],xbtick[ii]],[-0.025,np.sqrt(3)/2*(ulim/.5)],'--k',linewidth=.5)

        c1ps = np.array([2/3, 1/3, 0])
        x1ps = np.dot(xc, c1ps.transpose())
        y1ps = np.dot(yc, c1ps.transpose())
        c2ps = np.array([0, 0, 1])
        x2ps = np.dot(xc, c2ps.transpose())
        y2ps = np.dot(yc, c2ps.transpose())
        ax.plot([x1ps, x2ps], [y1ps, y2ps], '-k', linewidth=2)
        
        #for k in np.arange(3):
        #    ax.text(lbpx[k], lbpy[k], labels[k], ha='center', fontsize=12)
        label_side1 = 'Prolate'
        label_side2 = 'Oblate'
        label_side3 = 'Two-component'
        #axs.text((xc[1]+xc[2])/2, (yc[0]+yc[2])/2+0.08*lc, label_side1, ha='center', va='center', rotation=-65)
        #axs.text((xc[0]+xc[2])/2, (yc[1]+yc[2])/2+0.08*lc, label_side2, ha='center', va='center', rotation=65)
        #axs.text((xc[0]+xc[1])/2, (yc[0]+yc[1])/2-0.04*lc, label_side3, ha='center', va='center')
        print(vs)
        print(np.nanmin(error[i,j,0:,:]))
        print(np.nanmax(error[i,j,0:,:]))
        data=error[i,j,:,:]
        vmax=max(np.abs(np.nanpercentile(data,5)),np.abs(np.nanpercentile(data,95)))
        im=ax.pcolormesh(xbtrue[i,j,0:,:],ybtrue[i,j,0:,:],data,cmap='Spectral_r',shading='gouraud',vmin=-vmax,vmax=vmax)
        cb=fig.colorbar(im, cax=ax.inset_axes([0.95, 0.05, 0.05, .92]),label='$Bias$ $\Phi_'+v+'$')
        cb.ax.zorder=100
        ax.patch.set_alpha(0)
        ax.grid(False)
        if j==0:
            ax.set_yticks([0,np.sqrt(3)/4,np.sqrt(3)/2],['0',r'$\sqrt{3}/4$',r'$\sqrt{3}/2$'])
            ax.set_ylabel('$y_b$')
        if j==1:
            ax.set_yticks([],[])
        if i==2:
            ax.set_xticks([0,.2,.4,.6,.8,1],['',.2,.4,.6,.8,''])
            ax.set_xlabel('$x_b$')
        else:
            ax.set_xticks([],[])
        ax.spines[['left', 'bottom']].set_visible(False)
    plt.subplots_adjust(wspace=.4)
    fig.suptitle('      Unstable ($\zeta<0$)                     Stable ($\zeta>0$)      ')
plt.savefig('../../plot_output/a1_img_8bias.png', bbox_inches = "tight")

# %%

# %%
clrs=[]
xb_u=[]
xb_s=[]
yb_u=[]
yb_s=[]
k=0
for site in other[0]:
    try:
        color=class_colors[other[2][k]]
    except:
        color='darkgreen'
    m=bytes(site,'utf-8')==fpu['SITE'][:]
    clrs.append(color)
    xb_u.append(np.nanmean(fpu['ANI_XB'][m]))
    yb_u.append(np.nanmean(fpu['ANI_YB'][m]))
    m=bytes(site,'utf-8')==fps['SITE'][:]
    xb_s.append(np.nanmean(fps['ANI_XB'][m]))
    yb_s.append(np.nanmean(fps['ANI_YB'][m]))
    k=k+1


# %%

# %%
import matplotlib.colors as mcolors

def get_rgb(color_list):
    out=[]
    for c_c_ in color_list:
        out.append(mcolors.to_rgb(c_c_))
    return out
clrs=get_rgb(clrs)

# %%
plt.figure()
plt.scatter(xb_u,yb_u,c=clrs,s=10,zorder=10)
plt.figure()
plt.scatter(xb_s,yb_s,c=clrs,s=10,zorder=10)

# %%
sz=1.6
fig,axs=plt.subplots(4,2,figsize=(4*sz,4.5*sz))
vari=['A','U','V','W']
stbs=[-1,1]
for i in range(len(vari)):
    for j in range(2):
        vs=vari[i]+str(j)
        v=vari[i]
        #match vs:
        #    case :vmin=;vmax=
        ax=axs[i,j]
        xc = np.array([0, 1, 0.5])
        yc = np.array([0, 0, np.sqrt(3)*0.5])
        ccc=['k','r','b']
        for k in np.arange(3):
            ip1 = (k+1)%3
            ax.plot([xc[k], xc[ip1]], [yc[k], yc[ip1]], 'k', linewidth=2)
            ax.fill_between([xc[k], xc[ip1]], [yc[k], yc[ip1]],y2=[0,0],color=(0.9176470588235294, 0.9176470588235294, 0.9490196078431372, 1.0),zorder=0)
            
        xbtick=[.2,.4,.6,.8]
        ybtick=[0,np.sqrt(3)/8,np.sqrt(3)/4,3*np.sqrt(3)/8,np.sqrt(3)/2]
        for ii in range(5):
            if ii in[0,2,4]: nm=.05
            else: nm=0
            ax.plot([ybtick[ii]/(np.sqrt(3)/2)/2-nm,1-ybtick[ii]/(np.sqrt(3)/2)/2],[ybtick[ii],ybtick[ii]],'--k',linewidth=.5)
        for ii in range(4):
            if ii<2:
                ulim=xbtick[ii]
            else:
                ulim=1-xbtick[ii]
            ax.plot([xbtick[ii],xbtick[ii]],[-0.025,np.sqrt(3)/2*(ulim/.5)],'--k',linewidth=.5)

        c1ps = np.array([2/3, 1/3, 0])
        x1ps = np.dot(xc, c1ps.transpose())
        y1ps = np.dot(yc, c1ps.transpose())
        c2ps = np.array([0, 0, 1])
        x2ps = np.dot(xc, c2ps.transpose())
        y2ps = np.dot(yc, c2ps.transpose())
        ax.plot([x1ps, x2ps], [y1ps, y2ps], '-k', linewidth=2)
        
        #for k in np.arange(3):
        #    ax.text(lbpx[k], lbpy[k], labels[k], ha='center', fontsize=12)
        label_side1 = 'Prolate'
        label_side2 = 'Oblate'
        label_side3 = 'Two-component'
        #axs.text((xc[1]+xc[2])/2, (yc[0]+yc[2])/2+0.08*lc, label_side1, ha='center', va='center', rotation=-65)
        #axs.text((xc[0]+xc[2])/2, (yc[1]+yc[2])/2+0.08*lc, label_side2, ha='center', va='center', rotation=65)
        #axs.text((xc[0]+xc[1])/2, (yc[0]+yc[1])/2-0.04*lc, label_side3, ha='center', va='center')
        if v=='A':
            if j==0:
                ax.scatter(xb_u,yb_u,c=clrs,s=2,zorder=10)
            else:
                ax.scatter(xb_s,yb_s,c=clrs,s=2,zorder=10)
        else:
            data=error[i-1,j,:,:]
            vmax=max(np.abs(np.nanpercentile(data,5)),np.abs(np.nanpercentile(data,95)))
            im=ax.pcolormesh(xbtrue[i-1,j,0:,:],ybtrue[i-1,j,0:,:],data,cmap='Spectral_r',shading='gouraud',vmin=-vmax,vmax=vmax)
            cb=fig.colorbar(im, cax=ax.inset_axes([0.95, 0.05, 0.05, .92]),label='$Bias$ $\Phi_'+v+'$')
            cb.ax.zorder=100
        ax.patch.set_alpha(0)
        ax.grid(False)
        if j==0:
            ax.set_yticks([0,np.sqrt(3)/4,np.sqrt(3)/2],['0',r'$\sqrt{3}/4$',r'$\sqrt{3}/2$'])
            ax.set_ylabel('$y_b$')
        if j==1:
            ax.set_yticks([],[])
        if i==2:
            ax.set_xticks([0,.2,.4,.6,.8,1],['',.2,.4,.6,.8,''])
            ax.set_xlabel('$x_b$')
        else:
            ax.set_xticks([],[])
        ax.spines[['left', 'bottom']].set_visible(False)
    plt.subplots_adjust(wspace=.4)
    fig.suptitle('      Unstable ($\zeta<0$)                     Stable ($\zeta>0$)      ')
#plt.savefig('../../plot_output/a1_img_8bias.png', bbox_inches = "tight")

# %%

# %% [markdown]
# # Figure XB vs Error all sites

# %%
sz=1.3
fig,axs=plt.subplots(3,2,figsize=(5*sz,4.5*sz))
vari=['U','V','W']
stbs=[-1,1]
for i in range(len(vari)):
    for j in range(2):
        if j==0:
            d_=d_uxb
        elif j==1:
            d_=d_sxb
        v=vari[i]
        typ='MD_SC23_s'
        k=0
        for site in d_[v][typ].keys():
            try:
                color=class_colors[other[2][k]]
            except:
                color='darkgreen'
            xbins=(np.array(d_[v]['xbins_s'][site][1:])+np.array(d_[v]['xbins_s'][site][:-1]))/2
            axs[i,j].semilogy(xbins,np.abs(d_[v][typ][site][:]),color=color,marker='o',markerfacecolor='none',linewidth=.5,markersize=1,alpha=.6)
            k=k+1
        if i==2:
            axs[i,j].set_xlabel(r'$x_b$')
            axs[i,j].set_xticks([0,.25,.5,.75,1],['',0.25,0.5,0.75,''])
        else:
            axs[i,j].set_xticks([0,.25,.5,.75,1],['','','','',''])
        if j==0:
            if i==0:
                axs[i,j].set_ylabel(r'$MAD\ \Phi_u$')
            if i==1:
                axs[i,j].set_ylabel(r'$MAD\ \Phi_v$')
            if i==2:
                axs[i,j].set_ylabel(r'$MAD\ \Phi_w$')
        axs[i,j].set_ylim(-10,10)
        axs[i,j].set_xlim(.1,.9)
fig.suptitle('      Unstable ($\zeta<0$)                     Stable ($\zeta>0$)      ')
plt.savefig('../../plot_output/a1_img_11.png', bbox_inches = "tight")

# %%
axs[i,j].get_facecolor()

# %%

# %% [markdown]
# # FIGURE XB/YB CORRELATION

# %%
for i in range(45):
    print(str(fpst['site'][i])+': '+str(fpst['zd'][i]))

# %%
grass=[]
for i in range(45):
    if (fpst['zd'][i]<2):
        grass.append(str(fpst['site'][i])[2:-1])

data=dc_u['spearmanr'][:]
datas=dc_s['spearmanr'][:]
xvars=dc_u['xvars'][:]
yvars=dc_u['yvars'][:]
yvarnames={'zL':r'$\zeta$','USTAR':r'$u*$','Ustr':'$\overline{U}$','Wstr':r'$\overline{w}$',
           'mean_chm':r'$h_c$','std_chm':r'$\sigma_{h_c}$','std_dsm':r'$\sigma_{DSM}$','NETRAD':r'$R_{net}$','H':r'$H$','RH':r'$RH$',\
           'TA':r'$T$','PA':r'$P$'}
ynmlist=[]
mark=['+','s','^','+','s','^']
color=['blue','red','green','cornflowerblue','lightcoral','limegreen']

sz=1.6
#reduce data
dataxb=np.zeros((2,len(yvarnames.keys())))
datayb=np.zeros((2,len(yvarnames.keys())))

dataxbs=np.zeros((2,len(yvarnames.keys()),47))
dataybs=np.zeros((2,len(yvarnames.keys()),47))

i=0
for yv in yvarnames.keys():
    ynmlist.append(yvarnames[yv])
    print(yv)
    idx=np.where(np.array(yvars)==yv)[0][0]
    dataxb[0,i]=data[0,idx]
    dataxb[1,i]=datas[0,idx]
    datayb[0,i]=data[1,idx]
    datayb[1,i]=datas[1,idx]
    j=0
    for site in dc_u['sitelevel'].keys():
        if (yv in ['mean_chm','std_chm','std_dsm'])&(site in []):
            dataybs[0,i,j]=float('nan')
            dataxbs[0,i,j]=float('nan')
            dataybs[1,i,j]=float('nan')
            dataxbs[1,i,j]=float('nan')
        else:
            dataybs[0,i,j]=dc_u['sitelevel'][site]['spearmanr'][1,idx]
            dataxbs[0,i,j]=dc_u['sitelevel'][site]['spearmanr'][0,idx]
            dataybs[1,i,j]=dc_s['sitelevel'][site]['spearmanr'][1,idx]
            dataxbs[1,i,j]=dc_s['sitelevel'][site]['spearmanr'][0,idx]
        j=j+1
    i=i+1

# %%
sz=2
fig,axs=plt.subplots(3,2,figsize=(4*sz,4.5*sz))
for i in range(2):
    if i==0:
        dall=datayb
        dsite=dataybs
    else:
        dall=dataxb
        dsite=dataxbs

    #allplot
    axs[0,i].bar(ynmlist,dall[0,:],color='steelblue')
    axs[0,i].bar(ynmlist,dall[1,:],width=.45,color='lightsteelblue')
    N=len(ynmlist)
    axs[0,i].set_xticks(np.linspace(0,N-1,N),[])
    axs[0,i].legend([r'$\zeta<0$',r'$\zeta>0$'])
    axs[0,i].plot([-1,N+1],[0,0],'w',linewidth=4,zorder=0)
    axs[0,i].set_ylim(-.55,.61)
    axs[0,i].set_xlim(-1,N)
    axs[0,0].set_ylabel('Correlation')

    sns.boxplot(dsite[0,:].T,ax=axs[1,i])
    axs[1,i].set_xticks(np.linspace(0,N,N),[])
    axs[1,i].plot([-1,N+1],[0,0],'w',linewidth=4,zorder=0)
    axs[1,i].set_xlim(-1,N)
    axs[1,i].set_ylim(-.55,.61)
    axs[1,0].set_ylabel('Site Correlation ($\zeta<0$)')
    
    sns.boxplot(dsite[1,:].T,ax=axs[2,i])
    axs[2,i].set_xticks(np.linspace(0,N-1,N),ynmlist,rotation=45)
    axs[2,i].plot([-1,N+1],[0,0],'w',linewidth=4,zorder=0)
    axs[2,i].set_xlim(-1,N)
    axs[2,i].set_ylim(-.55,.61)
    axs[2,0].set_ylabel('Site Correlation ($\zeta>0$)')

axs[0,0].set_title('$y_b$')
axs[0,1].set_title('$x_b$')
plt.subplots_adjust(hspace=.1)
plt.savefig('../../plot_output/a1_img_12.png', bbox_inches = "tight")

# %%
dc_u['yvars']

# %%
grass=['NOGP','WOOD','OAES','JORN','ONAQ','KONA','KONZ','BARR','DCFS','MOAB']
forest=['UKFS','ABBY','BART','GRSM','HARV','HEAL','LENO','MLBS','ORNL','OSBS','PUUM','RMNP','SCBI','SERC','SOAP','STEI','TALL','TEAK','TREE','WREF']

# %%
dd=[]
for k in dc_u['sitelevel'].keys():
    d=dc_u['sitelevel'][k]['spearmanr'][1,-6]
    if np.isnan(d):
        pass
    else:
        dd.append(d)

# %%
site=b'ABBY'
color=
m=~(np.isnan(fpu['ANI_YB'][:])|(np.isnan(fpu['std_dsm'][:])))&(fpu['SITE'][:]==site)
a=fpu['ANI_YB'][m]
b=fpu['std_dsm'][m]
mp=.33
counts1,xbins1,ybins1,image = plt.hist2d(a,b,bins=50,density=True)
counts1=counts1/np.max(counts1)
plt.clf()
#levels=np.array([2,10,50,100,200,300,500,750,1000,2000,3000,4000])
levels=[.005,.15,.3,.45,.6,.75,.9]
plt.contour(np.transpose(counts1),extent=[xbins1.min(),xbins1.max(),ybins1.min(),ybins1.max()],levels=levels,linewidths=1,cmap=cmp)

# %%
counts=[]
clrs=[]
xbins=[]
ybins=[]
k=0
plt.figure(figsize=(4,8))
for site in other[0]:
    #if site not in forest:
    #    k=k+1
    #    pass
    try:
        color=class_colors[other[2][k]]
    except:
        color='darkgreen'
    k=k+1
    m=(bytes(site,'utf-8')==fpu['SITE'][:])&(~(np.isnan(fpu['ANI_YB'][:])|(np.isnan(fpu['std_dsm'][:]))))
    a=fpu['ANI_YB'][m]
    b=fpu['std_dsm'][m]
    counts1,xbins1,ybins1,image = plt.hist2d(a,b,bins=50,density=True)
    counts1=counts1/np.max(counts1)

    counts.append(counts1)
    xbins.append(xbins1)
    ybins.append(ybins1)
    clrs.append(color)
    
    plt.clf()

for i in range(len(counts)):
    #levels=np.array([2,10,50,100,200,300,500,750,1000,2000,3000,4000])
    levels=[.5]
    print(other[0][i]+': '+str(np.median(ybins[i])))
    plt.contour(np.transpose(counts[i]),extent=[xbins[i].min(),xbins[i].max(),ybins[i].min(),ybins[i].max()],levels=levels,linewidths=1,colors=clrs[i],alpha=.75,yscale='log')
    ax=plt.gca()
    ax.set_yscale('log')

plt.xlabel('$y_b$')
plt.ylabel('log($\sigma_{dsm}$)')
plt.xlim(0,.5)
plt.ylim(.5,70)
#plt.savefig('../../plot_output/test.png', bbox_inches = "tight")

# %%

# %%
plt.boxplot(dd)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Self Similarity 2

# %%
m=fpu['SITE'][:]==b'DCFS'
m=m&(fpu['zzd'][:]/fpu['L_MOST'][:]>-.1)
plt.hist(fpu['ANI_YB'][m]/fpu['ANID_YB'][m],bins=np.linspace(.5,1,100))
#plt.hist(fpu['ANID_YB'][:],bins=np.linspace(.5,1.2,100))

# %%
m=fpu['SITE'][:]==b'MLBS'
m=m&(fpu['zzd'][:]/fpu['L_MOST'][:]>-.1)
plt.hist(fpu['ANI_YB'][m]/fpu['ANID_YB'][m],bins=np.linspace(.5,1,100))

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Diurnal Cycle and Wind Speed

# %% [markdown]
# ### Diurnal Cycle Prep

# %%
truehour=[]
truedoy=[]
oldsite=b'SAFD'
s=-1
for t in range(len(fpu['TIME'][:])):
    site=fpu['SITE'][t]
    if site != oldsite:
        print('.',flush=True,end='')
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
hour=[]
sday=[]
for t in truehour:
    hour.append(t.hour)
    sday.append(t.hour*3600+t.minute*60+t.second)
sday=np.array(sday)

# %%
truehour=[]
truedoy=[]
oldsite=b'SAFD'
s=-1
for t in range(len(fps['TIME'][:])):
    site=fps['SITE'][t]
    if site != oldsite:
        print('.',flush=True,end='')
        s=s+1
        try:
            offset=float(fpst['utc_off'][s])
        except:
            if site==b'WREF':
                offset=-9
            elif site==b'YELL':
                offset=-8
        oldsite=site
    tt=fps['TIME'][t]
    dt=datetime.utcfromtimestamp(tt)+timedelta(hours=offset)
    truehour.append(dt)
    truedoy.append(dt.timetuple().tm_yday)
hour=[]
sdays=[]
for t in truehour:
    hour.append(t.hour)
    sdays.append(t.hour*3600+t.minute*60+t.second)
sdays=np.array(sdays)

# %%
g1=[]
g2=[]
g3=[]
for site in np.unique(fpu['SITE'][:]):
    m=fpu['SITE'][:]==site
    zzd=fpu['zzd'][m][0]
    if (zzd>8):#&(site not in [b'BONA',b'DEJU',b'SJER']):
        g1.append(site)
    elif site in []: #[b'JORN',b'OAES',b'ONAQ',b'SRER',b'MOAB',b'CPER',b'TOOL',b'HEAL']:
        g2.append(site)
    else:
        g3.append(site)

# %% [markdown]
# ### Plotting

# %%
sz=2
fig,axs=plt.subplots(4,1,figsize=(2*sz,5.5*sz))

frst_s=[.296,.321,.347]
grss_s=[.302,.319,.331]

# masks
m1=np.zeros((len(fpu['SITE']),),dtype=bool)
m2=np.zeros((len(fpu['SITE']),),dtype=bool)
m3=np.zeros((len(fpu['SITE']),),dtype=bool)
for site in g1:
    m1=m1|(fpu['SITE'][:]==site)
for site in g2:
    m2=m2|(fpu['SITE'][:]==site)
for site in g3:
    m3=m3|(fpu['SITE'][:]==site)

ctf=[1,2,4]

mi=[fpu['Ustr'][:]<ctf[0],(fpu['Ustr'][:]>ctf[0])&(fpu['Ustr'][:]<ctf[1]),\
    (fpu['Ustr'][:]>ctf[1])&(fpu['Ustr'][:]<ctf[2]),fpu['Ustr'][:]>ctf[2]]

#zl=-fpu['zzd'][:]/fpu['L_MOST'][:]
#ctf=[.01,.1,1]
#mi=[zl<ctf[0],(zl>ctf[0])&(zl<ctf[1]),\
#    (zl>ctf[1])&(zl<ctf[2]),zl>ctf[2]]


b=fpu['ANI_YB'][:]

for i in range(4):

    xbins=np.unique(sday)
    a1_25=[]
    a1_50=[]
    a1_75=[]
    a1_std=[]
    a2_25=[]
    a2_50=[]
    a2_75=[]
    a2_std=[]
    a3_25=[]
    a3_50=[]
    a3_75=[]
    a3_std=[]
    heat=[]

    for j in range(len(xbins)):
        m1_1=m1&(sday==xbins[j])&mi[i]
        m2_1=m2&(sday==xbins[j])&mi[i]
        m3_1=m3&(sday==xbins[j])&mi[i]
        
        a1_25.append(np.percentile(b[m1_1],25))
        a1_50.append(np.percentile(b[m1_1],50))
        a1_75.append(np.percentile(b[m1_1],70))
        a1_std.append(np.std(b[m1_1]))
        heat.append(np.nanmedian(-fpu['zzd'][m1_1|m3_1]/fpu['L_MOST'][m1_1|m3_1]))

        a3_25.append(np.percentile(b[m3_1],25))
        a3_50.append(np.percentile(b[m3_1],50))
        a3_75.append(np.percentile(b[m3_1],75))
        a3_std.append(np.std(b[m3_1]))

    a1_50=np.array(a1_50)
    #a2_50=np.array(a2_50)
    a3_50=np.array(a3_50)
    heat=np.array(heat)
    
    a1_std=np.array(a1_std)*.5
    #a2_std=np.array(a2_std)*.5
    a3_std=np.array(a3_std)*.5
    
    axs[i].plot(xbins,a1_50,color='green',label='Forest')
    axs[i].fill_between(xbins,a1_25,a1_75,color='green',alpha=.1,label='_extra1')
    #axs[i].fill_between(xbins,a1_50-a1_std,a1_50+a1_std,color='forestgreen',alpha=.2)
    
    #axs[i].plot(xbins,a2_50,color='orange')
    #axs[i].fill_between(xbins,a2_25,a2_75,color='orange',alpha=.1)
    #axs[i].fill_between(xbins,a2_50-a2_std,a2_50+a2_std,color='orange',alpha=.2)

    axs[i].plot(xbins,a3_50,color='darkorange',label='Non-Forest')
    axs[i].fill_between(xbins,a3_25,a3_75,color='darkorange',alpha=.1,label='_extra2')

    print(np.max(heat))
    
    axs[i].plot(xbins,heat/(2*np.max(heat)),'--',color='indianred',label='$H$')
    #axs[i].fill_between(xbins,a3_50-a3_std,a3_50+a3_std,color='indianred',alpha=.2)

    #axs[i].plot([0,70000],[frst_s[i],frst_s[i]],'--',color='forestgreen',linewidth=.5)
    #axs[i].plot([0,70000],[grss_s[i],grss_s[i]],'--',color='orange',linewidth=.5)
    
    #axs[i].plot([66500,70000],[frst_s[i],frst_s[i]],color='forestgreen',linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
    #axs[i].plot([66500,70000],[grss_s[i],grss_s[i]],color='orange',linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
                
    axs[i].set_ylim(0.025,.51)
    axs[i].set_xlim(18000,68500)
    axs[i].set_ylabel('$y_b$')

axs[0].legend(loc='upper right',fontsize=10)

axs[0].set_title('$\overline{U}<1\ \ (12.91\%)$')
axs[1].set_title('$\overline{U}\geq 1$, $\overline{U}<2\ \ (21.4\%)$')
axs[2].set_title('$\overline{U}\geq 2$, $\overline{U}<4\ \ (37.1\%)$')
axs[3].set_title('$\overline{U}\geq 4\ \ (28.5\%)$')

axs[0].set_xticks([6*3600,9*3600,12*3600,15*3600,3600*18],[])
axs[1].set_xticks([6*3600,9*3600,12*3600,15*3600,3600*18],[])
axs[2].set_xticks([6*3600,9*3600,12*3600,15*3600,3600*18],[])
axs[3].set_xticks([6*3600,9*3600,12*3600,15*3600,3600*18],['6:00','9:00','12:00','15:00','18:00'])
axs[3].set_xlabel('Local Time')

plt.subplots_adjust(hspace=.2)
#plt.savefig('../../plot_output/a1_img_14.png', bbox_inches = "tight")

# %%
np.sum(mi,axis=1)/np.sum(mi)*100

# %%
sz=2
fig,axs=plt.subplots(3,1,figsize=(2*sz,4.5*sz))

frst_s=[.296,.321,.347]
grss_s=[.302,.319,.331]

# masks
m1=np.zeros((len(fpu['SITE']),),dtype=bool)
m2=np.zeros((len(fpu['SITE']),),dtype=bool)
m3=np.zeros((len(fpu['SITE']),),dtype=bool)
for site in g1:
    m1=m1|(fpu['SITE'][:]==site)
for site in g2:
    m2=m2|(fpu['SITE'][:]==site)
for site in g3:
    m3=m3|(fpu['SITE'][:]==site)

mi=[fpu['Ustr'][:]<1,(fpu['Ustr'][:]>1)&(fpu['Ustr'][:]<2),fpu['Ustr'][:]>4]

b=fpu['ANI_XB'][:]

for i in range(3):

    xbins=np.unique(sday)
    a1_25=[]
    a1_50=[]
    a1_75=[]
    a1_std=[]
    a2_25=[]
    a2_50=[]
    a2_75=[]
    a2_std=[]
    a3_25=[]
    a3_50=[]
    a3_75=[]
    a3_std=[]

    for j in range(len(xbins)):
        m1_1=m1&(sday==xbins[j])&mi[i]
        m2_1=m2&(sday==xbins[j])&mi[i]
        m3_1=m3&(sday==xbins[j])&mi[i]
        
        a1_25.append(np.percentile(b[m1_1],25))
        a1_50.append(np.percentile(b[m1_1],50))
        a1_75.append(np.percentile(b[m1_1],70))
        a1_std.append(np.std(b[m1_1]))

        #a2_25.append(np.percentile(b[m2_1],25))
        #a2_50.append(np.percentile(b[m2_1],50))
        #a2_75.append(np.percentile(b[m2_1],75))
        #a2_std.append(np.std(b[m2_1]))

        a3_25.append(np.percentile(b[m3_1],25))
        a3_50.append(np.percentile(b[m3_1],50))
        a3_75.append(np.percentile(b[m3_1],75))
        a3_std.append(np.std(b[m3_1]))

    a1_50=np.array(a1_50)
    #a2_50=np.array(a2_50)
    a3_50=np.array(a3_50)
    
    a1_std=np.array(a1_std)*.5
    #a2_std=np.array(a2_std)*.5
    a3_std=np.array(a3_std)*.5
    
    axs[i].plot(xbins,a1_50,color='green',label='Forest')
    axs[i].fill_between(xbins,a1_25,a1_75,color='green',alpha=.1,label='_extra1')
    #axs[i].fill_between(xbins,a1_50-a1_std,a1_50+a1_std,color='forestgreen',alpha=.2)
    
    #axs[i].plot(xbins,a2_50,color='orange')
    #axs[i].fill_between(xbins,a2_25,a2_75,color='orange',alpha=.1)
    #axs[i].fill_between(xbins,a2_50-a2_std,a2_50+a2_std,color='orange',alpha=.2)

    axs[i].plot(xbins,a3_50,color='darkorange',label='Non-Forest')
    axs[i].fill_between(xbins,a3_25,a3_75,color='darkorange',alpha=.1,label='_extra2')
    #axs[i].fill_between(xbins,a3_50-a3_std,a3_50+a3_std,color='indianred',alpha=.2)

    #axs[i].plot([0,70000],[frst_s[i],frst_s[i]],'--',color='forestgreen',linewidth=.5)
    #axs[i].plot([0,70000],[grss_s[i],grss_s[i]],'--',color='orange',linewidth=.5)
    
    #axs[i].plot([66500,70000],[frst_s[i],frst_s[i]],color='forestgreen',linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
    #axs[i].plot([66500,70000],[grss_s[i],grss_s[i]],color='orange',linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
                
    axs[i].set_ylim(0.2,0.8)
    axs[i].set_xlim(18000,68500)
    axs[i].set_ylabel('$y_b$')
    axs[i].legend(loc='upper right')
    
axs[0].set_xticks([6*3600,9*3600,12*3600,15*3600,3600*18],[])
axs[1].set_xticks([6*3600,9*3600,12*3600,15*3600,3600*18],[])
axs[2].set_xticks([6*3600,9*3600,12*3600,15*3600,3600*18],['6:00','9:00','12:00','15:00','18:00'])
axs[2].set_xlabel('Local Time')

plt.subplots_adjust(hspace=.1)

# %%
m1=np.zeros((len(fps['SITE']),),dtype=bool)
m2=np.zeros((len(fps['SITE']),),dtype=bool)
m3=np.zeros((len(fps['SITE']),),dtype=bool)
for site in g1:
    m1=m1|(fps['SITE'][:]==site)
for site in g2:
    m2=m2|(fps['SITE'][:]==site)
for site in g3:
    m3=m3|(fps['SITE'][:]==site)

mi=[fps['Ustr'][:]<1,(fps['Ustr'][:]>1)&(fps['Ustr'][:]<2),fps['Ustr'][:]>2]
for i in range(3):
    print(np.nanmedian(fps['ANI_YB'][mi[i]&m1]))
print()
for i in range(3):
    print(np.nanmedian(fps['ANI_YB'][mi[i]&m3]))

# %%
sz=2
fig,axs=plt.subplots(3,1,figsize=(2*sz,4.5*sz))

# masks
m1=np.zeros((len(fps['SITE']),),dtype=bool)
m2=np.zeros((len(fps['SITE']),),dtype=bool)
m3=np.zeros((len(fps['SITE']),),dtype=bool)
for site in g1:
    m1=m1|(fps['SITE'][:]==site)
for site in g2:
    m2=m2|(fps['SITE'][:]==site)
for site in g3:
    m3=m3|(fps['SITE'][:]==site)

mi=[fps['Ustr'][:]<1,(fps['Ustr'][:]>1)&(fps['Ustr'][:]<2),fps['Ustr'][:]>4]

b=fps['ANI_XB'][:]

for i in range(3):

    xbins=np.unique(sday)
    xbin_=(xbins[1:]+xbins[0:-1])/2
    a1_25=[]
    a1_50=[]
    a1_75=[]
    a1_std=[]
    a2_25=[]
    a2_50=[]
    a2_75=[]
    a2_std=[]
    a3_25=[]
    a3_50=[]
    a3_75=[]
    a3_std=[]

    for j in range(len(xbins)-1):
        m1_1=m1&(sdays>xbins[j])&(sdays<xbins[j+1])&mi[i]
        m2_1=m2&(sdays>xbins[j])&(sdays<xbins[j+1])&mi[i]
        m3_1=m3&(sdays>xbins[j])&(sdays<xbins[j+1])&mi[i]
        
        a1_25.append(np.percentile(b[m1_1],25))
        a1_50.append(np.percentile(b[m1_1],50))
        a1_75.append(np.percentile(b[m1_1],70))
        a1_std.append(np.std(b[m1_1]))

        #a2_25.append(np.percentile(b[m2_1],25))
        #a2_50.append(np.percentile(b[m2_1],50))
        #a2_75.append(np.percentile(b[m2_1],75))
        #a2_std.append(np.std(b[m2_1]))

        a3_25.append(np.percentile(b[m3_1],25))
        a3_50.append(np.percentile(b[m3_1],50))
        a3_75.append(np.percentile(b[m3_1],75))
        a3_std.append(np.std(b[m3_1]))

    a1_50=np.array(a1_50)
    #a2_50=np.array(a2_50)
    a3_50=np.array(a3_50)
    
    a1_std=np.array(a1_std)*.5
    #a2_std=np.array(a2_std)*.5
    a3_std=np.array(a3_std)*.5
    
    axs[i].plot(xbin_,a1_50,color='forestgreen')
    axs[i].fill_between(xbin_,a1_25,a1_75,color='forestgreen',alpha=.1)
    #axs[i].fill_between(xbins,a1_50-a1_std,a1_50+a1_std,color='forestgreen',alpha=.2)
    
    #axs[i].plot(xbin_,a2_50,color='orange')
    #axs[i].fill_between(xbin_,a2_25,a2_75,color='orange',alpha=.1)
    #axs[i].fill_between(xbins,a2_50-a2_std,a2_50+a2_std,color='orange',alpha=.2)

    axs[i].plot(xbin_,a3_50,color='indianred')
    axs[i].fill_between(xbin_,a3_25,a3_75,color='indianred',alpha=.1)
    #axs[i].fill_between(xbins,a3_50-a3_std,a3_50+a3_std,color='indianred',alpha=.2)

    axs[i].set_ylim(0.3,.75)

# %%
x=sdays
xbins=np.linspace(np.nanpercentile(x,1),np.nanpercentile(x,99),47)
xtrue=[]
ytrue=[]
for i in range(46):
    m=(x>xbins[i])&(x<xbins[i+1])
    xtrue.append(np.nanmedian(x[m]))
    ytrue.append(np.sum(m))
plt.plot(xtrue,ytrue,color='black',linewidth=2)

# %%
a=sday
b=fpu['ANI_YB'][:]
bins=np.unique(a)
b25=[]
b50=[]
b75=[]
for i in range(len(bins)):
    m=(a==bins[i])
    b25.append(np.percentile(b[m],25))
    b50.append(np.percentile(b[m],50))
    b75.append(np.percentile(b[m],75))

# %%
plt.plot(bins,b50)
plt.fill_between(bins,b25,b75,color='blue',alpha=.2)

# %%
fpst.keys()

# %%
std_dsm=[]
data=[]
for i in range(45):
    site=fpst['site'][i]
    m=fpu['SITE'][:]==site
    data.append(np.sum(m))
    std_dsm.append(fpst['std_dsm'][i])

# %%
plt.scatter(data,std_dsm)

# %%
zb=np.logspace(-4,2)
zzd=fpu['zzd'][:]
zl=-fpu['zzd'][:]/fpu['L_MOST'][:]
ub=np.linspace(0,15)
u=fpu['Ustr'][:]
ani=fpu['ANI_YB'][:]
f_ani=np.zeros((49,49))
f_anicv=np.zeros((49,49))
g_ani=np.zeros((49,49))
g_anicv=np.zeros((49,49))
for i in range(49):
    for j in range(49):
        m=(zl>zb[i])&(zl<zb[i+1])&(u>ub[j])&(u<ub[j+1])
        m1=m&(zzd>8)
        m2=m&(zzd<8)
        f_ani[i,j]=np.median(ani[m1])
        f_anicv[i,j]=np.std(ani[m1])/np.mean(ani[m1])
        g_ani[i,j]=np.median(ani[m2])
        g_anicv[i,j]=np.std(ani[m2])/np.mean(ani[m2])

# %%
data=f_ani
data[3,:]=float('nan')

plt.imshow(f_ani,cmap='terrain',origin='lower',vmin=0,vmax=.5)
plt.xticks([0,10,20,30,40],[ub[0],ub[10],ub[20],ub[30],ub[40]])
plt.yticks([0,10,20,30,40],[zb[0],zb[10],zb[20],zb[30],zb[40]])
plt.colorbar()

# %%
data=g_ani

plt.imshow(g_ani,cmap='terrain',origin='lower',vmin=0,vmax=.5)
plt.xticks([0,10,20,30,40],[ub[0],ub[10],ub[20],ub[30],ub[40]])
plt.yticks([0,10,20,30,40],[zb[0],zb[10],zb[20],zb[30],zb[40]])
plt.colorbar()

# %%
zb=np.linspace(0,750)
zzd=fpu['zzd'][:]
zl=fpu['H'][:]
ub=np.linspace(0,2.5)
u=fpu['USTAR'][:]
ani=fpu['ANI_XB'][:]
f_ani=np.zeros((49,49))
f_anicv=np.zeros((49,49))
g_ani=np.zeros((49,49))
g_anicv=np.zeros((49,49))
for i in range(49):
    for j in range(49):
        m=(zl>zb[i])&(zl<zb[i+1])&(u>ub[j])&(u<ub[j+1])
        m1=m&(zzd>8)
        m2=m&(zzd<8)
        f_ani[i,j]=np.median(ani[m1])
        f_anicv[i,j]=np.std(ani[m1])/np.mean(ani[m1])
        g_ani[i,j]=np.median(ani[m2])
        g_anicv[i,j]=np.std(ani[m2])/np.mean(ani[m2])

# %%
plt.imshow(g_ani,cmap='terrain',origin='lower',vmin=.1,vmax=.75)
plt.xticks([0,10,20,30,40],[ub[0],ub[10],ub[20],ub[30],ub[40]])
plt.yticks([0,10,20,30,40],[zb[0],zb[10],zb[20],zb[30],zb[40]])
plt.colorbar()

# %%
plt.imshow(f_ani,cmap='terrain',origin='lower',vmin=.1,vmax=.75)
plt.xticks([0,10,20,30,40],[ub[0],ub[10],ub[20],ub[30],ub[40]])
plt.yticks([0,10,20,30,40],[zb[0],zb[10],zb[20],zb[30],zb[40]])
plt.colorbar()

# %%
plt.hexbin(fpu['H'][:],fpu['ANI_YB'][:],mincnt=1,cmap='terrain')

# %%
plt.hexbin(fpu['U'][:],fpu['ANI_YB'][:],mincnt=1,cmap='terrain')

# %%

# %% [markdown]
# # More Windspeed

# %% [markdown]
# ## Scaling with Windspeed Categories

# %% [markdown]
# ### Preparation

# %%
Na=9
Nz=20
anibinst=np.array([0,.6,1.2,1.7,2.4,3.3,4.3,5.5,8.1,30])
#anibinst=np.array([.1,.2,.3,.4,.5,.6,.7,.8])
anilvlt=(anibinst[0:-1]+anibinst[1:])/2

zLbu=-np.logspace(-3.6,2,Nz+1)
zLbs=np.logspace(-3.6,2,Nz+1)
#zLu=(zLbu[0:-1]+zLbu[1:])/2
#zLs=(zLbs[0:-1]+zLbs[1:])/2

upper=np.zeros((2,3,Na,Nz))
lower=np.zeros((2,3,Na,Nz))
mid=np.zeros((2,3,Na,Nz))

atrue=np.zeros((2,3,Na))
zLtrue=np.zeros((2,3,Na,Nz))

for s in range(2):
    for v in range(3):
        vs=['U','V','W'][v]
        if s==0:
            vs=vs+'U'
            zLb=zLbu
            fp=fpu
        else:
            vs=vs+'S'
            zLb=zLbs
            fp=fps
        print(vs)
        phi,_=get_phi(fp,vs)
        ani=fp['Ustr'][:]
        zL=fp['zzd'][:]/fp['L_MOST'][:]
        for i in range(Na):
            ma=(ani>anibinst[i])&(ani<anibinst[i+1])
            atrue[s,v,i]=np.nanmedian(ani[ma])
            for j in range(Nz):
                if s==0:
                    m=ma&(zL<zLb[j])&(zL>zLb[j+1])
                else:
                    m=ma&(zL>zLb[j])&(zL<zLb[j+1])
                if np.nansum(m)<100:
                    upper[s,v,i,j]=float('nan')
                    mid[s,v,i,j]=float('nan')
                    lower[s,v,i,j]=float('nan')
                    zLtrue[s,v,i,j]=float('nan')
                else:
                    phim=phi[m]
                    upper[s,v,i,j]=np.nanpercentile(phim,75)
                    mid[s,v,i,j]=np.nanpercentile(phim,50)
                    lower[s,v,i,j]=np.nanpercentile(phim,25)
                    zLtrue[s,v,i,j]=np.nanmedian(zL[m])

# %%

# %%

# %% [markdown]
# ### Plotting

# %%
sz=1.4
fig,axs=plt.subplots(3,2,figsize=(5*sz,4.5*sz),gridspec_kw={'width_ratios': [1,1]},dpi=400)
ss=.05
alph=.75
minpct=1e-04
mrkr='.'

ylabels=[r'$\Phi_{u}$',r'$\Phi_{v}$',r'$\Phi_{w}$']

j=0
for v in ['U','V','W']:
    # SETUP
    if v in ['U','V']:
        ymax=8
        ymin=.5
    else:
        ymax=3.5
        ymin=.2
    
    ##### UNSTABLE #####
    # SCATTER UNSTABLE
    # LINES UNSTABLE
    yplt=mid[0,j,:,:]
    for i in range(yplt.shape[0]):
        if i==0:
            continue
        anic=atrue[0,j,i]
        xplt=zLtrue[0,j,i,:]
        if v in ['U','V','W','H2O','CO2']:
            axs[j,0].semilogx(-xplt,yplt[i,:],'-o',color=pl.cm.turbo(anic/10),markersize=2,linewidth=.2)#,linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
        else:
            axs[j,0].loglog(-xplt,yplt[i,:],'-o',color=cani_norm(anic[i]),markersize=2,linewidth=.2)#,linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
        axs[j,0].fill_between(-xplt, lower[0,j,i,:], upper[0,j,i,:],color=pl.cm.turbo(anic/10),alpha=.15)
    _,yplt=get_phi(fpu,v+'U',zLbu)
    if v in ['U','V','W','H2O','CO2']:
        axs[j,0].semilogx(-zLbu,yplt,'--',color='k',linewidth=1.5,path_effects=[pe.Stroke(linewidth=3, foreground='w'), pe.Normal()],zorder=5)
    else:
        axs[j,0].loglog(-zLbu,yplt,'--',color='k',linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()],zorder=5)
    
    # LABELING
    if j==2:
        axs[j,0].tick_params(which="both", bottom=True)
        locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )) 
        axs[j,0].xaxis.set_minor_locator(locmin)
        axs[j,0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axs[j,0].set_xticks([10**-3,10**-2,10**-1,1,10],[r'$-10^{-3}$','',r'$-10^{-1}$','',r'$-10^{1}$'])
        axs[j,0].set_xlim(10**(-3.5),10**(1.3))
        axs[j,0].set_xlabel(r'$\zeta$')
    else:
        axs[j,0].set_xticks([10**-3,10**-2,10**-1,1,10],[])
        axs[j,0].set_xlim(10**(-3.5),10**(1.3))
    #axs[j,0].xaxis.set_minor_locator(tck.AutoMinorLocator())
    axs[j,0].set_ylabel(ylabels[j])
    axs[j,0].set_ylim(ymin,ymax)
    axs[j,0].invert_xaxis()
    
    ##### STABLE #####
    # LINES STABLE
    yplt=mid[1,j,:,:]
    anic=atrue[1,j,:]
    for i in range(yplt.shape[0]):
        if i==0:
            continue
        anic=atrue[1,j,i]
        xplt=zLtrue[1,j,i,:]
        if v in ['U','V','W','H2O','CO2']:
            axs[j,1].semilogx(xplt,yplt[i,:],'-o',color=pl.cm.turbo(anic/10),markersize=2,linewidth=.2)#,linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
        else:
            axs[j,1].loglog(xplt,yplt[i,:],'-o',color=cani_norm(anic[i]),markersize=2,linewidth=.2)#,linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
        axs[j,1].fill_between(xplt, lower[1,j,i,:], upper[1,j,i,:],color=pl.cm.turbo(anic/10),alpha=.15)
    _,yplt=get_phi(fps,v+'S',zLbs)
    if v in ['U','V','W','H2O','CO2']:
        axs[j,1].semilogx(zLbs,yplt[:],'--',color='k',linewidth=1.5,path_effects=[pe.Stroke(linewidth=3, foreground='w'), pe.Normal()],zorder=5)
    else:
        axs[j,1].loglog(zLbs,yplt[:],'--',color='k',linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()],zorder=5)

    # LABELING
    if j==2:
        axs[j,1].tick_params(which="both", bottom=True)
        locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )) 
        axs[j,1].xaxis.set_minor_locator(locmin)
        axs[j,1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axs[j,1].set_xticks([10**-3,10**-2,10**-1,1,10],[r'$10^{-3}$','',r'$10^{-1}$','',r'$10^{1}$'])
        axs[j,1].set_xlim(10**(-3.5),10**(1.3))
        axs[j,1].set_xlabel(r'$\zeta$')
    else:
        locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )) 
        axs[j,1].xaxis.set_minor_locator(locmin)
        axs[j,1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axs[j,1].set_xticks([10**-3,10**-2,10**-1,1,10],[])
        axs[j,1].set_xlim(10**(-3.5),10**(1.3))
    axs[j,1].tick_params(labelleft=False)
    #axs[j,1].grid(False)
    #axs[j,1].xaxis.grid(True, which='minor')
    axs[j,1].set_ylim(ymin,ymax)
    
    j=j+1

plt.subplots_adjust(hspace=.08,wspace=.02)
plt.savefig('../../plot_output/a1/a1_wind_scaling_v2.png', bbox_inches = "tight")

# %%

# %% [markdown]
# ## MAD versus Windspeed for all Relations

# %% [markdown]
# ### Preparation

# %%
from sklearn.metrics import mean_squared_error
def getbins(A,n):
     B=np.sort(A)
     bins=[]
     for i in np.linspace(0,len(A)-1,n):
         i=int(i)
         bins.append(B[i])
     return bins
def get_phi_sc23(fp,var):
    zL=fp['zzd'][:]/fp['L_MOST'][:]
    ani=fp['ANI_YB'][:]
    match var:
        case 'UU':
            a=.784-2.582*np.log10(ani)
            phin=a*(1-3*zL)**(1/3)
        case 'VU':
            a=.725-2.702*np.log10(ani)
            phin=a*(1-3*zL)**(1/3)
        case 'WU':
            a=1.119-0.019*ani-.065*ani**2+0.028*ani**3
            phin=a*(1-3*zL)**(1/3)
        case 'US':
            a_=np.array([2.332,-2.047,2.672])
            c_=np.array([.255,-1.76,5.6,-6.8,2.65])
            a=0
            c=0
            for i in range(3):
                a=a+a_[i]*ani**i
            for i in range(5):
                c=c+c_[i]*ani**i
            phin=a*(1+3*zL)**(c)
        case 'VS':
            a_=np.array([2.385,-2.781,3.771])
            c_=np.array([.654,-6.282,21.975,-31.634,16.251])
            a=0
            c=0
            for i in range(3):
                a=a+a_[i]*ani**i
            for i in range(5):
                c=c+c_[i]*ani**i
            phin=a*(1+3*zL)**(c)
        case 'WS':
            a_=np.array([.953,.188,2.243])
            c_=np.array([.208,-1.935,6.183,-7.485,3.077])
            a=0
            c=0
            for i in range(3):
                a=a+a_[i]*ani**i
            for i in range(5):
                c=c+c_[i]*ani**i
            phin=a*(1+3*zL)**(c)
    return phin
def get_phi_tsw(fp,var):
    zL=fp['zzd'][:]/fp['L_MOST'][:]
    ani=fp['ANI_YB'][:]
    lani=np.log10(ani)
    match var:
        case 'UU':
            a=.5-2.960*lani
            phin=a*(1-3*zL)**(1/3)
        case 'VU':
            a=.397-2.721*lani
            phin=a*(1-3*zL)**(1/3)
        case 'WU':
            a=0.878-.790*lani-.688*lani**2
            phin=a*(1-3*zL)**(1/3)
        case 'US':
            a=2.490-2.931*ani+3.967*ani**2
            c=-0.014+.441*ani-.473*ani**2
            phin=a*(1+3*zL)**(c)
        case 'VS':
            a=1.333+.584*ani+1.307*ani**2
            c=.223-.429*ani+.328*ani**2
            phin=a*(1+3*zL)**(c)
        case 'WS':
            a=.857+.385*ani+2.399*ani**2
            c=.096-.0043*ani-.038*ani**2
            phin=a*(1+3*zL)**(c)
    return phin


# %%
Ns=50
Nu=50
ubs=fps['Ustr'][:]
ubu=fpu['Ustr'][:]

ubsb=getbins(ubs,Ns+1)
ubub=getbins(ubu,Nu+1)

ubr=np.zeros((2,3,Nu))
mad=np.zeros((2,3,3,Nu))
md=np.zeros((2,3,3,Nu))
rmse=np.zeros((2,3,3,Nu))
rmdse=np.zeros((2,3,3,Nu))

for i in range(3):
    v=['U','V','W'][i]
    print(v)
    # unstable
    phi,phio=get_phi(fpu,v+'U')
    phin=get_phi_sc23(fpu,v+'U')
    phint=get_phi_tsw(fpu,v+'U')
    for j in range(Nu):
        m=(ubu>ubub[j])&(ubu<ubub[j+1])
        phim=phi[m]
        phiom=phio[m]
        phinm=phin[m]
        phin2m=phint[m]
        md[0,i,0,j]=np.nanmedian(phim-phiom)
        md[0,i,1,j]=np.nanmedian(phim-phinm)
        md[0,i,2,j]=np.nanmedian(phim-phin2m)
        mad[0,i,0,j]=np.nanmedian(np.abs(phim-phiom))
        mad[0,i,1,j]=np.nanmedian(np.abs(phim-phinm))
        mad[0,i,2,j]=np.nanmedian(np.abs(phim-phin2m))
        rmse[0,i,0,j]=np.sqrt(np.mean((phiom-phim)**2))
        rmse[0,i,1,j]=np.sqrt(np.mean((phinm-phim)**2))
        rmse[0,i,2,j]=np.sqrt(np.mean((phin2m-phim)**2))
        rmdse[0,i,0,j]=np.sqrt(np.median((phiom-phim)**2))
        rmdse[0,i,1,j]=np.sqrt(np.median((phinm-phim)**2))
        rmdse[0,i,2,j]=np.sqrt(np.median((phin2m-phim)**2))
        ubr[0,i,j]=np.nanmedian(ubu[m])
    phi,phio=get_phi(fps,v+'S')
    phin=get_phi_sc23(fps,v+'S')
    phint=get_phi_tsw(fps,v+'S')
    for j in range(Ns):
        m=(ubs>ubsb[j])&(ubs<ubsb[j+1])
        phim=phi[m]
        phiom=phio[m]
        phinm=phin[m]
        phin2m=phint[m]
        md[1,i,0,j]=np.nanmedian(phim-phiom)
        md[1,i,1,j]=np.nanmedian(phim-phinm)
        md[1,i,2,j]=np.nanmedian(phim-phin2m)
        mad[1,i,0,j]=np.nanmedian(np.abs(phim-phiom))
        mad[1,i,1,j]=np.nanmedian(np.abs(phim-phinm))
        mad[1,i,2,j]=np.nanmedian(np.abs(phim-phin2m))
        rmse[1,i,0,j]=np.sqrt(np.mean((phiom-phim)**2))
        rmse[1,i,1,j]=np.sqrt(np.mean((phinm-phim)**2))
        rmse[1,i,2,j]=np.sqrt(np.mean((phin2m-phim)**2))
        rmdse[1,i,0,j]=np.sqrt(np.median((phiom-phim)**2))
        rmdse[1,i,1,j]=np.sqrt(np.median((phinm-phim)**2))
        rmdse[1,i,2,j]=np.sqrt(np.median((phin2m-phim)**2))
        ubr[1,i,j]=np.nanmedian(ubs[m])


# %%

# %% [markdown]
# ### Plotting

# %%
sz=1
from matplotlib.ticker import FixedLocator
fig,axs=plt.subplots(3,2,figsize=(5*sz,4*sz),dpi=500)
labels=[r'$Bias$ $\Phi_u$',r'$Bias$ $\Phi_v$',r'$Bias$ $\Phi_w$']
for i in range(3):
    for s in range(2):
        ax=axs[i,s]
        #ax.plot(ubr[s,i,:],[0]*Nu,color='white',linewidth=2)
        ax.plot(ubr[s,i,:],-md[s,i,0,:],'-o',linewidth=.75,markersize=1.5,zorder=2,color='black',alpha=.3,label='MOST')
        ax.plot(ubr[s,i,:],-md[s,i,1,:],'-o',linewidth=.75,markersize=1.5,zorder=3,color='mediumpurple',alpha=.5,label='SC23 fit')
        ax.plot(ubr[s,i,:],-md[s,i,2,:],'-o',linewidth=.75,markersize=1.5,zorder=4,color='goldenrod',alpha=.5,label='Neon fit')
        #ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12],[0,'','','','',5,'','','','',10,'',''])
        ax.set_xticks([0,5,10])
        ax.xaxis.set_minor_locator(FixedLocator([0,1,2,3,4,5,6,7,8,9,10,11,12]))
        ax.grid(axis='both', which='major',linewidth=1)
        ax.grid(which='minor', linestyle='-', alpha=0.5,linewidth=.5)
        ax.tick_params(axis='both', which='major', labelsize=8,pad=.5)
        if s==0:
            ax.set_ylabel(labels[i],fontsize=10)
            if i==0:
                ax.set_title('Unstable')
        elif i==0:
            ax.set_title('Stable')
            ax.legend(fontsize=8)
        if i==2:
            ax.set_xlabel(r'Windspeed ($m\ s^{-1}$)',fontsize=10)
plt.subplots_adjust(hspace=.05,wspace=.25)
plt.savefig('../../plot_output/a1/a1_wind_v_bias.png', bbox_inches = "tight")

# %%
sz=1
from matplotlib.ticker import FixedLocator
fig,axs=plt.subplots(3,2,figsize=(5*sz,4*sz),dpi=500)
labels=[r'$MAD$ $\Phi_u$',r'$MAD$ $\Phi_v$',r'$MAD$ $\Phi_w$']
for i in range(3):
    for s in range(2):
        ax=axs[i,s]
        #ax.plot(ubr[s,i,:],[0]*Nu,color='white',linewidth=2)
        ax.plot(ubr[s,i,:],mad[s,i,0,:],'-o',linewidth=.75,markersize=1.5,zorder=2,color='black',alpha=.3,label='MOST')
        ax.plot(ubr[s,i,:],mad[s,i,1,:],'-o',linewidth=.75,markersize=1.5,zorder=3,color='mediumpurple',alpha=.5,label='SC23 fit')
        ax.plot(ubr[s,i,:],mad[s,i,2,:],'-o',linewidth=.75,markersize=1.5,zorder=4,color='goldenrod',alpha=.5,label='Neon fit')
        #ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12],[0,'','','','',5,'','','','',10,'',''])
        ax.set_xticks([0,5,10])
        ax.xaxis.set_minor_locator(FixedLocator([0,1,2,3,4,5,6,7,8,9,10,11,12]))
        ax.grid(axis='both', which='major',linewidth=1)
        ax.grid(which='minor', linestyle='-', alpha=0.5,linewidth=.5)
        ax.tick_params(axis='both', which='major', labelsize=8,pad=.5)
        if s==0:
            ax.set_ylabel(labels[i],fontsize=10)
            if i==0:
                ax.set_title('Unstable')
        elif i==0:
            ax.set_title('Stable')
            ax.legend(fontsize=8)
        if i==2:
            ax.set_xlabel(r'Windspeed ($m\ s^{-1}$)',fontsize=10)
plt.subplots_adjust(hspace=.05,wspace=.2)
plt.savefig('../../plot_output/a1/a1_wind_v_mad.png', bbox_inches = "tight")

# %%
print(matplotlib.colors.cnames["goldenrod"])

# %% [markdown]
# # Scratch

# %%
i=0
fig=plt.figure(figsize=(3,4))
sbf = fig.subfigures(2, 1, hspace=0,wspace=0,frameon=False)
ax=sbf[1].add_subplot(111)
fpsites=fps['SITE'][:]
for site in np.unique(fpsites):
    m=fpsites==site
    ss=str(site)[2:-1]
    ffu=h5py.File('/home/tswater/tyche/data/neon/foot_stats/'+ss+'_U.h5','r')
    c=stats.mode(ffu['nlcd_dom'])[0]
    try:
        colors=class_colors[c]
    except Exception as e:
        colors='darkgreen'
        print(e)
    y,binEdges=np.histogram(fps['Ustr'][m],bins=np.linspace(0,10),density=True)
    bincenters=.5*(binEdges[1:]+binEdges[:-1])
    ax.plot(bincenters,y,c=colors,linewidth=1,alpha=.75)
    i=i+1
ax.set_xlabel(r'$y_B$')
ax.set_ylabel('Frequency')
#ax.set_xticks([0,.25,.5,.75])
#ax.set_xlim(-.05,.8)

i=0
ax=sbf[0].add_subplot(111)
fpsites=fpu['SITE'][:]
for site in np.unique(fpsites):
    m=fpsites==site
    ss=str(site)[2:-1]
    ffu=h5py.File('/home/tswater/tyche/data/neon/foot_stats/'+ss+'_U.h5','r')
    c=stats.mode(ffu['nlcd_dom'])[0]
    try:
        colors=class_colors[c]
    except Exception as e:
        colors='darkgreen'
        print(e)
    y,binEdges=np.histogram(fpu['Ustr'][m],bins=np.linspace(0,10),density=True)
    bincenters=.5*(binEdges[1:]+binEdges[:-1])
    ax.plot(bincenters,y,c=colors,linewidth=1,alpha=.75)
    i=i+1
    #plt.hist(fpu['ANI_XB'][m],bins=np.linspace(0,1))
    #ss=str(site)[2:-1]
    #plt.title(str(site)+': '+str(d_u['U']['MAD_OLD_s'][ss])[0:5])
#ax.set_xlabel(r'$x_B$')
#ax.set_xticks([0,.25,.5,.75],[])
#ax.set_xlim(-.05,.8)
ax.set_ylabel('Frequency')
#sbf[0].colorbar(sm,cax=ax.inset_axes([0.95, 0, 0.05, 1]),label='$MAD$')

# %%
plt.hist(fpu['Ustr'][:],bins=np.linspace(0,20,200))

# %%
dt=list(fpu['Ustr'][:])
dt.extend(list(fps['Ustr'][::31]))
print(np.nanpercentile(dt,95))

# %%

# %%

# %%
.223-.429*.5+.328*.5**2

# %%
