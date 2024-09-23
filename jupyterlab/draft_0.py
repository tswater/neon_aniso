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
mpl.rcParams['figure.dpi'] = 100
sns.set_theme()
plt.rcParams.update({'figure.max_open_warning': 0})

# %% [markdown]
# # COLOR STUFF

# %%
vmx_a=.7
vmn_a=.1
vmx_a2=1
vmn_a2=0

# %%
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
def cani_norm2(x):
    try:
        x_=x.copy()
        x_[x_<vmn_a2]=vmn_a2
        x_[x_>vmx_a2]=vmx_a2
    except:
        x_=x.copy()
        if x_>vmx_a2:
            x_=vmx_a2
        elif x_<vmn_a2:
            x_=vmn_a2
    x_=(x_-vmn_a2)/(vmx_a2-vmn_a2)
    return cmap_ani(x_)


# %%

# %%
indir='/home/tsw35/soteria/neon_advanced/qaqc_data/'
#NEON_TW_S_CO2.h5
#NEON_TW_S_H2O.h5
#NEON_TW_S_UVWT.h5
#NEON_TW_U_CO2.h5
#NEON_TW_U_H2O.h5
#NEON_TW_U_UVWT.h5

# %%
fp=h5py.File(indir+'NEON_TW_U_UVWT.h5','r')
len(fp['TIME'][:])

# %%
fp['LE']


# %%

# %%
def binplot1d(xx,yy,ani,xbins,anibins=np.array([-.05,.05,.15,.25,.35,.45,.55,.65,.75,.85]),mincnt=10):
    xplot=(xbins[0:-1]+xbins[1:])/2
    yplot=np.zeros((len(anibins)-1,len(xplot)))
    aniplot=(anibins[0:-1]+anibins[1:])/2
    for i in range(len(anibins)-1):
        for j in range(len(xbins)-1):
            pnts=yy[(ani>=anibins[i])&(ani<anibins[i+1])&(xx>=xbins[j])&(xx<xbins[j+1])]
            if len(pnts)<mincnt:
                yplot[i,j]=float('nan')
            else:
                yplot[i,j]=np.nanmedian(pnts)
    return xplot,yplot,aniplot


# %%
sigu=np.sqrt(fp['UU'][:])
sigv=np.sqrt(fp['VV'][:])
sigw=np.sqrt(fp['WW'][:])
ani=fp['ANI_YB'][:]
ustar=fp['USTAR'][:]
lmost=fp['L_MOST'][:]
zzd=fp['tow_height'][:]-fp['zd'][:]

# %%
plt.hist(np.log(-zzd/lmost),bins=np.logspace(-5,1))
plt.title('')

# %%
xplt,yplt,aplt=binplot1d(zzd/lmost,sigu/ustar,ani,-(np.logspace(-4,2,21)[-1:0:-1]),np.linspace(vmn_a,vmx_a,11),mincnt=50)

# %%
xplt,yplt,aplt=binplot1d(zzd/lmost,sigu/ustar,ani,-(np.logspace(-4,2,21)[-1:0:-1]),np.linspace(vmn_a,vmx_a,11),mincnt=50)
xx=-(zzd/lmost)
yy=(sigu/ustar)
cc=cani_norm(ani)
m=np.zeros((len(cc),))
m[0:100000]=1
np.random.shuffle(m)
m=m.astype(bool)
plt.scatter(xx[m],yy[m],color=cc[m],s=1,alpha=.1)

for i in range(yplt.shape[0]):
    plt.semilogx(-xplt,yplt[i,:],color=cani_norm(aplt[i]),linewidth=2,path_effects=[pe.Stroke(linewidth=3, foreground='w'), pe.Normal()])

ax=plt.gca()
ax.tick_params(which="both", bottom=True)
locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )) 
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xticks([10**-3,10**-2,10**-1,1,10],[r'$-10^{-3}$','',r'$-10^{-1}$','',r'$-10^{1}$'])
ax.set_xlim(10**(-3.5),10**(1.1))
plt.ylim(0.9,8)
plt.gca().invert_xaxis()
#plt.legend(aplt)

# %%
fp.close()

# %%

# %%
x=np.linspace(0,np.sqrt(3)/2,50)
y=np.sin(x*np.pi/np.sqrt(3)*2)
plt.scatter(x,y,c=x,cmap=cmap,norm=mpl.colors.Normalize(vmin=0,vmax=np.sqrt(3)/2),s=100)

# %%
m=(ani<.1)&(ani>0)
xx=np.log(-(zzd/lmost)[m])
yy=(sigu/ustar)[m]
plt.hexbin(xx,yy,cmap='turbo',bins=400,mincnt=1,extent=(-7,4,0,15))

# %%
m=(ani<.15)&(ani>.05)
xx=np.log(-(zzd/lmost)[m])
yy=(sigu/ustar)[m]
plt.hexbin(xx,yy,cmap='turbo',bins=400,mincnt=1)#,extent=(-7,4,0,10))

# %%
plt.figure(dpi=300)
xx=np.log(-(zzd/lmost))
yy=(sigu/ustar)
cc=cani_norm(ani)
m=np.zeros((len(cc),))
m[0:100000]=1
np.random.shuffle(m)
m=m.astype(bool)
m=m&(ani<.15)
plt.scatter(xx[m],yy[m],color=cc[m],s=1,alpha=.1)
plt.ylim(0,10)
plt.xlim(-5,2)
plt.gca().invert_xaxis()

# %%

# %%
plt.hist(np.log(-(zzd/lmost)),bins=np.linspace(-6,4))

# %%
fpu=h5py.File(indir+'NEON_TW_U_UVWT.h5','r')
fps=h5py.File(indir+'NEON_TW_S_UVWT.h5','r')
ani_s=fps['ANI_YB'][:]
ani_u=fpu['ANI_YB'][:]

# %%
plt.hist(fpu['USTAR'][:],bins=np.linspace(0,5))
plt.title('')

# %%
plt.hist(fps['USTAR'][:],bins=np.linspace(0,5))
plt.title('')

# %%
np.sum((ani_s<0)|(ani_s>(np.sqrt(3)/2)))

# %%
plt.hist(fps['VV'][:],bins=50)
plt.title('')

# %%
np.log10(10)

# %%
m=(('ABBY'.encode('UTF-8'))==fpu['SITE'][:])

# %%
np.sum(m)

# %%
(3/2)**2

# %% [markdown]
# # First Figure

# %%
d_unst=pickle.load(open('/home/tsw35/soteria/neon_advanced/data/d_unst_v2.p','rb'))
d_stbl=pickle.load(open('/home/tsw35/soteria/neon_advanced/data/d_stbl_v2.p','rb'))
d_ani_ust=pickle.load(open('/home/tsw35/soteria/neon_advanced/data/d_ani_ust_v2.p','rb'))
d_ani_stb=pickle.load(open('/home/tsw35/soteria/neon_advanced/data/d_ani_stb_v2.p','rb'))

# %%
d_unst.keys()

# %%
d_stbl['U'].keys()

# %%

# %%
sz=1.4
fig,axs=plt.subplots(6,3,figsize=(5*sz,9*sz),gridspec_kw={'width_ratios': [1,1,.06]},dpi=400)
ss=.1
alph=.2
minpct=1e-04

anibins=np.linspace(vmn_a,vmx_a,11)
anilvl=(anibins[0:-1]+anibins[1:])/2
anic=cani_norm(anilvl)
ylabels=[r'$\Phi_{u}$',r'$\Phi_{v}$',r'$\Phi_{w}$',r'$\Phi_{\theta}$',r'$\Phi_{q}$',r'$\Phi_{C}$']

j=0
for v in d_unst.keys():
    # SETUP
    cc=cani_norm(d_unst[v]['ani'][:])
    cc_s=cani_norm(d_stbl[v]['ani'][:])
    phi=d_unst[v]['phi']
    phi_s=d_stbl[v]['phi']
    
    ymin=min(np.nanpercentile(phi,.1),np.nanpercentile(phi_s,.1))
    ymax=max(np.nanpercentile(phi,99),np.nanpercentile(phi_s,99))
    if v in ['H2O','CO2']:
        ymax=10
    
    ##### UNSTABLE #####
    # SCATTER UNSTABLE
    axs[j,0].scatter(-d_unst[v]['zL'][:],phi,color=cc,s=ss,alpha=alph,marker=".")
    
    # LINES UNSTABLE
    xplt=d_unst[v]['p_zL']
    yplt=d_unst[v]['p_phi']
    cnt=d_unst[v]['p_cnt']
    tot=np.sum(cnt)
    yplt[cnt/tot<minpct]=float('nan')
    for i in range(yplt.shape[0]):
        if v in ['U','V','W','H2O','CO2']:
            axs[j,0].semilogx(-xplt,yplt[i,:],color=anic[i],linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
        else:
            axs[j,0].loglog(-xplt,yplt[i,:],color=anic[i],linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
    
    # LABELING
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )) 
    axs[j,0].xaxis.set_minor_locator(locmin)
    axs[j,0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if j==5:
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
    axs[j,1].scatter(d_stbl[v]['zL'][:],phi_s,color=cc,s=ss,alpha=alph,marker=".")
    
    # LINES STABLE
    xplt=d_stbl[v]['p_zL']
    yplt=d_stbl[v]['p_phi']
    cnt=d_stbl[v]['p_cnt']
    tot=np.sum(cnt)
    yplt[cnt/tot<minpct]=float('nan')
    for i in range(yplt.shape[0]):
        if v in ['U','V','W','H2O','CO2']:
            axs[j,1].semilogx(xplt,yplt[i,:],color=anic[i],linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
        else:
            axs[j,1].loglog(xplt,yplt[i,:],color=anic[i],linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
    
    # LABELING
    if j==5:
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

plt.subplots_adjust(hspace=.08,wspace=.02)

# %%
sz=1.4
ss=.1
alph=.2
minpct=1e-04
v='H2O'
cc=cani_norm(d_unst[v]['ani'][:])
phi=d_unst[v]['phi']
plt.scatter(3.4*(1-9.5*d_unst[v]['zL'][:])**(-1/3),phi,color=cc,s=ss,alpha=alph,marker=".")
plt.plot([-1,11],[-1,11])
plt.ylim(0,10)
plt.xlim(0,10)

# %%
sz=1.4
ss=.1
alph=.2
minpct=1e-04
v='H2O'
cc=cani_norm(d_unst[v]['ani'][:])
phi=d_unst[v]['phi']
plt.scatter(np.sqrt(30)*(1-25*d_unst[v]['zL'][:])**(-1/3),phi,color=cc,s=ss,alpha=alph,marker=".")
plt.plot([-1,11],[-1,11])
plt.ylim(0,10)
plt.xlim(0,10)

# %%

# %%

# %%
a=np.array([1,2,2,1,2,3,1,5,2,10,6,5,9,3,6,2,3,4,3,2,3,5,3,2,4,6,3,4,5,3,2,6,2,7,2,1])

# %%
print(np.std(a))
print(np.std(a*3))
print(np.std(a)*3)

# %%
plt.imshow(anic.reshape(10,1,4),origin='lower',interpolation=None)
plt.xticks([],[])
inta=[anilvl[0]-(anilvl[1]-anilvl[0])]
inta.extend(anilvl)
inta.extend([anilvl[-1]+(anilvl[1]-anilvl[0])])
xtc=np.interp([.1,.2,.3,.4,.5,.6,.7],inta,np.linspace(-1,10,12))
plt.yticks(xtc,[.1,.2,.3,.4,.5,.6,.7])
plt.grid(False)
#plt.yticks([.2,.3,.4,.5,.6,.7]),[.2,.3,.4,.5,.6,.7])
plt.gca().yaxis.tick_right()

# %%
xtc

# %%
plt.hist(d_stbl['T']['phi'][:],bins=np.linspace(-10,2))
plt.title('')

# %%
plt.hist(d_stbl['H2O']['phi'][:],bins=np.linspace(-5000,5000))
plt.title('')

# %%
plt.hist(d_unst['CO2']['phi'][:],bins=np.linspace(-5,5))
plt.title('')

# %%
np.nanmedian(np.abs(d_unst['CO2']['phi'][:]))

# %%
np.logspace(-5,-4,5)

# %%
np.sum(d_stbl['CO2']['phi']<0)/len(d_stbl['CO2']['phi'])

# %% [markdown]
# # Stiperski Curves/MOST

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

# T Unstable
a=0.017+0.217*ani
t_u_stp=1.07*(0.05+np.abs(zL))**(-1/3)+\
        (-1.14+a*(np.abs(zL)**(-9/10)))*(1-np.tanh(10*(np.abs(zL))**(2/3)))
t_u_old=.99*(.067-zL)**(-1/3)
t_u_old[zL>-0.05]=.015*(-zL[zL>-0.05])**(-1)+1.76

# H2O Unstable 
a=0.017+0.217*ani
h_u_stp=1.07*(0.05+np.abs(zL))**(-1/3)+\
        (-1.14+a*(np.abs(zL)**(-9/10)))*(1-np.tanh(10*(np.abs(zL))**(2/3)))
h_u_old=.99*(.067-zL)**(-1/3)
h_u_old[zL>-0.05]=.15*(-zL[zL>-0.05])**(-1)+1.76

# CO2 Unstable
a=0.017+0.217*ani
c_u_stp=1.07*(0.05+np.abs(zL))**(-1/3)+\
        (-1.14+a*(np.abs(zL)**(-9/10)))*(1-np.tanh(10*(np.abs(zL))**(2/3)))
c_u_old=.99*(.067-zL)**(-1/3)
c_u_old[zL>-0.05]=.15*(-zL[zL>-0.05])**(-1)+1.76

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
print('Ustb')
print(np.mean(a))
print(np.mean(c))

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
print('Vstb')
print(np.mean(a))
print(np.mean(c))


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
print('Wstb')
print(np.mean(a))
print(np.mean(c))


# T Stable
a_=[.607,-.754]
b_=[-.353,3.374,-8.544,6.297]
c_=[0.195,-1.857,5.042,-3.874]
d_=[.0763,-1.004,2.836,-2.53]
#d_=[.0763,-.1004,-2.836,-2.53]
a=0
b=0
c=0
d=0
for i in range(2):
    a=a+a_[i]*ani**i
for i in range(4):
    b=b+b_[i]*ani**i
    c=c+c_[i]*ani**i
    d=d+d_[i]*ani**i
    
logphi=a+b*np.log10(zL)+c*np.log10(zL)**2+d*np.log10(zL)**3
#logphi=a+b*np.log10(zL)+c*np.log10(zL**2)+d*np.log10(zL**3)
t_s_stp=10**(logphi)
#t_s_stp=np.sqrt(10**(logphi))
t_s_old=0.00087*(zL)**(-1.4)+2.03

h_s_stp=t_s_stp
c_s_stp=t_s_stp
h_s_old=t_s_old
c_s_old=t_s_old

ust_stp=[u_u_stp,v_u_stp,w_u_stp,t_u_stp,h_u_stp,c_u_stp]
stb_stp=[u_s_stp,v_s_stp,w_s_stp,t_s_stp,h_s_stp,c_s_stp]
ust_old=[u_u_old,v_u_old,w_u_old,t_u_old,h_u_old,c_u_old]
stb_old=[u_s_old,v_s_old,w_s_old,t_s_old,h_s_old,c_s_old]

# %%

# %%

# %%
sz=1.4
fig,axs=plt.subplots(6,3,figsize=(5*sz,9*sz),gridspec_kw={'width_ratios': [1,1,.06]})
ss=.03
alph=.1
minpct=1e-05

anibins=np.linspace(vmn_a,vmx_a,11)
anilvl=(anibins[0:-1]+anibins[1:])/2
anic=cani_norm(anilvl)
ylabels=[r'$\Phi_{u}$',r'$\Phi_{v}$',r'$\Phi_{w}$',r'$\Phi_{\theta}$',r'$\Phi_{q}$',r'$\Phi_{C}$']

j=0
for v in d_unst.keys():
    # SETUP
    cc=cani_norm(d_unst[v]['ani'][:])
    cc_s=cani_norm(d_stbl[v]['ani'][:])
    phi=d_unst[v]['phi']
    phi_s=d_stbl[v]['phi']
    
    ymin=min(np.nanpercentile(phi,.1),np.nanpercentile(phi_s,.1))
    ymax=max(np.nanpercentile(phi,99),np.nanpercentile(phi_s,99))
    
    ##### UNSTABLE #####
    # SCATTER UNSTABLE
    axs[j,0].scatter(-d_unst[v]['zL'][:],phi,color=cc,s=ss,alpha=alph)
    
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
        axs[j,0].semilogx(-xplt,yplt,'--',color='k',linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()],zorder=5)
    else:
        axs[j,0].loglog(-xplt,yplt,'--',color='k',linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()],zorder=5)

    
    # LABELING
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )) 
    axs[j,0].xaxis.set_minor_locator(locmin)
    axs[j,0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if j==5:
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
    axs[j,1].scatter(d_stbl[v]['zL'][:],phi_s,color=cc_s,s=ss,alpha=alph)
    
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
        axs[j,1].semilogx(xplt,yplt[:],'--',color='k',linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()],zorder=5)
    else:
        axs[j,1].loglog(xplt,yplt[:],'--',color='k',linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()],zorder=5)

    
    # LABELING
    if j==5:
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
    xtc=np.interp([.1,.2,.3,.4,.5,.6,.7],inta,np.linspace(-1,10,12))
    axs[j,2].set_yticks(xtc,[.1,.2,.3,.4,.5,.6,.7])
    axs[j,2].grid(False)
    #plt.yticks([.2,.3,.4,.5,.6,.7]),[.2,.3,.4,.5,.6,.7])
    axs[j,2].yaxis.tick_right()
    
    j=j+1

plt.subplots_adjust(hspace=.08,wspace=.02)

# %%
zL.shape
zL.reshape(1,40).repeat(10,0).shape

# %%
u_u_old.shape

# %%
ani.reshape(10,1).repeat(40,1).shape

# %%
stb_old[1].shape

# %%
yplt=ust_old

# %%

# %%

# %%
10**(5*np.log10(2.4))

# %%
10**(5*np.log10(2.4))*10**(-1)

# %%
d_=[.0763,-.1004,-2.836,-2.53]
d=0
for i in range(4):
    d=d+d_[i]*ani**i
print(d)

# %%
print(a)
print(b)
print(c)
print(d)

# %%
zL=-np.logspace(-4,2,40)
t_u_old=.99*(.067-zL)**(-1/3)
t_u_old[zL>-0.05]=.15*(-zL[zL>-0.05])**(-1)+1.76

# %%
.15*(--0.05)**(-1)-1

# %%
.99*(.067--0.05)**(-1/3)

# %% [markdown]
# # Anisotropy Analysis

# %%
class_names={11:'Water',12:'Snow',21:'Open Dev',22:'Low Dev',23:'Med Dev',24:'High Dev',31:'Barren',\
             41:'Decid.',42:'Everg.',43:'Mix Forest',51:'Dwarf Scrub',52:'Shrub',\
             71:'Grass',72:'AK:Sedge',73:'AK:Lichen',74:'AK:Moss',81:'Pasture',82:'Crops',90:'Wetland',95:'Herb Wet',0:'NaN'}
class_colors={11:'royalblue',12:'whitesmoke',21:'pink',22:'lightcoral',23:'red',24:'maroon',31:'silver',\
              41:'limegreen',42:'darkgreen',43:'yellowgreen',51:'darkgoldenrod',52:'tan',\
              71:'wheat',72:'khaki',73:'darkkhaki',74:'darkseagreen',81:'gold',82:'sandybrown',90:'skyblue',95:'cadetblue',0:'white'}
nlcdd={}
nlcdd['BONA']=43
nlcdd['TOOL']=51
nlcdd['BARR']=72
nlcdd['HEAL']=52
nlcdd['DEJU']=42
nlcdd['LAJA']=81
nlcdd['GUAN']=42
nlcdd['PUUM']=42


# %%
def nlcd_plot(d_,zb,stab,bxplt=True,nowhis=True,meansort=False):
    # Make a list of sites (remove ALL)
    sites=[]
    for site in d_.keys():
        if len(site)==4:
            sites.append(site)

    # Get Ani, nlcd, and percentiles
    ani={}
    nlcd={}
    pct75={}
    pct25={}
    for site in sites:
        if meansort:
            ani[site]=d_[site][zb][stab]['mean']
        else:
            ani[site]=d_[site][zb][stab]['median']
        pct75[site]=d_[site][zb][stab]['pct75']
        pct25[site]=d_[site][zb][stab]['pct25']
        if site in nlcdd.keys():
            nlcd[site]=nlcdd[site]
        else:
            nlcd[site]=d_[site][zb][stab]['lc']
            
    # make a list sorted by anisotropies and a flipped dictionary linking ani to site
    ani_f={}
    for k in ani.keys():
        ani_f[ani[k]]=k
    ani_sorted=list(ani_f.keys())
    ani_sorted.sort()
   
    # make empty sorted lists
    pct75_sorted=[]
    pct25_sorted=[]
    nlcd_sorted=[]
    sites_sorted=[]
    means_sorted=[]
    
    # fill sorted lists
    for k in ani_sorted:
        site=ani_f[k]
        pct75_sorted.append(pct75[site])
        pct25_sorted.append(pct25[site])
        nlcd_sorted.append(nlcd[site])
        sites_sorted.append(site)
        means_sorted.append(d_[site][zb][stab]['mean'])
    
    if bxplt:
        stats=[]
        for i in range(len(sites_sorted)):
            site=sites_sorted[i]
            min_=d_[site][zb][stab]['min']
            max_=d_[site][zb][stab]['max']
            iqr=pct75_sorted[i]-pct25_sorted[i]
            
            whishi=min(pct75_sorted[i]+iqr*1.5,max_)
            whislo=max(pct25_sorted[i]-iqr*1.5,min_)
            
            if nowhis:
                sd={'med':ani_sorted[i], 'q1':pct25_sorted[i], 'q3':pct75_sorted[i],
                'whislo':pct25_sorted[i],'whishi':pct75_sorted[i]}
            else:
                sd={'med':ani_sorted[i], 'q1':pct25_sorted[i], 'q3':pct75_sorted[i],
                'whislo':whislo,'whishi':whishi}
            stats.append(sd)
        return stats, nlcd_sorted, sites_sortedf
        
    else:
        return ani_sorted, pct25_sorted, pct75_sorted, nlcd_sorted, sites_sorted



# %%
stats,nlcd,sites_=nlcd_plot(d_ani_ust,'yb','full')

# %%
stats,nlcd,sites_=nlcd_plot(d_ani_ust,'yb','full')
a=1
_, ax = plt.subplots(figsize=(13*a,3.5*a));
bplt=ax.bxp(stats, showfliers=False,patch_artist=True);
i=0
for patch in bplt['boxes']:
    patch.set_facecolor(class_colors[nlcd[i]])
    i=i+1
ax.set_xticks(np.linspace(1,47,47),sites_)
ax.tick_params(axis='x',labelrotation=45)
#plt.ylim(.2,.47)

# %%
stats,nlcd,sites_=nlcd_plot(d_ani_ust,'yb','hi')
a=1
_, ax = plt.subplots(figsize=(13*a,3.5*a));
bplt=ax.bxp(stats, showfliers=False,patch_artist=True);
i=0
for patch in bplt['boxes']:
    patch.set_facecolor(class_colors[nlcd[i]])
    i=i+1
ax.set_xticks(np.linspace(1,47,47),sites_)
ax.tick_params(axis='x',labelrotation=45)
#plt.ylim(.2,.47)

# %%
stats,nlcd,sites_=nlcd_plot(d_ani_ust,'yb','lo')
a=1
_, ax = plt.subplots(figsize=(13*a,3.5*a));
bplt=ax.bxp(stats, showfliers=False,patch_artist=True);
i=0
for patch in bplt['boxes']:
    patch.set_facecolor(class_colors[nlcd[i]])
    i=i+1
ax.set_xticks(np.linspace(1,47,47),sites_)
ax.tick_params(axis='x',labelrotation=45)
#plt.ylim(.2,.47)

# %%
stats,nlcd,sites_=nlcd_plot(d_ani_stb,'yb','full')
a=1
_, ax = plt.subplots(figsize=(13*a,3.5*a));
bplt=ax.bxp(stats, showfliers=False,patch_artist=True);
i=0
for patch in bplt['boxes']:
    patch.set_facecolor(class_colors[nlcd[i]])
    i=i+1
ax.set_xticks(np.linspace(1,47,47),sites_)
ax.tick_params(axis='x',labelrotation=45)
#plt.ylim(.2,.47)

# %%
stats,nlcd,sites_=nlcd_plot(d_ani_stb,'yb','hi')
a=1
_, ax = plt.subplots(figsize=(13*a,3.5*a));
bplt=ax.bxp(stats, showfliers=False,patch_artist=True);
i=0
for patch in bplt['boxes']:
    patch.set_facecolor(class_colors[nlcd[i]])
    i=i+1
ax.set_xticks(np.linspace(1,47,47),sites_)
ax.tick_params(axis='x',labelrotation=45)
#plt.ylim(.2,.47)

# %%
stats,nlcd,sites_=nlcd_plot(d_ani_stb,'yb','lo')
a=1
_, ax = plt.subplots(figsize=(13*a,3.5*a));
bplt=ax.bxp(stats, showfliers=False,patch_artist=True);
i=0
for patch in bplt['boxes']:
    patch.set_facecolor(class_colors[nlcd[i]])
    i=i+1
ax.set_xticks(np.linspace(1,47,47),sites_)
ax.tick_params(axis='x',labelrotation=45)
#plt.ylim(.2,.47)

# %%
#return ani_sorted, means_sorted, pct25_sorted, pct75_sorted, nlcd_sorted, sites_sorted


# %% [markdown]
# # SS by Stability

# %%
d_unst['U']['SSlo_s'].keys()

# %%
sz=1
j=0
fig,axs=plt.subplots(4,1,figsize=(6*sz,9*sz))
names=[]
names.append(r'$\zeta<-.1$')
names.append(r'$\zeta<0$')
names.append(r'$-.1<\zeta<0$')
names.append(r'$0<\zeta<.1$')
names.append(r'$0<\zeta$')
names.append(r'$.1<\zeta$')

for v in list(d_unst.keys())[0:4]:
    ss=[]
    ss.append(d_unst[v]['SShi'])
    ss.append(d_unst[v]['SS'])
    ss.append(d_unst[v]['SSlo'])
    
    ss.append(d_stbl[v]['SSlo'])
    ss.append(d_stbl[v]['SS'])
    ss.append(d_stbl[v]['SShi'])
    
    axs[j].bar(names,ss)
    axs[j].plot([2.5,2.5],[-1,1],'k--')
    if j<3:
        axs[j].set_xticks([1,2,3,4,5,6],[])
    axs[j].set_ylabel(ylabels[j])
    axs[j].set_ylim(-.45,.75)
    axs[j].tick_params(axis='x',labelrotation=45)
    j=j+1

# %%

# %% [markdown]
# # SS by Site and Stability

# %%
sz=1
j=0
fig,axs=plt.subplots(4,1,figsize=(6*sz,9*sz))
names=[]
names.append(r'$\zeta<-.1$')
names.append(r'$\zeta<0$')
names.append(r'$-.1<\zeta<0$')
names.append(r'$0<\zeta<.1$')
names.append(r'$0<\zeta$')
names.append(r'$.1<\zeta$')

for v in list(d_unst.keys())[0:4]:
    ss=[]
    ss.append(list(d_unst[v]['SShi_s'].values()))
    ss.append(list(d_unst[v]['SS_s'].values()))
    ss.append(list(d_unst[v]['SSlo_s'].values()))
    
    ss.append(list(d_stbl[v]['SSlo_s'].values()))
    ss.append(list(d_stbl[v]['SS_s'].values()))
    ss.append(list(d_stbl[v]['SShi_s'].values()))
    
    ss=np.array(ss)
    axs[j].plot([0,7],[0,0],color='w',linewidth=3)
    axs[j].boxplot(ss.T,labels=names)
    axs[j].plot([3.5,3.5],[-1,1],'k--')
    if j<3:
        axs[j].set_xticks([1,2,3,4,5,6],[])
    axs[j].set_ylabel(ylabels[j])
    axs[j].set_ylim(-.8,.8)
    axs[j].set_xlim(.5,6.5)
    axs[j].tick_params(axis='x',labelrotation=45)
    j=j+1

# %%

# %% [markdown]
# # SS by Site and Stability Heatmap 

# %%
data=np.zeros((47,8))
a=1.01
_, ax = plt.subplots(figsize=(13*a,3.5*a))
sites=list(d_unst[v]['SShi_s'].keys())
sites.sort()
var_=list(d_unst.keys())[0:4]
for i in range(len(sites)):
    site=sites[i]
    for j in range(4):
        v=var_[j]
        data[i,j*2]=d_unst[v]['SS_s'][site]
        data[i,j*2+1]=d_stbl[v]['SS_s'][site]

plt.imshow(data.T,cmap='seismic',vmin=-.6,vmax=.6)
plt.xticks(np.linspace(0,46,47),sites,rotation=45)
plt.yticks([-.5,1.5,3.5,5.5,7.5],['\n\nU   $\zeta>0$\n$\zeta<0$','\n\nV   $\zeta>0$\n$\zeta<0$','\n\nW   $\zeta>0$\n$\zeta<0$','\n\n$θ$   $\zeta>0$\n$\zeta<0$',''])
plt.grid(False,axis='x')
plt.grid(True,axis='y',color='k',linewidth=2)

# %%
bounds=[]
clist=[]
names=[]
cck=list(class_colors.keys())
cck.sort()
for i in cck:
    bounds.append(i)
    clist.append(class_colors[i])
    names.append(class_names[i])
bounds.append(100)
cmap = matplotlib.colors.ListedColormap(clist)
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

# %%
nlcds=[]
for site in sites:
    if site in nlcdd.keys():
        nlcds.append(nlcdd[site])
    else:
        nlcds.append(d_ani_ust[site]['yb']['lo']['lc'])

# %%
plt.figure(figsize=(13*a,3.5*a),dpi=300)
plt.imshow(np.array(nlcds).reshape(1,47),cmap=cmap,norm=norm)
plt.grid(False)
plt.yticks([0],['NLCD'])
plt.xticks([],[])

# %%
nnn=np.unique(nlcd)
clist=[]
names=[]
for i in nnn:
    clist.append(class_colors[i])
    names.append(class_names[i])

# %%
names

# %%
fig=plt.figure(figsize=(20,1.5),dpi=500)
ax = fig.add_subplot(1, 1, 1)
line=[0,1]
for i in clist:
    plt.bar(line,line,color=i)
ax.set_facecolor('white')
plt.legend(names,ncol=10,handletextpad=.1,fontsize=15,framealpha=0,columnspacing=0.7)
plt.grid(False)
plt.ylim(0,10)

# %% [markdown]
# # Focus Sites Plot

# %%

# %%

# %%

# %%

# %% [markdown]
# # Curve Fit Testing

# %%

# %%

# %%

# %%
40*10**6/1440/365/47

# %%
1.1*10**6/48/365/47

# %%
fp=h5py.File('/home/tsw35/soteria/neon_advanced/qaqc_data/NEON_TW_U_UVWT.h5','r')


# %%

def U_ust(zL,a):
    return a*(1-3*zL)**(1/3)


# %%
sigu=np.sqrt(fp['UU'][:])
sigv=np.sqrt(fp['VV'][:])
sigw=np.sqrt(fp['WW'][:])
ani=fp['ANI_YB'][:]
ustar=fp['USTAR'][:]
lmost=fp['L_MOST'][:]
zzd=fp['tow_height'][:]-fp['zd'][:]
zL=zzd/lmost
phi=sigu/ustar
fpsites=fp['SITE'][:]

# %%
count={}
for site in np.unique(fpsites):
    count[site]=np.sum(fpsites==site)


# %%
def C_ust(zL,a,c):
    return a*(1-25*zL)**(-1/3)


# %%
for site in count.keys():
    print(str(site)+': '+str(count[site]/20000))

# %%
np.mean(list(count.values()))


# %%
def evenbin(a_,bins=15):
    aa=a_[(a_>=vmn_a)&(a_<=vmx_a)]
    binedge=[]
    for i in range(bins+1):
        binedge.append(np.percentile(aa,i/bins*100))
    return binedge


# %%
abin=evenbin(ani,bins=122)

# %%

# %%
abin=evenbin(ani,bins=100)
abin=np.array(abin)
params=np.ones((len(abin)-1,1))
pvars=np.ones((len(abin)-1,1))
for i in range(len(abin)-1):
    m=(ani>abin[i])&(ani<abin[i+1])
    params[i,:],pcov=sci.optimize.curve_fit(U_ust,zL[m],phi[m])

# %%
pcov[0,0]

# %%
zL_plot=-np.logspace(-4,2)
clrs=cani_norm(np.array((abin[0:-1]+abin[1:])/2))
for i in range(len(clrs)):
    plt.semilogx(-zL_plot,U_ust(zL_plot,params[i][0]),color=clrs[i])
plt.gca().invert_xaxis()

# %%
plt.plot((abin[0:-1]+abin[1:])/2,params.reshape(100))

# %%
plt.hexbin(np.log10(-zL),ani,mincnt=1,cmap='turbo',gridsize=500,extent=(-4.5,2,0,.8))

# %%
fps=h5py.File('/home/tsw35/soteria/neon_advanced/qaqc_data/NEON_TW_S_UVWT.h5','r')

# %%
ani2=fps['ANI_YB'][:]
zL2=(fps['tow_height'][:]-fps['zd'][:])/fps['L_MOST']

# %%
plt.hexbin(np.log10(zL2),ani2,mincnt=1,cmap='turbo',gridsize=500,extent=(-4.5,2,0,.8))


# %%
aaa=evenbin(ani2,25)

# %%
len(ani2[(ani2>vmn_a)&(ani2<vmx_a)])

# %%
abin

# %%
len(ani[(ani>vmn_a)&(ani<vmx_a)])/10000

# %%
fp=h5py.File('/home/tsw35/soteria/neon_advanced/qaqc_data/NEON_TW_U_H2O.h5','r')


# %%
def C_ust(zL,a,c):
    return a*(1+c*zL)**(-1/3)


# %%
def C_ust2(zL,a):
    return a*(1-20*zL)**(-1/3)


# %%
molh2o=18.02*10**(-3)
moldry=28.97*10**(-3)
kgdry_m3=fp['RHO'][:]/(fp['H2O'][:]/1000+1)
rr = fp['H2O_SIGMA'][:]*molh2o/moldry
lhv=2500827 - 2360*(fp['T_SONIC'][:]-273)
phi=np.abs(rr/(fp['LE'][:]/lhv/fp['USTAR'][:]))*kgdry_m3/10**3
ani=fp['ANI_YB'][:]
ustar=fp['USTAR'][:]
lmost=fp['L_MOST'][:]
zzd=fp['tow_height'][:]-fp['zd'][:]
zL=zzd/lmost

# %%
abin=evenbin(ani,bins=110)
abin=np.array(abin)
params=np.ones((len(abin)-1,2))
for i in range(len(abin)-1):
    m=(ani>abin[i])&(ani<abin[i+1])
    params[i,:],_=sci.optimize.curve_fit(C_ust,zL[m],phi[m],bounds=([-np.inf,-30],[np.inf,-1]))

# %%
zL_plot=-np.logspace(-4,2)
clrs=cani_norm(np.array((abin[0:-1]+abin[1:])/2))
for i in range(len(clrs)):
    plt.semilogx(-zL_plot,C_ust(zL_plot,params[i][0],params[i][1]),color=clrs[i])
plt.gca().invert_xaxis()
plt.ylim(0,15)

# %%
abin=evenbin(ani,bins=110)
abin=np.array(abin)
params=np.ones((len(abin)-1,1))
for i in range(len(abin)-1):
    m=(ani>abin[i])&(ani<abin[i+1])
    params[i,:],_=sci.optimize.curve_fit(C_ust2,zL[m],phi[m])

# %%
zL_plot=-np.logspace(-4,2)
clrs=cani_norm(np.array((abin[0:-1]+abin[1:])/2))
for i in range(len(clrs)):
    plt.semilogx(-zL_plot,C_ust2(zL_plot,params[i][0]),color=clrs[i])
plt.gca().invert_xaxis()
plt.ylim(0,15)

# %%
fp=h5py.File('/home/tsw35/soteria/neon_advanced/qaqc_data/NEON_TW_S_UVWT.h5','r')

# %%
ani=fp['ANI_YB'][:]
ustar=fp['USTAR'][:]
lmost=fp['L_MOST'][:]
zzd=fp['tow_height'][:]-fp['zd'][:]
zL=zzd/lmost


# %%
def U_stb(zL,a,c):
    phi=a*(1+3*zL)**(c)
    return phi
def T_stb(zL,a,b,c,d):
    logphi=a+b*np.log10(zL)+c*np.log10(zL)**2+d*np.log10(zL)**3
    phi=10**(logphi)
    return phi


# %%
phiu=np.sqrt(fp['UU'][:])/ustar
phiv=np.sqrt(fp['VV'][:])/ustar
phiw=np.sqrt(fp['WW'][:])/ustar
phiT=fp['T_SONIC_SIGMA'][:]*fp['USTAR'][:]/fp['WTHETA'][:]

# %%
abin=evenbin(ani,bins=110)
abin=np.array(abin)
params=np.ones((len(abin)-1,2))
for i in range(len(abin)-1):
    print('.',end='',flush=True)
    m=(ani>abin[i])&(ani<abin[i+1])
    try:
        params[i,:],_=sci.optimize.curve_fit(U_stb,zL[m],phiu[m],p0=[2,.08])
    except:
        print(abin[i])

# %%
zL_plot=np.logspace(-4,2)
clrs=cani_norm(np.array((abin[0:-1]+abin[1:])/2))
for i in range(len(clrs)):
    if params[i][0]==1:
        continue
    plt.semilogx(zL_plot,U_stb(zL_plot,params[i][0],params[i][1]),color=clrs[i])

# %%

# %%
params=np.ones((len(abin)-1,2))
for i in range(len(abin)-1):
    print('.',end='',flush=True)
    m=(ani>abin[i])&(ani<abin[i+1])
    try:
        params[i,:],_=sci.optimize.curve_fit(U_stb,zL[m],phiv[m],p0=[2,.08])
    except:
        print(abin[i])

# %%
zL_plot=np.logspace(-4,2)
clrs=cani_norm(np.array((abin[0:-1]+abin[1:])/2))
for i in range(len(clrs)):
    if params[i][0]==1:
        continue
    plt.semilogx(zL_plot,U_stb(zL_plot,params[i][0],params[i][1]),color=clrs[i])

# %%

# %%
params2=np.ones((len(abin)-1,2))
for i in range(len(abin)-1):
    print('.',end='',flush=True)
    m=(ani>abin[i])&(ani<abin[i+1])
    try:
        params2[i,:],_=sci.optimize.curve_fit(U_stb,zL[m],phiw[m],p0=[1.4,.03])
    except:
        print(abin[i])

# %%
zL_plot=np.logspace(-4,2)
clrs=cani_norm(np.array((abin[0:-1]+abin[1:])/2))
for i in range(len(clrs)):
    if params[i][0]==1:
        continue
    plt.semilogx(zL_plot,U_stb(zL_plot,params2[i][0],params2[i][1]),color=clrs[i])


# %%

# %%

# %%
def T_stb(zL,a,b,c,d):
    logphi=a+b*np.log10(zL)+c*np.log10(zL)**2+d*np.log10(zL)**3
    phi=10**(logphi)
    return phi


# %%
def T_stb2(zL,a,d):
    logphi=a+d*np.log10(zL)**3
    phi=10**(logphi)
    return phi


# %%
def T_stb3(zL,a,b,c):
    #t_s_old=0.00087*(zL_plot)**(-1.4)+2.03
    phi=a*(zL)**(b)+c
    return phi


# %%
#1+10**(-.005*np.log10(tt)**3)
def T_stb31(zL,a,b,c):
    #t_s_old=0.00087*(zL_plot)**(-1.4)+2.03
    logphi=a+b*5/6*np.log10(zL)+c*np.log10(zL)**2+b*np.log10(zL)**3
    phi=10**(logphi)
    return phi


# %%
#1+10**(-.005*np.log10(tt)**3)
def T_stb32(zL,a,b,c):
    #t_s_old=0.00087*(zL_plot)**(-1.4)+2.03
    logphi=a+c*np.log10(zL)-b*5/6*np.log10(zL)**2+b*np.log10(zL)**3
    phi=10**(logphi)
    return phi


# %%
#1+10**(-.005*np.log10(tt)**3)
def T_stb2(zL,a,b):
    #t_s_old=0.00087*(zL_plot)**(-1.4)+2.03
    logphi=a+b*5/6*np.log10(zL)-5/6*b*np.log10(zL)**2+b*np.log10(zL)**3
    phi=10**(logphi)
    return phi


# %%
fp=h5py.File('/home/tsw35/soteria/neon_advanced/qaqc_data/NEON_TW_S_UVWT.h5','r')

# %%
#abin=evenbin(ani,bins=25)
#abin=np.array(abin)
#params=np.ones((len(abin)-1,4))
#for i in range(len(abin)-1):
#    print('.',end='',flush=True)
#    m=(ani>abin[i])&(ani<abin[i+1])&m2
#    try:
        # [-1,-1,-1,-1],[1,1,1,1]
        # [-.1,-.4,-.2,-.4],[.7,.4,.2,.2]
#        params[i,:],_=sci.optimize.curve_fit(T_stb,zL[m],phiT[m],p0=[.5,-.1,.05,.1],bounds=([-.1,-.4,-.2,-.4],[.7,.4,.2,.2]),loss='cauchy')
#    except Exception as e:
#        print(e)

# %%
phiT=np.abs(fp['T_SONIC_SIGMA'][:]*fp['USTAR'][:]/fp['WTHETA'][:])
ani=fp['ANI_YB'][:]
ustar=fp['USTAR'][:]
lmost=fp['L_MOST'][:]
zzd=fp['tow_height'][:]-fp['zd'][:]
zL=zzd/lmost
m2=np.zeros((len(ani),))
m2[0:1000000]=1
np.random.shuffle(m2)
m2=m2.astype(bool)

# %%
#abin=evenbin(ani,bins=25)
#abin=np.array(abin)
params=np.ones((len(abin)-1,2))
for i in range(len(abin)-1):
    print('.',end='',flush=True)
    m=(ani>abin[i])&(ani<abin[i+1])&m2
    try:
        # [-1,-1,-1,-1],[1,1,1,1]
        # [-.1,-.4,-.2,-.4],[.7,.4,.2,.2]
        params[i,:],_=sci.optimize.curve_fit(T_stb2,zL[m],phiT[m],p0=[.5,-.1],bounds=([-1,-1,],[1,1]),loss='cauchy')
    except Exception as e:
        print(e)

# %%
plt.figure(dpi=300)
zL_plot=np.logspace(-4,2)
phi_s=d_stbl['T']['phi']
cc_s=cani_norm(d_stbl['T']['ani'][:])
ss=.1
alph=.1
clrs=cani_norm(np.array((abin[0:-1]+abin[1:])/2))
plt.scatter(d_stbl['T']['zL'][:],phi_s,color=cc_s,s=ss,alpha=alph)
for i in range(len(clrs)):
    if params[i][0]==1:
        continue
    plt.loglog(zL_plot,T_stb2(zL_plot,params[i][0],params[i][1]),color=clrs[i])
    #plt.loglog(zL_plot,T_stb(zL_plot,params[i][0],params[i][1],params[i][2],params[i][3]),color=clrs[i])
t_s_old=0.00087*(zL_plot)**(-1.4)+2.03
plt.semilogx(zL_plot,t_s_old,'k--')
plt.ylim(.5,100)


# %%

# %%
fp=h5py.File('/home/tsw35/soteria/neon_advanced/qaqc_data/NEON_TW_S_H2O.h5','r')
ani=fp['ANI_YB'][:]
ustar=fp['USTAR'][:]
lmost=fp['L_MOST'][:]
zzd=fp['tow_height'][:]-fp['zd'][:]
zL=zzd/lmost
molh2o=18.02*10**(-3)
moldry=28.97*10**(-3)
kgdry_m3=fp['RHO'][:]/(fp['H2O'][:]/1000+1)
rr = fp['H2O_SIGMA'][:]*molh2o/moldry
lhv=2500827 - 2360*(fp['T_SONIC'][:]-273)
phiH=np.abs(rr/(fp['LE'][:]/lhv/fp['USTAR'][:]))*kgdry_m3/10**3
#molh2o=18.02*10**(-3)
#moldry=28.97*10**(-3)
#co2 = fp['CO2_SIGMA'][:]
#co2=co2/10**6
#kgdry_m3=fp['RHO'][:]/(fp['H2O'][:]/1000+1)
#moldry_m3=kgdry_m3/moldry
#phi=np.abs(co2/(fp['CO2FX'][:]/fp['USTAR'][:]))*moldry_m3


# %%
def C_stb(zL,a,b,c):
    #logphi=a+b*np.log10(zL)+c*np.log10(zL)**2+d*np.log10(zL)**3
    logphi=a+b*np.log10(zL)+c*np.log10(zL)**3
    logphi=a+c*np.log10(zL)**3
    phi=10**(logphi)
    return phi


# %%
m2=np.zeros((len(ani),))
m2[0:1000000]=1
np.random.shuffle(m2)
m2=m2.astype(bool)
m2=m2&(~np.isnan(phiH))

# %%
#abin=evenbin(ani,bins=25)
#abin=np.array(abin)
params=np.ones((len(abin)-1,3))
for i in range(len(abin)-1):
    print('.',end='',flush=True)
    m=(ani>abin[i])&(ani<abin[i+1])&m2
    try:
        #[-1,-1,-1,-1],[1,1,1,1]
        # [-.1,-.4,-.2,-.4],[.7,.4,.2,.2]
        params[i,:],_=sci.optimize.curve_fit(C_stb,zL[m],phiH[m],p0=[.5,0,0],bounds=([-1,-1,-1],[1,1,1]),loss='cauchy')
    except Exception as e:
        print(e)

# %%
plt.figure(dpi=300)
zL_plot=np.logspace(-4,2)
clrs=cani_norm(np.array((abin[0:-1]+abin[1:])/2))
phi_s=d_stbl['H2O']['phi']
cc_s=cani_norm(d_stbl[v]['ani'][:])
plt.scatter(d_stbl['H2O']['zL'][:],phi_s,color=cc_s,s=ss,alpha=alph)
#params=para0.copy()
#params[:,2]=0
for i in range(len(clrs)):
    if params[i][0]==1:
        continue
    #plt.semilogx(zL_plot,T_stb(zL_plot,params[i][0],params[i][1],params[i][2],params[i][3]),color=clrs[i])
    plt.semilogx(zL_plot,C_stb(zL_plot,params[i][0],params[i][1],params[i][2]),color=clrs[i])
t_s_old=0.00087*(zL_plot)**(-1.4)+2.03
plt.semilogx(zL_plot,t_s_old,'k--')
plt.ylim(.5,10)

# %%
params

# %%

# %%

# %%
params[params==1]=float('nan')

# %%
np.nanmin(params,axis=0)

# %%
np.nanmax(params,axis=0)

# %%
#abin=evenbin(ani,bins=25)
#abin=np.array(abin)
params=np.ones((len(abin)-1,3))
for i in range(len(abin)-1):
    print('.',end='',flush=True)
    m=(ani>abin[i])&(ani<abin[i+1])&m2
    try:
        params[i,:],_=sci.optimize.curve_fit(T_stb3,zL[m],phiT[m],p0=[.00087,-1.4,2],bounds=([0,-2,0],[np.inf,-1,np.inf]))
    except Exception as e:
        print(e)

# %%
zL_plot=np.logspace(-4,2)
clrs=cani_norm(np.array((abin[0:-1]+abin[1:])/2))
for i in range(len(clrs)):
    if params[i][0]==1:
        continue
    plt.loglog(zL_plot,T_stb3(zL_plot,params[i][0],params[i][1],params[i][2]),color=clrs[i])
t_s_old=0.00087*(zL_plot)**(-1.4)+2.03
plt.loglog(zL_plot,t_s_old,'k--')
plt.ylim(.5,100)

# %%

# %%
#abin=evenbin(ani,bins=25)
#abin=np.array(abin)
params=np.ones((len(abin)-1,2))
for i in range(len(abin)-1):
    print('.',end='',flush=True)
    m=(ani>abin[i])&(ani<abin[i+1])&m2
    try:
        params[i,:],_=sci.optimize.curve_fit(T_stb4,zL[m],phiT[m],p0=[0,-.01],bounds=([0,-2],[np.inf,0]))
    except Exception as e:
        print(e)

# %%
zL_plot=np.logspace(-4,2)
clrs=cani_norm(np.array((abin[0:-1]+abin[1:])/2))
phi_s=d_stbl['T']['phi']
cc_s=cani_norm(d_stbl[v]['ani'][:])
plt.scatter(d_stbl['T']['zL'][:],phi_s,color=cc_s,s=ss,alpha=alph)
for i in range(len(clrs)):
    if params[i][0]==1:
        continue
    plt.plot(zL_plot,T_stb4(zL_plot,params[i][0],params[i][1]),color=clrs[i])
t_s_old=0.00087*(zL_plot)**(-1.4)+2.03
plt.semilogx(zL_plot,t_s_old,'k--')
plt.ylim(.5,100)

# %%

# %%

# %%

# %%

# %%
plt.hexbin(np.log10(zL),np.log10(phiT),mincnt=1,cmap='turbo',gridsize=500,extent=(-4.5,2,-1,2))

# %%

# %%
ani1=np.linspace(0,np.sqrt(3)/2)
a_=[.607,-.754]
b_=[-.353,3.374,-8.544,6.297]
c_=[0.195,-1.857,5.042,-3.874]
d_=[.0763,-1.004,2.836,-2.53]
#d_=[.0763,-.1004,-2.836,-2.53]
a=0
b=0
c=0
d=0
for i in range(2):
    a=a+a_[i]*ani1**i
for i in range(4):
    b=b+b_[i]*ani1**i
    c=c+c_[i]*ani1**i
    d=d+d_[i]*ani1**i

# %%
plt.plot(a)
plt.plot(b)
plt.plot(c)
plt.plot(d)
plt.legend(['1','2','3','4'])

# %%
tt=np.logspace(-4,2)

# %%
#plt.loglog(tt,T_stb3(tt,.001,-1.5,1))

#plt.loglog(tt,10**(-.05*np.log10(tt)**3))
plt.loglog(tt,1+10**(-.005*np.log10(tt)**3))


# %%
def T_stb(zL,a,b,c,d):
    logphi=a+b*np.log10(zL)+c*np.log10(zL)**2+d*np.log10(zL)**3
    phi=10**(logphi)
    return phi

def T_ust(zL,a):
    phi=1.07*(0.05+np.abs(zL))**(-1/3)+\
        (-1.14+a*(np.abs(zL)**(-9/10)))*(1-np.tanh(10*(np.abs(zL))**(2/3)))
    return phi

def U_stb(zL,a,c):
    phi=a*(1+3*zL)**(c)
    return phi

def U_ust(zL,a):
    return a*(1-3*zL)**(1/3)

def W_stb(zL,a,c):
    return U_stb(zL,a,c)

def W_ust(zL,a):
    return U_ust(zL,a)

def C_stb(zL,a,b,c,d):
    logphi=a+b*np.log10(zL)+c*np.log10(zL)**2+d*np.log10(zL)**3
    phi=10**(logphi)
    return phi

def C_ust(zL,a):
    #return a*(1+b*zL)**(-1/3)
    return a*(1-25*zL)**(-1/3)



# %%
d_fit=pickle.load(open('/home/tsw35/soteria/neon_advanced/data/d_fit_v1.p','rb'))
#dftc=pickle.load(open('/home/tsw35/soteria/neon_advanced/data/d_fit_CO2_v0.p','rb'))
#d_fit['CO2s']=dftc['CO2s']

# %%
d_fit['Uu'].keys()

# %%
d_fit=pickle.load(open('/home/tsw35/soteria/neon_advanced/data/d_fit_v1.p','rb'))

def C_ust(zL,a):
    #return a*(1+b*zL)**(-1/3)
    return a*(1-25*zL)**(-1/3)


# %%

# %%

# %%

# %%

# %%
for k in d_fit.keys():
    param=d_fit[k]['param']
    anibins=d_fit[k]['anibins']
    anilvls=(anibins[1:]+anibins[0:-1])/2
    pall=np.nanmean(param,axis=0)
    plt.figure(figsize=(4,3*len(pall)),dpi=200)
    for i in range(len(pall)):
        plt.subplot(len(pall),1,i+1)
        plt.scatter(anilvls,param[:,i])
        plt.ylim(np.nanpercentile(param[:,i],1),np.nanpercentile(param[:,i],99))
        plt.title(k)


# %%
param=d_fit['Ts']['param']

# %%
plt.hist(param[:,0],bins=20)

# %%
plt.subplot(len(pall),1,i+1)
plt.scatter(anilvls,param[:,i])
plt.ylim(np.nanpercentile(param[:,i],1),np.nanpercentile(param[:,i],99))
plt.title(k)

# %%
k='Ts'
param=d_fit[k]['param']
anibins=d_fit[k]['anibins']
anilvls=(anibins[1:]+anibins[0:-1])/2

# %%
m=(~np.isnan(param[:,3]))&(param[:,3]<0)

# %%
for i in range(4):
    plt.figure()
    plt.scatter(anilvls[m],param[:,i][m])

# %%
plt.scatter(param[:,2][m],param[:,3][m])

# %%
from numpy.polynomial import polynomial as P
deg=1
x=anilvls[m]
y=param[:,0][m]
c, stats = P.polyfit(x,y,deg,full=True)
plt.scatter(x,y)
y2=c[0]+c[1]*x
if deg>1:
    y2=y2+c[2]*x**2
if deg>2:
    y2=y2+c[3]*x**3
plt.plot(x,y2)

# %%
plt.scatter(param[:,1],param[:,3],c=param[:,2],cmap='turbo',vmax=0,vmin=-.07)
xx=np.linspace(-1,1)
plt.plot(xx,-xx*1.2)
plt.ylim(-.13,.01)
plt.xlim(-.01,.17)
plt.colorbar()

# %%
plt.scatter(param[:,2],param[:,3],c=param[:,1],cmap='turbo',vmin=-.01,vmax=.17)
xx=np.linspace(-1,1)
plt.plot(xx,xx*1.2)
plt.xlim(-.07,.01)
plt.ylim(-.13,.01)
plt.colorbar()

# %%
stats

# %%
degs={'Uu':[2],'Vu':[2],'Wu':[3],'Tu':[2],'H2Ou':[2,2],'CO2u':[2,3],
      'Us':[2,2],'Vs':[2,3],'Ws':[2,3],'Ts':[1,3,3,2],'H2Os':[1,3,3,3],'CO2s':[1,3,3,3]}
mins={}
maxs={'Tu':[.75],'H2Ou':[14],'CO2u':[40],'Ts':[100,100,100,10],'H2Os':[100,100,.025,.02],'CO2s':[100,100,125,.02]}

# %%
letters=['a','b','c','d']
for var in degs.keys():
    param=d_fit[var]['param']
    anibins=d_fit[var]['anibins']
    anilvls=(anibins[1:]+anibins[0:-1])/2
    pall=np.nanmean(param,axis=0)
    m=~np.isnan(param[:,0])
    if var in maxs.keys():
        for mx in range(len(maxs[var])):
            m=m&(param[:,mx]<maxs[var][mx])
    x=anilvls[m]
    absz=anibins[1:]-anibins[0:-1]
    absz=absz[m]
    plt.figure(figsize=(4,3*len(pall)),dpi=200)
    print("'"+str(var)+"':{",end='')
    for i in range(len(pall)):
        y=param[:,i][m]
        c = P.polyfit(x,y,degs[var][i])
        plt.subplot(len(pall),1,i+1)
        plt.scatter(anilvls[m],param[:,i][m])
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
        plt.plot(x,y2,'k--')
        #plt.ylim(np.nanpercentile(param[:,i],1)*.9,np.nanpercentile(param[:,i],99))
        plt.title(var)
    print('},')

# %%
plt.scatter(param[:,2],param[:,3])
xx=np.linspace(-.025,.025)
plt.plot(xx,xx*.2-.002)
plt.xlim(-.025,.005)
plt.ylim(-.01,0)

# %%
plt.scatter(param[:,1],param[:,2])
xx=np.linspace(-.025,.025)
plt.plot(xx,xx*.2-.002)
plt.xlim(-.07,.01)
plt.ylim(-.025,.005)


# %%
def C_ust(zL,a):
    #return a*(1+b*zL)**(-1/3)
    return a*(1-25*zL)**(-1/3)



# %%
zll=-np.logspace(-4,2)
plt.semilogx(-zll,C_ust(zll,7))
plt.semilogx(-zll,C_ust(zll,12))
plt.semilogx(-zll,C_ust(zll,15))
plt.semilogx(-zll,C_ust(zll,20))
plt.semilogx(-zll,C_ust(zll,30))
plt.gca().invert_xaxis()

# %%
len(d_fit.keys())

# %% [markdown]
# # DSM/DTM/CHM

# %%
dsm_dir='/home/tsw35/soteria/data/NEON/dsm/_tifs/'
dtm_dir='/home/tsw35/soteria/data/NEON/dtm/_tifs/'

# %%
os.listdir(dtm_dir)

# %%
os.listdir(dsm_dir)

# %%
fp=h5py.File(indir+'NEON_TW_U_UVWT.h5','r')
fp.keys()

# %%
sites=np.unique(fp['SITE'][:])
lats={}
lons={}
tows={}
for site0 in sites:
    site=str(site0)[2:-1]
    m=fp['SITE'][:]==site0
    lats[site]=fp['lat'][m][0]
    lons[site]=fp['lon'][m][0]
    tows[site]=fp['tow_height'][m][0]

# %%
import rasterio

# %%
idx=np.where(sites=='ABBY')

# %%
site='ABBY'
idx=np.where(sites==site.encode('UTF-8'))[0][0]
fp=rasterio.open('/home/tsw35/soteria/data/NEON/dsm/_tifs/'+site+'_dsm.tif')
data=fp.read(1)
xx,yy=fp.index(lons[idx],lats[idx])
data[data<0]=float('nan')
dsm=data[xx-2000:xx+2000,yy-2000:yy+2000]
fp=rasterio.open('/home/tsw35/soteria/data/NEON/dtm/_tifs/'+site+'_dtm.tif')
data=fp.read(1)
xx,yy=fp.index(lons[idx],lats[idx])
data[data<0]=float('nan')
dtm=data[xx-2000:xx+2000,yy-2000:yy+2000]
chm=dsm-dtm
print(np.std(dsm))
print(np.std(dtm))
print(np.std(chm))

# %%
sig_dsm={}
sig_dtm={}
sig_chm={}
ani_u={}
ani_s={}
sites_in=[]
#ss_u={}
#ss_s={}
for site0 in sites:
    site=str(site0)[2:-1]
    print(site,end=',')
    ani_u[site]=d_ani_ust[site]['yb']['full']['median']
    ani_s[site]=d_ani_stb[site]['yb']['full']['median']
    sites_in.append(site)
    fp=rasterio.open('/home/tsw35/soteria/data/NEON/dsm/_tifs/'+site+'_dsm.tif')
    data=fp.read(1)
    xx,yy=fp.index(lons[site],lats[site])
    data[data<0]=float('nan')
    dsm=data[xx-2000:xx+2000,yy-2000:yy+2000]
    fp=rasterio.open('/home/tsw35/soteria/data/NEON/dtm/_tifs/'+site+'_dtm.tif')
    data=fp.read(1)
    xx,yy=fp.index(lons[site],lats[site])
    data[data<0]=float('nan')
    dtm=data[xx-2000:xx+2000,yy-2000:yy+2000]
    chm=dsm-dtm
    sig_dsm[site]=np.nanstd(dsm)
    sig_dtm[site]=np.nanstd(dtm)
    sig_chm[site]=np.nanstd(chm)

# %%
print(sig_dsm)
print()
print(sig_dtm)
print()
print(sig_chm)


# %%
def nlcd_plot2(sites,sortr,colr,txtr):
    # sites is list, sortr,colr,txtr are dictionaries
    d_r={}
    for site in sites:
        d_r[sortr[site]]=site
    
    r_str=list(d_r.keys())
    r_str.sort()
    r_sit=[]
    r_clr=[]
    r_txt=[]
    for i in r_str:
        site=d_r[i]
        r_sit.append(site)
        r_clr.append(colr[site])
        r_txt.append(txtr[site])
    
    return r_sit,r_str,r_clr,r_txt
    


# %%
sig_dsm={'ABBY': 68.810265, 'BARR': 1.9357907, 'BART': 93.49701, 'BLAN': 9.830146, 'BONA': 125.8275, 'CLBJ': 12.345337, 'CPER': 9.308821, 'DCFS': 20.139002, 'DEJU': 28.84339, 'DELA': 9.64386, 'DSNY': 5.154887, 'GRSM': 110.06048, 'GUAN': 61.176243, 'HARV': 27.139366, 'HEAL': 75.396286, 'JERC': 8.731858, 'JORN': 4.723422, 'KONA': 26.386044, 'KONZ': 26.856789, 'LAJA': 6.9870534, 'LENO': 12.000937, 'MLBS': 80.65106, 'MOAB': 33.593365, 'NIWO': 114.98512, 'NOGP': 11.918743, 'OAES': 13.990836, 'ONAQ': 28.982918, 'ORNL': 31.701296, 'OSBS': 10.324455, 'PUUM': 89.462364, 'RMNP': 114.17059, 'SCBI': 78.13488, 'SERC': 15.168535, 'SJER': 44.64709, 'SOAP': 114.08914, 'SRER': 29.227484, 'STEI': float('nan'), 'STER': 5.638673, 'TALL': 23.096083, 'TEAK': 128.07854, 'TOOL': 34.194275, 'TREE': 14.203682, 'UKFS': 23.62557, 'UNDE': 13.276412, 'WOOD': 9.882522, 'WREF': 112.56092, 'YELL': 77.05183}

# %%
for site in sig_dsm.keys():
    print(site+': '+str(sig_dsm[site]))

# %%
st,an,sg,_=nlcd_plot2(sites_in,ani_s,sig_dsm,sig_chm)

# %%
cmap=plt.get_cmap('bone')

# %%
a=1
_, ax = plt.subplots(figsize=(13*a,3.5*a))
sgnorm=sg-np.nanmin(sg)
sgnorm=sgnorm/np.nanmax(sgnorm)
plt.bar(st,an,color=cmap(sgnorm))
ax.set_xticks(np.linspace(0,46,47),st)
ax.tick_params(axis='x',labelrotation=45)
plt.ylim(.25,.38)

# %%
d_fit=pickle.load(open('/home/tsw35/soteria/neon_advanced/data/d_fit_HCuv1.p','rb'))

# %%
degs={'H2Ou':[3],'CO2u':[3]}
mins={}
maxs={}
#maxs={'Tu':[.75],'H2Ou':[14],'CO2u':[40],'Ts':[100,100,100,10],'H2Os':[100,100,.025,.02],'CO2s':[100,100,125,.02]}

# %%
letters=['a','b','c','d']
for var in degs.keys():
    param=d_fit[var]['param']
    anibins=d_fit[var]['anibins']
    anilvls=(anibins[1:]+anibins[0:-1])/2
    pall=np.nanmean(param,axis=0)
    m=~np.isnan(param[:,0])
    if var in maxs.keys():
        for mx in range(len(maxs[var])):
            m=m&(param[:,mx]<maxs[var][mx])
    x=anilvls[m]
    absz=anibins[1:]-anibins[0:-1]
    absz=absz[m]
    plt.figure(figsize=(4,3*len(pall)),dpi=200)
    print("'"+str(var)+"':{",end='')
    for i in range(len(pall)):
        y=param[:,i][m]
        c = P.polyfit(x,y,degs[var][i])
        plt.subplot(len(pall),1,i+1)
        plt.scatter(anilvls[m],param[:,i][m])
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
        plt.plot(x,y2,'k--')
        #plt.ylim(np.nanpercentile(param[:,i],1)*.9,np.nanpercentile(param[:,i],99))
        plt.title(var)
    print('},')

# %%
np.nanmedian(param[:,2])

# %%
np.nanmedian(d_fit['H2Ou']['param'][:,1])

# %%
d_ss=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_stbl_site_v0.p','rb'))
d_us=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_unst_site_v0.p','rb'))

# %%
d_us.keys()

# %%
d_ss=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_stbl_site_v0.p','rb'))
d_us=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_unst_site_v0.p','rb'))
v='U'
n=0
for site in d_us[v]['p_ani'].keys():
    plt.figure()
    cc=cani_norm(d_us[v]['p_ani'][site])
    xx=d_us[v]['p_zL'][site][:]
    yy=d_us[v]['p_phi'][site][:]
    for i in range(len(cc)):
        #plt.loglog(-xx,yy[i,:],color=cc[i,:])
        plt.semilogx(-xx,yy[i,:],color=cc[i,:])
    plt.title(site)
    plt.xticks([10**-3,10**-2,10**-1,1,10],[r'$-10^{-3}$','',r'$-10^{-1}$','',r'$-10^{1}$'])
    plt.xlim(10**(-3.5),10**(1.5))
    plt.ylim(.5,10)
    plt.gca().invert_xaxis()


# %%

# %%
yy.shape

# %%

# %%
yy[i,:].shape

# %%
xx

# %%
d_log=pickle.load(open('/home/tsw35/soteria/neon_advanced/data/d_fit_scalarlog.p','rb'))
d_C=pickle.load(open('/home/tsw35/soteria/neon_advanced/data/d_fit_scalarC.p','rb'))

# %%
d_fit=pickle.load(open('/home/tsw35/soteria/neon_advanced/data/d_fit_scalar2.p','rb'))

# %%
d_fit['CO27']=d_log['CO27']
d_fit['H2O7']=d_log['H2O7']
d_fit['CO24']=d_C['CO24']
d_fit['CO26']=d_C['CO26']

# %%
degs={'T2':[2,1],'T3':[3,3],'T4':[2,1],'T5':[3,3,3],'T6':[3,1],'T7':[3,3],\
      'H2O2':[1,2],'H2O3':[3,3,3],'H2O4':[2,2],'H2O5':[3,3],'H2O6':[3,2],'H2O7':[3,3],\
      'CO22':[1,3],'CO23':[3,3,3],'CO24':[2,1],'CO25':[2,1],'CO26':[2,1],'CO27':[3,3]}
mins={}
maxs={}

# %%
letters=['a','b','c','d']
for var in degs.keys():
    param=d_fit[var]['param']
    anibins=d_fit[var]['anibins']
    anilvls=(anibins[1:]+anibins[0:-1])/2
    pall=np.nanmean(param,axis=0)
    m=~np.isnan(param[:,0])
    if var in maxs.keys():
        for mx in range(len(maxs[var])):
            m=m&(param[:,mx]<maxs[var][mx])
    x=anilvls[m]
    absz=anibins[1:]-anibins[0:-1]
    absz=absz[m]
    plt.figure(figsize=(4,3*len(pall)),dpi=200)
    print("'"+str(var)+"':{",end='')
    for i in range(len(pall)):
        y=param[:,i][m]
        c = P.polyfit(x,y,degs[var][i])
        plt.subplot(len(pall),1,i+1)
        plt.scatter(anilvls[m],param[:,i][m])
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
prms={'T2':{'a':[0.0016086400426415747,-0.009046112626421346,0.01674919857041958],'b':[3.1069569610314867,-2.721337687613489]},
'T3':{'a':[3.817709220055608,-7.473760573874225,10.68634838019546,-7.025544971110323],'b':[0.02447044229928632,-0.14745178845621343,0.42535783717801423,-0.3239796865035902]},
'T4':{'a':[0.014058713018066155,-0.06846731062292509,0.13165525017433005],'b':[3.04131851476864,-2.719792181824897]},
'T5':{'a':[0.0027252543145951896,0.007027843331210413,-0.09196590181888623,0.2007361171500964],'b':[-1.1780935040635225,-0.08447995241669148,0.5614236224538129,-0.037235841944141534],'c':[3.4996684250079277,-6.108768265650669,8.098673174827468,-5.931630938506805]},
'T6':{'a':[0.4976992048433562,-2.928653210710675,6.925549603692723,-4.14831462620025],'b':[2.9179364060632915,-2.94339427218065]},
'T7':{'a':[3.839115631186413,-7.649826422085375,11.027420453875179,-7.042463168602552],'b':[0.14869411390544263,-1.1222215055483538,3.0368093230353854,-2.089468545484071]},
'H2O2':{'a':[4.449311924864436,-3.3938685113439004],'b':[-0.004799144374535944,0.013445137767661895,0.42770933832593444]},
'H2O3':{'a':[4.516964373063404,-4.678113366682572,5.568586331475535,-6.724319316848667],'b':[1.939228279791965,-20.013030457028464,65.73647069042104,-57.93802424528405]},
'H2O4':{'a':[0.17164376117746769,-0.959576303167761,1.7094510064278852],'b':[3.8863009623656986,-0.38722338061937195,-4.991252724918581]},
'H2O5':{'a':[3.5838553759472704,-25.592585974821585,63.95066854368746,-39.75572561544258],'b':[0.42382342649041355,24.70142140535896,-68.94899093260442,41.594606950484305]},
'H2O6':{'a':[0.35374232006278505,-2.193667390301957,4.621210678138279,-1.5999742874276064],'b':[3.759576715945993,0.22519297180658115,-6.101208969383122]},
'H2O7':{'a':[-0.4670570378346629,3.4401437730645763,-8.872475673971953,5.8987360354368406],'b':[4.006581533986369,-0.8088944763047118,-5.408152462596361,2.341242489018091]},
'CO22':{'a':[4.151638501380521,-3.0218430389529396],'b':[0.29741981059947653,-3.099397858364962,10.872794546556168,-9.24653064441252]},
'CO23':{'a':[4.76121840576042,-7.531477900201103,10.344768180144898,-8.257197014885465],'b':[0.9999999934394734,-2.7413746281936976e-08,1.8825029827145572e-07,-2.0912782735169094e-07]},
'CO24':{'a':[0.6144831470128158,-0.5816213540763281,1.9126834735747011],'b':[3.4110851117407783,-3.9160561695026623]},
'CO25':{'a':[2.8485080017971396,0.23174295182456373,3.435543197123571],'b':[1.3502468207333613,-5.775976388063307]},
'CO26':{'a':[0.8845657789177256,-0.5835075212388595,2.237996599356271],'b':[3.1701608838942703,-4.123288804470784]},
'CO27':{'a':[-0.34844121229497904,0.025891502693535007,-0.7386773911350409,0.5377195462781004],'b':[4.487121702528974,-8.712771635114335,13.986857651209343,-10.518618476354495]}}

# %%
fxns_={'T2':T_stb2,'T4':T_stb4,'T6':T_stb6,\
       'H2O2':C_stb2,'H2O3':C_stb3,'H2O4':C_stb4,'H2O5':C_stb5,'H2O6':C_stb6,'H2O7':C_stb7,\
       'CO22':C_stb2,'CO23':C_stb3,'CO24':C_stb4,'CO25':C_stb5,'CO26':C_stb6,'CO27':C_stb7}

# %%
abin=np.linspace(.1,.7,11)


# %%
def C_stb7(zL,a,b):
    return a*np.log10(zL)+b

def C_stb2(zL,a,b): # ramana 2004
    return a*(1+b*zL)**(-.33)

def C_stb3(zL,a,b): # ramana 2004 another degree of freedom
    return a*(1+b*zL)**(-.05)


def C_stb4(zL,a,b):
    return a*(zL)**(-.2)+b

def C_stb5(zL,a,b):
    return a*(zL)**(-.05)+b #Pahlow 2001

def C_stb6(zL,a,b):
    return a*(zL)**(-.15)+b #Pahlow 2001



def T_stb2(zL,a,b):
    return a*(zL)**(-1.4)+b #sfyri reference

def T_stb4(zL,a,b):
    return a*(zL)**(-1)+b #Pahlow 2001

def T_stb6(zL,a,b):
    return a*(zL)**(-1/3)+b # Quan and Hu 2009



# %%
ss=.1
alph=.2
v='CO2'
num='6'
zL_plot=np.logspace(-4,2)
clrs=cani_norm(np.array((abin[0:-1]+abin[1:])/2))
phi_s=d_stbl[v]['phi']
cc_s=cani_norm(d_stbl[v]['ani'][:])
plt.scatter(d_stbl[v]['zL'][:],phi_s,color=cc_s,s=ss,alpha=alph)

letters=['a','b','c','d']

vn=v+num
an=np.array((abin[0:-1]+abin[1:])/2)
for i in range(len(clrs)):
    pm=[]
    for x in range(len(degs[vn])):
        d=degs[vn][x]
        pp=0
        for j in range(d+1):
            pj=prms[vn][letters[x]][j]
            pp=pp+pj*an[i]**j
        pm.append(pp)
    if len(pm)==1:
        phi_plot=fxns_[vn](zL_plot,pm[0])
    elif len(pm)==2:
        phi_plot=fxns_[vn](zL_plot,pm[0],pm[1])
    elif len(pm)==3:
        phi_plot=fxns_[vn](zL_plot,pm[0],pm[1],pm[2])
    plt.semilogx(zL_plot,phi_plot,color=clrs[i])
plt.ylim(.5,10)
plt.xlim(10**-3.5,10**2)

# %%
ss=.1
alph=.2
v='H2O'
num='3'
zL_plot=np.logspace(-4,2)
clrs=cani_norm(np.array((abin[0:-1]+abin[1:])/2))
phi_s=d_stbl[v]['phi']
cc_s=cani_norm(d_stbl[v]['ani'][:])
plt.scatter(d_stbl[v]['zL'][:],phi_s,color=cc_s,s=ss,alpha=alph)

letters=['a','b','c','d']

vn=v+num
an=np.array((abin[0:-1]+abin[1:])/2)
plt.semilogx(zL_plot,Ctest(zL_plot,param[0],param[1],param[2]))
plt.semilogx(zL_plot,np.log10(zL_plot)*-.5+3)
#plt.semilogx(zL_plot,Ctest(zL_plot,3.5,-.1)-1)
#plt.semilogx(zL_plot,Ctest(zL_plot,3,-.1))
#plt.semilogx(zL_plot,Ctest(zL_plot,2.5,-.1)+1)
plt.ylim(.5,10)
plt.xlim(10**-4,10**2)

# %%
np.log10(.01)


# %%
def Ctest(zL,a,b,c):
    return a*zL**b+c
    
p0=[2,-1/3,0]
bounds=([.01,-1,-5],[10,-.01,5])
m=(~np.isnan(phi_s))
phi_s=phi_s[m]
zL2=d_stbl[v]['zL'][m]

# %%
from scipy import optimize

# %%
param,pcov=optimize.curve_fit(Ctest,zL2,phi_s,p0,bounds=bounds,loss='cauchy')

# %%
param

# %%
list(range(2,8))

# %%

# %%
degs={'H2O5':[4,4]}

letters=['a','b','c','d']
for var in ['H2O5']:
    param=d_fit[var]['param']
    anibins=d_fit[var]['anibins']
    anilvls=(anibins[1:]+anibins[0:-1])/2
    pall=np.nanmean(param,axis=0)
    m=~np.isnan(param[:,0])
    if var in maxs.keys():
        for mx in range(len(maxs[var])):
            m=m&(param[:,mx]<maxs[var][mx])
    x=anilvls[m]
    absz=anibins[1:]-anibins[0:-1]
    absz=absz[m]
    plt.figure(figsize=(4,3*len(pall)),dpi=200)
    print("'"+str(var)+"':{",end='')
    for i in range(len(pall)):
        y=param[:,i][m]
        c = P.polyfit(x,y,degs[var][i])
        plt.subplot(len(pall),1,i+1)
        plt.scatter(anilvls[m],param[:,i][m])
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
prms={'H2O5':{'a':[-1.5543523880884509,43.16620958655561,-250.69778274067278,552.1368018570441,-390.2005743437812],'b':[7.3055971055254645,-67.38953544703286,352.47021390257044,-751.146870902566,522.6086926376627]}}

# %%
ss=.1
alph=.2
v='H2O'
num='5'
zL_plot=np.logspace(-4,2)
clrs=cani_norm(np.array((abin[0:-1]+abin[1:])/2))
phi_s=d_stbl[v]['phi']
cc_s=cani_norm(d_stbl[v]['ani'][:])
plt.scatter(d_stbl[v]['zL'][:],phi_s,color=cc_s,s=ss,alpha=alph)

letters=['a','b','c','d']

vn=v+num
an=np.array((abin[0:-1]+abin[1:])/2)
for i in range(len(clrs)):
    pm=[]
    for x in range(len(degs[vn])):
        d=degs[vn][x]
        pp=0
        for j in range(d+1):
            pj=prms[vn][letters[x]][j]
            pp=pp+pj*an[i]**j
        pm.append(pp)
    if len(pm)==1:
        phi_plot=fxns_[vn](zL_plot,pm[0])
    elif len(pm)==2:
        phi_plot=fxns_[vn](zL_plot,pm[0],pm[1])
    elif len(pm)==3:
        phi_plot=fxns_[vn](zL_plot,pm[0],pm[1],pm[2])
    elif len(pm)==4:
        phi_plot=fxns_[vn](zL_plot,pm[0],pm[1],pm[2],pm[3])
    plt.semilogx(zL_plot,phi_plot,color=clrs[i])
plt.ylim(.5,10)
plt.xlim(10**-4,10**2)

# %%
letters=['a','b','c','d']
for var in ['T2']:
    param=d_fit[var]['param']
    anibins=d_fit[var]['anibins']
    anilvls=(anibins[1:]+anibins[0:-1])/2
    pall=np.nanmean(param,axis=0)
    m=~np.isnan(param[:,0])
    if var in maxs.keys():
        for mx in range(len(maxs[var])):
            m=m&(param[:,mx]<maxs[var][mx])
    x=anilvls[m]
    absz=anibins[1:]-anibins[0:-1]
    absz=absz[m]
    plt.figure(figsize=(4,3*len(pall)),dpi=200)
    print("'"+str(var)+"':{",end='')
    for i in range(len(pall)):
        y=param[:,i][m]
        c = P.polyfit(x,y,degs[var][i])
        plt.subplot(len(pall),1,i+1)
        plt.scatter(anilvls[m],param[:,i][m])
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
        plt.ylabel(letters[i]+'($y_b$)')
        if i>0:
            plt.xlabel(r'$y_b$')
    plt.subplots_adjust(hspace=.2)
    print('},')

# %%

# %%

# %%

# %%

# %%
'T2':{'a':[0.0016086400426415747,-0.009046112626421346,0.01674919857041958],'b':[3.1069569610314867,-2.721337687613489]},
'T3':{'a':[3.817709220055608,-7.473760573874225,10.68634838019546,-7.025544971110323],'b':[0.02447044229928632,-0.14745178845621343,0.42535783717801423,-0.3239796865035902]},
'T4':{'a':[0.014058713018066155,-0.06846731062292509,0.13165525017433005],'b':[3.04131851476864,-2.719792181824897]},
'T5':{'a':[0.0027252543145951896,0.007027843331210413,-0.09196590181888623,0.2007361171500964],'b':[-1.1780935040635225,-0.08447995241669148,0.5614236224538129,-0.037235841944141534],'c':[3.4996684250079277,-6.108768265650669,8.098673174827468,-5.931630938506805]},
'T6':{'a':[0.4976992048433562,-2.928653210710675,6.925549603692723,-4.14831462620025],'b':[2.9179364060632915,-2.94339427218065]},
'T7':{'a':[3.839115631186413,-7.649826422085375,11.027420453875179,-7.042463168602552],'b':[0.14869411390544263,-1.1222215055483538,3.0368093230353854,-2.089468545484071]},
'H2O2':{'a':[4.4715024138701205,-3.4837301794203026],'b':[0.01210089094033287,-0.07568044147521406,0.0885267494534989,0.7224095404252947,-1.0115298405833557]},
'H2O3':{'a':[4.509159591557627,-4.479272837210766,4.3511657156262,-4.693451722618339],'b':[0.37607677898539743,-3.4948560497532375,9.039634622417152,-5.128947074865075],'c':[1.2454812247221345,-18.348719604746194,52.49222280913729,-42.87422835362976]},
'H2O4':{'a':[4.510474177607806,-3.690333539066831]},
'H2O5':{'a':[-0.00018239280292816124,0.0027092262789102694,-0.010668487858307155,0.012755102659479184],'b':[4.356751745391893,-3.2495297068255544,1.411483975413329,-3.467998055015781]},
'CO22':{'a':[99.99999999999969,1.6746580498899245e-12,-4.110478577267167e-12,3.2741920424242564e-12],'b':[-9.668563987548445e-17,1.1028994146142744e-15,-3.089846771928226e-15,2.534331913632339e-15]},
'CO23':{'a':[499.99999999997897,9.925372144008048e-12,2.40303102281431e-11,-1.789650747790153e-11],'b':[-1.6415000891271577e-11,2.3687957866166365e-10,-7.154648526084222e-10,6.094206788292886e-10],'c':[-0.19999999499298377,-9.590326479948412e-08,2.7704461474536396e-07,-2.2113624335238422e-07]},
'CO24':{'a':[9.999999992114589,9.662101866828152e-09,-6.524216014450997e-09,5.711342806758071e-09]},
'CO25':{'a':[11.60784036359944,-33.839482594753946,92.0482873145598,-75.00714126429136],'b':[111.4434376801427,-422.96713148124434,1113.6463690757507,-875.6118725211406]},

# %%
v='U'
plt.figure()
cc=cani_norm2(np.linspace(0,1,7))
xx=d_us[v]['p_zL'][:]
yy=d_us[v]['p_phi'][:]
for i in range(len(cc)):
    #plt.loglog(-xx,yy[i,:],color=cc[i,:])
    plt.semilogx(-xx,yy[i,:],color=cc[i,:])
plt.title(site)
plt.xticks([10**-3,10**-2,10**-1,1,10],[r'$-10^{-3}$','',r'$-10^{-1}$','',r'$-10^{1}$'])
plt.xlim(10**(-3.5),10**(1.5))
plt.ylim(.5,10)
plt.gca().invert_xaxis()

# %%
d_ss=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_stbl_xb.p','rb'))
d_us=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_unst_xb.p','rb'))
v='U'
n=0data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAiQAAAG5CAYAAAC+4y9wAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8fJSN1AAAACXBIWXMAAA9hAAAPYQGoP6dpAABYT0lEQVR4nO39d5gb5b0+/t9T1Mvuansv9nq97tjGYJodG9N7KCYkOSENDiEnJ4T8Uk4g5ZBPQgr5HXJISM8BAiGQhCQk1ACmhQ7uZXvv0qrXmfn+Ia12l13ba3vXo5Xv13XpGmk0Go301ki3npl5RtA0TQMRERGRjkS9F4CIiIiIgYSIiIh0x0BCREREumMgISIiIt0xkBAREZHuGEiIiIhIdwwkREREpDsGEiIiItIdAwkRERHpjoGEiI7aDTfcgLVr16Kvr2/KfaOjozjjjDOwdetW/Otf/0JDQ8NBL3/605/Sj/vIRz6Ciy666JDP++Mf/xgNDQ1wu92z/pqISB+y3gtARPPXHXfcgYsuughf+9rX8Ktf/WrSff/93/+NYDCIO++8E/39/QCAW265BaeccsqU+VRVVR2X5SWizMVAQkRHrbCwEF//+tfx+c9/Hr///e+xdetWAMAzzzyDxx9/HF//+tdRXV2dDiTV1dVYtWqVjktMRJmKm2yI6JhccMEFuPDCC3HnnXeiu7sbHo8HX//613H66afjQx/6kN6LR0TzBFtIiOiY3X777XjjjTfw1a9+FS6XC/F4HP/v//2/KdOpqopEIjFlvCzzq4joRMdvASI6Zrm5ufj2t7+NT3/60wCA733veygpKZky3ec///lpH79t27ZppyeiEwcDCRHNig0bNmDVqlUYHR3FpZdeOu00t956K0499dQp4/Pz8+d68YgowzGQENGsMRqNMBgMB72/srISy5cvP45LRETzBXdqJSIiIt0xkBAREZHuuMmGiI6bjo4OvPfee1PGl5SUTNqpNRAI4Mknn5wyncvlwrp169K3n3/+edhstinTnXfeebOzwER03DCQENFxc9ddd007/sYbb5x0BE5fXx8+97nPTZlu3bp1uP/++9O3v/rVr047v/379x/jkhLR8SZomqbpvRBERER0YuM+JERERKS7Iw4kHR0duP3223HppZdiyZIlBz0r57Zt23DZZZdh+fLl2LJlC373u98d88ISERFRdjriQNLU1IRt27ahuroaCxYsmHaad999FzfddBOWLFmCX/ziF7j88stxxx134JFHHjnmBSYiIqLsc8T7kKiqClFM5pgvf/nL2LVrFx5//PFJ03zyk5+E1+udFEBuu+02PP/883jxxRfTjyciIiICjqKF5HBhIhaL4bXXXsOFF144afzFF1+MoaEh7Nmz50ifkoiIiLLcrDdVdHZ2Ih6Po66ubtL4hQsXAgBaWlpm+ymJiIhonpv1QOL1egEATqdz0vix22P3ExEREY2Zs505BEE4ovEzwS5TiIiIstOs99Sak5MDYGpLiM/nAzC15eRIqKoGny909AunM0kS4XRa4POFoSiq3otzQmMtMgvrkTlYi8yRLbVwOi2QpMO3f8x6IKmqqoLBYEBrayvOOuus9Pjm5mYAOOihwjOVSMzfooxRFDUrXkc2YC0yC+uROViLzHGi1GLWN9kYjUaceuqpeOKJJyaNf/zxx1FYWIglS5bM9lMSERHRPHfELSThcBjbtm0DAPT09Ew6K+e6devgcrnwmc98Bh/+8Ifxta99DRdffDHeeecdPPLII/jWt77FPkiIiIhoiiPuGK27uxubN2+e9r777rsPp5xyCoBk1/F33XUXWlpaUFJSguuvvx7XXXfdMS2soqhwu4PHNA89ybKIvDwbPJ7gCdH8lslYi8zCemQO1iJzZEstXC7b3OxDUlFRMaNTe2/YsAEbNmw40tkTERHRCYjbT4iIiEh3DCRERESkOwYSIiIi0h0DCREREemOgYSIiIh0x0BCREREumMgISIiIt0xkBAREZHuGEiIiIhIdwwkREREpDsGEiIiItIdAwkRERHpjoGEiIiIdMdAQkRERLpjICEiIiLdMZAQERGR7hhIiIiISHcMJERERKQ7BhIiIiLSHQMJERER6Y6BhIiIiHTHQEJERES6YyAhIiIi3TGQEBERke4YSIiIiEh3DCRERESkOwYSIiIi0h0DCREREemOgYSIiIh0x0BCREREumMgISIiIt0xkBAREZHuGEiIiIhIdwwkREREpDsGEiIiItIdAwkRERHpjoGEiIiIdMdAQkRERLpjICEiIiLdMZAQERGR7hhIiIiISHcMJERERKQ7BhIiIiLSHQMJERER6Y6BhIiIiHTHQEJERES6YyAhIiIi3TGQEBERke4YSIiIiEh3DCRERESkOwYSIiIi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iIiISHcMJERERKQ7BhIiIiLSHQMJERER6Y6BhIiIiHTHQEJERES6YyAhIiIi3TGQEBERke4YSIiIiEh3DCRERESkOwYSIiIi0h0DCREREemOgYSIiIh0x0BCREREumMgISIiIt0xkBAREZHuGEiIiIhIdwwkREREpDsGEiIiItIdAwkRERHpjoGEiIiIdMdAQkRERLpjICEiIiLdMZAQERGR7hhIiIiISHcMJERERKQ7BhIiIiLSHQMJERER6Y6BhIiIiHTHQEJERES6YyAhIiIi3TGQEBERke4YSIiIiEh3DCRERESkOwYSIiIi0h0DCREREemOgYSIiIh0x0BCREREumMgISIiIt0xkBAREZHuGEiIiIhIdwwkREREpLs5DyTBYBBnnXUWGhoasHPnzrl+OiIiIpqH5jyQ/OQnP4GiKHP9NERERDSPzWkgaWlpwYMPPojPfvazc/k0RERENM/NaSD59re/ja1bt6K2tnYun4aIiIjmOXmuZvzkk09i3759uPvuu7F79+5Zm68sz9/9cCVJnDQk/bAWmYX1yBysReY40WoxJ4EkHA7ju9/9Lm655RbY7fZZm68oCsjLs83a/PTidFr0XgRKYS0yC+uROViLzHGi1GJOAslPf/pT5Ofn44orrpjV+aqqBp8vNKvzPJ4kSYTTaYHPF4aiqHovzgmNtcgsrEfmYC0yR7bUwum0zKiVZ9YDSU9PD37961/jnnvuQSAQAACEQqH0MBgMwmY7+laORGL+FmWMoqhZ8TqyAWuRWViPzMFaZI4TpRazHki6u7sRj8fx6U9/esp9H/3oR7Fy5Ur84Q9/mO2nJSIionls1gNJY2Mj7rvvvknj9u7di+985zv45je/ieXLl8/2UxIREdE8N+uBxOl04pRTTpn2vqVLl2Lp0qWz/ZREREQ0z50YxxIRERFRRpuzfkgmOuWUU7B///7j8VREREQ0D7GFhIiIiHTHQEJERES6YyAhIiIi3TGQEBERke4YSIiIiEh3DCRERESkOwYSIiIi0h0DCREREemOgYSIiIh0x0BCREREumMgISIiIt0xkBAREZHuGEiIiIhIdwwkREREpDsGEiIiItIdAwkRERHpjoGEiIiIdMdAQkRERLpjICEiIiLdMZAQERGR7hhIiIiISHcMJERERKQ7BhIiIiLSHQMJERER6Y6BhIiIiHTHQEJERES6YyAhIiIi3TGQEBERke4YSIiIiEh3DCRERESkOwYSIiIi0h0DCREREemOgYSIiIh0x0BCREREujvhAomSiCLsH0QiFoSmqXovDhEREQGQ9V6A4yUW8aF3/7Poa3oOSjycHCmIMBjtMJjskE02GEwOyCY7DEZ7cmgau2/sugOSwQJBEPR9MURERFkm6wNJNOhG974nMdDyElQlBgAQRBmamgA0FfGoD/Gob8bzEyUjcooakFe6DHmly2B2FDOgEBERHaOsDSQhXz+69/wDQ+2vQdMUAIDdVYvKpRfAVb4KmqogHg0gEQ0kh7HkMHnxIxENpsb70+PVRBSqEoOnbyc8fTsBACZbQTqc5BQvhmyw6PmyiYiI5qWsCyQBdwe69/wDw11vA9AAADnFi1G55ELkFDemWzMESYTJmgeTNW/G81aVOML+AXj6dmO0bxe8QwcQDQ6jv/kF9De/AEGQ4ChcmA4ottxKtp4QERHNwLwKJGoiCk1JQJCmLrZ3qAndu/+ebrkAAFf5KlQsuQDOggWz8vyiZIAttwK23ApUNJ4LJRGFd2AfPH274OnfjYh/AL7B/fAN7kfH9j/CYHYiryQZTnJLl0CWc2ZlOYiIiLLNvAokMX8/3vnOajhqToe1dDkMjhLEIGJ4uAVBb29yIkFAYdU6VCy5ALbcijldHkk2wVW+Eq7ylQCAsH8Qo/274OnbhdGBfYhHfBhsfxWD7a8CEODIr0HlolPhKF4Fg6VgTpeNiIhoPhE0TdP0XoiZCnoH8OyPLoDkboZmckIpWAzNktrkoioQvV0wR70w21wwOEpgdJbB6CyFITU0OkphcJZCMlrnfFlVJQ7fcEtqf5NdCI12T7rfnl+Lwqp1KKhaC5PVNefLQ5PJsoi8PBs8niASCR7+rTfWI3OwFpkjW2rhctkgSYfvZWRetZAAgOpaANU1YROMpkL29UDo3w4hEUECQGDk0PMQDRYIkhGCZIQomyBIhtR14/h4yQhBNkKU3n+/ARAkCIIIiKmhIE5zWwIEESZRRIktF4rViXA0hIQkwD3UjsBIGwIjbWh792E4C+tRWL0O+ZVrYTQ75/T9IyIiykTzKpAYLTlwla+Cu2c7ZKMFpYvORtmizTCY7FATMcQDA4j5+hD39SLm708OfX3Jcf7kUI0FocbDwFhfJDrIrVoHY90mhFUV/pFW+Iaa4BtqQsvbDyK3qBEF1etQULkastGm2zISEREdT/Nqk42iqHC7g0jEQhAlA0TJcESP1zQNStQHJTwKVYlDS0ShKbH3XY9BU2LQEnGoSmpcIjVOiUFT4tBUBdDUZE+vmgJNU6Gpampc6j514lCDpqlQwiPwNj2X7AMFgGi0IWfZZZDKVsPn7UPA3Z5eVkGUkFuyDIXV6+AqX8nDiWdZtjSFZgvWI3OwFpkjW2ox00028zKQzFeyLMIqBdH0/C8x8Mb/ITLclL7PUtSI3JXXQHFWwN0/eZ8TUTIgr2wFiuvOQF7pch5KPAuyZUXPFqxH5mAtMke21IKBJANN/HDF4woCHa9h6O374N715+RmJACCZETekovgWHIJIhow3Pkmwv6B9DzsrlpULb+EweQYZcuKni1Yj8zBWmSObKkFA0kGOtiHKxHxwr3jUQy+dR9Cve+lxxtzq1Cw+iOwLdwEz3Az+pu3pbu/d+TXoWrZJcgtXcZgchSyZUXPFqxH5mAtMke21IKBJAPN5MMV7N2Oobfvw8j2R6BEvMmRgoicRVuQt+IqBBJx9E84L48jvw5Vyy9FbslSBpMjkC0rerZgPTIHa5E5sqUWWRlIEgkFHk/omOahaRo0TYMoHv7NmW1H8uFSYiF4dv8VQ2/fB3/7K+nxkiUXOUsvRSK3Fu7B/VCVOADAUbAQ1csvndQ9Ph1ctqzo2YL1yBysRebIllpkZSDxD/bCF1RgtOce8WNjsQj27n0LO3b+Cz6fB1arHXabEzZ7Dmw2J2w2Z/q23eaEzZYDo9F0xM+jaRpisSii0RAikTCi0TAi0RCikTAUJYb6+kVwOIqhKDOfZ3ioCcPv3I/h7Y8g7utNjzfkL4BUuwH+aCh95I6zsB5Vyy5FTvFiBpNDyJYVPVuwHpmDtcgc2VKLrAwko12t+OO/X4TNd/4RtqKZdQvv83uwY8er2Lv3TcRi0SN6PoPBBHsqsNhTocVstiIaiyAaDSMaSYWNaBiRSAjRaHK8ph36g2O352DhghWor1+JwsKyGQcHTVXga3sJI+/9Ae7df4EaCyTHy2aINWciZsqFljqhoLNwUXJTTvHiI3rNJ4psWdGzBeuROViLzJEttcjKQOLr7cAvz6mBtaAMG+94CPkNJx102v7+Dry3/RW0tu7C2EvMzS3EyhWno6ZmMcLhAAIBL4JBHwJBX3KYuh0Meo84vLyfLBtgMllgMllgNlthMlkgiSK6upsRjUbS0+Xk5KO+fiUW1a9EXl7RjOevxEIY3fcPDL/3MLzN/wRUBZpshlK4BGpeLZAKOTlFDahafilyihqO6fVkm2xZ0bMF65E5WIvMkS21yMpAEo/FcN/lK+Dt2A8AKFt3Nhqv/AxK126CIAhQVQUtrbuxffvLGBjoSj+uomIhVq48HdVVi5Jdu89ALBZFMOhNhpWALx1UItEwjEYzzCYLTGbr5OGE8CHLUzttk2URdocR7733Dvbtew/t7XuhKIn0/fn5pVhUvxIL61fA6cib+fsSGMTIzj9h5L3fI9jzLjTZAqWwAWpuHZDaV8ZZUI/KZRdx59eUbFnRswXrkTlYi8yRLbXIykCiKCr62rvw+l2fR8eLjwGpRXcsWAbb5ivRG48hEEgemSKKEhYtWoWVK05HQUGpjks97v0frlgsira2PWhq3o6uriao6vgHriAnH8UWO3IiIcQGuhDoa0c85IcSj0NLxKEkYqnhxNsJiFIUDlcczgINst0KpXAx1NzadDARVQmlC89B1dpLIR1hT7fZJFtW9GzBemQO1iJzZEstsjaQjB326+9pxXt/uhf7O/YjWrkQMBgBAAYIWLp0HU46eTOsVoeeizvF2IdreNADb087Ar3tCPR3wN/bjtGBLgxFgvBZ7Yi7itObXKCpkAZ7YOhsgqGnBUI8NuPnszgBZyFgL7UAxQ3JYCIlT1+kRcIQQxLKllyG8nVbYLSdWCf1y5YVPVuwHpmDtcgc2VKLrAwkkUgE23dsh6aqOND0Htra9qT3D5GDPsh73oKh8wAkSULt5qvReNVnkFe3VOelTh550/3qEzjwl5/D39UE/0B3unVnOqrZCiw6CfHKhYha7enxoiCgyOlCRX4pKgrLYTBZIBkMECQDJIMRgixDko0QZAMk2ZAcGozQVAUDb/0e/W88gKgYhpq/AJDNyZkmYoh2tEDzmVG86iJUnHIOcuuyf7NOtqzo2YL1yBysRebIllpkZSDxeIbxP3d/Y9K4qsp6rFx5BsrLatH98uPY++g9GNr9Rvr+ktUb0XjlTSg/5RwIOvQ90v/ui3j3l9/E8J43J42XzFY4SmtgL62GvbQG9pJq2Mtqk7dLqmGwJM/06/WOoKl5B5qatsPtHu9CXpYNqK5uQP3Claiubph2n5XpqPEwhnb+Bd3vPIKwLAOmVOBRFWjD7QjuO4Coz4KiVeejbN0WlK7ZCKM9Z3bejAySLSt6tmA9MgdrkTmypRZZGUh8vlH87sGfQVNVFBSWY+WK0+ByFU+ZbmjPG9j7yE/Q+eJfkmfbBeCsrEfjlTeh7pxrIZutc76sI/vfwbu//Cb63noeQDKALPngv2PpBVcCjmLI9vwjboUYcQ+gORVOvN6R9HiDwYjamiWor1+Bysp6SKnNMoeTiPrR9eZ9GGh/A4mxx2gaBF8P1O798He64XdLyFu4HtUbL0f1xsthzi04omXOVNmyomcL1iNzsBaZI1tqkZWB5Ei7jg8OdGHfn3+Gpsf/D/FgcmdXozMPC8//KKo3Xob8htWzvmnC27Ef7/3qv9H50l8BAKJsQP3F12P5h78IR1HprHy4NE3D8HAfmpq3o7l5B/z+0fR9JpMZdbXLUF+/AuXldRBFaUbzG+15D+3vPoxgYCg9XggOQRrej1h/H0b7AL9bQumazag9+2pUnH5huhVnPsqWFT1bsB6Zg7XIHNlSCwaSCeIhP5qfeAD7/vRTBHrb0+OtRRWoOuMiVJ55MYqWr4c4w5aF6QT6O7Dj/76L1qcfgqaqgCCgbstWrPjYV+AorQEwNx8uTdMwMNCVCic7EQr50/dZLDYsqFuG+vqVKC2tntEhz0FvD7p3/Q1DnW8BqU7WhIgX4vABKEMdGO3R4B0ABIMVladfiNrNV6Hs5M0QZ7jJKFNky4qeLViPzMFaZI5sqQUDyTRURUHPv55E23OPoOe1p5EIB9L3mXLyUXn6hag682KUrN4IaYbdxofdg9j5wPfR9LdfQ00kzytTecZFWPXx25Bb2zhp2rn+cKmqir6+djQ170BLyy5EIuPvlc3mxKJFq9DQsBr502zmer9oyIPe/c+ir/l5qIlUJ3HxEKSRJmC4HaPdMXj6ACUOmJwuVG+8HLVnX4PCpet02VfnSGXLip4tWI/MwVpkjmypBQPJ4eYVi6Dv7efR+eLf0PXq3xHzedL3GawOlJ96DqrOvARlp2yBwWKf8vhYYBS7H74b+x79KRKpH/6S1Rtx0idvR0Hj2mmf83h+uFRVQXd3C5qbd6K1bdek3mELC8vQsGg16utXwmqd+tomSsRC6G/ehp79zyA+dvZhJQbR3QrJ3YrAoIqhtjDiqdnbiqtQu/kq1J599ZRAlkmyZUXPFqxH5mAtMke21IKB5AioSgID219B10t/RefLjyM83Je+TzSYUHbyZlSdeTEq1p8HyWTBvj//DLsf+hFiqX038hevwUmf/DpK12w85PPo9eFSlAQ6OvZj//530d6xD2pqR19BEFFVVY+GhtWorWk85JE6qhLHYPtr6Nn7JML+/rGREL0dkIaboSk56Nk1hJA7nH5MXt0y1J59NWrP2QprfsmcvsYjlS0rerZgPTIHa5E5sqUWDCRHSVNVDO97G10v/Q2dL/0V/p7W9H2CKMFgc6SDSE71Yqz6xG2oPOOiGe0cmwkfrkgkiKbmHdi//91J3esbjWYsXLgcDQ2rUVpSfdDXo2kq3D070L33H/APt4yNhODvhTS8HyZLKXxuG7pffxtqItktvigbUL3xciy+4saDth4db5lQCxrHemQO1iJzZEstdA0kTzzxBP72t79h9+7d8Hq9qKysxLXXXoutW7dCPIb9CxRFxchIAG//8wk4XflYtPqUWVzqqTRNw2jbHnS+9Fd0vfg3eFp3AQBsJdVY+bGvoPbsayBKhz+KZUymfbg8niHsP/Au9u9/F4HAaHq805mHRYtOQkPDScjNOfhhvr6hJnTvfRLunvfS45JH5hyA1V4IybEGXW/umtQvTEHjWiy+4kZUbbgMUqp3XT1kWi1OdKxH5mAtMke21ELXQHL11VejrKwMW7ZsQX5+Pl5//XX8/Oc/x0c/+lF86UtfOur5KoqKh/7nR/j1N24FAKzasAXXfvEbqGlcPluLfkj+nlYE+jtQtOL0o/oxzdQPl6ap6O1tx77976ClZSfiE7qnLympRkPDSahfuBImk3nax4e8vejZ9xQG216FpiVflxDxQhw5AItkgHPhxejb1Y325x+Dmpq3xVWMRZd8AvUXfxwW18zPcjxbMrUWJyrWI3OwFpkjW2qhayBxu91wuVyTxn3nO9/BQw89hLfeegtG49H9M46EQri4IgdKIgFBEKBpGgRBwOmXXIVrPv81FFXWzMLSz5358OGKx2Noa9+D/fvfRVdX03jX/LIBCxYsw5LGk1FaWjPtJp1oyIPeA8+iv+l5KOkjc8KQ3M2QgyMoWH4lfCNGNP39YYRHkvuhiLIBNR/4IBZ/8EbkN6w+bq9zPtTiRMJ6ZA7WInNkSy0ybh+Sxx57DF/60pfw0ksvoajo6P4RD3R14NqlNTj1gsux9Zbb8PCP7sC//v4nAIBkMGDLtR/HFZ/5/yGnoHA2F33WzLcPVzDow4Gm7di37+1J3dbn5hagsXEtFjesTp/AMOgbxf63X8Oe119B07uvwmYNYOX6BbA5LMkHqQmIo50QRpoREatgyFkN357dGNn3Vnq+hUvXoeGKG1F91qVz3q/JfKtFtmM9MgdrkTmypRYZF0huu+02PP3003j11VchHcF+FxP1trXiyx+8AN/9ywuw2JM/hC0738XvvvcNbH/xnwAAs82OSz71WVzyqf9IT5MpJEmE02mBzxeGoszdhysaCaPrwF6079mJjr070bFvF0I+H4oqq1FcXYfiqhqUpIaF5VWQDYf+8dc0Df39ndi9+00caNqe3qQjQIAUU9G3Yw/2vfRiskO4CYqrqrFu81q48hNw5o63igmBAUgjTeht7cdAoARW1QpxqB1IHf1jLShFw2WfRMMl18OSNzebc45XLWhmWI/MwVpkjmyphdNpyZxAsnPnTmzduhWf+cxncNNNNx31fPo72hCLRlG1aPGU+97Z9hx+8fUvY/87yZPY5RYU4sNf/Bou+vgNMJpm1snZfKNpGob7etGycztad21Hy64daN21HV1N+6GqM/vwiqKIosoqlNbUoax2QXpYlhrac3MBACP9fdjxyot47+VtaO08AENBLpxl44fyRv1+hHoGUV5YgVXrz8KK089CSVV1ejn723dg18sPw+9uRnprT9QPyd0MX1cbdu5X4B8EquwijEJqPxTZgMUXXIvTbvoGcipqZ+19IyKizDPngWRoaAhXX301iouLcf/998NwmH/jhxLy+xFNHDxlaZqG1/7xGB78wbfQ29oEACiqqMY1X/gazrz06qNumQEAVVPR0tmJ9p5u2CwWOO0OOO12OO12OGw2SDM4Z8yxpN14LIauA3vRsW9XuuWjfe9O+D3uaad35LlQ07gc1Y3LUbNkOey5eRjq6kB/ZxsGOtvR39GKwc52xCZ0mDYde04eLA4Hhro7p9xXt2Ytqk9eAzhMUFKtG4CAqsqFWLr0ZNTVLYUsT+6OP+wfQs/+f6KvaRuUROq5lThETxuUgWbs3BtETxdQLAN5qRypQYBt8alYf/MdqFo5O0dWZcs/j2zBemQO1iJzZEstMqKFxO/34yMf+Qii0SgefPBB5OXlHdP8ZtoPiZJI4IVHH8Ajd38HnoFkJ2dVDUtx7Re/jpM2nntEJ9TrHx7CW7u2463dO+H2jk47jQABdqsVDrsdTttYSEkOnbZUcLHZkZfrRFlJPkZHQ0e0PfDAO6/jhzd9GKNDA1PuEyUJZXX1qGpYiurG5ahevAzVjcuRV1Ry2NepqipGhwYw2NWOgY5WDHS1Y6CzDYOd7Rjoaod3eHD8NQoCqpesQOPJp2HJujOweO16OPOThwQrSgKtrXuwd++b6OpuTj/GbLaiYdFJaGhYjYKC0knLo8QjGGh7Fb37n0Zk7IR+qf5MxJEWuEcN2LHTD6PHi6Kx3VA0wC3noeisK7HmgitRv+pkSPLRnX8oW7bNZgvWI3OwFpkjW2qh+z4k0WgUn/jEJ9De3o6HH34Y5eXlxzzPI+0YLRoO4cn7foa/3PsjBH2jAIDFJ5+GLR/6ODRVRTQcRjQSQiwcRjQUQjQSQjQcRjAShkeUEbDYkDBb0/PT4nGIoyMoq18MwWSBLxiAPxjAkbyFBoOcDik5dke6pSXH7pwwzg6bxQpBEPDm04/jf/7z44hHI7A6clCzZDx0VC9ehor6xTCaLTN+/iMRCQYw2N0Bv8eNmiXLYXPmHvYxPp8be/e9jb1730Iw6EuPz80twMKFK1C/cCVcEw7z1TQVnr5d6N3/DEb796THC+FRiO4mmGQHBkZz0fXadliiyfkpGtAZAPqQg8Yzz8GaD5yHlWdthj138pFdh5ItK3q2YD0yB2uRObKlFroGkkQigZtvvhlvvfUWHnjgASxePHWfj6MRiUTxjxdexpDHDbPRhNLCIpQVFqHA5TrkJpOA14O/3PsjPPF/9yJ+sE0Usgxz9UKYFzbCWFaVPkGcpiqIdbUj3LIP0c4WQFEgCAKu+cLtuOzGW6BpGgKhEPzBAHzBAHyBAHxBP/yBCbcDyeASPszmkYkkSYIRgK+vG0ooAFdePs6+5ErUVdWivLgEpqM8dPp4UVUVnV0HsG/v22jv2AdFSaTvy88vRX39CtQvXAGnczxEhLy96D3wLAZaX4GmpqZPRCGOtsMQHII5fxWaX92LQHs7AEBRgfYA0OID4hDRsPoUrN50Hk7aeC4qFzUesoUoW1b0bMF6ZA7WInNkSy10DSS33347Hn74YXzxi1/E2rWTuwpfuHAh7PZDn9DtYAZGRvCfd3x7ynhJklCcX4DSwqLUpRhlhUXIy8mBKIy/Ce7+Xjx27w/RtnsHTBYrjBYrlJw8hGxO+GUj1Ak/YHkmI6pzc1FXVIwchzM1vQUvPPIAnnnwVwCA9RdegRu/ew/MVtuMll/VFEBU0NUzALfXB1/AD2/AnwotyaE34EcwHDrkfARBQGlBESpLy1CVupQVlcBwlJsv5losFkFb2140NW9HV1fTpB1ui4srUb9wBRYuXAGbzQkAiEcDGGh5Eb37n0Vs7IR+AAR/PyRPC8ymfIy0h9C3owWaBigQ0ObTksEkNevC8iqs3nQezrj0atSvOnlKOMmWFT1bsB6Zg7XIHNlSC10DyaZNm9DT0zPtfffddx9OOeXodkwc9nhw74MPo9CVj0g0gr6hQfQNDSE2oWfRiYwGI0oLC9NBpaQg2aLiCwbw5q4deGf3TviCgfT0BXkurF26AicvW4FCV/5Bl+OZB3+N33zzViiJBGqWrMCt9z6IwvKqwy7/TD5ciVgM9371s3jl6cchWm048+p/w5KzzoYv4Efv0CA6+3rhC/inPE4SJZQWFaGqpAxVZeWoKi1DaUHRMe3IOxcikRBaW3ejqXk7enpaJ2zuElBeVouF9SuwoG4ZLBYbNFWBu3cH+pqex2j/7vGZxEOQPG2QA8MIukX07+lHPAIIBhN8tjK8eaAXoUg0PXlpzQKccdk1OPPSa1BclTxaJ1tW9GzBemQO1iJzZEstdN+HZC5Mtw+JqqnweL3oHRpE39BAKqQMYmBkGIqiHGRO42wWK05qXIqTl61ETXnFjHd43fvmq7jrpg/D5x6Gw5WPW+55AEvWnX7IxxzuwxUO+HHXZz6MHS8/D1GS8Olv340PXPWRKdN5/T509vWmLj3o7OudtlVFlmSUF5egurQMtRVVWLpwEcwZdAh0MORHS/NONDXvQH9/R3q8IIiorFyI+oUrUFu7FCaTGWH/IPqbt2Gg5UUk4qnXqqnJk/q5W5FwhzDU6kfAA8hmO3LXnosWr4Y3nnsK0dD4Z2bx2vU487KtOOOSK1BZUzHvV/RskS1fvNmAtcgc2VKLEyaQHHxaBUMedzqgjF2GPCOQRAnL6htw8rIVaFywELJ0dJs6hnu78IMbr0Pb7vcgyTI+dvv3cM51nzzo9If6cHkG+/HdT1yJ9j3JzUmf/9/7cNLGc2a0HJqmwePzoqO3B139yaDS1dc7ZZ8Vg2zAsvpFWLNkORoX1GfUJh6f34Pm5p1obt6OoaHe9HhJklFb04iGhpNQWbkIAlQMd72Nvqbnxs82DCT7NPG0QhvpxmhXCN5+QJAdqDjzEoRzKvHGv17Dzn+Nd94mG4047fyLsf6iq7DijLMhZ/g+OdkuW754swFrkTmypRYnfCA5mHgiDiD54zwbouEQfvaVm/HK3x4FAJx97cdx/e3fm/YH7mAfrp6WA/jO9VdgqKcTOfmF+NIvH8GCFcd2XhdVUzHs8aCrrxcdfT3Y03wAg+6R9P0WkxkrGhqxZuly1FfXzKgfleNldHQITc070NS0Ax7P+KHHZrMN9fUr0LDoJBQVVSDk7UF/8wsYaH0FqpLabKcqEH1dEEZaEe4dgW8ICLgBZ2UDSs+4FL0hES8/9Tg69+1Kz9eR58L6C6/AmZdtnXZ/E5p72fLFmw1Yi8yRLbVgIDmONE3DX3/+P3jo+1+HpmlYvHY9Pn/P/cgtmNzt+XQfrn1vvYbv33ANAqMelNYswJd//UeUVNfNyTJ29ffhnT078faeXfD6xw/JdVhtOKlxKdYsXXFEm63mmqZpGB7uxf797+JA03aEw+P7++TmFqBh0UlYtOgkWC1mDHe8gb6m5xAc7U5PI4RHIXo7AU8XggMh+IeBkFdE2SnnIXflBjR39+Gfj/wensH+9GOm29+E5l62fPFmA9Yic2RLLbI2kIyMBODz+2GzWqf0Aqq3d59/Cnd//pMI+b3IL63Arfc+iLplq9L3v//D9cZTf8Pdn/8E4tEIFq5aiy/9/A/pzsbmkqqpaO3qxNu7d+K9fXsm7X/iysnF6iXLsGbpcpQVFmdMOFFVBV1dzdh/4F20te1BItXSBQClpTVoaDgJdXXLEA/0ob/pBQx1vg5trPdYTYMQGobo7YTm7kJwMA7fEACpEDVbtiJRWI83X30Fbzz1N0QnvBfLT/8Atnzo41iz+YLDnu+Hjk22fPFmA9Yic2RLLbIykAwODeP6mz6DeDwOp8OBD5y5EZs3bEK+a+YdYs213tYmfP+GrehtbYLBZMa/f/cenH7JVQAmf7j+/pt78ZtvfhGapmHN5vPxuf/5DUwW62HmPvsURcH+9la8tXsHdh7Yh2hs/IilkoJCrFmyHKuXLkdhXua8x7FYFK2tu7D/wLvo7m4FkPwIi6KE2ppGLGo4CWXF5fD0vIuhjtfgG2oaf7CqQgj0Q/R2QnX3IjCkwD8E2CrWoHrzNRiIGfDKE3/FrldfSB8BlFdUgk1XfxSbrvkYCsoqdHjF2S9bvnizAWuRObKlFlkZSPr6+/FvN04+OZ8oilh70hqcs+lsLK5vyIh/9CG/F3f/5yfw7gtPAwAuveHz2PqF22E0GZCTY8H/fuWLeOyndwFI7nPy8W/84Ki7QJ9NsXgMu5ub8PbundjdcmDSUUolBYVYunARlixYhLqKyow5nDgQ8OJA03vYv/9duN3jXeubzVYsqFuGqqpFyM/Nhb9/B4Y6Xpu0SQdKAqK/B6K3C4mRfgSGNAR9JpSsuQyuk8/Bu9t344VH7od3JNm1vSCKWLPpPGz50Cew4szNEMXDr2A0M9nyxZsNWIvMkS21yMpAEg5H0NrWg9ycXLy3czuefu5Z7D2wL31/VUUVztm0GaetWw+Tzoe3qoqCh390Bx776Q8BAKs2bMF/3PUzPPi92/Hsww8AAK655TZcftOtGRGi3i8ciWDHgb14e/dOHGhvg6qNrwwWkxkNtQtSAWUhHLaj6+huNmmahpGR/tT+Ju8hFBrvq0UQBBQWlqOiYiEK83Ihhrvg7nwTkeDw+AwSUYi+bojeTiRGhuEbAuJqAUrWXo6AuQAvb3sJe954OT15UWUNzr72emz84IeRU1B4PF9qVsqWL95swFpkjmypRVYGkul2au3s7sTTzz2LV17/F2KpzQ02qw0bz9yAszdsQlGhvj8Wr/ztUdz75c8gFgnDYDIhHo1CkmV8+ts/xsYrr9N12WYqFA5jX1sLdjcfwJ6Wpkn7nAgQUFlahqUL67F04SJUlJRO6h1XD6qqoru7Ge3te9Hd3QLP6NCk+yVJRllZLSpLCyGEBhEZ2olEbEJnc7FgMpz4epAYHUHIAyQUFyx1H0DHqIJt/3wOodROwZLBgFPPuxRnf+gTaDz5tIwMl/NBtnzxZgPWInNkSy1OmEAyJhAMYNvLL+KZ5/+JoZHkP19BEHDSilU4d9MWLG1ccsQ/FuFIGEPDQxgcGsLA0CAGh4YQj8ewecMHsKB2wYzn07rrPfzgxg9hpK8bZpsNt/70d1h++qYjWpZMoapq6jDiJuxuOYDu/r5J9ztsdixZsBBLFizC4toFsJjNOi3puEDAi+7uZnR3t6Cru3lS6wkAmEwWFOUXwiaFIQZbIathpD8qiSjEQD8Efx/g60fEE0c05kTMvhTbm0ewa8+B9Hwq6hfj7Gs/jg1XXAurI+c4vsL5L1u+eLMBa5E5sqUWJ1wgGaOqKt7buR1PPfcMdu0Z7268rLQM53zgbJyx/jRYUmfHVVUVntFRDA4NYnA4GTjGwsfQ8CB8/qldtI85bd16XHPFVSjIP3gX8xN5h4fw/CP/h7OvvAq5pbXz+sM1kTfgx56WJuxpPoB9ba2Ixsa7bBdFEXUVVaitqERZYTFKC4tQlJ9/1B3RzQZN0+DxDKK3rxX9/W1oaz2AWDw6aRqrxYociwBTYhhWIQCDmFpFNBVCaASivw9ioA8xjw+RkBXuaCFeea8PA75kC53JYsXqTedhzebzcdKGLUd0FuITVbZ88WYD1iJzZEstTthAMlFvXy+efv5ZvPTqK4ikei21WCxYUFuHEbcbQ8NDSCQSh5yH3W5HUUEhiguLUFRYhGH3CF557VUAgMFgwAVbzsPF51+YDjmHki0froNJKAm0dnVid/MB7G5uwqB7eMo0kiihKD8/GVCKilBWWIyyomLkOXOO6+aOsVqMjPjQ29uZbj3p7++Eqk4+5YDdYoZdjsKsemA3xCEJqVUmFkyHE9U7iLBPRu+IAW/tDaI/1WWKKEloWHMq1mw6H2s2n4+yuvrj9hrnk2xfN+YT1iJzZEstGEgmCIXDePHVl/DMc8+if3Bg0n2SJKHAlY/CwkIUFRShuKgIRQWFKCpMDq3WqYfitnW04Xd/+H16h1qnw4mrLrsCG04/65BHn2TLh2umhjxu7GttRu/gAHqHBtA7ODipBWUik9GE0sLkyQ/LioqTZ2wuKoJtjg6FPlgt4vEY+vra0ZXaxDM83Iexw4qB5GZAu1mCFX7YpAhschyiAEBNQAgMQgz0QfD1IeaLwuuX0NkbR+sA0DOanEtpzQKs3nw+1mw6H4vXrs+Io6sywYm2bmQy1iJzZEstGEimoaoq9uzbi6GRIRQWJANIvst1VIewapqGt997Bw8++jAGUiGnorwC1121FSuWLp/2Mcfy4YrFY5AleV4fajp2zp2xgNI3NIi+wdSJENXpT4SYY3egvLgEZUXJlpTyohIUufKP+bDjmdYiEgmiu6cV3d0t6O5uhtc7Mul+URBgNyiwiiE4DDFYpERy/5OoH2LYDSHshhAaQcLrhW9UQ9+whvYBoGUIMNtzsWrD2Viz6Xys3HA27Dl5x/Sa5rNs+eLNBqxF5siWWjCQHCeJRALPvvAc/vS3xxBMnVV2xbLluO7Kragon9yJ1pF8uGLxGA40N2HPvr3YtXcPWttbkeN04rzN52LTho2wWW1z9pqOt4SSwODICPqGBtNBpXdwAG7v6LTTS5KE0oKiSSGlrKgYDtvM35OjXdH9/lF097Skd5J9/w6ykgjYpChschwWKQGzFE/ug6IqEMIeCGE3xPAItMAIAiNhDI4AnYNAy4CAypVnJPc72XgOyurqT6gjdrLlizcbsBaZI1tqwUBynAWCATz2+F/x9PPPQlEUCIKATWdtxAcvuQI5TieAQ3+4FEVBW0c7du/dg9379uBAc1P6RIDvZzabsemsjThv87kZ1UvtbAtHI+lw0jPQn9rsMzCpN9mJnHZHKqAUo6yoBCUFBcixO2C32qa0LM3Gip7cQXYI3T3JcNLT04pYLDJlOoOowizGYZETsEjJi1FUICTCyYASGgFCI4iOeDA8rKBvBPAlcpDbuBFLTj8HS9efhaKK6qNaxvkiW754swFrkTmypRYMJDrpH+jHQ3/8A956920AyfBw6QUX47yzz4HVYk5/uOJxBd29PakAsht7D+xHOByeNK/cnFwsbVyCpYuXYHH9IuxvbsLfn/oHunt7ACRbCk5btx4Xnns+KstPjC7NVU2Fe3Q0GVIG+5ObfwYHMORxH/QxgiDAbrXBabPDYbfDabMj1+FAcZELBskEm9kGZ2q82WQ66pYJVVUxNNSDnt5WDA31Yni4F6OjI5i4D8oYEWo6oJhTIcUsxiBFvRAioxAiXghRL+Kjo/CPxhCImCHm1KBg6WYsveCTKKyc+WHn80G2fPFmA9Yic2RLLRhIdLb3wD787g8Poa2jHQBQkF+Aqy//IEwmEa+/9S527dkD34Qz7gKA1WrFkobGdAgpKymd8uOoqiq279qBvz/1xKRealctX4mLzr0AixdlRvf5x1s0FkXvxNaUVEgJhII4ko+4QZbhsNnhtNuRY3cgx+FEjsOBXIcTOXYnch3JcSajcUbzi8WjGBnux/BIL4aH+jA83IsR9wAUZbqju7R0ODGJCsxSAmZJgVENQox6IUR9ECJeIDKKuDeAWNwEOacGRY0bULrmQuTUngxR1reH4qOVLV+82SDbauH3+/HGO29h+ZJluneUeaSOVy1UVUXXgT3Y89pL2P36y9j/1r+QV1yKq/7jK1i75cJj/k1hIMkAqqrildf/hT/8+VG4p/kHbzQa0bBwEZY2LsGyxiWorqw+op1Wm1tb8PhT/8Bb776d/tFdUFuHi869AGtPWjOvd4CdLaqqIhAKwRf0wxcIwB8MwBcIIBAOIhILY9g9Cq8/OT4cnbq55WAsJjNyHI50aBkLKmPX7VYb7FYrjIapwUVRFIyODmF4uBdDw8mQMjzch2g0PM0zAQI0mEQFplRAMUsJmKQETAlvskUlFVSEiA+JYBSibIfBUQJTTgXMBbWwlzbCXrUCtuJ6SCbHUb+XcynbfgTns2yphc/vw9+fegLPvPBPRKNRWCwW3PCxT+Lk1Wtn9HhN0+B3j2Cwux2JeBxGswVmqw1GswUmiwUmixUGk3lO/gCqqopELApJElBSVjjrtVBVFd1Ne7H7Xy9i9+svY+8bLyMw6pl22kWrT8F1X/oWFq9df9TPx0CSQaLRKP7xzJPY9vKLKCoqwOL6xWhsaMTC2gUwzMJp7fsH+vGPZ57Ei6+8nN7vpLioGBdsOQ9nnXYGjDP8N38ime5LNxaPJwNLKrR4/T6M+n3w+v2p6354A76D7sMyHYMsw2axJi9WK+wWK6wWC+zW8XE2ixU2swWalkA44IHf58bo6BDcnkGMegaRmLY1BQA0GEVlPKSICsxiHAYlCEPCDzERAuIhCGOXWAhqNALABNGYC4O9GMacCliKFsJe1ghH5QoYncWQTE4IxznMZsKPoKZpiESjCIZDqUsYwXAIgVAIodRwbFwwHEIoHIYoirCYzDCbTLCYU0OTGRaTCebU0GK2pMePTWcxmSBLMhRVhaIqUBV1/Lqauq5MuP6+aQBAlmQYDTIMsgEGgwFGWYbBYIBBNkCWpKP+ocyEWryfqqlQVXVGnSp6fV48/tQ/8M8XnkuvqzarLX3Qwflnn4utH7wasiwjHo1iqLcTg53tGOhqnzBsw2B3OyKBAAQASL2VE99RAYAgAEaLFSaTGSaLBUaLBUaTGSazBUazGUaLFZIoIhGLQolFoMRiSMSjUGIxqPEYlEQMaiKeusSgJhLQEnFoqgpRSD6HwWKBPa8AdlchnIXFyCksRV5JOfJKyuEqq4IjvwiyyQTJaIZoMEIymCAaTBAlCUosgljAi4jPg56929Hyzmvo2vMuBlr3QYmEYBABgwAYRMBokOCw2yBbbfDnFEONxyEOtEMIB5FQAVt+EaqXngRncTkMVgcMNgcMFjsMNicMVntynNUOyWgCBBXxUATxoBdRrxsrL7kGlhkcRchAchzN9Yru9fnwzPPP4unnnk2vfE6HA+ds2oKzN26Gw67/SfAyxbHUIhKNTgooU0KL34dAODTpbMlHwiDLsFttcNhssFttMBlkSCIgqHEo8ShiUT8iwVFoSgySAEz/u6NBFlQYxORl/LoCgxZNhxZDwgcxFVYQD0FQItDiMUABNMiAYAIkC0SDDaLRAcmcA9mSB9nqgsFRBJOzGMbcUphzy2Cw50MyWSEZzZCMJgjizH8UD1cPRVUQjcUQjcUQi8UQjcfGb8fjSCgJJBIJJBQldT01VBQkEgnED3FfKBxGIBUyVDUzfoBng0GSYJBEyKIEWRJhEEXIoghZEmGVJdgMBlhlCVZZhFUWYBUFWETALGqwmGUE/SEk4nFoSgKqkpgwVKCpyqTrmqpCUxLJ60rqPlVNTqcqiCkqYioQ1wTEBBEJQUQcIuKClLwuSIgLIhKChIQgpcZLiItSehwEAeZEBPZ4ELZYALaoH9aYH7aIF1IihrAqYp9ciDa5AIqQ7BYgN+FHQ7ADhbER7LFUo9VekxwfGcGKgTdhVsLpYDH2SRUmrlNjQy150VLXNW3C8GgI488jCIAgThgnTr5vzEGf6mB3CEKy5Vwbn06bML02YRiyuODJr4PHVQt/TllqgZIsEQ9yAj1whXrgivTArnkhyYAkA6KUGsrjQ1EcX2hV0aAqwKY7W+Eoqj3828JAcvwcr38ekUgEL7zyIp545ikMTzivT0lRMaoqKlFdWY2qykpUV1YhLzfvhNznZK5roWkaorHYpH/bk/9ppy6pf92B1PWDt4YcnMkgwyBLyR5k1Tg0VYEIDaIAiMLYEBPGjY+XBA1GQYVBGg8ukqBCFhRIagyyFoWsRmBQwpDUEAxKCEIiAkGJAokYoETT14XUN52qaNA0QFUFxDUjooIZcZgRF8yICybERTPiggVxwYSYaEZcNEERjFBEI+KCjARkJNJDCXFIUHH8WmxkQYNZVGEWVZgEFWZRgUlIwCwkh8nrcRi1GDRVQ0wFoqqAmCoiqiWHMU1CHCJimowYJMQhIwYZcRgQhwztICegFDQFIlQImgoBKkQteX1snIjkeEHToAjylAtm4cSWopaAWQ3CogZh1saGIVi0ICxa8vs3LhiTtRRMiMGEuGBETDBNGp++DuPBUvOsUBMJJEaHEAhEoGnJ58k3+HGqbS8WGHsnPXVLpARP+9YgphlgEaI4N+ctVJuGoAFQICdfA1KvBab0bQNiyFHdcGgeyJj6R2Psh3/ij3wytCSffCxkQAAyYUu6AhGDYgW6pVr0SLUIiLmT7s9RhyFChUconFI7ixpAkdqDQq0fRRhEjuiHIBoASYYmGpKpRBAhKFGEEhL2qIvxmdseQFFBwWGXi4HkODreTaGKouD1t97A359+Au2dHdNOY7fbUV1ZhaqKKlRXJi9lJaWQs7wH0UxsltY0DbF4DIFQCIFQEP5gEP5QEIH3Df3BAALBIAKhEFRtdpZdSIWVsX+L4+MBQEs3W6e+UyEIyXGioEFM3VY1QFEFJDRA0UQkNCE9h9kiaGrqJz2OZGRJDiUoEKEko4umTL6N1G3tfbehQkICJkRh1KIwIwIjotP+4Ey/MGPvzMS/swcbihh7EzUIiEOGKkgQkTziSpj43gsT3reJf9Wn/G2fTNOQenXJS0IbfxcUTUq/SwlNRkQzIQITwpoZYc2UvK2ZEMPUTciapkGNx5GIx6HE44CqHWQxhGlvTlxqWQLMJhFWkwizrMIgqDAICgxiAkZBSV5PXxIwCAqMYgIGIQFogF8xw5sww6dYMBw1oW9EQdAfTP/rl00mWHJzYbAk9+2wCmHkiAE4hSA0IRmQ/XER7X0xRKPJOttzHTDl5s0wzGmwyQk45TgchhiccgxOQwwOOQqTqGJy80lyekCD8P4mlbHbE8YJggBFExFRZURVAyKqAXFNgklSYRHjsIgxmEQl+TEYmw0ATVOTgUhNDjVNhaam5i0IEEUBoiAiCgN6o070Rp3oi+cgoY13LilCRb4cRL4hiFw5AoMIaKIEDTL8ignehAmjcSN8cXns2yDNIKpwmWIoNEVRaI7CZYohporY53Wg1W+Hogn4/3/tv1A8g/O+MZAcR3r+CHp9PnR2d6KjqxOdXclhb3/ftE3Usiyjoqw83ZpSXVWFBTV1WbUvSiYGkiOlairC4UgqpAQRCCUvyU0ZUUSiUUTefz0aRSQWRTQWQyQagXpcVn8N0oSWmUnXMXGcBklMBpzkOA2SmGzZkcamEdMRAIA2uakdY1/s77udHjfhNgCkQlX6HdAw6ct24jujJdPHlPFjJjb3T3ye9P3CNOOmmdv7GxLevzQT79YwvrzahGVP/kgJ6TlPup56V5LvgTYpXKqaikBYxWhQhS+sIBBWEYwoiESVo980cQhmgwCnVUCOVUSuVUCuTUSOVYTZMDn0TqwVAASjGnZ0JnCgT01nozy7hMoSC2SzGYGEAYG4jKh68N6cNVVF0O1G1J886ZTBbIa9qBCiJMIgjm3u1CCLGgyCipgqIhCXEdcOHlpMogKHIQGnIQ6nMQFHamgUVEQUEWFFQkSRksOEhIgqIqpIiCoioqqImCIicYj5A8maGUUVRmlsR3c1fTGLCkxycmiWVYjQMBozYiBswlDEBF9cwsRPkCxosBlU2GQVNkPyD8nhqBoQSQgIKQLCCRHhhDAloIx92sbG22QV3/vK1+BysYUko2Taj2AsHkNPbw86OjvR0d2Fjq4OdHZ3TekPBUiGlPq6hWhsWIwli2dvh1y9ZFot9KBpGuKJBCLRKKKxKKLxGFRVhapqUFP/spI7EyaHiqIgFosgGo0gEg0j4Pci4BtFMOhDJBSApiaS5/SBAkFTkpsWkHpvx37PhdQX2Ps3kGc5TUu2IGmalnpfU+NUDWp6CKiq9r7x041LbhITRWH8kvonLE0zbnx88jECBERiCYQiCYRTl1Akjkj04C1DoijAYpZhNcmQZXHCvghjPz5jL3TCa550RYMGIJFQEYokEI0d/LkMsgirWYbFYoDVLKcuBiiqiu6BAAZHQumA5LQZUVHqQI7dOGXTszK2KU0REFOFdIvexEDsHQ2hq2cUqqbBaBBRX52HHMf0h85rGqBoQEwRJs03pgip1sDZIaQCvCwmQ4KiAglNgHKEzyFAmxIWTJIKu6zBblBhkrT0Kmg0GGAyGmCaMBSlZGuIqiZft6pqqZ2q1fT+V0G/D75QFBHIqbA1vpxmSUWBWYFV1vCfn/sG8vIYSDLKfPgR1DQNQ8PDk1pTWtpb4RkdnTSd0WhE/YKFWNLQiCUNjairqZ1Xm3nmQy2yWfIHVkEiEUc8HoMgaHA4zRj1BBCNxaHGk+Pj8WhyU0EiuckgEY9BSSSQSMSRSMShJBJQlHjqR1pNbsJS1XSQSjZfjzdlq5qWup28Dk2FIEoQRQmiJKaGEqTUeaNESYYkSalpRARCEYx4fRjxeDHiGUU0Hk8eoplQxo+MUZJHwigThvPla9ZoNMCV60SBKxcOuxVOuw1OhwUWswnahCN8xkwMAcL7ttEIY9v4JtyrQYOqqojGovB6Axj1BeH1Jy8+fwjhyMyOYMt1WlBbmQ9XjhWiKKafSxCE1DIJ6WUTBHFySxi0dD00TYM/GMF7e7oQDEUhCMCi2mJUl7tSLyS1b0hqubXU0T5j19O3DxJUYurYax8LGYBBEmCURRhlCSaDDLPRALPRCKvJCGvqSC2DwQQ5ddSU0SAhFIogGosjHI0iFIkgHIsjHIshEosjGosjElcQSyQQTaiIJ9T0axUFwGW3oLwgF9UlpcjPc8FqscFiscNqdcBitcNitkIUj/7cYLFIGE/e93M8du9diISDMOQXwlFQhPBAFySjEZLRiHv+/iKKyysPOy8GkuNovv4IapqG/oEB7Nm/N3nZt3dKp24mkwkNCxclA8rixaipqjnmE+DNpflai2yVafWIx+Po7u1BR2cHOlKbODu6OxGJzLyvmkORJAmyLMNoMCYP1TWMHbprmHQ7Pd5ggCE1rdFggCAIiCfiiMfiiMaiiMXjiKWOQIrFYoiljkKKT7oeT3cL4HQ4UF5ajvKyMpSXlqGstAzlpeXIzcmBwSDpVotwJIzevl509/agpzc57O7twYg7eVLLJQ2NuOLiy9DYsHhWnzcSieCX9/0a/3rzdQDA2pPW4IaPfXLas71PZ3JQGQsvCuKpI7zsVhtk2XjE34lHs16omopgKIxAKIj83DwYj1NLdsDrwV/u/RGe+O1PEU+d1b18YQPqV52MW370v7Dn5Bx2Hgwkx1GmfekeLU3T0NPXmw4new/sQyAQmDSN2WzG4voGLFnciIqycuS78lHgyofZbNZpqSfLllpkC1kWkZtrxciIH5FIDEr6EN1E8nrqUF0l1VScHq8koKpqsu+NsX44DDLksR92ebyfDlme/mzZwWAQHd2d6fDR3tmB3v6+aQ/bNsgGVJZXoLqqClWVVcjLyYUkycnnlGTIsgxZliDLhtRtKTXOkLyeanHRq9PCZGtO4pD7g2XiuhGOhBEKhef03F2apuHZF57D/Q//DoqioLiwCP9x482oqdLnPFJ+vx8jnmE01NdCUcSMqcXheAb70dfWjKrFS9NnMDeZkyH4cBhIjqNMXNFng6qq6O7txp59+7BnfzKghEKhaae122zId+UnA0p+AQpcrkm3c5zO4/Jlna21yHSapmHYPZJscZjQ+uD1epFQEnO+aUOSpGR4McgwyDI0DRg9yFmlbVYbaqqSO3Unj0CrRllJaUa3/M2GE33daGlrxd0/uwfDI8MwyAb824c+jI1nbJjT7hHCkTDaOtrR2t6G1rZWtLa3YSjVZQMA5ObkoKK8ApVlFclheTnKS8sz5g/ewWiahkce+yOuuOR8lJaUHHZ6BpLj6ERZ0VVVRUdXJ/bs34v9TQcwODSIEbcbofD0IWUiSZKQ73KhwFWAfFc+XHl5cOXlIS93/DIboSVTaxGLx6CqGsym+XlOmokURUFvfx/ax4JHajjWad9MSJKU3LyRam0Yvy6nN3skN18kkIjHk03k8XiqM7TkcCZfcYX5BcnQUVWdDh/5Lhf76MmgdeN4CgQD+Omvfo73dm4HkDwlR0lRMVwuF/Lz8pHvcsGVl/wzZbfZjuhzEo/H0dndiZb2NrS2taGlvRV9/X3Tfk5znDnw+rwHnVdhQSEqy8tRMSGolBaXzuiAA1VVEYlEEI5EEIlGEIkkL+FoBJFIGFaLFWWlZSgqKDyq71tN0/Dgo7/HP55+Ev93708YSDLNib6ih0IhjHjcGB4Zxoh7BMPuEYyMuDHsTt52ezwz+vEQRRG5ObnIy81FXm4eXLl5yM3NhSs3D3l5Y7fzYLVYDjqPTKhFNBpNbSJoR1tH8tLT1wtVVSe3JLnykZ+fj/y88dakvNzcjDlXkaZpCIfD6OnrTYWPDnR0dqKrpzu9z8JEkiShvLQcNVVV6cPKS4qKUFDgRDAQgwZh1jZtaJoGRVGS4SSeSPbqmkgGl7HNPiVFxbDZbMf0PNkkE9aNTKCqKh5/6h/4w58fPeT3kslohMs1tn6OBZVkcHG5XFBVNdny0Z5s+ejs7pp2c2C+y4W6mjrU1dSirqYWtdU1yHE6YDaL2LWnCe2dXeju6UZXTze6e7sx6p0+qEiShJKiYpSXlUMQBESjUYQj4fHAEUkeKTfTU2AYZANKS0pQVlqGspJSVJSVo6y0DCVFxQcNPqqq4r7fP4Bnnv8nAOCPD9wPh/3w6xgDyXHEFf3QFEWBZ3R0PLCMjMA96oFn1AOPxwOP14NRr3fGzfomoxFWqw12mw02mw221HWr1QaH3YaiQhdEwQCz2ZKcxpq8WK3WWW+Wj0Sj6OjqQFtHO9onhI+jXf1EUZz0xZef70KBKx85zpz0/hLj+zRMuKRaGgyyAZIsQ061QIz9w0skEvAHAggEU5ex64HgpHH+YADB4Ni4IBKJ6XuYNZvMk1oeaqqqUV5aNuWLjOtG5mAtJusf6EdbRztGPG6MuN1wu0fS19+/c/9M2e121NXUYkFNHepq6rCgthY5zqk7fR6qFn6/H929Pejq7UZ3Tze6e5LXD7a5/GBEUYTZbE6di8kMc2roDwTQN9CHeHzqn4qxxxUVFKK8rBxlpaUoL0nuHF1aXILfPfp7vPDSNgiCgI9/+GO48rILeXK9TMMV/dgpigKvzwvP6Cg8ox64Rz0YHR2F25MKLqMeeEZHZ7R56FAsFgusFgssZkvqevKkeJb00DLpttViTY8zmUzo6+9HW6rlo72jHb0HaZLNzclBTVUNaqvHLyaTCSNuN0bcIxOGqRYltxueUc9RnyfnYMZ2+IwdwYkD3y83JzcdOsZCyEybe7luZA7WYuZi8RjcHs+UoOL2pNZbjxuqqqK2qibd8lFXU4fCgoIZbeY50lpomgbPqAfdPT3o6e+FIAiTg8b7g4fZDINsOOiyqKqKoeEh9PT1obe/N3UEVC96+3un7a9qIkEQcMPHPokzTzuDZ/vNRFzRj59INAqfz4tgKIhgMIRAMJA8b0yqC/ZwOIRYPAKPx4tAMIhgKDnNbB3WOZ3cnNxJwaO2ugZ5uXlHPB9VVTHqHcWI250KKePBxef3pY9CSShKsu+O1Anl4ok4lNTwUARBgM1mg8Nmh81mg91mh8Nuhz11e+y6fWyYun4s+71w3cgcrEXmyNRaaJqGUe8oenp70dOXDCg9fb3o6e2Fz++DQTbgho9/CutPPgUAZhxI5k9PVkRHwGwywVxYdND7D7aiK4qCUCiUDC2RMELhEMLh5DAUDqeuhxFO3U7eH0nfDodDiMZiyMvNS4eOsRaQvNzcWXltY5trXHku1C9YeMSPH+snIZ5IJDsZUxKIxxNQVQVWqxVWizVj9k8hoswjCEL6IINlS5ZOui8QDECW5KM6AoiBhGgCSZLgcDjgmMEx8wejqmpG/6ALgpA+egVZcDQPEWUOu81+1I/N3G9Nonkqk8MIEVGm4jcnERER6Y6BhIiIiHTHQEJERES6YyAhIiIi3TGQEBERke4YSIiIiEh3DCRERESkOwYSIiIi0h0DCREREemOgYSIiIh0x0BCREREumMgISIiIt0xkBAREZHuBE3TNL0XYqY0TYOqzpvFnZYkiVAUVe/FILAWmYb1yBysRebIhlqIogBBEA473bwKJERERJSduMmGiIiIdMdAQkRERLpjICEiIiLdMZAQERGR7hhIiIiISHcMJERERKQ7BhIiIiLSHQMJERER6Y6BhIiIiHTHQEJERES6YyAhIiIi3TGQEBERke4YSIiIiEh3DCRERESkO1nvBaCZ+epXv4qXXnoJg4OD2L17N2SZpSOaDtcVoqluuOEG9PX1QRAEFBQU4I477kBpaaneizWJoGmapvdC0OG98cYbqKurw+mnn84vWaJD4LpCNJXf74fD4QAA3Hfffdi+fTt++MMf6rxUk3GTzRzp6OjA7bffjksvvRRLlizBRRdddNBp29ra8IlPfAKrVq3C+vXrcccddyASiUyaZt26dSgoKJjrxSaaVbO9HswE1xXKdHqsF2NhBAACgcBRLfdc41+HOdLU1IRt27Zh5cqVUFUVB2uI8vl8+Ld/+zeUlZXh7rvvhtvtxne+8x2Mjo7iBz/4wXFeaqLZxfWAaCq91osvfOELeP3115GTk4Pf/OY3x/oyZh0DyRzZtGkTzj77bADAl7/8ZezatWva6X7/+9/D5/Phscceg8vlAgBIkoRbb70V//7v/44FCxYct2Ummm2zvR5ce+21GBgYmPL4BQsW4Be/+MUcvQqi2aXXevHDH/4Qmqbhl7/8JX7yk5/gG9/4xiy/smPDQDJHRHFmW8NefPFFrF+/Pv1hA4Bzzz0XX/3qV7Ft2zYGEprXZns9eOihh+ZkOYmOJz3XC0EQcM0112DDhg0ZF0i4D4nOWlpapoQOo9GIqqoqtLS06LRURMcX1wOiqWZrvQgGg+jv70/ffvLJJ1FfXz9ryzlb2EKiM5/PB6fTOWW80+mE1+tN3/7iF7+I119/HUCyuW/t2rW46667jttyEs2lma4HM8F1hbLFbK0X4XAYN998M6LRKACgtLQU3//+92dtOWcLA8kM+f1+DA4OHna6yspKGI3GY34+TdMgCEL6diZ+eOjEo/d6MBNcV+h4y/T1oqCgAI8++ugxP+9cYyCZoWeeeQZf+cpXDjvdY489hsbGxhnP1+l0wufzTRnv9/u5/whlHK4HRFNxvZgdDCQzdMUVV+CKK66Y9fkuWLBgyrbAWCyGzs5OfPCDH5z15yM6FlwPiKbiejE7uFOrzs466yy89tpr8Hg86XHPPPMMYrEYNmzYoOOSER0/XA+IpjrR1gu2kMyRcDiMbdu2AQB6enoQCATw5JNPAkj2JDl2GNfWrVvxwAMP4KabbsJNN92EkZERfPe738XFF1+clU1ydGLhekA0FdeL6fFcNnOku7sbmzdvnva+++67D6ecckr6dltbG+644w68/fbbMJvNuOiii3DrrbfCbDYfr8UlmhNcD4im4noxPQYSIiIi0h33ISEiIiLdMZAQERGR7hhIiIiISHcMJERERKQ7BhIiIiLSHQMJERER6Y6BhIiIiHTHQEJERES6YyAhIiIi3TGQEBERke4YSIiIiEh3DCRERESku/8PCLlVzczqx3wAAAAASUVORK5CYII=
for site in d_us[v]['p_phi_s'].keys():
    plt.figure()
    cc=cani_norm2(np.linspace(0,1,7))
    xx=d_us[v]['p_zL_s'][site][:]
    yy=d_us[v]['p_phi_s'][site][:]
    for i in range(len(cc)):
        #plt.loglog(-xx,yy[i,:],color=cc[i,:])
        plt.semilogx(-xx,yy[i,:],color=cc[i,:])
    plt.title(site)
    plt.xticks([10**-3,10**-2,10**-1,1,10],[r'$-10^{-3}$','',r'$-10^{-1}$','',r'$-10^{1}$'])
    plt.xlim(10**(-3.5),10**(1.5))
    plt.ylim(.5,10)
    plt.gca().invert_xaxis()

# %%
d_ss['U'].keys()

# %%
len(cc)

# %%
