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
mpl.rcParams['figure.dpi'] = 100ff
sns.set_theme()
plt.rcParams.update({'figure.max_open_warning': 0})

# %%

# %% [markdown]
# # Test Tif

# %%
idir='/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data/'
dsm=rasterio.open('/home/tswater/Documents/tyche/data/neon/dsm/BART/dsm_BART.tif').read(1)
fpdsm=rasterio.open('/home/tswater/Documents/tyche/data/neon/dtm/BART/dtm_BART.tif')
fpu=h5py.File(idir+'/NEON_TW_U_UVWT.h5','r')
mu=fpu['SITE'][:]==bytes('BART','utf-8')
lat=fpu['lat'][mu][0]
lon=fpu['lon'][mu][0]
transformer = Transformer.from_crs("EPSG:4326", fpdsm.crs, always_xy=True)
xx_, yy_ = transformer.transform(lon, lat)

# %%
transformer = Transformer.from_crs("EPSG:4326", fpdsm.crs, always_xy=True)
xx_, yy_ = transformer.transform(lon, lat)

# %%
print(xx)
print(yy)

# %%
xx,yy=fpdsm.index(xx_,yy_)

# %%
plt.imshow(dsm,cmap='terrain',vmin=200)
plt.scatter([xx],[yy],c='k')
plt.colorbar()

# %%
lat

# %%

# %% [markdown]
# # Test Footprint

# %%
fp=nc.Dataset('/home/tswater/Documents/tyche/data/neon/dp4ex/ABBY/NEON.D16.ABBY.DP4.00200.001.nsae.2023-06-23.nc','r')
fpdsm=rasterio.open('/home/tswater/Documents/tyche/data/neon/dsm/ABBY/dsm_ABBY.tif')
fpu=h5py.File(idir+'/NEON_TW_U_UVWT.h5','r')
mu=fpu['SITE'][:]==bytes('ABBY','utf-8')
lat=fpu['lat'][mu][0]
lon=fpu['lon'][mu][0]
transformer = Transformer.from_crs("EPSG:4326", fpdtm.crs, always_xy=True)
xx_, yy_ = transformer.transform(lon, lat)
xx,yy=fpdsm.index(xx_,yy_)

# %%
proj4=fpdsm.crs.to_proj4()
nlcdf='/home/tswater/Downloads/nlcd_2021_land_cover_l48_20230630/nlcd_2021_land_cover_l48_20230630.img'
ex_str=str(int(xx_-dx*150.5))+' '+str(int(yy_-dx*150.5))+' '+str(int(xx_+dx*150.5))+' '+str(int(yy_+dx*150.5))
cmd="gdalwarp -t_srs '"+proj4+"' -tr "+str(dx)+' -te '+ex_str+' '+nlcdf+' temp.tif'

print(cmd)


# %%
def databig(data,dx):
    dout=np.zeros((dx*301,dx*301))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            dout[i*dx:(i+1)*dx,j*dx:(j+1)*dx]=data[i,j]
    return dout


# %%
dx=fp.dx

# %%
data=fp['footprint'][:]
#data[data<=0.001]=float('nan')
#contour=data.copy()
#contour[data<.1]=float('nan')
#contour[data>=.1]=1

# %%
time=10
plt.imshow(data[time,100:200,100:200],cmap='terrain')
plt.contour(data[time,100:200,100:200]*100)
plt.colorbar()
print(np.nansum(data[time]))

# %%
np.nansum(data[data>1])

# %%
sms=[]
cutoff=[]
for t in range(48):
    for i in np.logspace(-2,-6,100):
        dd=data[t,:,:]
        sm=np.nansum(dd[dd>i])
        if sm>=.95:
            sms.append(sm)
            cutoff.append(i)
            break

# %%
plt.semilogx(cutoff,sms,'o')

# %%
d2=data.copy()
d2[d2<.0004]=0
d2[d2>=.0004]=1
dx=fp.dx
dtm=fpdtm.read(1)[xx-int(dx*150.5):xx+int(dx*150.5),yy-int(dx*150.5):yy+int(dx*150.5)]
dsm=fpdsm.read(1)[xx-int(dx*150.5):xx+int(dx*150.5),yy-int(dx*150.5):yy+int(dx*150.5)]
dtm2=dtm.copy()
dtm2[foot==0]=float('nan')
foot=databig(d2[0],int(dx))

# %%
plt.imshow(dtm2,cmap='terrain')

# %%
vari=[]
vari2=[]
dsm=dsm[xx-int(dx*150.5):xx+int(dx*150.5),yy-int(dx*150.5):yy+int(dx*150.5)]
dtm=dtm[xx-int(dx*150.5):xx+int(dx*150.5),yy-int(dx*150.5):yy+int(dx*150.5)]
for t in range(48):
    print('.',end='',flush=True)
    foot=databig(d2[t],int(dx))
    a=dsm[foot==1]
    b=dtm[foot==1]
    vari.append(np.nanstd(a))
    vari2.append(np.nanstd(b))

# %%
plt.scatter(vari,np.array(vari)-np.array(vari2))

# %%
fpdsm=rasterio.open('/home/tswater/Documents/tyche/data/neon/dsm/ABBY/dsm_ABBY.tif')
fpdtm=rasterio.open('/home/tswater/Documents/tyche/data/neon/dtm/ABBY/dtm_ABBY.tif')
dsm=fpdsm.read(1)
dtm=fpdtm.read(1)

# %%

# %%
import timeit


# %%
def f():
    times[99]
times=fpu['TIME'][:]
# %timeit f()

# %%
import copy

# %%
test={'hello':[],'goodbye':[]}
test2=copy.deepcopy(test)
print(test2)
test['hello'].append(78)
print(test2)


# %%
def f():
    times[11199]
times=fpu['TIME'][:]
# %timeit f()

# %%
def f():
    list(times).pop()
times=fpu['TIME'][:]
# %timeit f()

# %%
from pyproj import CRS

# %%
CRS.to_proj4
fpdsm.crs.to_proj4()

# %%
fp=nc.Dataset('/home/tswater/Documents/tyche/data/neon/dp4ex/NOGP/NEON.D09.NOGP.DP4.00200.001.nsae.2023-06-23.nc','r')

# %%
int(fp.dx)

# %%
from scipy import stats

# %%
a=np.array([1,1,2,3,4,2,3,4,1,2,3,2,3,2,3,45,5,6,8,9,3,2,1,2,1,23,4,3,2,3,4,3,2,3,5,4,6,4,5,5,6,55,9,float('nan')])
print(np.unique(a,return_counts=True))
stats.mode(a)

# %% [markdown]
# # PDFs With MET Variables

# %%
fps=h5py.File('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data/NEON_TW_S_UVWT.h5','r')
fpu=h5py.File('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data/NEON_TW_U_UVWT.h5','r')

# %%
lmost=fps['L_MOST'][:]
zzd=fps['tow_height'][:]-fps['zd'][:]
zL_s=zzd/lmost

lmost=fpu['L_MOST'][:]
zzd=fpu['tow_height'][:]-fpu['zd'][:]
zL_u=zzd/lmost

ani_u=fpu['ANI_YB'][:]
ani_s=fps['ANI_YB'][:]


# %%
def time2hour(time,utcf):
    hour_=[]
    doy_=[]
    d0=datetime(1970,1,1,0)
    yold=1970
    N=len(time)
    milestones=np.linspace(0,N,10).astype(int)
    for t in range(N):
        if t in milestones:
            print('.',end='',flush=True)
        df=d0+timedelta(hours=utcf[t])+timedelta(seconds=time[t])
        hour_.append(df.hour)
        doy_.append(df.timetuple().tm_yday)
    print()
    return np.array(hour_),np.array(doy_)


# %%
fps.keys()
varlist=['CO2','CO2FX','CO2_SIGMA','G','H','H2O','HOUR','DOY','G','H','H2O','H2O_SIGMA',\
         'LE','LW_OUT','NETRAD','P','PA','RH','RHO','TA','T_SONIC','T_SONIC_SIGMA','USTAR','VELO',\
         'VPD','VPT']
static_vars=['elevation','canopy_height','tow_height','nlcd_dom','zd']

# %%
hru,doyu=time2hour(fpu['TIME'][:],fpu['utc_off'][:])
hrs,doys=time2hour(fps['TIME'][:],fps['utc_off'][:])


# %%

for var in varlist:
    print(var,flush=True)
    if var=='HOUR':
        Xu=hru
        Xs=hrs
    elif var=='DOY':
        Xu=doyu+hru/24
        Xs=doys+hrs/24
    elif var=='VELO':
        Xu=np.sqrt(fpu['U'][:]**2+fpu['V'][:]**2)
        Xs=np.sqrt(fps['U'][:]**2+fps['V'][:]**2)
    else:
        Xu=fpu[var][:]
        Xs=fps[var][:]

        Xu[Xu==-9999]=float('nan')
        Xs[Xs==-9999]=float('nan')
        
    fig=plt.figure(figsize=(6,3))
    grid=ImageGrid(fig, 111,  # similar to subplot(111)
                nrows_ncols=(1, 2),
                axes_pad=0.05,
                cbar_mode=None,
                cbar_pad=.02,
                cbar_size="5%")

    xmin=min(np.nanpercentile(Xu,.5),np.nanpercentile(Xs,.5))
    xmax=max(np.nanpercentile(Xu,99.5),np.nanpercentile(Xs,99.5))
    xrange=xmax-xmin
    mp=xrange/(np.sqrt(3)/2)*.66

    grid[0].set_ylabel(var)
    
    im=grid[0].hexbin(Xu,np.array(ani_u)*mp,mincnt=10,cmap='terrain',gridsize=200,extent=(xmin,xmax,0,np.sqrt(3)/2*mp),vmin=0,vmax=600)
    im=grid[1].hexbin(Xs,np.array(ani_s*mp),mincnt=10,cmap='terrain',gridsize=900,extent=(xmin,xmax,0,np.sqrt(3)/2*mp),vmin=0,vmax=600)
    grid[0].set_yticks(np.array([.2,.4,.6,.8])*mp,[.2,.4,.6,.8])
    grid[1].set_yticks(np.array([.2,.4,.6,.8])*mp,[.2,.4,.6,.8])

# %%
fpu.keys()

# %%
# ls /home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/

# %%
a=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_unst_tw_v2.p','rb'))
print(a.keys())
print(a['U'].keys())
print(a['ABBY'].keys())

# %%
fpu['SITE'][0]

# %%
bytes('ABBY','utf-8')

# %% [markdown]
# # Test MAD vs site characteristics

# %%
d_u=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_unst_v3.p','rb'))
d_s=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_stbl_v3.p','rb'))
fpst=h5py.File('/home/tswater/tyche/data/neon/static_data.h5','r')

# %%
d_s['U'].keys()

# %%
float('Inf')


# %%
def fix(d1,d2):
    new1=[]
    new2=[]
    for i in range(len(d1)):
        if (np.isnan(d2[i])) or (d2[i]==float('Inf')):
            continue
        else:
            new1.append(d1[i])
            new2.append(d2[i])
    return np.array(new1),np.array(new2)


# %%
from scipy import stats

# %%
sites=fpst['site'][:]
var='W'
mad_u=[]
mad_s=[]
sites.sort()
for site in sites:
    ss=str(site)[2:-1]
    mad_u.append(d_u[var]['MAD_SC23_s'][ss])
    mad_s.append(d_s[var]['MAD_SC23_s'][ss])
mad_u=np.array(mad_u)
mad_s=np.array(mad_s)
for k in ['elevation','canopy_height','mean_U','mean_chm','mean_dsm','mean_dtm','mean_xb','std_xb','mean_yb','q_jja','T_jja','T_std','std_chm','std_dtm','range_chm','range_dtm','tow_height','zd','towzd']:
    plt.figure(figsize=(8,3))
    plt.suptitle(k)
    plt.subplot(1,2,1)
    if k=='mean_xb':
        k='mean_u_xb'
        k2='mean_s_xb'
    elif k=='mean_yb':
        k='mean_u_yb'
        k2='mean_s_yb'
    elif k=='std_xb':
        k='std_u_xb'
        k2='std_s_xb'
    else:
        k2=k
    if k=='towzd':
        data=fpst['tow_height'][:]-fpst['zd'][:]
    else:
        data=fpst[k][:]
    plt.scatter(data,mad_u)
    
    try:
        xx,yy=fix(mad_u,data)
        plt.xlabel(str(stats.pearsonr(xx,yy)[0])[0:6])
    except:
        pass
    plt.subplot(1,2,2)
    if k2=='towzd':
        data=fpst['tow_height'][:]-fpst['zd'][:]
    else:
        data=fpst[k2][:]
    plt.scatter(data,mad_s)
    try:
        xx,yy=fix(mad_s,data)
        plt.xlabel(str(stats.pearsonr(xx,yy)[0])[0:6])
    except:
        pass

# %%
sites=fpst['site'][:]
mad_u=[]
mad_s=[]
sites.sort()
vari=['U','V','W','T','H2O','CO2']
varj=['mean_U','mean_xb','std_xb','mean_yb','q_jja','T_jja','T_std','mean_chm','mean_dsm','mean_dtm','std_chm','std_dtm','range_chm','range_dtm','tow_height','elevation','canopy_height','zd','towzd']
stb_sc=np.zeros((6,len(varj)))
ust_sc=np.zeros((6,len(varj)))
for i in range(6):
    var=vari[i]
    mad_u=[]
    mad_s=[]
    sites.sort()
    for site in sites:
        ss=str(site)[2:-1]
        mad_u.append(d_u[var]['MAD_SC23_s'][ss])
        mad_s.append(d_s[var]['MAD_SC23_s'][ss])
    mad_u=np.array(mad_u)
    mad_s=np.array(mad_s)
    for j in range(len(varj)):
        k=varj[j]
        if k=='mean_xb':
            k='mean_u_xb'
            k2='mean_s_xb'
        elif k=='mean_yb':
            k='mean_u_yb'
            k2='mean_s_yb'
        elif k=='std_xb':
            k='std_u_xb'
            k2='std_s_xb'
        else:
            k2=k
        if k=='towzd':
            datau=fpst['tow_height'][:]-fpst['zd'][:]
        else:
            datau=fpst[k][:]
        
        try:
            xx,yy=fix(mad_u,datau)
            ust_sc[i,j]=stats.pearsonr(xx,yy)[0]
        except:
            ust_sc[i,j]=float('nan')
            pass
        if k2=='towzd':
            datas=fpst['tow_height'][:]-fpst['zd'][:]
        else:
            datas=fpst[k2][:]
        try:
            xx,yy=fix(mad_s,datas)
            stb_sc[i,j]=stats.pearsonr(xx,yy)[0]
        except:
            stb_sc[i,j]=float('nan')

# %%

# %%
N=len(varj)
x=np.linspace(0,N-1,N)
plt.figure(figsize=(12,4))
plt.scatter(x,ust[0,:],c='blue')
plt.scatter(x,stb[0,:],c='cornflowerblue')

plt.scatter(x,ust[1,:],c='red',marker='s')
plt.scatter(x,stb[1,:],c='lightcoral',marker='s')

plt.scatter(x,ust[2,:],c='green',marker='P')
plt.scatter(x,stb[2,:],c='limegreen',marker='P')
plt.plot([0,N],[0,0],'k-')
plt.legend(['U (Unstable)','U (Stable)','V (Unstable)','V (Stable)','W (Unstable)','W (Stable)'],framealpha=1)

plt.xticks(x,varj,rotation=45)
plt.ylabel('Corr. MAD vs Var')
print()

# %%
N=len(varj)
x=np.linspace(0,N-1,N)
plt.figure(figsize=(12,4))
plt.scatter(x,-(np.abs(ust[0,:])-np.abs(ust_sc[0,:])),c='blue')
plt.scatter(x,-(np.abs(stb[0,:])-np.abs(stb_sc[0,:])),c='cornflowerblue')

plt.scatter(x,-(np.abs(ust[1,:])-np.abs(ust_sc[1,:])),c='red',marker='s')
plt.scatter(x,-(np.abs(stb[1,:])-np.abs(stb_sc[1,:])),c='lightcoral',marker='s')

plt.scatter(x,-(np.abs(ust[2,:])-np.abs(ust_sc[2,:])),c='green',marker='P')
plt.scatter(x,-(np.abs(stb[2,:])-np.abs(stb_sc[2,:])),c='limegreen',marker='P')
plt.plot([0,N],[0,0],'k-')
plt.legend(['U (Unstable)','U (Stable)','V (Unstable)','V (Stable)','W (Unstable)','W (Stable)'],framealpha=1)

plt.xticks(x,varj,rotation=45)
plt.ylabel(r'$\Delta$ Correlation')
print()

# %%
d_u['U'].keys()


# %%
def sortXbyY(X,Y):
    return [x for _, x in sorted(zip(Y, X))]
Z=sortXbyY(fpst['site'][:],fpst['mean_u_xb'][:])
for s in Z:
    print(str(s)+': '+str(d_u['U']['MAD_SC23_s'][str(s)[2:-1]]*1.7)[0:6])

# %%

# %%
for var in fpst.keys():
    try:
        xx,yy=fix(fpst['mean_u_xb'][:],fpst[var][:])
        a=stats.pearsonr(xx,yy)[0]
    except:
        a=0
        pass
    if np.abs(a)<.4:
        continue
    else:
        plt.figure()
        plt.scatter(xx,yy)
        plt.ylabel(var)

# %%
len(fpst['mean_s_xb'][0::2])

# %%
fpst.keys()

# %% [markdown]
# # XB Stuff

# %%
vmx_a=.7
vmn_a=.1
vmx_a2=1
vmn_a2=0

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
#binplot1d(zzd/lmost,sigu/ustar,ani,-(np.logspace(-4,2,21)[-1:0:-1]),np.linspace(vmn_a,vmx_a,11),mincnt=50)

# %%
fps=h5py.File('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data/NEON_TW_S_UVWT.h5','r')
fpu=h5py.File('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data/NEON_TW_U_UVWT.h5','r')

# %%
fpu.keys

# %%
minpct=1e-4
phi=np.sqrt(fpu['VV'][:])/fpu['USTAR'][:]
zL=(fpu['tow_height'][:]-fpu['zd'][:])/fpu['L_MOST'][:]
ani=fpu['ANI_YB'][:]
xplt,yplt,aplt,cnt=binplot1d(zL,phi,ani,False)
tot=np.sum(cnt)
yplt[cnt/tot<minpct]=float('nan')

phi=np.sqrt(fps['VV'][:])/fps['USTAR'][:]
zL=(fps['tow_height'][:]-fps['zd'][:])/fps['L_MOST'][:]
ani=fps['ANI_YB'][:]

minpct=1e-4
xplts,yplts,aplts,cnts=binplot1d(zL,phi,ani,True)
tot=np.sum(cnts)
yplts[cnts/tot<minpct]=float('nan')

# %%
zL_u=-np.logspace(-4,2,40)
zL=zL_u.copy()

# U Unstable
u_u_old=2.55*(1-3*zL)**(1/3)

# V Unstable
v_u_old=2.05*(1-3*zL)**(1/3)

# W Unstable 
w_u_old=1.35*(1-3*zL)**(1/3)

t_u_old=.99*(.067-zL)**(-1/3)
t_u_old[zL>-0.05]=.015*(-zL[zL>-0.05])**(-1)+1.76

zL_s=np.logspace(-4,2,40)
zL=zL_s.copy()

# U Stable
u_s_old=2.06*np.ones(zL.shape)

# V Stable
v_s_old=2.06*np.ones(zL.shape)

# T STABLE
t_s_old=0.00087*(zL)**(-1.4)+2.03


# W Stable
w_s_old=1.6*np.ones(zL.shape)


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


# %%
plotit(xplt,yplt,aplt,xplts,yplts,aplts,w_u_old,w_s_old,ylim=[.9,3.5])
#plt.gca().invert_xaxis()
#plt.legend(aplt)

# %%
minpct=1e-4
phi=np.sqrt(fpu['VV'][:])/fpu['USTAR'][:]
zL=(fpu['tow_height'][:]-fpu['zd'][:])/fpu['L_MOST'][:]
ani=fpu['ANI_YB'][:]
xb=fpu['ANI_XB'][:]
m=xb>.7

xplt,yplt,aplt,cnt=binplot1d(zL[m],phi[m],ani[m],False)
tot=np.sum(cnt)
yplt[cnt/tot<minpct]=float('nan')

phi=np.sqrt(fps['VV'][:])/fps['USTAR'][:]
zL=(fps['tow_height'][:]-fps['zd'][:])/fps['L_MOST'][:]
ani=fps['ANI_YB'][:]
xb=fps['ANI_XB'][:]
m=xb>.7

minpct=1e-4
xplts,yplts,aplts,cnts=binplot1d(zL[m],phi[m],ani[m],True)
tot=np.sum(cnts)
yplts[cnts/tot<minpct]=float('nan')

# %%
plotit(xplt,yplt,aplt,xplts,yplts,aplts,v_u_old,v_s_old)

# %%
minpct=1e-4
phi=np.sqrt(fpu['VV'][:])/fpu['USTAR'][:]
zL=(fpu['tow_height'][:]-fpu['zd'][:])/fpu['L_MOST'][:]
ani=fpu['ANI_YB'][:]
xb=fpu['ANI_XB'][:]
m=(xb>.55)&(xb<.7)

xplt,yplt,aplt,cnt=binplot1d(zL[m],phi[m],ani[m],False)
tot=np.sum(cnt)
yplt[cnt/tot<minpct]=float('nan')

phi=np.sqrt(fps['VV'][:])/fps['USTAR'][:]
zL=(fps['tow_height'][:]-fps['zd'][:])/fps['L_MOST'][:]
ani=fps['ANI_YB'][:]
xb=fps['ANI_XB'][:]
m=(xb>.55)&(xb<.7)

minpct=1e-4
xplts,yplts,aplts,cnts=binplot1d(zL[m],phi[m],ani[m],True)
tot=np.sum(cnts)
yplts[cnts/tot<minpct]=float('nan')

# %%
plotit(xplt,yplt,aplt,xplts,yplts,aplts,v_u_old,v_s_old)

# %%
minpct=1e-4
phi=np.sqrt(fpu['VV'][:])/fpu['USTAR'][:]
zL=(fpu['tow_height'][:]-fpu['zd'][:])/fpu['L_MOST'][:]
ani=fpu['ANI_YB'][:]
xb=fpu['ANI_XB'][:]
m=(xb>.45)&(xb<.55)

xplt,yplt,aplt,cnt=binplot1d(zL[m],phi[m],ani[m],False)
tot=np.sum(cnt)
yplt[cnt/tot<minpct]=float('nan')

phi=np.sqrt(fps['VV'][:])/fps['USTAR'][:]
zL=(fps['tow_height'][:]-fps['zd'][:])/fps['L_MOST'][:]
ani=fps['ANI_YB'][:]
xb=fps['ANI_XB'][:]
m=(xb>.45)&(xb<.55)

minpct=1e-4
xplts,yplts,aplts,cnts=binplot1d(zL[m],phi[m],ani[m],True)
tot=np.sum(cnts)
yplts[cnts/tot<minpct]=float('nan')

# %%
plotit(xplt,yplt,aplt,xplts,yplts,aplts,v_u_old,v_s_old)

# %%
minpct=1e-4
phi=np.sqrt(fpu['VV'][:])/fpu['USTAR'][:]
zL=(fpu['tow_height'][:]-fpu['zd'][:])/fpu['L_MOST'][:]
ani=fpu['ANI_YB'][:]
xb=fpu['ANI_XB'][:]
m=(xb>.3)&(xb<.45)

xplt,yplt,aplt,cnt=binplot1d(zL[m],phi[m],ani[m],False)
tot=np.sum(cnt)
yplt[cnt/tot<minpct]=float('nan')

phi=np.sqrt(fps['VV'][:])/fps['USTAR'][:]
zL=(fps['tow_height'][:]-fps['zd'][:])/fps['L_MOST'][:]
ani=fps['ANI_YB'][:]
xb=fps['ANI_XB'][:]
m=(xb>.3)&(xb<.45)

minpct=1e-4
xplts,yplts,aplts,cnts=binplot1d(zL[m],phi[m],ani[m],True)
tot=np.sum(cnts)
yplts[cnts/tot<minpct]=float('nan')

# %%
plotit(xplt,yplt,aplt,xplts,yplts,aplts,v_u_old,v_s_old)

# %%
minpct=1e-4
phi=np.sqrt(fpu['VV'][:])/fpu['USTAR'][:]
zL=(fpu['tow_height'][:]-fpu['zd'][:])/fpu['L_MOST'][:]
ani=fpu['ANI_YB'][:]
xb=fpu['ANI_XB'][:]
m=(xb>0)&(xb<.3)

xplt,yplt,aplt,cnt=binplot1d(zL[m],phi[m],ani[m],False)
tot=np.sum(cnt)
yplt[cnt/tot<minpct]=float('nan')

phi=np.sqrt(fps['VV'][:])/fps['USTAR'][:]
zL=(fps['tow_height'][:]-fps['zd'][:])/fps['L_MOST'][:]
ani=fps['ANI_YB'][:]
xb=fps['ANI_XB'][:]
m=(xb>0)&(xb<.3)

minpct=1e-4
xplts,yplts,aplts,cnts=binplot1d(zL[m],phi[m],ani[m],True)
tot=np.sum(cnts)
yplts[cnts/tot<minpct]=float('nan')

# %%
plotit(xplt,yplt,aplt,xplts,yplts,aplts,v_u_old,v_s_old)

# %%
anibins=np.linspace(vmn_a,vmx_a,7)
zLbins=np.logspace(-4,2,20)[-1:0:-1]
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
siteX='BONA'
lim1=.7
lim2=.4
lim3=0

##################

minpct=1e-4
phi=np.sqrt(fpu['VV'][:])/fpu['USTAR'][:]
zL=(fpu['tow_height'][:]-fpu['zd'][:])/fpu['L_MOST'][:]
ani=fpu['ANI_YB'][:]
xb=fpu['ANI_XB'][:]
fpsites=fpu['SITE'][:]

phi2=np.sqrt(fps['VV'][:])/fps['USTAR'][:]
zL2=(fps['tow_height'][:]-fps['zd'][:])/fps['L_MOST'][:]
ani2=fps['ANI_YB'][:]
xb2=fps['ANI_XB'][:]
fpsites2=fps['SITE'][:]

#######################

m=(xb>lim1)&(fpsites==bytes(siteX,'utf-8'))
m2=(xb2>lim1)&(fpsites2==bytes(siteX,'utf-8'))

xplt,yplt,aplt,cnt=binplot1d(zL[m],phi[m],ani[m],False)
tot=np.sum(cnt)
yplt[cnt/tot<minpct]=float('nan')

minpct=1e-4
xplts,yplts,aplts,cnts=binplot1d(zL2[m2],phi2[m2],ani2[m2],True)
tot=np.sum(cnts)
yplts[cnts/tot<minpct]=float('nan')
plotit(xplt,yplt,aplt,xplts,yplts,aplts,u_u_old,u_s_old)

##########################

m=(xb>lim2)&(fpsites==bytes(siteX,'utf-8'))&(xb<lim1)
m2=(xb2>lim2)&(fpsites2==bytes(siteX,'utf-8'))&(xb2<lim1)

xplt,yplt,aplt,cnt=binplot1d(zL[m],phi[m],ani[m],False)
tot=np.sum(cnt)
yplt[cnt/tot<minpct]=float('nan')

minpct=1e-4
xplts,yplts,aplts,cnts=binplot1d(zL2[m2],phi2[m2],ani2[m2],True)
tot=np.sum(cnts)
yplts[cnts/tot<minpct]=float('nan')
plotit(xplt,yplt,aplt,xplts,yplts,aplts,u_u_old,u_s_old)

############################
m=(xb>lim3)&(fpsites==bytes(siteX,'utf-8'))&(xb<lim2)
m2=(xb2>lim3)&(fpsites2==bytes(siteX,'utf-8'))&(xb2<lim2)

xplt,yplt,aplt,cnt=binplot1d(zL[m],phi[m],ani[m],False)
tot=np.sum(cnt)
yplt[cnt/tot<minpct]=float('nan')

minpct=1e-4
xplts,yplts,aplts,cnts=binplot1d(zL2[m2],phi2[m2],ani2[m2],True)
tot=np.sum(cnts)
yplts[cnts/tot<minpct]=float('nan')
plotit(xplt,yplt,aplt,xplts,yplts,aplts,u_u_old,u_s_old)

# %%
import matplotlib.pyplot as pl

# %%
plt.hist(mads)

# %%
np.max(mads)

# %%
mads=[]
for site in np.unique(fpsites):
    mads.append(d_s['U']['MAD_OLD_s'][str(site)[2:-1]])
mads=np.array(mads)
#mad_norm=(mads-.45)/(1.4-.45) UNSTABLE
mad_norm=(mads-.35)/(.6-.35)
cc=pl.cm.coolwarm(mad_norm)
sm = plt.cm.ScalarMappable(cmap=pl.cm.coolwarm, norm=plt.Normalize(vmin=.35, vmax=.6))
i=0
fig,ax=plt.subplots(1,1,figsize=(4,3),dpi=200)
for site in np.unique(fpsites):
    m=fpsites2==site
    y,binEdges=np.histogram(fps['ANI_XB'][m],bins=np.linspace(0,1),density=True)
    bincenters=.5*(binEdges[1:]+binEdges[:-1])
    ax.plot(bincenters,y,c=cc[i],linewidth=2,alpha=.75)
    i=i+1
    #plt.hist(fpu['ANI_XB'][m],bins=np.linspace(0,1))
    #ss=str(site)[2:-1]
    #plt.title(str(site)+': '+str(d_u['U']['MAD_OLD_s'][ss])[0:5])
plt.xlabel(r'$x_B$')
plt.ylabel('Frequency')
fig.colorbar(sm,cax=ax.inset_axes([0.95, 0.1, 0.05, 0.8]),label='$MAD$')
fig.savefig('trash.png',bbox_inches = "tight")

# %%
np.max(mads)

# %%
mads=[]
for site in np.unique(fpsites):
    mads.append(d_u['T']['MAD_OLD_s'][str(site)[2:-1]])
mads=np.array(mads)
#mad_norm=(mads-.45)/(1.4-.45) UNSTABLE
mad_norm=(mads-.35)/(.6-.35)
cc=pl.cm.coolwarm(mad_norm)
sm = plt.cm.ScalarMappable(cmap=pl.cm.coolwarm, norm=plt.Normalize(vmin=.2, vmax=1.3))
i=0
fig,ax=plt.subplots(1,1,figsize=(4,3),dpi=200)
for site in np.unique(fpsites):
    m=fpsites==site
    y,binEdges=np.histogram(fpu['ANI_XB'][m],bins=np.linspace(0,1),density=True)
    bincenters=.5*(binEdges[1:]+binEdges[:-1])
    ax.plot(bincenters,y,c=cc[i],linewidth=2,alpha=.75)
    i=i+1
    #plt.hist(fpu['ANI_XB'][m],bins=np.linspace(0,1))
    #ss=str(site)[2:-1]
    #plt.title(str(site)+': '+str(d_u['U']['MAD_OLD_s'][ss])[0:5])
plt.xlabel(r'$x_B$')
plt.ylabel('Frequency')
fig.colorbar(sm,cax=ax.inset_axes([0.95, 0.1, 0.05, 0.8]),label='$MAD$')
fig.savefig('trash.png',bbox_inches = "tight")

# %%
m=fpsites==b'BONA'
plt.hist(fpu['ANI_XB'][m],bins=np.linspace(0,1),density=True,alpha=.5)
print(np.mean(fpu['ANI_XB'][m]))
m=fpsites==b'ABBY'
plt.hist(fpu['ANI_XB'][m],bins=np.linspace(0,1),density=True,alpha=.5)


# %%
def plotit_t(xplt,yplt,aplt,xplts,yplts,aplts,old,olds,ylim=[.1,20]):
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
        plt.loglog(xplts,yplts[i,:],color=cani_norm(aplts[i]),linewidth=2,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])
    plt.loglog(zL,olds,'k--')
    
    ax=plt.gca()
    ax.tick_params(which="both", bottom=True)
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )) 
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([10**-3,10**-2,10**-1,1,10],[r'$10^{-3}$','',r'$10^{-1}$','',r'$10^{1}$'])
    ax.set_xlim(10**(-3.5),10**(1.6))
    plt.ylim(ylim[0],ylim[1])


# %%
siteX='ALL'
lim1=.65
lim2=.3
lim3=0
if siteX=='ALL':
    mmm=True
    mmm2=True
else:
    mmm=(fpsites==bytes(siteX,'utf-8'))
    mmm2=(fpsites2==bytes(siteX,'utf-8'))

##################

minpct=1e-4
phi=np.abs(fpu['T_SONIC_SIGMA'][:]/(fpu['WTHETA'][:]/fpu['USTAR'][:]))
zL=(fpu['tow_height'][:]-fpu['zd'][:])/fpu['L_MOST'][:]
ani=fpu['ANI_YB'][:]
xb=fpu['ANI_XB'][:]
fpsites=fpu['SITE'][:]

phi2=np.abs(fps['T_SONIC_SIGMA'][:]/(fps['WTHETA'][:]/fps['USTAR'][:]))
zL2=(fps['tow_height'][:]-fps['zd'][:])/fps['L_MOST'][:]
ani2=fps['ANI_YB'][:]
xb2=fps['ANI_XB'][:]
fpsites2=fps['SITE'][:]

#######################

m=(xb>lim1)&mmm
m2=(xb2>lim1)&mmm2

xplt,yplt,aplt,cnt=binplot1d(zL[m],phi[m],ani[m],False)
tot=np.sum(cnt)
yplt[cnt/tot<minpct]=float('nan')

minpct=1e-4
xplts,yplts,aplts,cnts=binplot1d(zL2[m2],phi2[m2],ani2[m2],True)
tot=np.sum(cnts)
yplts[cnts/tot<minpct]=float('nan')
plotit_t(xplt,yplt,aplt,xplts,yplts,aplts,t_u_old,t_s_old)

##########################

m=(xb>lim2)&mmm&(xb<lim1)
m2=(xb2>lim2)&mmm2&(xb2<lim1)

xplt,yplt,aplt,cnt=binplot1d(zL[m],phi[m],ani[m],False)
tot=np.sum(cnt)
yplt[cnt/tot<minpct]=float('nan')

minpct=1e-4
xplts,yplts,aplts,cnts=binplot1d(zL2[m2],phi2[m2],ani2[m2],True)
tot=np.sum(cnts)
yplts[cnts/tot<minpct]=float('nan')
plotit_t(xplt,yplt,aplt,xplts,yplts,aplts,t_u_old,t_s_old)

############################
m=(xb>lim3)&mmm&(xb<lim2)
m2=(xb2>lim3)&mmm2&(xb2<lim2)

xplt,yplt,aplt,cnt=binplot1d(zL[m],phi[m],ani[m],False)
tot=np.sum(cnt)
yplt[cnt/tot<minpct]=float('nan')

minpct=1e-4
xplts,yplts,aplts,cnts=binplot1d(zL2[m2],phi2[m2],ani2[m2],True)
tot=np.sum(cnts)
yplts[cnts/tot<minpct]=float('nan')
plotit_t(xplt,yplt,aplt,xplts,yplts,aplts,t_u_old,t_s_old)

# %% [markdown]
# # Trying EQ Ideas

# %%
anibins=np.linspace(vmn_a,vmx_a,11)
anilvl=(anibins[0:-1]+anibins[1:])/2
anic=cani_norm(anilvl)

zL_u=-np.logspace(-4,2,40)
zL=zL_u.copy()
zL=zL.reshape(1,40).repeat(10,0)

ani=anilvl.copy()
ani=ani.reshape(10,1).repeat(40,1)

# U Unstable
a=.784-2.582*np.log10(ani)
u_u_stp=a*(1-3*zL)**(1/3)
u_u_old=2.55*(1-3*zL)**(1/3)

# %%
plt.plot(ani[:,0],a[:,0])

# %%
b=.1
c=-.1
d=1/3
uu=a[0]*(b+c*zL[0])**d

# %%
plt.semilogx(-zL[0],u_u_old[0],'-k',linewidth=2)
plt.semilogx(-zL[0],u_u_stp[0])
plt.semilogx(-zL[0],u_u_stp[4])
plt.semilogx(-zL[0],u_u_stp[8])
#plt.semilogx(-zL[0],uu)
uu=a[4]*(b+c*zL[0])**d
plt.semilogx(-zL[0],uu)
plt.gca().invert_xaxis()


# %%
def getbins(A,n):
     B=np.sort(A)
     bins=[]
     for i in np.linspace(0,len(A)-1,n):
         i=int(i)
         bins.append(B[i])
     return bins


# %%
def getbins2D(A,B,n):
    bina=getbins(A,n)
    binb=np.zeros((n-1,n))
    for i in range(n-1):
        m=(A>bina[i])&(A<bina[i+1])
        binb[i,:]=getbins(B[m],n)
    return bina,binb


# %%
def U_ust(zL,a):
     return a*(1-3*zL)**(1/3)
fxns_={}
fxns_['Uu']=U_ust

# %%
from scipy import optimize

# %%
p0s={'Uu':[2.5],'Vu':[2],'Wu':[1.3],'Tu':[.1],'H2Ou':[-1/3],'CO2u':[-.25],\
 #p0s={'Uu':[2.5],'Vu':[2],'Wu':[1.3],'Tu':[.1],'H2Ou':[4,-25,-1/3],'CO2u':[4,-80,-.25],\
      'Us':[2,.08],'Vs':[2,.08],'Ws':[1.4,.03],'Ts':[.35,0.025,-.03,-.025],\
      'H2Os':[.5,-.1,.05,.1],'CO2s':[.5,-.1,.05,.1]}
 
bounds={'Uu':([0],[10]),\
         'Vu':([0],[10]),\
         'Wu':([0],[10]),\
         'Tu':([0],[10]),\
         'H2Ou':([-1],[-.1]),\
         'CO2u':([-1],[-.1]),\
         'Us':([0,0],[5,1]),\
         'Vs':([0,0],[5,1]),\
         'Ws':([0,0],[5,1]),\
         'Ts':([.05,-.1,-.1,-.1],[.6,.1,.1,0]),\
         'H2Os':([0,-.1,-.1,-.1],[1,0,.1,.1]),\
         'CO2s':([0,-.1,-.1,-.1],[1,0,.1,.1])}

# %%
ybine,xbine=getbins2D(fpu['ANI_YB'][:],fpu['ANI_XB'][:],21)
phi_=np.sqrt(fpu['UU'][:])/fpu['USTAR'][:]
zL_=(fpu['tow_height'][:]-fpu['zd'][:])/fpu['L_MOST'][:]
xb_=fpu['ANI_XB'][:]
yb_=fpu['ANI_YB'][:]

Np=1
params=np.ones((len(ybine)-1,len(xbine[0])-1,Np))*float('nan')
p_vars=np.ones((len(ybine)-1,len(xbine[0])-1,Np))*float('nan')
var='Uu'

for i in range(len(ybine)-1):
    for j in range(len(xbine[0])-1):
        print(str(i)+','+str(j)+'   ',end='',flush=True)
        m=(xb_<xbine[i,j+1])&(xb_>xbine[i,j])&(yb_<ybine[i+1])&(yb_>ybine[i])
        try:
            if var in bounds.keys():
                params[i,j,:],pcov=optimize.curve_fit(fxns_[var],zL_[m],phi_[m],p0s[var],bounds=bounds[var],loss='cauchy')
            else:
                params[i,j,:],pcov=optimize.curve_fit(fxns_[var],zL_[m],phi_[m],p0s[var])
            for p in range(Np):
                p_vars[i,j,p]=pcov[p,p]
        except Exception as e:
            print(e)


# %%
ybin=(np.array(ybine[0:-1])+np.array(ybine[1:]))/2
xbin=(np.array(xbine[:,0:-1])+np.array(xbine[:,1:]))/2

# %%
for i in range(1,20):
    plt.plot(xbin[i,:],params[i,:],'o-')
plt.xlabel(r'$x_b$')
plt.ylabel('a')

# %%

# %% [markdown]
# # Error by XB

# %%
d_u=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_unst_xbbins.p','rb'))
d_s=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_stbl_xbbins.p','rb'))

# %%
xbins=(np.array(d_s['U']['xbins'][1:])+np.array(d_s['U']['xbins'][:-1]))/2
plt.scatter(xbins,d_s['U']['MAD_OLD'][:])

# %%
d_=d_s
v='H2O'
typ='MAD_SC23_s'
for site in d_u['U'][typ].keys():
    xbins=(np.array(d_[v]['xbins_s'][site][1:])+np.array(d_[v]['xbins_s'][site][:-1]))/2
    plt.plot(xbins,d_[v][typ][site][:],marker='o',markerfacecolor='none',linewidth=.5)
plt.xlabel(r'$x_b$')
plt.ylabel(r'$MAD$')

# %%
xbins

# %%
d_
