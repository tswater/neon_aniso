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
from neonutil.nutil import nscale,SITES
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import h5py
import sys
mpl.rcParams['figure.dpi'] = 100
sns.set_theme()
plt.rcParams.update({'figure.max_open_warning': 0})

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Test 15 Minute Files

# %%
l1dir='/home/tswater/Documents/tyche/data/neon/L1/'
#l1dir='/run/media/tylerwaterman/Elements/NEON/L1/'

# %% [markdown]
# ### Basic

# %%
fa15=h5py.File(l1dir+'neon_15m/ABBY_15m.h5','r')
fg15=h5py.File(l1dir+'neon_15m/GUAN_15m.h5','r')

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
fa30=h5py.File(l1dir+'neon_30m/ABBY_30m.h5','r')
fg30=h5py.File(l1dir+'neon_30m/GUAN_30m.h5','r')
abby=[fa15,fa30]
guan=[fg15,fg30]


# %%
# check nan frequency similarity
def getnan(data):
    return np.sum(np.isnan(data)|(data==-9999))/len(data)*100

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
        else:
            var3=var
        if var3 not in f30.keys():
            continue
        fq5=getnan(f15[var][:])
        fq3=getnan(f30[var3][:])
        if np.abs(fq5-fq3)>5:
            print(var+':   '+str(fq5)[0:5]+' vs '+str(fq3)[0:5])
    print()
    print()


# %%

# %%
# check quality flags
convert={'THETA':'qT_SONIC','Q':'qH2O','C':'qCO2'}
i=0

def getbad(data):
    return np.sum((data!=0))/len(data)*100

for pair in [abby,guan]:
    print(['ABBY','GUAN'][i])
    i=i+1
    f15=pair[0]
    f30=pair[1]
    for var in f15.keys():
        if 'q' not in var:
            continue
        if 'profile' in var:
            continue
        if var in convert.keys():
            var3=convert[var]
        else:
            var3=var
        if var3 not in f30.keys():
            continue
        fq5=getgood(f15[var][:])
        fq3=getgood(f30[var3][:])
        if np.abs(fq5-fq3)>5:
            print(var+':   '+str(fq5)[0:5]+' vs '+str(fq3)[0:5])
    print()
    print()

# %% jupyter={"outputs_hidden": true, "source_hidden": true}
# compute differences and plot histogram
convert={'THETA':'T_SONIC','Q':'H2O','C':'CO2'}
i=0
for pair in [abby,guan]:
    print(['ABBY','GUAN'][i])
    i=i+1
    f15=pair[0]
    f30=pair[1]
    j=0
    for var in f15.keys():
        if 'q' in var:
            continue
        if 'profile' in var:
            continue
        if var in convert.keys():
            var3=convert[var]
        else:
            var3=var
        if var3 not in f30.keys():
            continue
        j=j+1
        #if (j<10):
        #    continue
        d15=(f15[var][:][0:-1:2]+f15[var][:][1::2])/2
        d30=f30[var3][:-1]
        m=(d15==-9999)|(d30==-9999)|np.isnan(d15)|np.isnan(d30)
        diff=d15-d30
        dout=diff
        dmx=np.nanpercentile(np.abs(diff[~m]),98)
        plt.figure()
        plt.title(var)
        plt.hist(dout[~m],bins=np.linspace(-dmx,dmx,151))


# %%
# Plot against eachother
idx0=82988
idxf=83188
s=5
i=0
for pair in [abby,guan]:
    print(['ABBY','GUAN'][i])
    i=i+1
    f15=pair[0]
    f30=pair[1]
    j=0
    for var in f15.keys():
        if 'q' in var:
            continue
        if 'profile' in var:
            continue
        if var in convert.keys():
            var3=convert[var]
        else:
            var3=var
        if var3 not in f30.keys():
            continue
        elif var in ['GCC90_D']:
            continue
        d15=f15[var][:][idx0*2:idxf*2]
        d30=f30[var3][:][idx0:idxf]
        d15[d15==-9999]=float('nan')
        d30[d30==-9999]=float('nan')

        miny=np.nanpercentile([d15[::2],d30],2)
        maxy=np.nanpercentile([d15[::2],d30],98)
        
        t15=f15['TIME'][:][idx0*2:idxf*2]
        t30=f30['TIME'][:][idx0:idxf]
        plt.figure(figsize=(12,3),dpi=250)
        plt.title(var)
        plt.plot(t30,d30,'-o',linewidth=.2*s)
        plt.plot(t15,d15,'-o',linewidth=.1*s)
        plt.ylim(miny,maxy)

# %%
1.49503034e-05 1.02440652e+00

# %%
b=b0*.994142481+2.67966518e-06

# %%
b=b0*1.02440652e+00+1.49503034e-05

# %%
a=fa30['CO2'][-48*31:]
b0=(fa15['C'][-48*31*2:]+fa15['C'][-48*31*2-1:-1])/2
b=b0*.994142481+2.67966518e-06
b=b0*1.02440652e+00+1.49503034e-05
c=(fa15['Q'][-48*31*2:]+fa15['Q'][-48*31*2-1:-1])/2
a[a==-9999]=float('nan')
b[b0==-9999]=float('nan')
print(np.nanmedian(a))
print(np.nanmedian(b0*10**6))
print(np.nanmedian(b*10**6))
print(np.nanmedian((b*1/(1+c))*10**6))

# %%
d=b*10**6#*1/(1+c)*10**6
plt.scatter(a[:48],d[:48*2:2])

# %%

# %%
plt.hist(a-d[::2],bins=np.linspace(-20,20))

# %%
idx0=72988
idxf=73188
s=5
a=fa30['CO2FX'][:]
b=fa30['WC'][:]*dmolairdry*10**6
a[a==-9999]=float('nan')
b[b==-9999]=float('nan')
#t15=fa15['TIME'][:][idx0*2:idxf*2]
t30=fa30['TIME'][:][idx0:idxf]
plt.figure(figsize=(12,3),dpi=250)
plt.title(var)
plt.plot(t30,a[idx0:idxf],'-o',linewidth=.2*s)
plt.plot(t30,b[idx0:idxf],'-o',linewidth=.1*s)
#plt.plot(t15,b[idx0*2:idxf*2],'-o',linewidth=.1*s)
plt.ylim(-10,5)

# %%
np.nanmedian(fa15['DMOLAIRDRY'][::2][dmolairdry>0])

# %%
idx0=62988
idxf=73188
s=5
a=fa30['CO2FX'][:]
b=fa30['WC'][:]*fa15['DMOLAIRDRY'][::2]*10**6
a[a==-9999]=float('nan')
b[b==-9999]=float('nan')
#t15=fa15['TIME'][:][idx0*2:idxf*2]
t30=fa30['TIME'][:][idx0:idxf]
plt.figure(figsize=(12,3),dpi=250)
plt.title(var)
#plt.plot(t30,a[idx0:idxf],'-o',linewidth=.2*s)
#plt.plot(t30,b[idx0:idxf],'-o',linewidth=.1*s)
plt.plot(t30,(a[idx0:idxf]-b[idx0:idxf])/a[idx0:idxf])
#plt.plot(t15,b[idx0*2:idxf*2],'-o',linewidth=.1*s)
plt.ylim(-1,1)

# %%
fm=h5py.File('/run/media/tylerwaterman/Elements/NEON/neon_processed/L2_scalar/L2mask_U_C.h5','r')
m=fm['ABBY'][:]



# %%
a=fa30['CO2FX'][:]
b=fa30['WC'][:]*fa15['DMOLAIRDRY'][::2]*10**6
a[a==-9999]=float('nan')
b[b==-9999]=float('nan')
a=a[m]
b=b[m]

dmolair=fa15['DMOLAIRDRY'][::2]+fa15['Q'][::2]
dmolairdry=dmolair/fa15['PA'][::2]*fa30['PA'][:]*1000/fa15['T'][::2]*fa15['THETA'][::2]-fa15['Q'][::2]
c=fa30['WC'][:]*dmolairdry*10**6
c=c[m]

# %%
plt.hist((a-c)/a,bins=np.linspace(-.1,.2))

# %%
# Test RH fixes

# %%
fpo=fa15
e=fpo['PA'][:]*fpo['Q'][:]/(1+fpo['Q'][:])
e[fpo['PA'][:]==-9999]=-9999
e[fpo['Q'][:]<=0]=-9999
ts=fpo['THETA'][:]-273.14
svp=.61121*np.exp((18.678-(ts)/234.5)*((ts)/(257.14+(ts))))*1000
svp[ts==-9999]=-9999
rh=e/svp*100
rh[e==-9999]=-9999
rh[svp==-9999]=-9999

# %%
plt.hist(rh[rh>0],bins=np.linspace(0,100))

# %%
plt.hist(fa30['RH'][:],bins=np.linspace(0,100))

# %%
np.nanmedian(fpo['T'][:])

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Testing CO2Flux Validation STuff

# %%
idir='/home/tylerwaterman/Downloads/NEON_eddy-flux(2)/NEON_eddy-flux/NEON.D16.ABBY.DP4.00200.001.2023-12.expanded.20250129T000730Z.RELEASE-2025/'

# %%
fp=h5py.File(idir+'NEON.D16.ABBY.DP4.00200.001.nsae.2023-12-30.expanded.20250122T235215Z.h5','r')

# %%
fp['ABBY/dp01/data/co2Turb/000_050_01m/rtioMoleDryCo2Vali'].attrs['evalCoef']

# %%
d1=fp['ABBY/dp01/data/co2Turb/000_050_01m/rtioMoleDryCo2']['mean'][:]
d2=fp['ABBY/dp01/data/co2Turb/000_050_01m/rtioMoleDryCo2Cor']['mean'][:]
d3=fp['ABBY/dp01/data/co2Turb/000_050_01m/rtioMoleDryCo2Raw']['mean'][:]

# %%
d2=fp['ABBY/dp04/data/fluxCo2/turb']['fluxCor'][:]
d3=fp['ABBY/dp04/data/fluxCo2/turb']['fluxRaw'][:]

# %%
plt.figure(figsize=(12,3))
#plt.plot(d1)
plt.plot(d2)
plt.plot(d3)
#plt.ylim(0,100)
#plt.ylim(400,500)

# %%

# %%

# %% [markdown]
# # Testing Spatial Additions

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Testing Adding Function

# %%
if 'neonutil.l1tools' in sys.modules:
    del sys.modules['neonutil.l1tools']
from neonutil.l1tools import add_spatial

# %%
bdir='/run/media/tswater/Elements/NEON/downloads/'
ndir='/home/tswater/Documents/tyche/data/neon/L1/neon_30m/'
add_spatial(ndir,bdir+'dsm/',bdir+'dtm/',bdir+'lai/',fdir=bdir+'dp4ex/',\
        nlcd_ak='/home/tswater/Downloads/NLCD_2016_Land_Cover_AK_20200724.img',\
        nlcd_hi='/home/tswater/Downloads/hi_landcover_wimperv_9-30-08_se5.img',\
        nlcd_pr='/home/tswater/Downloads/pr_landcover_wimperv_10-28-08_se5.img',\
        nlcd_us='/home/tswater/Downloads/nlcd_2021_land_cover_l48_20230630/nlcd_2021_land_cover_l48_20230630.img',\
        debug=True,sites=['ABBY'])




# %%
test.close()

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Debugging issues with DSM/DTM/LAI footprints for STEI, LAJA, KONZ and KONA

# %%
import rasterio
from pyproj import Transformer
from rasterio.windows import Window
from scipy import stats
dsmdir='/run/media/tswater/Elements/NEON/downloads/dsm/'
l1dir ='/run/media/tswater/Elements/NEON/L1/neon_30m/'

# %%
site='STEI'
fpr=rasterio.open(dsmdir+site+'/dsm_'+site+'.tif')
fps=h5py.File(l1dir+site+'_30m.h5','r')

# %%
lat=fps.attrs['lat']
lon=fps.attrs['lon']

# %%
transformer=Transformer.from_crs('EPSG:4326',fpr.crs,always_xy=True)
xx_,yy_=transformer.transform(lon,lat)
xx,yy=fpr.index(xx_,yy_)

# %%
wn=Window(yy-2500,xx-2500,5000,5000)

# %%
dsm=fpr.read(1,boundless=True,fill_value=float('nan'),window=wn)
#dsm=fpr.read(1)

# %%
plt.imshow(dsm,cmap='terrain',vmin=150)
#plt.scatter(xx,yy)

# %%
for site in SITES:
    fpr=rasterio.open(dsmdir+site+'/dsm_'+site+'.tif')
    fps=h5py.File(l1dir+site+'_30m.h5','r')
    lat=fps.attrs['lat']
    lon=fps.attrs['lon']
    transformer=Transformer.from_crs('EPSG:4326',fpr.crs,always_xy=True)
    xx_,yy_=transformer.transform(lon,lat)
    xx,yy=fpr.index(xx_,yy_)
    a=np.sqrt((xx-yy)**2+(yy-xx)**2)
    print(site+': '+str(a))

# %%
lon

# %%
SITES[0:14]

# %%
vlist=['zL','USTAR','U','W','f_chm','f_std_dtm','f_std_chm','f_std_dsm','NETRAD','H','RH','THETA']
vnames=[r'$\zeta$',r'$u*$',r'$\overline{U}$',r'$\overline{w}$',\
        r'$h_c$',r'$\sigma_{hc}$',r'$\sigma_{DTM}$',r'$\sigma_{DSM}$',r'$R_{net}$',r'$H$',r'$RH$',r'$T$']
corr=np.zeros((2,2,12))
corrs=np.zeros((2,2,12,47))
Ns=350000
for e in range(2):
    ani=['ANI_YB','ANI_XB'][e]
    for s in range(2):
        f=[fu,fs][s]
        ms=np.zeros((len(f['TIME'][:]),)).astype(bool)
        ms[0:Ns]=True
        np.random.shuffle(ms)
        fpsites=f['SITE'][:][ms]
        an=f[ani][:][ms]
        for v in range(12):
            vs=vlist[v]
            data=f[vs][:][ms]
            corr[e,s,v]=stats.spearmanr(an,data,nan_policy='omit')[0]
            for i in range(47):
                #print('.',end='',flush=True)
                site=SITES[i]
                m=fpsites==bytes(site,'utf-8')
                data2=data[m]
                try:
                    corrs[e,s,v,i]=stats.spearmanr(an[m],data2,nan_policy='omit')[0]
                except Exception as e2:
                    corrs[e,s,v,i]=float('nan')
            print('*',end='',flush=True)

# %%

# %% [markdown]
# ### Test Script For Adding LAI spatiotemporal

# %%
if 'neonutil.l1tools' in sys.modules:
    del sys.modules['neonutil.l1tools']
from neonutil.l1tools import add_spatiotemporal

# %%
bdir='/run/media/tswater/Elements/NEON/downloads/'
ndir='/home/tswater/Documents/tyche/data/neon/L1/neon_30m/'
add_spatiotemporal(ndir,bdir+'lai/',bdir+'dp4ex/',scl=30,sites=['ABBY'],debug=True)

# %%
test=h5py.File(ndir+'ABBY_30m.h5','r')
test.keys()

# %%
test['LAI_fmedian'].attrs['dates']

# %%
