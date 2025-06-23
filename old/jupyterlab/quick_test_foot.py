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
fpdsm=rasterio.open('/home/tswater/Documents/tyche/data/neon/dsm/STEI/dsm_STEI.tif')
stei=fpdsm.read(1)
transformer=Transformer.from_crs('EPSG:4326',fpdsm.crs,always_xy=True)
lat=45.50895
lon=-89.58636
xx_,yy_=transformer.transform(lon,lat)
xx,yy=fpdsm.index(xx_,yy_)

# %%
plt.imshow(stei,cmap='terrain',vmin=0)
plt.scatter(xx,yy)
plt.colorbar()

# %%
ffu=h5py.File('/home/tswater/tyche/data/neon/foot_stats/ABBY_U.h5','r')
ffs=h5py.File('/home/tswater/tyche/data/neon/foot_stats/ABBY_S.h5','r')

# %%
idir='/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data/'
fpu=h5py.File(idir+'/NEON_TW_U_UVWT.h5','r')
fps=h5py.File(idir+'/NEON_TW_S_UVWT.h5','r')
d_u=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_unst_v3.p','rb'))
d_s=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_stbl_v3.p','rb'))
d_utw=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_unst_tw_v3.p','rb'))
d_stw=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_stbl_tw_v3.p','rb'))
fpst=h5py.File('/home/tswater/tyche/data/neon/static_data.h5','r')

# %%
ms=fps['SITE'][:]==b'ABBY'
mu=fpu['SITE'][:]==b'ABBY'

# %%
fpst.keys()

# %%
ffu.keys()

# %%
plt.hist(ffu['mean_chm'][:]-fpst['mean_chm'][0])

# %%
plt.hist(ffs['mean_chm'][:]-fpst['mean_chm'][0])

# %%
phi=np.sqrt(fpu['UU'][:][mu])/fpu['USTAR'][:][mu]
zL=(fpu['tow_height'][:][mu]-fpu['zd'][:][mu])/fpu['L_MOST'][:][mu]
ani=fpu['ANI_YB'][:][mu]
fpsite=fpu['SITE'][:][mu]
a=.784-2.582*np.log10(ani)
phi_stp=a*(1-3*zL)**(1/3)
phi_old=2.55*(1-3*zL)**(1/3)
anix=fpu['ANI_XB'][:][mu]

# %%
ffu.keys()

# %%
ad_most=np.abs(phi-phi_old)
ad_sc23=np.abs(phi-phi_stp)
plt.hexbin(ffu['range_chm'][:],ad_most,mincnt=1,cmap='terrain',gridsize=100,extent=(8,55,0,5),vmin=0,vmax=50)


# %%
def getbins(A,n):
    A2=A[~np.isnan(A)]
    B=np.sort(A2)
    bins=[]
    for i in np.linspace(0,len(A2)-1,n):
        i=int(i)
        bins.append(B[i])
    return np.array(bins)


# %%
def binmean(a,b,bins=50):
    bine=getbins(a,bins+1)
    binc=(bine[1:]+bine[0:-1])/2
    mean=[]
    q1=[]
    q3=[]
    for i in range(bins):
        q1.append(np.nanpercentile(b[(a<bine[i+1])&(a>bine[i])],25))
        q3.append(np.nanpercentile(b[(a<bine[i+1])&(a>bine[i])],75))
        mean.append(np.nanmean(b[(a<bine[i+1])&(a>bine[i])]))
    return binc,mean,q1,q3
    


# %%
ffu=h5py.File('/home/tswater/tyche/data/neon/foot_stats/STEI_U.h5','r')
mu=fpu['SITE'][:]==b'STEI'
phi=np.sqrt(fpu['UU'][:][mu])/fpu['USTAR'][:][mu]
zL=(fpu['tow_height'][:][mu]-fpu['zd'][:][mu])/fpu['L_MOST'][:][mu]
ani=fpu['ANI_YB'][:][mu]
fpsite=fpu['SITE'][:][mu]
a=.784-2.582*np.log10(ani)
phi_stp=a*(1-3*zL)**(1/3)
phi_old=2.55*(1-3*zL)**(1/3)
anix=fpu['ANI_XB'][:][mu]
ad_most=np.abs(phi-phi_old)
ad_sc23=np.abs(phi-phi_stp)

# %%
np.sum(~np.isnan(ffu['std_dtm'][:]))

# %%
for k in ffu.keys():
    print(k)
    if 'nlcd' in k:
        continue
    if 'std_slope' in k:
        continue
    if 'aspect' in k:
        continue
    x,mean,q1,q3=binmean(ffu[k][:],ad_most,20)
    plt.figure()
    plt.plot(x,mean)
    plt.plot(x,q1,'--')
    plt.plot(x,q3,'--')
    plt.title(k)

# %%
ffu=h5py.File('/home/tswater/tyche/data/neon/foot_stats/STEI_U.h5','r')
print(ffu.keys())

# %%
sites=[]
for f in os.listdir('/home/tswater/tyche/data/neon/foot_stats/'):
    ss=f[0:4]
    if ss=='test':
        continue
    elif ss in sites:
        continue
    else:
        sites.append(ss)

# %%
from scipy import stats

# %%
varlist=['aspect_dtm', 'mean_chm', 'nlcd_2', 'nlcd_3', 'nlcd_dom', 'nlcd_iqv', 'range_chm', 'range_dtm', 'slope_dtm', 'std_aspect', 'std_chm', 'std_dsm', 'std_dtm', 'std_slope', 'treecover_lidar']
scl_vars=['U_stbl','U_unst','V_stbl','V_unst','W_stbl','W_unst']
mad=np.zeros((len(scl_vars),len(sites)))
xdata=np.zeros((2,len(varlist),len(sites)))
cors=np.zeros((6,len(varlist)))


for i in range(len(sites)):
    site=sites[i]
    ffu=h5py.File('/home/tswater/tyche/data/neon/foot_stats/'+site+'_U.h5','r')
    ffs=h5py.File('/home/tswater/tyche/data/neon/foot_stats/'+site+'_U.h5','r')
    j=0
    for svar in ['U','V','W']:
        mad[j+1,i]=d_u[svar[0:1]]['MAD_OLD_s'][site]
        mad[j,i]=d_s[svar[0:1]]['MAD_OLD_s'][site]
        j=j+2
    j=0
    for var in varlist:
        if 'nlcd' in var:
            xdata[0,j,i]=stats.mode(ffu[var][:])[0]
            xdata[1,j,i]=stats.mode(ffu[var][:])[0]
        else:
            xdata[0,j,i]=np.nanmean(ffs[var][:])
            xdata[1,j,i]=np.nanmean(ffs[var][:])
        j=j+1

# %%
varlist=['aspect_dtm', 'mean_chm', 'nlcd_2', 'nlcd_3', 'nlcd_dom', 'nlcd_iqv', 'range_chm', 'range_dtm', 'slope_dtm', 'std_aspect', 'std_chm', 'std_dsm', 'std_dtm', 'std_slope', 'treecover_lidar']
scl_vars=['U_stbl','U_unst','V_stbl','V_unst','W_stbl','W_unst']
mad_sc=np.zeros((len(scl_vars),len(sites)))
#xdata=np.zeros((2,len(varlist),len(sites)))
cors_sc=np.zeros((6,len(varlist)))


for i in range(len(sites)):
    site=sites[i]
    ffu=h5py.File('/home/tswater/tyche/data/neon/foot_stats/'+site+'_U.h5','r')
    ffs=h5py.File('/home/tswater/tyche/data/neon/foot_stats/'+site+'_U.h5','r')
    j=0
    for svar in ['U','V','W']:
        mad_sc[j+1,i]=d_u[svar[0:1]]['MAD_SC23_s'][site]
        mad_sc[j,i]=d_s[svar[0:1]]['MAD_SC23_s'][site]
        j=j+2


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
plt.scatter(xdata[1,10,:]/xdata[1,1,:],mad_sc[0])
plt.xlim(0,10)

# %%
for i in range(len(scl_vars)):
    for j in range(len(varlist)):
        #plt.figure()
        if i in [0,2,4]:
            k=1
        xx,yy=fix(mad_sc[i,:],xdata[k,j,:])
        try:
            cors_sc[i,j]=stats.spearmanr(xx,yy)[0]
        except:
            pass
        if np.abs(cors_sc[i,j])<.5:
            continue
        plt.figure()
        plt.scatter(xdata[k,j,:],mad_sc[i,:])
        tstr=scl_vars[i]+': '+str(varlist[j])+' vs MAD '+str(cors_sc[i,j])[0:5]
        plt.title(tstr)

# %% [markdown]
# # Point by Point

# %%
varlist=['aspect_dtm', 'mean_chm', 'nlcd_dom', 'range_chm', 'range_dtm', 'slope_dtm', 'std_chm', 'std_dsm', 'std_dtm', 'treecover_lidar']
phi=np.sqrt(fpu['UU'][:])/fpu['USTAR'][:]
zL=(fpu['tow_height'][:]-fpu['zd'][:])/fpu['L_MOST'][:]
ani=fpu['ANI_YB'][:]
fpsite=fpu['SITE'][:]
a=.784-2.582*np.log10(ani)
phi_stp=a*(1-3*zL)**(1/3)
phi_old=2.55*(1-3*zL)**(1/3)
anix=fpu['ANI_XB'][:]
ad_most=np.abs(phi-phi_old)
ad_sc23=np.abs(phi-phi_stp)

# %%
phi=np.sqrt(fps['UU'][:])/fps['USTAR'][:]
zL=(fps['tow_height'][:]-fps['zd'][:])/fps['L_MOST'][:]
ani=fps['ANI_YB'][:]
fpsite=fps['SITE'][:]
a=.784-2.582*np.log10(ani)
a_=np.array([2.332,-2.047,2.672])
c_=np.array([.255,-1.76,5.6,-6.8,2.65])
a=0
c=0
for i in range(3):
    a=a+a_[i]*ani**i
for i in range(5):
    c=c+c_[i]*ani**i
phi_stp=a*(1+3*zL)**(c)
phi_old=2.06
anix_s=fps['ANI_XB'][:]
ad_most_s=np.abs(phi-phi_old)
ad_sc23_s=np.abs(phi-phi_stp)

# %%
sitelist=list(np.unique(fpsite))
sitelist.sort()

# %%
lenlist=[]

# %%
fdata={}
fdata['U_stb']={}
fdata['U_ust']={}
for var in varlist:
    fdata['U_stb'][var]=[]
    fdata['U_ust'][var]=[]
    for site in sitelist:
        ss=str(site)[2:-1]
        ffu=h5py.File('/home/tswater/tyche/data/neon/foot_stats/'+ss+'_U.h5','r')
        ffs=h5py.File('/home/tswater/tyche/data/neon/foot_stats/'+ss+'_S.h5','r')
        fdata['U_stb'][var].extend(ffs[var][:])
        fdata['U_ust'][var].extend(ffu[var][:])
        if var=='std_chm':
            lenlist.append(len(ffs[var][:]))


# %%
len(ad_most_s)

# %%
ffs.keys()

# %%

# %%
corr_s_old=[]
corr_s_sc23=[]
corr_u_old=[]
corr_u_sc23=[]
for var in varlist:
    #xx,yy=fix(ad_most_s,fdata['U_stb'][var][:])
    #corr_s_old.append((stats.pearsonr(xx,yy)[0]))
    #xx,yy=fix(ad_sc23_s,fdata['U_stb'][var][:])
    #corr_s_sc23.append((stats.pearsonr(xx,yy)[0]))
    xx,yy=fix(ad_most,fdata['U_ust'][var][:])
    corr_u_old.append((stats.spearmanr(xx,yy)[0]))
    xx,yy=fix(ad_sc23,fdata['U_ust'][var][:])
    corr_u_sc23.append((stats.spearmanr(xx,yy)[0]))

# %%
plt.bar(varlist,corr_u_sc23)
plt.xticks(rotation=45)

# %%
ffu.keys()

# %%
fpu.keys()

# %%
stats.spearmanr(ad_sc23,anix)

# %%
m=fpsite!=b'TALL'
len(phi_old[m])

# %%
len(fdata['U_stb'][var])

# %%
stats.mode(ffu['nlcd_dom'][:])[0]

# %% [markdown]
# # Test NLCD Dom differences

# %%
print('XXX:SITE: OLD : NEW : VAR')
fpsite=fpu['SITE'][:]
for site in np.unique(fpsite):
    m=fpsite==site
    nlcd_old=fpu['nlcd_dom'][m][0]
    ss=str(site)[2:-1]
    ffu=h5py.File('/home/tswater/tyche/data/neon/foot_stats/'+ss+'_U.h5','r')
    nlcd_new=stats.mode(ffu['nlcd_dom'])[0]
    nlcd_var=stats.mode(ffu['nlcd_2'])[0]
    
    if nlcd_new!=nlcd_old:
        print('XXX'+ss+': '+str(nlcd_old)[0:2]+'    '+str(nlcd_new)[0:2]+'    '+str(nlcd_var)[0:5])
    else:
        print('   '+ss+':             '+str(nlcd_var)[0:5])

# %%
ffu.keys()

# %%
# check: OSBS (forest to Shrub), NIWO (Shrub to Grass), SOAP( Forest to Grassland), YELL (Forest to Shrub), UNDE (Forest to Woody Wetlands)
# color sparse forests?

# %%
sn,we=np.indices([301,301])
plt.imshow(we-150)
plt.colorbar()

# %%
we=we-150

# %%
mask=np.zeros((301,301))
insert=np.ones((15,19))
mask[130:145,146:165]=insert[:]

# %%

# %%
print(np.mean(sn[mask.astype(bool)]))
print(np.mean(we[mask.astype(bool)]))

# %%
plt.imshow(mask)
plt.scatter([150],[150])
plt.scatter([150+5],[150-13])
plt.xlim(120,180)
plt.ylim(180,120)

# %%
nlcd_class={41:'Decid',
            42:'Everg',
            43:'Mixed',
            51:'Scrub',
            52:'Shrub',
            71:'Grass',
            72:'Sedge',
            73:'Lichn',
            74:'Moss_',
            81:'Hay  ',
            82:'Crops',
            90:'WetWo',
            95:'WetGr'}

# %%
fpsite=fpu['SITE'][:]
for site in np.unique(fpsite):
    m=fpsite==site
    nlcd_old=fpu['nlcd_dom'][m][0]
    ss=str(site)[2:-1]
    try:
        print(ss+': '+nlcd_class[nlcd_old])
    except:
        print(ss+': ')

# %%
d_u['U'].keys()

# %%
std_chm=[]
mean_chm=[]
range_chm=[]
treecover=[]
std_dsm=[]
for site in np.unique(fpsite):
    m=fpsite==site
    nlcd_old=fpu['nlcd_dom'][m][0]
    ss=str(site)[2:-1]
    ffu=h5py.File('/home/tswater/tyche/data/neon/foot_stats/'+ss+'_U.h5','r')
    std_chm.append(np.nanmedian(ffu['std_chm'][:]))
    mean_chm.append(np.nanmedian(ffu['mean_chm'][:]))
    range_chm.append(np.nanmedian(ffu['range_chm'][:]))
    treecover.append(np.nanmedian(ffu['treecover_lidar'][:]))
    std_dsm.append(np.nanmedian(ffu['std_dsm'][:]))

# %%
std_chm=np.array(std_chm)
range_chm=np.array(range_chm)
mean_chm=np.array(mean_chm)
for i in range(len(np.unique(fpsite))):
    site=str(np.unique(fpsite)[i])[2:-1]
    if (treecover[i]>.1)&(treecover[i]<.8)&(mean_chm[i]>3):
        print(site+': '+str(treecover[i])+'  '+str(mean_chm[i]))

# %%
for i in range(len(np.unique(fpsite))):
    site=str(np.unique(fpsite)[i])[2:-1]
    m=fpsite==bytes(site,'utf-8')
    tow_height=fpu['tow_height'][m][0]
    if (std_dsm[i]>tow_height*.5)&(mean_chm[i]>.99):
        print(site+': '+str(std_dsm[i])+'  '+str(mean_chm[i]))


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


# %%
# sorter: MAD_most
# sortee: MAD_SC23, land cover, IsSparse, site, 
# sortee: site, issparse, landcover, madSC23

# %%
from scipy import stats

# %%
var='U'
mad_most=[]
other=[[],[],[],[]]
for i in range(47):
    site=np.unique(fpsite)[i]
    other[0].append(str(site)[2:-1])
    ss=str(site)[2:-1]
    ffu=h5py.File('/home/tswater/tyche/data/neon/foot_stats/'+ss+'_U.h5','r')
    std=np.nanmedian(ffu['std_dsm'][:])
    mean=np.nanmedian(ffu['mean_chm'][:])
    if (std>1.5*mean)&(mean>1):
        other[1].append(1)
    else:
        other[1].append(0)
    other[2].append(stats.mode(ffu['nlcd_dom'])[0])
    other[3].append(d_u[var]['MAD_SC23_s'][ss])
    mad_most.append(d_u[var]['MAD_OLD_s'][ss])

# %%
d_stw['U']['U']['U']['U']['U']

# %%

# %%

# %%
class_names={11:'Water',12:'Snow',21:'Open Dev',22:'Low Dev',23:'Med Dev',24:'High Dev',31:'Barren',\
             41:'Decid.',42:'Everg.',43:'Mix Forest',51:'Dwarf Scrub',52:'Shrub',\
             71:'Grass',72:'AK:Sedge',73:'AK:Lichen',74:'AK:Moss',81:'Pasture',82:'Crops',90:'Wetland',95:'Herb Wet',0:'NaN'}
class_colors={11:'royalblue',12:'whitesmoke',21:'pink',22:'lightcoral',23:'red',24:'maroon',31:'silver',\
              41:'limegreen',42:'darkgreen',43:'yellowgreen',51:'darkgoldenrod',52:'tan',\
              71:'wheat',72:'khaki',73:'darkkhaki',74:'darkseagreen',81:'gold',82:'sandybrown',90:'skyblue',95:'cadetblue',0:'white'}

class_names={41:'Decid.',42:'Everg.',43:'Mix Forest',51:'Dwarf Scrub',52:'Shrub',\
             71:'Grass',72:'AK:Sedge',73:'AK:Lichen',81:'Pasture',82:'Crops',90:'Wetland',95:'Herb Wet'}
class_colors={11:'royalblue',12:'whitesmoke',21:'pink',22:'lightcoral',23:'red',24:'maroon',31:'silver',\
              41:'limegreen',42:'darkgreen',43:'yellowgreen',51:'darkgoldenrod',52:'tan',\
              71:'wheat',72:'khaki',73:'darkkhaki',74:'darkseagreen',81:'gold',82:'sandybrown',90:'skyblue',95:'cadetblue',0:'white'}

# %%
import matplotlib.patches as mpatches

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
        hatch.append('//')
    else:
        hatch.append('')
yerr=[np.zeros((len(X),)),X[:]-Y2[3]]
yerr=np.array(yerr)
yerr[0,:][yerr[1,:]<0]=yerr[1,:][yerr[1,:]<0]*(-1)
yerr[1,:][yerr[1,:]<0]=0
a=plt.bar(Y2[0],Y2[3],color=colors,hatch=hatch,edgecolor='black',yerr=yerr,capsize=4)
plt.xticks(rotation=45)
plt.xlim(-.5,47)
plt.ylim(.2,1.8)
plt.ylabel(r'$MAD$')
leg=[]
for clas in class_names.keys():
    ptch=mpatches.Patch(color=class_colors[clas],label=class_names[clas])
    leg.append(ptch)
plt.legend(handles=leg,ncol=2,loc='upper left')

# %%

# %%

# %%
fpu.keys()

# %%
