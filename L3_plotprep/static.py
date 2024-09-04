import h5py
import numpy as np
import os
from subprocess import run
import rasterio
from scipy import stats
from osgeo import gdal
from pyproj import Transformer

from datetime import datetime,timedelta

# Note: meteorological means are for valid, unstable conditions. Stable is neglected

odir='/home/tswater/Documents/tyche/data/neon/'
dsmdir='/home/tswater/Documents/tyche/data/neon/dsm/'
dtmdir='/home/tswater/Documents/tyche/data/neon/dtm/'
idir='/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data/'

ovar={'elevation':[],'canopy_height':[],'site':[],'lat':[],'zd':[],'lon':[],\
        'nlcd_dom':[],'tow_height':[],'utc_off':[],'mean_U':[],\
        'std_dsm':[],'std_dtm':[],'std_chm':[],\
        'mean_dsm':[],'mean_dtm':[],'mean_chm':[],\
        'slope_dtm':[],'range_dtm':[],'range_chm':[],\
        'E_aspect_pct':[],'NE_aspect_pct':[],'N_aspect_pct':[],'NW_aspect_pct':[],\
        'W_aspect_pct':[],'SW_aspect_pct':[],'S_aspect_pct':[],'SE_aspect_pct':[],\
        'T_jja':[],'T_son':[],'T_djf':[],'T_mam':[],'T_std':[],\
        'q_jja':[],'q_son':[],'q_djf':[],'q_mam':[],'q_std':[],\
        'mean_u_xb':[],'mean_u_yb':[],'std_u_xb':[],'std_u_yb':[],'skw_u_xb':[],'skw_u_yb':[],\
        'mean_s_xb':[],'mean_s_yb':[],'std_s_xb':[],'std_s_yb':[],'skw_s_xb':[],'skw_s_yb':[]}

###### HELPER FUNCTIONS ########
def time2hour(time,utcf):
    hour_=[]
    d0=datetime(1970,1,1,0)
    yold=1970
    N=len(time)
    milestones=np.linspace(0,N,10).astype(int)
    for t in range(N):
        if t in milestones:
            print('.',end='',flush=True)
        df=d0+timedelta(hours=utcf)+timedelta(seconds=time[t])
        hour_.append(df.month)
    print()
    return np.array(hour_)

def calculate_slope(DEM):
    gdal.DEMProcessing('slope.tif', DEM, 'slope')
    with rasterio.open('slope.tif') as dataset:
        slope=dataset.read(1)
    run('rm slope.tif',shell=True)
    return slope

def calculate_aspect(DEM):
    gdal.DEMProcessing('aspect.tif', DEM, 'aspect')
    with rasterio.open('aspect.tif') as dataset:
        aspect=dataset.read(1)
    run('rm aspect.tif',shell=True)
    return aspect

def mean_aspect(data):
    data[data>=337.5]=data[data>=337.5]-360
    data[data==0]=float('nan')
    bins=np.array([0,45,90,135,180,225,270,315,360])-22.5
    counts=[0,0,0,0,0,0,0,0]
    for i in range(len(counts)):
        counts[i]=np.nansum((data>bins[i])&(data<bins[i+1]))
    counts=np.array(counts)
    counts=counts/np.sum(counts)
    return counts

fp=h5py.File(odir+'static_data.h5','w')

# build list of sites
sites_=os.listdir(dsmdir)
sites_.sort()
sites=[]
for site in sites_:
    if len(site)>4:
        continue
    else:
        sites.append(site)
fps=h5py.File(idir+'/NEON_TW_S_UVWT.h5','r')
fpu=h5py.File(idir+'/NEON_TW_U_UVWT.h5','r')

# get data
for site in sites:
    print('Begin '+site,flush=True)
    ms=fps['SITE'][:]==bytes(site,'utf-8')
    mu=fpu['SITE'][:]==bytes(site,'utf-8')

    # simple
    ovar['elevation'].append(fpu['elevation'][mu][0])
    ovar['canopy_height'].append(fpu['canopy_height'][mu][0])
    ovar['site'].append(bytes(site,'utf-8'))
    ovar['lat'].append(fpu['lat'][mu][0])
    ovar['lon'].append(fpu['lon'][mu][0])
    ovar['zd'].append(fpu['zd'][mu][0])
    ovar['nlcd_dom'].append(fpu['nlcd_dom'][mu][0])
    ovar['tow_height'].append(fpu['tow_height'][mu][0])
    ovar['utc_off'].append(fpu['utc_off'][mu][0])

    # met
    print('MET',end=',',flush=True)
    months=time2hour(fpu['TIME'][mu],ovar['utc_off'][-1])
    mu_jja=(months==6)|(months==7)|(months==8)
    mu_son=(months==9)|(months==10)|(months==11)
    mu_djf=(months==12)|(months==1)|(months==2)
    mu_mam=(months==3)|(months==4)|(months==5)

    T=fpu['TA'][mu]
    q=fpu['H2O'][mu]
    ovar['T_std'].append(np.std(T))
    ovar['q_std'].append(np.std(q))
    ovar['T_jja'].append(np.mean(T[mu_jja]))
    ovar['T_son'].append(np.mean(T[mu_son]))
    ovar['T_djf'].append(np.mean(T[mu_djf]))
    ovar['T_mam'].append(np.mean(T[mu_mam]))
    ovar['q_son'].append(np.nanmean(q[mu_son]))
    ovar['q_djf'].append(np.nanmean(q[mu_djf]))
    ovar['q_mam'].append(np.nanmean(q[mu_mam]))
    ovar['q_jja'].append(np.nanmean(q[mu_jja]))
    ovar['mean_U'].append(np.mean(np.sqrt(fpu['U'][mu]**2+fpu['V'][mu]**2)))

    # aniso
    print('ANI',end=',',flush=True)
    ybu=fpu['ANI_YB'][mu]
    xbu=fpu['ANI_XB'][mu]
    ybs=fps['ANI_YB'][ms]
    xbs=fps['ANI_XB'][ms]

    ovar['mean_u_xb'].append(np.mean(xbu))
    ovar['mean_u_yb'].append(np.mean(ybu))
    ovar['mean_s_xb'].append(np.mean(xbs))
    ovar['mean_s_yb'].append(np.mean(ybs))
    ovar['std_u_xb'].append(np.std(xbu))
    ovar['std_u_yb'].append(np.std(ybu))
    ovar['std_s_xb'].append(np.std(xbs))
    ovar['std_s_yb'].append(np.std(ybs))
    ovar['skw_u_xb'].append(stats.skew(xbu))
    ovar['skw_u_yb'].append(stats.skew(ybu))
    ovar['skw_s_xb'].append(stats.skew(xbs))
    ovar['skw_s_yb'].append(stats.skew(ybs))

    # spatial mean, std, slope, range
    print('DEM',end=',',flush=True)
    fpdsm=rasterio.open(dsmdir+site+'/dsm_'+site+'.tif')
    fpdtm=rasterio.open(dtmdir+site+'/dtm_'+site+'.tif')
    transformer=Transformer.from_crs('EPSG:4326',fpdsm.crs,always_xy=True)
    xx_,yy_=transformer.transform(ovar['lon'][-1],ovar['lat'][-1])
    xx,yy=fpdsm.index(xx_,yy_)
    dsm=fpdsm.read(1)[xx-2000:xx+2000,yy-2000:yy+2000]
    dsm[dsm<0]=float('nan')
    dtm=fpdtm.read(1)[xx-2000:xx+2000,yy-2000:yy+2000]
    dtm[dtm<0]=float('nan')
    chm=dsm-dtm

    ovar['mean_dsm'].append(np.nanmean(dsm))
    ovar['mean_dtm'].append(np.nanmean(dtm))
    ovar['mean_chm'].append(np.nanmean(chm))
    ovar['std_dsm'].append(np.nanstd(dsm))
    ovar['std_dtm'].append(np.nanstd(dtm))
    ovar['std_chm'].append(np.nanstd(chm))
    ovar['range_dtm'].append(np.nanpercentile(dtm,95)-np.nanpercentile(dtm,5))
    ovar['range_chm'].append(np.nanpercentile(chm,95)-np.nanpercentile(chm,5))

    print('SLP/ASP',end=',',flush=True)
    slp=calculate_slope(dtmdir+site+'/dtm_'+site+'.tif')[xx-2000:xx+2000,yy-2000:yy+2000]
    asp=calculate_aspect(dtmdir+site+'/dtm_'+site+'.tif')[xx-2000:xx+2000,yy-2000:yy+2000]
    slp[slp>45]=float('nan')
    asp[slp>45]=float('nan')
    slp[slp==0]=float('nan')
    asp[slp==0]=float('nan')

    ovar['slope_dtm'].append(np.mean(slp))
    aspects=mean_aspect(asp)
    ovar['E_aspect_pct'].append(aspects[0])
    ovar['NE_aspect_pct'].append(aspects[1])
    ovar['N_aspect_pct'].append(aspects[2])
    ovar['NW_aspect_pct'].append(aspects[3])
    ovar['W_aspect_pct'].append(aspects[4])
    ovar['SW_aspect_pct'].append(aspects[5])
    ovar['S_aspect_pct'].append(aspects[6])
    ovar['SE_aspect_pct'].append(aspects[7])

    print()

print('#### OUTPUT ####',flush=True)
for var in ovar.keys():
    print(var)
    print(np.shape(ovar[var]))
    fp.create_dataset(var,data=np.array(ovar[var][:]))

fp.close()




