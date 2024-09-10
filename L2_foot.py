# save flux footprint based statistics

import rasterio
import numpy as np
import netCDF4 as nc
import h5py
import copy
from pyproj import Transformer
import os
from subprocess import run
from scipy import stats
from osgeo import gdal
from datetime import datetime,timedelta
from rasterio.windows import Window
import sys
from numba import jit
site_1=sys.argv[1]

odir='/home/tswater/Documents/tyche/data/neon/foot_stats/'
dsmdir='/home/tswater/Documents/tyche/data/neon/dsm/'
dtmdir='/home/tswater/Documents/tyche/data/neon/dtm/'
footdir='/home/tswater/Documents/tyche/data/neon/dp4ex/'
idir='/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data/'
nlcdf='/home/tswater/Downloads/nlcd_2021_land_cover_l48_20230630/nlcd_2021_land_cover_l48_20230630.img'

ovar={'slope_dtm':[],'aspect_dtm':[],'std_dsm':[],'std_dtm':[],'std_chm':[],'mean_chm':[],\
        'treecover_lidar':[],'range_dtm':[],'range_chm':[],'std_aspect':[],'std_slope':[],\
        'nlcd_iqv':[],'nlcd_dom':[],'nlcd_2':[],'nlcd_3':[]}

ctf=.95

############# HELPER FUNCTIONS #####################
def calculate_slope(DEM,xx,yy,dx):
    gdal.DEMProcessing('slope.tif', DEM, 'slope')
    dxi=int(dx*301/2)
    with rasterio.open('slope.tif') as dataset:
        slope=dataset.read(1,boundless=True,fill_value=float('nan'),window=Window(xx-dxi,yy-dxi,dx*301,dx*301))
    run('rm slope.tif',shell=True)
    return slope

def calculate_aspect(DEM,xx,yy,dx):
    gdal.DEMProcessing('aspect.tif', DEM, 'aspect')
    dxi=int(dx*150.5)
    with rasterio.open('aspect.tif') as dataset:
        aspect=dataset.read(1,boundless=True,fill_value=float('nan'),window=Window(xx-dxi,yy-dxi,dx*301,dx*301))
    run('rm aspect.tif',shell=True)
    return aspect

def load_nlcd(x,y,dx,proj4):
    #xx-int(dx*150.5):xx+int(dx*150.5),yy-int(dx*150.5):yy+int(dx*150.5)
    ex_str=str(int(x-int(int(dx*301)/2)))+' '+str(int(y-int(int(dx*301)/2)))+' '+str(int(x+int(int(dx*301)/2)))+' '+str(int(y+int(int(dx*301)/2)))
    cmd="gdalwarp -t_srs '"+proj4+"' -tr "+str(dx)+' '+str(dx)+' -te '+ex_str+' '+nlcdf+' temp.tif'
    run(cmd,shell=True)
    nlcd=rasterio.open('temp.tif').read(1)
    run('rm temp.ti*',shell=True)
    return nlcd

@jit(nopython=True)
def databig(data,dx):
    dout=np.zeros((int(dx*301+.5),int(dx*301+.5)))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            dout[i*dx:(i+1)*dx,j*dx:(j+1)*dx]=data[i,j]
    return dout

def get_masked_data(foot,dx,dtm,dsm,asp,slp):
    footbig=databig(foot,dx)
    foot=foot.astype(np.bool)
    footbig=footbig.astype(np.bool)
    chm=dsm-dtm
    dsmf=dsm[footbig]
    dtmf=dtm[footbig]
    chmf=chm[footbig]
    aspf=asp[footbig]
    slpf=slp[footbig]
    return dsmf,dtmf,chmf,aspf,slpf

def get_footmask(data,ctf=ctf):
    val=7
    for i in np.logspace(-2,-8,100):
        sm=np.nansum(data[data>i])
        if sm>=ctf:
            val=i
            break
    if val==7:
        val=10**(-8)
    data[data>=val]=1
    data[data<1]=0
    return data.data

def iqv(data):
    types,counts=np.unique(data,return_counts=True)
    return len(types)*(100**2-np.sum(counts/(np.sum(counts)*100))**2)/(100**2*(len(types)-1))


# site by site (30min) and mean statistics
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

ovaru=copy.deepcopy(ovar)
ovars=copy.deepcopy(ovar)

# get data
Ns=len(fps['TIME'][:])
Nu=len(fpu['TIME'][:])
sites=fps['SITE'][:]
siteu=fpu['SITE'][:]
times=fps['TIME'][:]
timeu=fpu['TIME'][:]

d0=datetime(1970,1,1,0)
js=0
ju=0
oldname='asldkjfa'
site='ABCD'
basename='hellokitty'
use_u=(timeu[0]<times[0])
update=True
nlcd_loaded=False
foot_index=-1
oy=2010

while ((js<Ns)&(ju<Nu)):
    # check if site (dsm/dtm/nlcd) correct
    # check if day (foot) correct
    # check if half hour (foot) correct
    if not (str(sites[js])[2:-1]==site_1):
        js=js+1
        if not (str(siteu[ju])[2:-1]==site_1):
            ju=ju+1
            continue
        else:
            use_u=True
    elif not (str(siteu[ju])[2:-1]==site_1):
        ju=ju+1
        use_u=False
        continue
    else:
        use_u=(timeu[ju]<times[js])

    ### LOAD INFO ###
    if use_u:
        print('.',end='',flush=True)
        if not siteu[ju]==site:
            site=siteu[ju]
            update=True
            basename=os.listdir(footdir+str(site)[2:-1])[0][0:33]
            lat=fpu['lat'][ju]
            lon=fpu['lon'][ju]
            # load dsm, dtm, nlcd, lat, lon
            # compute aspect/slope maps
            # change site; update basename
        # check if the same day
        udt_=d0+timedelta(seconds=timeu[ju])
        if udt_.year!=oy:
            print(udt_.year)
            oy=udt_.year
        fname=basename+udt_.strftime('%Y-%m-%d.nc')
        if not fname==oldname:
            oldname=fname
            ff=nc.Dataset(footdir+str(site)[2:-1]+'/'+fname,'r')
            dx=ff.dx

        if update:
            print(site,flush=True)
            # dsm/dtm stuff
            fpdsm=rasterio.open(dsmdir+str(site)[2:-1]+'/dsm_'+str(site)[2:-1]+'.tif')
            fpdtm=rasterio.open(dtmdir+str(site)[2:-1]+'/dtm_'+str(site)[2:-1]+'.tif')
            transformer=Transformer.from_crs('EPSG:4326',fpdsm.crs,always_xy=True)
            xx_,yy_=transformer.transform(lon,lat)
            xx,yy=fpdsm.index(xx_,yy_)

            dxi=int(dx*301/2)
            dtm=fpdtm.read(1,boundless=True,fill_value=float('nan'),window=Window(xx-dxi,yy-dxi,dx*301,dx*301))
            dsm=fpdsm.read(1,boundless=True,fill_value=float('nan'),window=Window(xx-dxi,yy-dxi,dx*301,dx*301))

            ss=str(site)[2:-1]
            slp=calculate_slope(dtmdir+ss+'/dtm_'+ss+'.tif',xx,yy,dx)
            asp=calculate_aspect(dtmdir+ss+'/dtm_'+ss+'.tif',xx,yy,dx)
            slp[slp>45]=float('nan')
            asp[slp>45]=float('nan')
            slp[slp==0]=float('nan')
            asp[slp==0]=float('nan')

            # nlcd stuff
            if nlcd_loaded:
                continue
            else:
                nlcd=load_nlcd(xx_,yy_,dx,fpdtm.crs.to_proj4())
            update=False


        # load footprint
        foot_index=int((udt_.hour)*2+udt_.minute/30+.01)
        foot=get_footmask(ff['footprint'][foot_index,:,:])
        dsmf,dtmf,chmf,aspf,slpf=get_masked_data(foot,dx,dtm,dsm,asp,slp)

    else:
        if not sites[js]==site:
            site=sites[js]
            update=True
            basename=os.listdir(footdir+site)[0][0:33]
            lat=fps['lat'][js]
            lon=fps['lon'][js]
            # load dsm, dtm, nlcd, lat, lon
            # compute aspect/slope maps
            # change site; update basename

        # check if the same day
        udt_=d0+timedelta(seconds=times[js])
        fname=basename+udt_.strftime('%Y-%m-%d.nc')
        if update:
            print(site,flush=True)
            # dsm/dtm stuff
            fpdsm=rasterio.open(dsmdir+str(site)[2:-1]+'/dsm_'+str(site)[2:-1]+'.tif')
            fpdtm=rasterio.open(dtmdir+str(site)[2:-1]+'/dtm_'+str(site)[2:-1]+'.tif')
            transformer=Transformer.from_crs('EPSG:4326',fpdsm.crs,always_xy=True)
            xx_,yy_=transformer.transform(lon,lat)
            xx,yy=fpdsm.index(xx_,yy_)
            dxi=int(dx*301/2)
            dtm=fpdtm.read(1,boundless=True,fill_value=float('nan'),window=Window(xx-dxi,yy-dxi,dx*301,dx*301))
            dsm=fpdsm.read(1,boundless=True,fill_value=float('nan'),window=Window(xx-dxi,yy-dxi,dx*301,dx*301))
            ss=str(site)[2:-1]
            slp=calculate_slope(dtmdir+ss+'/dtm_'+ss+'.tif',xx,yy,dx)
            asp=calculate_aspect(dtmdir+ss+'/dtm_'+ss+'.tif',xx,yy,dx)

            slp[slp>45]=float('nan')
            asp[slp>45]=float('nan')
            slp[slp==0]=float('nan')
            asp[slp==0]=float('nan')

            if nlcd_loaded:
                continue
            else:
                nlcd=load_nlcd(xx_,yy_,dx,fpdtm.crs.to_proj4())
            update=False


        if not fname==oldname:
            oldname=fname
            ff=nc.Dataset(footdir+str(site)[2:-1]+'/'+fname,'r')
            dx=ff.dx
            # load footprint
            foot_index=int((udt_.hour)*2+udt_.minute/30+.01)
            foot=get_footmask(ff['footprint'][foot_index,:,:])
            dsmf,dtmf,chmf,aspf,slpf=get_masked_data(foot,dx,dtm,dsm,asp,slp)
        else:
            fi=int((udt_.hour)*2+udt_.minute/30+.01)
            if not (foot_index==fi):
                foot_index=fi
                foot=get_footmask(ff['footprint'][foot_index,:,:])
                dsmf,dtmf,chmf,aspf,slpf=get_masked_data(foot,dx,dtm,dsm,asp,slp)

    # now actual analysis
    if use_u:
        ova=ovaru
    else:
        ova=ovars

    # dsm/dtm/chm processing
    ova['slope_dtm'].append(np.nanmean(slpf))
    ova['aspect_dtm'].append(stats.circmean(aspf/180*np.pi,nan_policy='omit'))
    ova['std_aspect'].append(stats.circstd(aspf/180*np.pi,nan_policy='omit'))
    ova['mean_chm'].append(np.nanmean(dsmf-dtmf))
    ova['std_dtm'].append(np.nanstd(dtmf))
    ova['std_dsm'].append(np.nanstd(dsmf))
    ova['std_chm'].append(np.nanstd(dsmf-dtmf))
    ova['range_dtm'].append(np.nanpercentile(dtmf,95)-np.nanpercentile(dtmf,5))
    ova['range_chm'].append(np.nanpercentile(chmf,95)-np.nanpercentile(chmf,5))
    ova['treecover_lidar'].append(np.nansum(chmf>0.1)/len(chmf))

    # nlcd processing
    nlcds=nlcd[foot.astype(bool)]
    ova['nlcd_iqv'].append(iqv(nlcds))
    ova['nlcd_dom'].append(stats.mode(nlcds,axis=None)[0])
    ova['nlcd_2'].append(stats.mode(nlcds[nlcds!=ova['nlcd_dom'][-1]],axis=None)[0])
    ova['nlcd_3'].append(stats.mode(nlcds[(nlcds!=ova['nlcd_dom'][-1])&(nlcds!=ova['nlcd_2'][-1])],axis=None)[0])

    # go up one
    if use_u:
        ju=ju+1
    else:
        js=js+1


# export data
fpout_u=h5py.File(odir+site_1+'_U.h5','w')
fpout_s=h5py.File(odir+site_1+'_S.h5','w')
for var in ova.keys():
    try:
        fpout_u.create_dataset(var,data=np.array(ovaru[var][:]))
        fpout_s.create_dataset(var,data=np.array(ovars[var][:]))
    except Exception as e:
        print(e)
        continue
