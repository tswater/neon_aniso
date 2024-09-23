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
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from mpi4py import MPI
# MPI4PY Stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

odir='/home/tswater/Documents/tyche/data/neon/foot_stats/'
dsmdir='/home/tswater/Documents/tyche/data/neon/dsm/'
dtmdir='/home/tswater/Documents/tyche/data/neon/dtm/'
footdir='/home/tswater/Documents/tyche/data/neon/dp4ex/'
idir='/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data/'
nlcdf='/home/tswater/Downloads/nlcd_2021_land_cover_l48_20230630/nlcd_2021_land_cover_l48_20230630.img'
#nlcdf='/home/tswater/Downloads/NLCD_2016_Land_Cover_AK_20200724.img'
#nlcdf='/home/tswater/Downloads/pr_landcover_wimperv_10-28-08_se5.img'
#nlcdf='/home/tswater/Downloads/hi_hawaii_2010_ccap_hr_land_cover20150120.img'

ovar={'slope_dtm':[],'aspect_dtm':[],'std_dsm':[],'std_dtm':[],'std_chm':[],'mean_chm':[],\
        'treecover_lidar':[],'range_dtm':[],'range_chm':[],'std_aspect':[],'std_slope':[],\
        'nlcd_iqv':[],'nlcd_dom':[],'nlcd_2':[],'nlcd_3':[],'area':[],'center_we':[],'center_sn':[]}

ctf=.95

############# HELPER FUNCTIONS #####################
def calculate_slope(DEM,xx,yy,dx):
    suffix = datetime.now().strftime("%M%S%f")
    gdal.DEMProcessing('slope'+suffix+'.tif', DEM, 'slope')
    dxi=int(dx*301/2)
    with rasterio.open('slope'+suffix+'.tif') as dataset:
        slope=dataset.read(1,boundless=True,fill_value=float('nan'),window=Window(xx-dxi,yy-dxi,dx*301,dx*301))
    run('rm slope'+suffix+'.tif',shell=True)
    return slope

def calculate_aspect(DEM,xx,yy,dx):
    suffix = datetime.now().strftime("%M%S%f")
    gdal.DEMProcessing('aspect'+suffix+'.tif', DEM, 'aspect')
    dxi=int(dx*150.5)
    with rasterio.open('aspect'+suffix+'.tif') as dataset:
        aspect=dataset.read(1,boundless=True,fill_value=float('nan'),window=Window(xx-dxi,yy-dxi,dx*301,dx*301))
    run('rm aspect'+suffix+'.tif',shell=True)
    return aspect

def load_nlcd(x,y,dx,proj4,site):
    #xx-int(dx*150.5):xx+int(dx*150.5),yy-int(dx*150.5):yy+int(dx*150.5)
    if site in [b'BONA',b'TOOL',b'DEJU',b'HEAL',b'BARR']:
        nlcdf='/home/tswater/Downloads/NLCD_2016_Land_Cover_AK_20200724.img'
    elif site in [b'LAJA',b'GUAN']:
        nlcdf='/home/tswater/Downloads/pr_landcover_wimperv_10-28-08_se5.img'
    elif site==b'PUUM':
        nlcdf='/home/tswater/Downloads/hi_hawaii_2010_ccap_hr_land_cover20150120.img'
    else:
        nlcdf='/home/tswater/Downloads/nlcd_2021_land_cover_l48_20230630/nlcd_2021_land_cover_l48_20230630.img'
    ex_str=str(int(x-int(int(dx*301)/2)))+' '+str(int(y-int(int(dx*301)/2)))+' '+str(int(x+int(int(dx*301)/2)))+' '+str(int(y+int(int(dx*301)/2)))
    suffix = datetime.now().strftime("%M%S%f")
    cmd="gdalwarp -t_srs '"+proj4+"' -tr "+str(dx)+' '+str(dx)+' -te '+ex_str+' '+nlcdf+' temp'+suffix+'.tif'
    run(cmd,shell=True)
    nlcd=rasterio.open('temp'+suffix+'.tif').read(1)
    run('rm temp'+suffix+'.ti*',shell=True)
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

def get_footmask(ff,idx,ctf=ctf):
    if ff=='NOFILE':
        data=np.zeros((301,301))
        return data
    else:
        data=ff['footprint'][idx,:,:]
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

ovars=copy.deepcopy(ovar)

# get data
sites=fps['SITE'][:]
times=fps['TIME'][:]

d0=datetime(1970,1,1,0)
dt=copy.copy(d0)
site='hellokitty'
export=False
load_site=False
load_footfile=False
load_foot=False
buffer=0

sitelist=[]
for s_ in np.unique(sites):
     sss=str(s_)[2:-1]+'_S.h5'
     if sss in os.listdir(odir):
         pass
     else:
         sitelist.append(s_)


for s_ in sitelist[rank::size]:
    for i in range(len(times)):
        # figure out what needs to be loaded
        site_=sites[i]
        if i==(len(times)-1):
            load_site=True
        elif site_!=s_:
            continue
        t=times[i]
        dp=copy.copy(dt)
        dt=d0+timedelta(seconds=t)
        if site_!=site:
            load_site=True
            load_footfile=True
            load_foot=True
        elif dp.month!=dt.month:
            load_footfile=True
            load_foot=True
        elif (dp.day!=dt.day)|(dp.hour!=dt.hour)|((dp.minute<30)&(dt.minute>30)):
            load_foot=True
        else:
            noload=True
        basename=os.listdir(footdir+str(site_)[2:-1])[0][0:33]

        # load in footprint file
        if load_footfile:
            print()
            print('   '+str(dt.year)+'-'+str(dt.month),flush=True)
            buffer=0
            fname=basename+dt.strftime('%Y-%m-%d.nc')
            try:
                ff=nc.Dataset(footdir+str(site_)[2:-1]+'/'+fname,'r')
                dx=ff.dx
            except:
                ff='NOFILE'
                dx=float('nan')

        # export then load new site
        if load_site:
            # export if needed
            if len(ovars['nlcd_dom'][:])>0:
                print('Exporting '+str(site)[2:-1],flush=True)
                fpout_s=h5py.File(odir+str(site)[2:-1]+'_S.h5','w')
                for var in ovars.keys():
                    try:
                        fpout_s.create_dataset(var,data=np.array(ovars[var][:]))
                    except Exception as e:
                        print(e)
                fpout_s.close()
                print()
            site=site_

            # load in the new site info
            print('Loading in '+str(site)[2:-1],flush=True)
            lat=fps['lat'][i]
            lon=fps['lon'][i]
            fpdsm=rasterio.open(dsmdir+str(site)[2:-1]+'/dsm_'+str(site)[2:-1]+'.tif')
            fpdtm=rasterio.open(dtmdir+str(site)[2:-1]+'/dtm_'+str(site)[2:-1]+'.tif')
            transformer=Transformer.from_crs('EPSG:4326',fpdsm.crs,always_xy=True)
            xx_,yy_=transformer.transform(lon,lat)
            xx,yy=fpdsm.index(xx_,yy_)

            if ff=='NOFILE':
                raise Exception('No proper dx definition to load site; must assign manually')
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
            nlcd=load_nlcd(xx_,yy_,dx,fpdtm.crs.to_proj4(),site)

            ovars=copy.deepcopy(ovar)
            print('   site info loaded',flush=True)

        # load actual footprint
        if load_foot:
            if buffer==0:
                print('   ',end='')
            print('.',end='',flush=True)
            buffer=buffer+1
            if buffer>70:
                print()
                buffer=0
            foot_index=int((dt.hour)*2+dt.minute/30+.01)
            foot=get_footmask(ff,foot_index)
            idx_sn,idx_we=np.indices([301,301])
            idx_sn=150-idx_sn
            idx_we=idx_we-150
            dsmf,dtmf,chmf,aspf,slpf=get_masked_data(foot,dx,dtm,dsm,asp,slp)


        # reset all loads
        load_foot=False
        load_site=False
        load_footfile=False
        noload=False

        if noload:
            for var in ovars.keys():
                try:
                    ovars[var].append(ovars[var][-1])
                except:
                    ovars[var].append(float('nan'))

        else:
            ovars['center_we'].append(np.mean(idx_we[foot.astype(bool)]))
            ovars['center_sn'].append(np.mean(idx_sn[foot.astype(bool)]))
            ovars['area'].append(np.sum(foot*dx*dx))
            ovars['slope_dtm'].append(np.nanmean(slpf))
            ovars['aspect_dtm'].append(stats.circmean(aspf/180*np.pi,nan_policy='omit'))
            ovars['std_aspect'].append(stats.circstd(aspf/180*np.pi,nan_policy='omit'))
            ovars['mean_chm'].append(np.nanmean(dsmf-dtmf))
            ovars['std_dtm'].append(np.nanstd(dtmf))
            ovars['std_dsm'].append(np.nanstd(dsmf))
            ovars['std_chm'].append(np.nanstd(dsmf-dtmf))
            ovars['range_dtm'].append(np.nanpercentile(dtmf,95)-np.nanpercentile(dtmf,5))
            ovars['range_chm'].append(np.nanpercentile(chmf,95)-np.nanpercentile(chmf,5))
            ovars['treecover_lidar'].append(np.nansum(chmf>0.1)/len(chmf))

            # nlcd processing
            nlcds=nlcd[foot.astype(bool)]
            ovars['nlcd_iqv'].append(iqv(nlcds))
            ovars['nlcd_dom'].append(stats.mode(nlcds,axis=None)[0])
            ovars['nlcd_2'].append(stats.mode(nlcds[nlcds!=ovars['nlcd_dom'][-1]],axis=None)[0])
            ovars['nlcd_3'].append(stats.mode(nlcds[(nlcds!=ovars['nlcd_dom'][-1])&(nlcds!=ovars['nlcd_2'][-1])],axis=None)[0])

