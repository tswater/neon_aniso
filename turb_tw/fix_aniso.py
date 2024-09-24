import numpy as np
import os
from subprocess import run
import h5py
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from datetime import datetime
from mpi4py import MPI
from datetime import timedelta
from calendar import monthrange
import math
import warnings
import tracemalloc
import gc

tracemalloc.start()

warnings.filterwarnings('ignore',category=RuntimeWarning)

# MPI4PY Stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tsw35/soteria/software/miniconda/envs/eddy4R/lib/R/modules

#### USER INPUT ####
raw_dir='/home/tsw35/xSot_shared/NEON_raw_data/'
dp4_dir='/home/tsw35/soteria/data/NEON/dp04ex/'
pfdir='/home/tsw35/soteria/data/NEON/pfvalues/'
out_dir='/home/tsw35/soteria/data/NEON/aniso_1mv2/'
sites=['ALL']
years=['ALL']
dp4type=['expanded','basic'][0]
overwrite=False # if true, will overwrite already created output files
testing=True # if testing, will output more information

# timesteps per day for 1m and 30m intervals
n1m=1440
n30m=48


#### UNITS and DESCRIPTION ####
units={}
units['CO2FX_1m']='umolCo2 m-2 s-1'
desc={}



##########################
#### HELPER FUNCTIONS ####
##########################



####################################
####################################


#### SETUP ####

# Load in R packages
Rbase=importr('base')
Re4Rq=importr('eddy4R.qaqc')
Re4Rb=importr('eddy4R.base')

# Setup the list of sites and years
if 'ALL' in sites:
    sites=[]
    for file in os.listdir(raw_dir):
        if len(file)!=4:
            continue
        else:
            sites.append(file)
if 'ALL' in years:
    years=[2017,2018,2019,2020,2021,2022,2023]

if rank==0:
    print('Processing the following sites: ')
    print(sites)
    print('Processing the following years: ')
    print(years)

sites.sort()
years.sort()

# compile lists of NEON domains and sonic levels
domain={} #i.e. 16 for ABBY, as in D16
s_level={} #i.e. 050 for ABBY, as in 000_050_01m
for site in sites:
    file=os.listdir(dp4_dir+site)[1]
    domain[site]=file[6:8]
    fp=h5py.File(dp4_dir+site+'/'+file,'r')
    lvls=list(fp[site+'/dp01/data/amrs/'].keys())
    s_level[site]=lvls[0][4:7]
    fp.close()

# Make an array of site years to loop through
st_yr=[]
for site in sites:
    for year in years:
        st_yr.append(site+'_'+str(year))


#### CORE LOOP ####
for sty in st_yr[rank::size]:

    # get site, year, NEON domain and sonic level
    site=sty[0:4]
    year=int(sty[5:])
    dt0=datetime(year,1,1,0)
    dom=domain[site]
    lvl=s_level[site]
    fp4=h5py.File(pfdir+site+'_pfvalues.nc','r')#nc.Dataset(pfdir+site+'_pfvalues.nc','r')
    _day=fp4['day'][:]
    _month=fp4['month'][:]
    _year=fp4['year'][:]
    _ax=fp4['ax'][:]
    _ay=fp4['ay'][:]
    _ofst=fp4['ofst'][:]

    # check to see if this site-year can be skipped
    if not overwrite:
        try:
            outlist=os.listdir(out_dir+site+'/')
        except Exception as e:
            run('mkdir '+out_dir+site,shell=True)
            outlist=os.listdir(out_dir+site+'/')
        if len(outlist)==0:
            pass
        else:
            yrlist=[]
            for file in outlist:
                if str(year) in file:
                    yrlist.append(file)
            yrlist.sort()
            if len(yrlist)<1:
                pass
            else:
                lastname=str(year)+'-12'
                if lastname in yrlist[-1]:
                    if os.path.getsize(out_dir+site+'/'+yrlist[-1])>100000:
                        continue

    # make a list of days of the year to loop through
    if year in [2016,2020,2024,2028]:
        doys=list(range(366))
    else:
        doys=list(range(365))

    
    # core action loop
    skip=False
    
    for doy in doys:
        dt=dt0+timedelta(days=doy)
        dt_str=dt.strftime('%Y-%m-%d')
        day=dt.day

        #### DEAL WITH MONTHLY STUFF ####
        # if first of month, try to load in monthly dp4 file
        # otherwise, skip day if no dp4 file available
        # also output data from previous month if available, and initialize new
        if (day==1):

            # Initialize output data
            N30m=monthrange(year,dt.month)[1]*n30m
            N1m=monthrange(year,dt.month)[1]*n1m

            output={}
            output['UU_1m']=np.ones((N1m,))*float('nan')
            output['VV_1m']=np.ones((N1m,))*float('nan')
            output['WW_1m']=np.ones((N1m,))*float('nan')
            output['UV_1m']=np.ones((N1m,))*float('nan')
            output['UW_1m']=np.ones((N1m,))*float('nan')
            output['VW_1m']=np.ones((N1m,))*float('nan')
            output['Ustr_1m']=np.ones((N1m,))*float('nan')
            output['Vstr_1m']=np.ones((N1m,))*float('nan')
            output['Wstr_1m']=np.ones((N1m,))*float('nan')

            output['UV']=np.ones((N30m,))*float('nan')
            output['UU']=np.ones((N30m,))*float('nan')
            output['VV']=np.ones((N30m,))*float('nan')
            output['WW']=np.ones((N30m,))*float('nan')
            output['UW']=np.ones((N30m,))*float('nan')
            output['VW']=np.ones((N30m,))*float('nan')
            output['Ustr']=np.ones((N30m,))*float('nan')
            output['Vstr']=np.ones((N30m,))*float('nan')
            output['Wstr']=np.ones((N30m,))*float('nan')
            
            output['USTAR']=np.ones((N30m,))*float('nan')
            output['USTAR_1m']=np.ones((N1m,))*float('nan')

            skip=False
        elif skip:
            continue

        #### LOAD IN IF EXPANDED ####
        try:
            idx=np.where((dt.day==_day)&(dt.year==_year)&(dt.month==_month))[0][0]
            if (np.sum(np.isnan(_ax[idx]))+np.sum(np.isnan(_ay[idx]))+np.sum(np.isnan(_ofst[idx])))>0:
                raise Exception('error')

        except Exception as e:
            print('Skipping '+site+':'+str(rank)+':'+dt_str+' due to pf missing/filename error...\n'+str(e),flush=True)
            continue

        logstr=site+':'+str(rank)+':'+dt_str+':: '

        #### LOAD IN DATA ####
        # load in the raw turbulence file and data
        fname='NEON.D'+dom+'.'+site+'.IP0.00200.001.ecte.'+dt_str[0:10]+'.l0p.h5'
        try:
            fp0=h5py.File(raw_dir+site+'/'+fname,'r')
            anz=float(fp0[site+'/dp0p/data/soni/'].attrs['AngNedZaxs'][0])
            dp0_u=fp0[site+'/dp0p/data/soni/000_'+lvl+'/veloXaxs'][:]
            dp0_v=fp0[site+'/dp0p/data/soni/000_'+lvl+'/veloYaxs'][:]
            dp0_w=fp0[site+'/dp0p/data/soni/000_'+lvl+'/veloZaxs'][:]
        except Exception as e:
            print('Skipping '+site+':'+str(rank)+':'+dt_str+' due to DP0P missing/filename error...\n'+str(e),flush=True)
            continue

        # fix nan situation
        dp0_u[np.abs(dp0_u)>50]=float('nan')
        dp0_v[np.abs(dp0_v)>50]=float('nan')
        dp0_w[np.abs(dp0_w)>50]=float('nan')

        # check all nans in u,v,w
        if (np.sum(np.isnan(dp0_u))>=.995*len(dp0_u)):
            print('Skipping '+site+':'+dt_str+' due to raw data for U missing > 99.5% ',flush=True)
            continue
        elif (np.sum(np.isnan(dp0_v))>=.995*len(dp0_v)):
            print('Skipping '+site+':'+dt_str+' due to raw data for V missing > 99.5% ',flush=True)
            continue
        elif (np.sum(np.isnan(dp0_w))>=.995*len(dp0_w)):
            print('Skipping '+site+':'+dt_str+' due to raw data for W missing > 99.5% ',flush=True)
            continue

        print(logstr+'files loaded successfully',flush=True)


        #### R Workflow Beginning: DESPIKE and LAG CORRECT ####

        # convert to r object
        r_u=robjects.FloatVector(dp0_u.tolist())
        r_v=robjects.FloatVector(dp0_v.tolist())
        r_w=robjects.FloatVector(dp0_w.tolist())

        # despike
        try:
            out=Re4Rq.def_dspk_br86(r_u)
            r_u=out[-2]
            out=Re4Rq.def_dspk_br86(r_v)
            r_v=out[-2]
            out=Re4Rq.def_dspk_br86(r_w)
            r_w=out[-2]
        except Exception as e:
            print(logstr+'ERROR despiking failed; will continue processing \n'+str(e),flush=True)

        print(logstr+'R Processing complete',flush=True)

        #### PLANAR FIT CORRECTION #####

        # rotate into meteorological coordinates
        ason=np.radians(anz)-math.pi
        if ason<0:
            ason=ason+2*math.pi
        amet=math.pi/2-ason
        if amet<0:
            amet=amet+2*math.pi
        ld_u=np.array(r_u)*math.cos(amet)-np.array(r_v)*math.sin(amet)
        ld_v=np.array(r_u)*math.sin(amet)+np.array(r_v)*math.cos(amet)
        ld_w=np.array(r_w)

        a_x=float(_ax[idx])
        a_y=float(_ay[idx])
        ofst=float(_ofst[idx])

        ld_w=ld_w-ofst
        mat_pitch=np.matrix([[math.cos(a_y),0,-math.sin(a_y)],[0,1,0],[math.sin(a_y),0,math.cos(a_y)]])
        mat_roll=np.matrix([[1,0,0],[0,math.cos(a_x),math.sin(a_x)],[0,-math.sin(a_x),math.cos(a_x)]])
        mat_rot=np.dot(mat_pitch,mat_roll)

        combo=np.array([ld_u,ld_v,ld_w])
        [ld_u,ld_v,ld_w]=np.dot(mat_rot,combo)

        ld_u=np.squeeze(np.array(ld_u[0,:]).T)
        ld_v=np.squeeze(np.array(ld_v[0,:]).T)
        ld_w=np.squeeze(np.array(ld_w[0,:]).T)


        #### PREPARE FOR FLUX CALCULATIONS ####

        #### LOOP THROUGH THE 1m INTERVALS ####
        # recall day-1= index of day of month

        for t in range(n1m):
            if (rank==0) and (size<=1):
                print('.',end='',flush=True)

            # rotate into the mean wind (eddy4R.turb def.rot.ang.zaxs.erth.R)
            uin=ld_u[t*20*60:(t+1)*20*60]
            vin=ld_v[t*20*60:(t+1)*20*60]

            a_ertm=np.arctan2(np.nanmean(vin),np.nanmean(uin))

            rot_u=uin*np.cos(a_ertm)+vin*np.sin(a_ertm)
            rot_v=-uin*np.sin(a_ertm)+vin*np.cos(a_ertm)
            rot_w=np.array(ld_w[t*20*60:(t+1)*20*60])

            # mean remove
            wp=rot_w-np.nanmean(rot_w)
            up=rot_u-np.nanmean(rot_u)
            vp=rot_v-np.nanmean(rot_v)

            # momentum flux
            ustar=(np.nanmean(up*wp)**2+np.nanmean(vp*wp)**2)**(1/4)

            # load data into output arrays, checking for at least 10% real
            if (np.sum(np.isnan(up*vp))<.9*20*60):
                output['UV_1m'][(day-1)*n1m+t]=np.nanmean(up*vp)
            if (np.sum(np.isnan(up*wp))<.9*20*60):
                output['UW_1m'][(day-1)*n1m+t]=np.nanmean(up*wp)
            if (np.sum(np.isnan(vp*wp))<.9*20*60):
                output['VW_1m'][(day-1)*n1m+t]=np.nanmean(vp*wp)
            if (np.sum(np.isnan(up*up))<.9*20*60):
                output['UU_1m'][(day-1)*n1m+t]=np.nanmean(up*up)
                output['Ustr_1m'][(day-1)*n1m+t]=np.nanmean(rot_u)
            if (np.sum(np.isnan(vp*vp))<.9*20*60):
                output['VV_1m'][(day-1)*n1m+t]=np.nanmean(vp*vp)
                output['Vstr_1m'][(day-1)*n1m+t]=np.nanmean(rot_v)
            if (np.sum(np.isnan(up*up))<.9*20*60):
                output['WW_1m'][(day-1)*n1m+t]=np.nanmean(wp*wp)
                output['Wstr_1m'][(day-1)*n1m+t]=np.nanmean(rot_w)
            if (np.sum(np.isnan(ustar))<.9*20*60):
                output['USTAR_1m'][(day-1)*n1m+t]=ustar
            gc.collect()

        if (rank==0) and (size<=1):
            print('',flush=True)
        print(logstr+'1 minute flux processing complete',flush=True)

        for t in range(n30m):
            # rotate into the mean wind (eddy4R.turb def.rot.ang.zaxs.erth.R)
            uin=ld_u[t*20*60*30:(t+1)*20*60*30]
            vin=ld_v[t*20*60*30:(t+1)*20*60*30]

            a_ertm=np.arctan2(np.nanmean(vin),np.nanmean(uin))

            rot_u=uin*np.cos(a_ertm)+vin*np.sin(a_ertm)
            rot_v=-uin*np.sin(a_ertm)+vin*np.cos(a_ertm)
            rot_w=np.array(ld_w[t*20*60*30:(t+1)*20*60*30])

            # mean remove
            wp=rot_w-np.nanmean(rot_w)
            up=rot_u-np.nanmean(rot_u)
            vp=rot_v-np.nanmean(rot_v)

            # momentum flux
            ustar=(np.nanmean(up*wp)**2+np.nanmean(vp*wp)**2)**(1/4)

            # load data into output arrays, checking for at least 10% real
            if (np.sum(np.isnan(up*vp))<.9*20*60*30):
                output['UV'][(day-1)*n30m+t]=np.nanmean(up*vp)
            if (np.sum(np.isnan(up*wp))<.9*20*60*30):
                output['UW'][(day-1)*n30m+t]=np.nanmean(up*wp)
            if (np.sum(np.isnan(vp*wp))<.9*20*60*30):
                output['VW'][(day-1)*n30m+t]=np.nanmean(vp*wp)
            if (np.sum(np.isnan(up*up))<.9*20*60*30):
                output['UU'][(day-1)*n30m+t]=np.nanmean(up*up)
                output['Ustr'][(day-1)*n30m+t]=np.nanmean(rot_u)
            if (np.sum(np.isnan(vp*vp))<.9*20*60*30):
                output['VV'][(day-1)*n30m+t]=np.nanmean(vp*vp)
                output['Vstr'][(day-1)*n30m+t]=np.nanmean(rot_v)
            if (np.sum(np.isnan(wp*wp))<.9*20*60*30):
                output['WW'][(day-1)*n30m+t]=np.nanmean(wp*wp)
                output['Wstr'][(day-1)*n30m+t]=np.nanmean(rot_w)
            if (np.sum(np.isnan(ustar))<.9*20*60*30):
                output['USTAR'][(day-1)*n30m+t]=ustar
            gc.collect()


        #print(logstr+str(tracemalloc.get_traced_memory()),flush=True)

        if (dt0+timedelta(days=doy+1)).day==1:
            #### OUTPUT MONTH ####
            outname='NEON_TW_'+dt_str[0:7]+'.h5'
            run('rm '+out_dir+site+'/'+outname,shell=True)
            fp=h5py.File(out_dir+site+'/'+outname,'w')
            for k in output.keys():
                fp.create_dataset(k,data=np.array(output[k][:]))
                fp[k].attrs['source']='NEON_dp0p'
                if k in units.keys():
                    fp[k].attrs['units']=units[k]
                if k in desc.keys():
                    fp[k].attrs['description']=desc[k]
            fp.close()






