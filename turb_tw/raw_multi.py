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
out_dir='/home/tsw35/soteria/data/NEON/multiscale/'
sites=['ALL']
years=['ALL']
scales=[2,5,10,15,20,45,60,90,120,180]
cvars=['U','V','W','Us','Vs','UU','UV','UW','VV','VW','WW',\
        'UsUs','VsVs','UsVs','UsW','VsW',
        'WTHETA','WQ','WC','THETATHETA','QQ','CC','QC','TQ','TC',\
        'THETAC','THETAQ','Q','C','PA','THETA','WT','TT','T','DMOLAIRDRY']
dp4type=['expanded','basic'][0]
center30=True # if true, for scales greater than 30, will use moving window centered on 30 minutes
debugp=True # if true, will print more failure messages and skipped days/months
overwrite=False # if true, will redo processing and overwrite old files

# timesteps per day for 1m and 30m intervals
n1m=1440
n30m=48
nday={} # timesteps per day
dn={} # raw timesteps per interval
for s in scales:
    dn[s]=20*60*s
    if (center30)&(s>30):
        nday[s]=n30m
    else:
        nday[s]=int(1440/s)

#### UNITS and DESCRIPTION ####
units={}
units['CO2FX_1m']='umolCo2 m-2 s-1'
desc={}

##########################
#### HELPER FUNCTIONS ####
##########################

def output_month(t_dir,t_dt_str,t_output):
    outname='NEON_TW_'+t_dt_str[0:7]+'.h5'
    run('rm '+t_dir+'/'+outname,shell=True)
    fp=h5py.File(t_dir+'/'+outname,'w')
    for k in t_output.keys():
        fp.create_dataset(k,data=np.array(t_output[k][:]))
        fp[k].attrs['source']='NEON_dp0p'
        if k in units.keys():
            fp[k].attrs['units']=units[k]
        if k in desc.keys():
            fp[k].attrs['description']=desc[k]
    fp.close()



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

# Make an array of site-year-months to loop through
st_yr=[]
mnths=['01','02','03','04','05','06','07','08','09','10','11','12']

for site in sites:
    rlist=os.listdir(raw_dir+site)
    rlist.sort()
    r0=rlist[0]
    olist=os.listdir(out_dir+site)
    for year in years:
        for mnth in mnths:
            dt_str=str(year)+'-'+mnth
            outname='NEON_TW_'+dt_str+'.h5'
            if (int(r0[33:37])>year):
                if rank==0:
                    print('F1: '+site+' '+outname)
            elif (int(r0[38:40])>int(mnth))&(int(r0[33:37])==year):
                if rank==0:
                    print('F2: '+site+' '+outname)
            elif outname in olist:
                if rank==0:
                    print('F3: '+site+' '+outname)
            else:
                st_yr.append(site+'_'+str(year)+'_'+mnth)

mprank=int(len(st_yr)/size+.999) # months per process

if rank==0:
    print()
    print('Setup Done... Total months to evaluate: '+str(len(st_yr))+' or per process: '+str(mprank))
    print('',flush=True)

mprocessed=0

#### CORE LOOP ####
for sty in st_yr[rank::size]:

    # get site, year, NEON domain and sonic level
    site=sty[0:4]
    year=int(sty[5:9])
    dt0=datetime(year,int(sty[10:12]),1,0)
    dom=domain[site]
    lvl=s_level[site]
    fp4=h5py.File(pfdir+site+'_pfvalues.nc','r')#nc.Dataset(pfdir+site+'_pfvalues.nc','r')
    _day=fp4['day'][:]
    _month=fp4['month'][:]
    _year=fp4['year'][:]
    _ax=fp4['ax'][:]
    _ay=fp4['ay'][:]
    _ofst=fp4['ofst'][:]
    fp4.close()

    # core action loop
    skip=False
    nmonth=monthrange(year,int(sty[10:12]))[1]
    for day in range(1,nmonth+1):
        dt=dt0+timedelta(days=day-1)
        dt_str=dt.strftime('%Y-%m-%d')

        #### DEAL WITH MONTHLY STUFF ####
        # if first of month, try to load in monthly dp4 file
        # otherwise, skip day if no dp4 file available
        # also output data from previous month if available, and initialize new
        if (day==1):

            # Timesteps in month
            NMn={}
            for s in scales:
                NMn[s]=nmonth*nday[s]

            # Initialize Output
            output={}

            for s in scales:
                for v in cvars:
                    output[v+'_'+str(s)+'m']=np.ones((NMn[s],))*float('nan')

            skip=False
            fname='NEON.D'+dom+'.'+site+'.DP4.00200.001.nsae.'+dt_str[0:7]+'.'+dp4type+'.h5'
            outname='NEON_TW_'+dt_str[0:7]+'.h5'
            if outname in os.listdir(out_dir+site):
                skip = True
                continue
        elif skip:
            continue

        #### LOAD IN IF EXPANDED ####
        try:
            idx=np.where((dt.day==_day)&(dt.year==_year)&(dt.month==_month))[0][0]
            if (np.sum(np.isnan(_ax[idx]))+np.sum(np.isnan(_ay[idx]))+np.sum(np.isnan(_ofst[idx])))>0:
                raise Exception('error')

        except Exception as e:
            if debugp:
                print('Skipping '+site+':'+str(rank)+':'+dt_str+' due to pf missing/filename error...\n'+str(e),flush=True)
            if (dt0+timedelta(days=day)).day==1:
                output_month(out_dir+site,dt_str,output)
            continue

        logstr=site+':'+str(rank)+':'+dt_str+':: '


        #### LOAD IN DATA ####
        # load in the raw turbulence file and data
        fname='NEON.D'+dom+'.'+site+'.IP0.00200.001.ecte.'+dt_str[0:10]+'.l0p.h5'
        try:
            fp0=h5py.File(raw_dir+site+'/'+fname,'r')
            anz=float(fp0[site+'/dp0p/data/soni/'].attrs['AngNedZaxs'][0])
            dp0_h2od=fp0[site+'/dp0p/data/irgaTurb/000_'+lvl+'/densMoleH2o'][:]
            dp0_pres=fp0[site+'/dp0p/data/irgaTurb/000_'+lvl+'/presAtm'][:]
            dp0_tm=fp0[site+'/dp0p/data/irgaTurb/000_'+lvl+'/tempMean'][:]
            dp0_h2o=fp0[site+'/dp0p/data/irgaTurb/000_'+lvl+'/rtioMoleDryH2o'][:]
            dp0_co2=fp0[site+'/dp0p/data/irgaTurb/000_'+lvl+'/rtioMoleDryCo2'][:]
            dp0_ts=fp0[site+'/dp0p/data/soni/000_'+lvl+'/veloSoni'][:]
            dp0_u=fp0[site+'/dp0p/data/soni/000_'+lvl+'/veloXaxs'][:]
            dp0_v=fp0[site+'/dp0p/data/soni/000_'+lvl+'/veloYaxs'][:]
            dp0_w=fp0[site+'/dp0p/data/soni/000_'+lvl+'/veloZaxs'][:]
        except Exception as e:
            if debugp:
                print('Skipping '+site+':'+str(rank)+':'+dt_str+' due to DP0P missing/filename error...\n'+str(e),flush=True)
            if (dt0+timedelta(days=day)).day==1:
                output_month(out_dir+site,dt_str,output)
            continue

        # adjust tvelo to tsoni
        dp0_ts=dp0_ts**2/1.4/(8.314462175/28.97/10**(-3))

        # fix nan situation
        dp0_u[np.abs(dp0_u)>50]=float('nan')
        dp0_v[np.abs(dp0_v)>50]=float('nan')
        dp0_w[np.abs(dp0_w)>50]=float('nan')
        dp0_ts[dp0_ts<235]=float('nan')
        dp0_h2o[dp0_h2o<0]=float('nan')
        dp0_co2[dp0_co2<.00025]=float('nan')
        dp0_co2[dp0_co2>.00055]=float('nan')

        # check all nans in u,v,w
        if (np.sum(np.isnan(dp0_u))>=.995*len(dp0_u)):
            if debugp:
                print('Skipping '+site+':'+dt_str+' due to raw data for U missing > 99.5% ',flush=True)
            if (dt0+timedelta(days=day)).day==1:
                output_month(out_dir+site,dt_str,output)
            continue
        elif (np.sum(np.isnan(dp0_v))>=.995*len(dp0_v)):
            if debugp:
                print('Skipping '+site+':'+dt_str+' due to raw data for V missing > 99.5% ',flush=True)
            if (dt0+timedelta(days=day)).day==1:
                output_month(out_dir+site,dt_str,output)
            continue
        elif (np.sum(np.isnan(dp0_w))>=.995*len(dp0_w)):
            if debugp:
                print('Skipping '+site+':'+dt_str+' due to raw data for W missing > 99.5% ',flush=True)
            if (dt0+timedelta(days=day)).day==1:
                output_month(out_dir+site,dt_str,output)
            continue

        print(logstr+'files loaded successfully',flush=True)


        #### R Workflow Beginning: DESPIKE and LAG CORRECT ####

        # convert to r object
        r_h2o=robjects.FloatVector(dp0_h2o.tolist())
        r_co2=robjects.FloatVector(dp0_co2.tolist())
        r_u=robjects.FloatVector(dp0_u.tolist())
        r_v=robjects.FloatVector(dp0_v.tolist())
        r_w=robjects.FloatVector(dp0_w.tolist())
        r_ts=robjects.FloatVector(dp0_ts.tolist())
        r_h2od=robjects.FloatVector(dp0_h2od.tolist())
        r_pres=robjects.FloatVector(dp0_pres.tolist())
        r_tm=robjects.FloatVector(dp0_tm.tolist())

        # despike
        try:
            out=Re4Rq.def_dspk_br86(r_h2o)
            r_h2o=out[-2]
            out=Re4Rq.def_dspk_br86(r_co2)
            r_co2=out[-2]
            out=Re4Rq.def_dspk_br86(r_u)
            r_u=out[-2]
            out=Re4Rq.def_dspk_br86(r_v)
            r_v=out[-2]
            out=Re4Rq.def_dspk_br86(r_w)
            r_w=out[-2]
            out=Re4Rq.def_dspk_br86(r_ts)
            r_ts=out[-2]
            out=Re4Rq.def_dspk_br86(r_h2od)
            r_h2od=out[-2]
            out=Re4Rq.def_dspk_br86(r_pres)
            r_pres=out[-2]
            out=Re4Rq.def_dspk_br86(r_tm)
            r_tm=out[-2]
        except Exception as e:
            if debugp:
                print(logstr+'ERROR despiking failed; will continue processing \n'+str(e),flush=True)

        ### LAG CORRECTION ###
        try:
            ld_pres=np.ones((1728000,))*float('nan')
            out=Re4Rb.def_lag(r_w,r_pres,20)
            lag_=np.array(out[1])
            ld_pres[0:len(lag_)]=lag_[:]
        except Exception as e:
            if debugp:
                print(logstr+'ERROR lag correction H2O failed; pressure will be reset\n'+str(e),flush=True)
            ld_pres=np.array(r_pres)

        try:
            ld_h2o=np.ones((1728000,))*float('nan')
            out=Re4Rb.def_lag(r_w,r_h2o,20)
            lag_=np.array(out[1])
            ld_h2o[0:len(lag_)]=lag_[:]
        except Exception as e:
            if debugp:
                print(logstr+'ERROR lag correction H2O failed; no latent heat flux will be computed\n'+str(e),flush=True)

        # lag correction for CO2
        try:
            ld_co2=np.ones((1728000,))*float('nan')
            out=Re4Rb.def_lag(r_w,r_co2,20)
            lag_=np.array(out[1])
            ld_co2[0:len(lag_)]=lag_[:]
        except Exception as e:
            if debugp:
                print(logstr+'ERROR lag correction H2O failed; no latent heat flux will be computed\n'+str(e),flush=True)

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

        ld_ts=np.array(r_ts)

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

        # compute specific humidity
        moldry=28.97*10**(-3)
        molh2o=18.02*10**(-3)
        Rg=8.314462175
        dmolair=ld_pres/Rg/np.array(r_tm)
        dmolairdry=np.array(dmolair)-np.array(r_h2od)
        q=np.array(r_h2od)*molh2o/(dmolairdry*moldry+np.array(r_h2od)*molh2o)

        # translate sonic temperature to air temperature
        ld_th=ld_ts[:]
        ld_ta=(ld_ts/(1+.51*q))

        # compute volumetric heat capacity
        cpdry=1004.64
        cph2o=1846
        cp_=cpdry*dmolairdry*moldry + cph2o*np.array(r_h2od)*molh2o

        # compute latent heat of vaporization
        lhv = 2500827 - 2360*(ld_ts-273)

        #### LOOP THROUGH THE INTERVALS ####
        # recall day-1= index of day of month

        for s in scales:
            for t in range(nday[s]):
                if (rank==0) and (size<=1):
                    print('.',end='',flush=True)

                # check if window big enough; compute starting and ending
                if (center30)&(s>30):
                    i0=int((t*30*20*60)+(15*20*60)-dn[s]/2)
                    ie=i0+dn[s]
                    if (i0<0)|(ie>nday[s]*60*20*30):
                        continue
                else:
                    i0=t*dn[s]
                    ie=(t+1)*dn[s]

                # rotate into the mean wind (eddy4R.turb def.rot.ang.zaxs.erth.R)
                uin=ld_u[i0:ie]
                vin=ld_v[i0:ie]

                up=uin-np.nanmean(uin)
                vp=vin-np.nanmean(vin)

                a_ertm=np.arctan2(np.nanmean(vin),np.nanmean(uin))

                rot_u=uin*np.cos(a_ertm)+vin*np.sin(a_ertm)
                rot_v=-uin*np.sin(a_ertm)+vin*np.cos(a_ertm)
                rot_w=np.array(ld_w[i0:ie])

                # extract other variables
                tain=ld_ta[i0:ie]
                thin=ld_th[i0:ie]
                qin=ld_h2o[i0:ie]
                cin=ld_co2[i0:ie]
                pin=ld_pres[i0:ie]

                # mean remove
                wp=rot_w-np.nanmean(rot_w)
                usp=rot_u-np.nanmean(rot_u)
                vsp=rot_v-np.nanmean(rot_v)
                thp=thin-np.nanmean(thin)
                tap=tain-np.nanmean(tain)
                qp=qin-np.nanmean(qin)
                cp=cin-np.nanmean(cin)

                # iterate through mean variables
                vtr={'Us':rot_u,'Vs':rot_v,'W':rot_w,'THETA':thin,'T':tain,'PA':pin,'Q':qin,'C':cin,'U':uin,'V':vin}
                for v in vtr.keys():
                    output[v+'_'+str(s)+'m'][(day-1)*nday[s]+t]=np.nanmean(vtr[v])

                # iterate through compound variables
                vtr={'THETA':thp,'T':tap,'Q':qp,'C':cp,'Us':usp,'Vs':vsp,'W':wp,'U':up,'V':vp}
                for v1 in vtr.keys():
                    for v2 in vtr.keys():
                        vstr=v1+v2+'_'+str(s)+'m'
                        if vstr in output.keys():
                            out=vtr[v1]*vtr[v2]
                            if np.sum(np.isnan(out))>.9*dn[s]:
                                continue
                            output[vstr][(day-1)*nday[s]+t]=np.nanmean(out)

                # finally, unique variables
                if np.sum(np.isnan(dmolairdry[i0:ie]))<.9*dn[s]:
                    output['DMOLAIRDRY_'+str(s)+'m'][(day-1)*nday[s]+t]=np.nanmean(dmolairdry[i0:ie])

            if (rank==0) and (size<=1):
                print('',flush=True)
            print(logstr+str(s)+' minute flux processing complete',flush=True)

        ###### OUTPUT MONTH #############
        if (dt0+timedelta(days=day)).day==1:
            output_month(out_dir+site,dt_str,output)

    mprocessed=mprocessed+1
    print()
    print(str(rank)+':_:_: SITEYEAR '+sty+' COMPLETE with '+str(mprocessed)+'/'+str(mprank)+' months done',flush=True)






