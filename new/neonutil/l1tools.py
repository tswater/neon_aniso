# Tools for L1 processing, mainly used by the L1 driver script
import numpy as np
import h5py
import datetime
import os
from subprocess import run
from nutil import SITES,nscale
import pytz

############# "PRIVATE" FUNCTIONS ###################
def _bij(uu,vv,ww,uv,uw,vw):
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

def _aniso(bij):
    N=bij.shape[0]
    yb=np.ones((N,))*-9999
    xb=np.ones((N,))*-9999
    for t in range(N):
        if np.sum(bij[t,:,:]==-9999)>0:
            continue
        if np.sum(np.isnan(bij[t,:,:]))>0:
            continue
        lams=np.linalg.eig(bij[t,:,:])[0]
        lams.sort()
        lams=lams[::-1]
        xb[t]=lams[0]-lams[1]+.5*(3*lams[2]+1)
        yb[t]=np.sqrt(3)/2*(3*lams[2]+1)
    return yb,xb

def _out_to_h5(_fp,_ov,overwrite):
    for k in _ov.keys():
        try:
            _fp.create_dataset(key,data=np.array(_ov[key][:]))
        except:
            if overwrite:
                _fp[key][:]=np.array(_ov[key][:])
            else:
                print('Skipping output of '+str(key))
        _fp[key].attrs['missing_value']=-9999
    _fp.attrs['last_updated_utc']=str(datetime.datetime.utcnow())

# convert NEON timestamp to seconds since 1970 utc
def _dpt2utc(tm):
    tm2=[]
    t0=datetime.datetime(1970,1,1,0,0)
    t0=pytz.utc.localize(t0)
    for t in tm:
        dt=datetime.strptime(str(t)[2:-1],"%Y-%m-%dT%H:%M:%S.000Z")
        dt=pytz.utc.localize(dt)
        tm2.append((dt-t0).total_seconds())
    return np.array(tm2)

############# MAKE BASE ###############
# make base h5 file to add onto for L1
def make_base(scl,odir,dlt=None,d0=None,df=None,overwrite=False,sites=SITES):
    ''' scl   averaging scale in minutes, int
        odir  directory for output
        d0    datetime for first timestep,
        df    datetime for last timestep
        dlt   time between each timestep, None defaults to scl
    '''
    if dlt in [None]:
        dlt=scl
    for site in sites:
        fname=file+'_L'+str(scl)+'.h5'
        if fname in os.listdir(odir):
            if overwrite:
                print('Replacing '+fname)
                try:
                    run('rm '+odir+fname,shell=True)
                except:
                    pass
            else:
                print('Skipping '+fname)
        else:
            print('Creating '+fname)
        fp=h5py.File(odir+fname,'w')

        fp.attrs['creation_time_utc']=str(datetime.datetime.utcnow())
        fp.attrs['last_updated_utc']=str(datetime.datetime.utcnow())
        fp.attrs['description']='NEON tower data and colocated data from '+\
                'other sources with a '+str(scl)+'min averaging period'

        # add time
        tzutc=datetime.timezone.utc
        times=[]
        if d0 in [None]:
            d0=datetime.datetime(start_year,start_month,1,0,0,tzinfo=tzutc)
        if df in [None]:
            df=datetime.datetime(end_year,end_month,end_day,23,30,tzinfo=tzutc)

        dt=datetime.timedelta(minutes=dlt)
        time=d0
        while time<=df:
            times.append(time.timestamp())
            time=time+dt
        fp.create_dataset('TIME',data=np.array(times))
        fp['TIME'].attrs['units']='seconds since 1970,UTC'

####################### ADD TURB WRAPPER ########################
# Wraper for adding turbulence information
def add_turb(scl,ndir,tdir,ivars=None,overwrite=False,dlt=None,sites=SITES):
    if (scl==30)|(scl==1):
        return add_turb24(scl,ndir,tdir,ivars,overwrite,dlt,sites)
    else:
        return add_turb25(scl,ndir,tdir,ivars,overwrite,dlt,sites)

def add_turb24(scl,ndir,tdir,ivars=None,overwrite=False,dlt=None,sites=SITES):
    print('TURB NOT ADDED because Tyler did not write this code yet')

###################### ADD TURB25 ###############################
# add turbulence from 2025 files
def add_turb25(scl,ndir,tdir,ivars=None,overwrite=False,dlt=None,sites=SITES):
    '''Add core turbulence stats from 2025 produced file (raw_multi.py)
       scl   : averaging scale in minutes
       ndir  : directory of L1 base files
       tdir  : directory of turbulence files from raw_multi.py
       ivars : variables to process; None will add all
    '''

    # setup
    _ivars = ['U','V','W','Ustr','Vstr','UU','UV','UW','VV','VW','WW',\
              'WTHETA','WQ','WC','THETATHETA','QQ','CC','QC','TQ','TC',\
              'THETAC','THETAQ','Q','C','PA','THETA','WT','TT','T',\
              'DMOLAIRDRY']
    if dlt in [None]:
        dlt=scl
    if ivars in [None]:
        ivars=_ivars
    outvar={}
    for var in ivars:
        if var in _ivars:
            outvar[var]=[]

    for site in sites:
        ovar=outvar.copy()
        fpo=h5py.File(ndir+site+'_L'+str(scl)+'.h5','r+')
        time=fpo['TIME'][:]
        for k in ovar.keys():
            ovar[k]=np.ones((len(time),))*-9999

        # loop through time
        for file in flist:
            fp_in=h5py.File(neon_dir+site+'/'+file,'r')
            stdt=datetime.datetime(int(file[8:12]),int(file[13:15]),1,0,tzinfo=datetime.timezone.utc)
            try:
                a=np.where(time==stdt.timestamp())[0][0]
            except Exception as e:
                print(e)
                continue

            add='_'+str(scl)+'m'
            N=len(fp_in['UU'+add][:])
            # fix timing for december
            if (dlt<30) and (stdt.year==2023) and (stdt.month==12):
                N=N-int(30/dlt)+1

            for var in ovar.keys():
                ovar[var][a:a+N]=fp_in[var+add][0:N]
        _out_to_h5(fpo,ovar,overwrite)



####################### ADD DERIVED #############################
# Add variables derived from basic/turb information already in h5
def add_derived(scl,ndir,ivars=None,overwrite=False,sites=SITES):
    ''' Add derived variables, such as anisotropy
       scl   : averaging scale in minutes
       ndir  : directory of L1 base files
       ivars : variables to process; None will add all
    '''
    #### SETUP
    _ivars = ['ANI_YB','ANID_YB','ANI_XB','ANID_XB','RH','TD','RHO',\
              'USTAR','H','LE','L_MOST']
    if ivars in [None]:
        ivars=_ivars
    outvar={}
    for var in ivars:
        if var in _ivars:
            outvar[var]=[]

    #### RUN ALL
    for site in sites:
        ovar=outvar.copy()
        fpo=h5py.File(ndir+site+'_L'+str(scl)+'.h5','r+')
        tmp={}
        for var in ovar.keys():
            match var:
                case 'ANI_YB':
                    if 'YB' in tmp.keys():
                        ovar[var]=tmp['YB'][:]
                    else:
                        rst=_bij(fpo['UU'][:],fpo['VV'][:],fpo['WW'][:],\
                                fpo['UV'][:],fpo['UW'][:],fpo['VW'][:])
                        yb,xb=_aniso(rst)
                        ovar[var]=yb
                        tmp['XB']=xb
                case 'ANID_YB':
                    if 'dYB' in tmp.keys():
                        ovar[var]=tmp['dYB'][:]
                    else:
                        cero=np.zeros((len(fpo['UU'][:]),))
                        rstd=_bij(fpo['UU'][:],fpo['VV'][:],fpo['WW'][:],\
                                cero,cero,cero)
                        yb,xb=_aniso(rstd)
                        ovar[var]=yb
                        tmp['dXB']=xb
                case 'ANI_XB':
                    if 'XB' in tmp.keys():
                        ovar[var]=tmp['XB'][:]
                    else:
                        rst=_bij(fpo['UU'][:],fpo['VV'][:],fpo['WW'][:],\
                                fpo['UV'][:],fpo['UW'][:],fpo['VW'][:])
                        yb,xb=_aniso(rst)
                        ovar[var]=xb
                        tmp['YB']=yb
                case 'ANID_XB':
                    if 'dXB' in tmp.keys():
                        ovar[var]=tmp['dXB'][:]
                    else:
                        cero=np.zeros((len(fpo['UU'][:]),))
                        rstd=_bij(fpo['UU'][:],fpo['VV'][:],fpo['WW'][:],\
                                cero,cero,cero)
                        yb,xb=_aniso(rstd)
                        ovar[var]=xb
                        tmp['dYB']=yb
                case 'RH':
                    if 'e' not in tmp.keys():
                        e=fpo['PA'][:]*fpo['Q'][:]/(1+fpo['Q'][:])
                        e[fpo['PA'][:]==-9999]=-9999
                        e[fpo['Q'][:]<=0]=-9999
                        tmp['e']=e
                    if 'svp' not in tmp.keys():
                        ts=fpo['THETA'][:]
                        svp=.61121*np.exp((18.678-(ts)/234.5)*((ts)/(257.14+(ts))))
                        svp[ts==-9999]=-9999
                        tmp['svp']=svp
                    e=tmp['e']
                    svp=tmp['svp']
                    rh=e/svp*100
                    rh[e==-9999]=-9999
                    rh[svp==-9999]=-9999
                    ovar['RH']=rh
                case 'TD':
                    a=17.27
                    b= 237.7
                    if 'e' not in tmp.keys():
                        e=fpo['PA'][:]*fpo['Q'][:]/(1+fpo['Q'][:])
                        e[fpo['PA'][:]==-9999]=-9999
                        e[fpo['Q'][:]<=0]=-9999
                        tmp['e']=e
                    if 'svp' not in tmp.keys():
                        ts=fpo['THETA'][:]
                        svp=.61121*np.exp((18.678-(ts)/234.5)*((ts)/(257.14+(ts))))
                        svp[ts==-9999]=-9999
                        tmp['svp']=svp
                    e=tmp['e']
                    svp=tmp['svp']
                    rh=e/svp*100
                    ta=fpo['T'][:]
                    gam=np.log(rh/100)+(a*ta)/(b+ta)
                    td=b*gam/(a-gam)
                    ovar['TD']=td
                case 'RHO':
                    if 'e' not in tmp.keys():
                        e=fpo['PA'][:]*fpo['Q'][:]/(1+fpo['Q'][:])
                        e[fpo['PA'][:]==-9999]=-9999
                        e[fpo['Q'][:]<=0]=-9999
                        tmp['e']=e
                    if 'svp' not in tmp.keys():
                        ts=fpo['THETA'][:]
                        svp=.61121*np.exp((18.678-(ts)/234.5)*((ts)/(257.14+(ts))))
                        svp[ts==-9999]=-9999
                        tmp['svp']=svp
                    e=tmp['e']
                    svp=tmp['svp']
                    Ra = 286.9
                    Rw = 461.5
                    ts = fpo['THETA'][:]
                    rho = ((fpo['PA'][:]-e)/(Ra*(ts+273))+(e)/(Rw*(ts+273)))*1000
                    rho[fpo['PA'][:]<=0]=-9999
                    rho[ts<=-25]=-9999
                    ovar['RHO']=rho
                case 'USTAR':
                    if 'ustar' not in tmp.keys():
                        ustar=(fpo['UW'][:]**2+fpo['VW'][:]**2)**(1/4)
                        tmp['ustar']=ustar
                    ustar=tmp['ustar']
                    ovar['USTAR']=ustar
                case 'H':
                    wth=fpo['WTHETA'][:]
                    cpdry=1004.64
                    moldry=28.97*10**(-3)
                    cph2o=1846
                    molh2o=18.02*10**(-3)
                    dmair=fpo['DMOLAIRDRY'][:]
                    cp_=cpdry*dmair*moldry + cph2o*fpo['Q'][:]*molh2o
                    ovar['H']=cp_*wth
                case 'LE':
                    lhv = 2500827 - 2360*(fpo['T'][:]-273)
                    molh2o=18.02*10**(-3)
                    le=fpo['DMOLAIRDRY'][:]*lhv*molh2o*fpo['WQ'][:]
                    ovar['LE']=le
                case 'L_MOST':
                    if 'ustar' not in tmp.keys():
                        ustar=(fpo['UW'][:]**2+fpo['VW'][:]**2)**(1/4)
                        tmp['ustar']=ustar
                    ustar=tmp['ustar']
                    L=-ustar**3*fpo['THETA'][:]/(.4*9.81*fpo['WTHETA'][:])
                    ovar['L_MOST']=L
        _out_to_h5(fpo,ovar,overwrite)


####################### ADD STATIC ATTRIBUTES #############################
# Add static attributes by copying them from other averaging scales
def add_core_attrs(scl,ndir,nbdir=None,bscl=30,ivars=None,sites=SITES):
    ''' Add static site attributes
        scl   : averaging scale in minutes
        ndir  : directory of L1 base files
        nbdir : directory of base files in another scale that have attrs
               if None, will try to recompute them
        bscl  : as with nbdir, but scale. Assumes 30
        ivars : variables to process; None will add all
    '''
    # for now, we assume we already have these
    #### SETUP
    _ivars = ['canopy_height','elevation','lat','lon','lvls','lvls_u',
              'nlcd_dom','tow_height','utc_off','zd','nlcdXX']
    pctn=[11,12,21,22,23,24,31,42,43,51,52,71,72,73,74,81,82,90,95]
    if ivars in [None]:
        ivars=_ivars
    outvar={}
    for var in ivars:
        if var=='nlcdXX':
            for i in pctn:
                outvar['nlcd'+str(i)]=''
        elif var in _ivars:
            outvar[var]=''

    if nbdir==None:
        print('Base directory not specified; recomputing static variables '+\
                'is currently not supported. FAILURE')
        return None

    #### RUN ALL
    for site in sites:
        fpo=h5py.File(ndir+site+'_L'+str(scl)+'.h5','r+')
        fpi=h5py.File(nbdir+site+'_L'+str(bscl)+'.h5','r')
        for k in outvar.keys():
            fpo.attrs[k]=fpi.attrs[k]
        fpo.attrs['last_updated_utc']=str(datetime.datetime.utcnow())



##################################################################
def add_profile(scl,ndir,idir1,idir2=None,new=False,addprof=True,addqaqc=True,\
                ivars=None,overwrite=False,sites=SITES):
    ''' Add profile information (new or old)
        idir1   : if new, dp4 directory, else 1m L1 directory
        idir2   : if new, 2dwind directory, else None
        new     : if True, will recompute them, if false will pull from 1m
        addprof : if True, will add actual profile, if false will skip
        addqaqc : if True, will add qaqc flags to L1, if false will skip
    '''
    if new:
        add_profile_wind(scl,ndir,idir2,addprof,addqaqc,ivars,overwrite,sites)
        add_profile_tqc(scl,ndir,idir1,addprof,addqaqc,ivars,overwrite,sites)
    else:
        add_profile_old(scl,ndir,idir1,addprof,addqaqc,ivars,overwrite,sites)

def add_profile_old(scl,ndir,idir,addprof=True,addqaqc=True,\
                ivars=None,overwrite=False,sites=SITES):
    # add profiles using the 1 minute file
    _ivars = ['profile_t','profile_q','profile_c','profile_u']
    if ivars in [None]:
        ivars=_ivars
    outvar={}
    for var in ivars:
        if var in _ivars:
            outvar[var]={}

    #### RUN ALL
    for site in sites:
        ovar=outvar.copy()
        fpo=h5py.File(ndir+site+'_L'+str(scl)+'.h5','r+')
        fpi=h5py.File(idir+site_'_L1.h5','r')
        tout=fpo['TIME'][:]
        tin=fpi['TIME'][:]

        # copy attrs
        fpo.attrs['lvls']=fpi.attrs['lvls']
        fpo.attrs['lvls_u']=fpi.attrs['lvls_u']

        # create hdf5 group
        for var in ovar.keys():
            if overwrite:
                try:
                    del fpo[var]
                except:
                    pass
            try:
                fpo.create_group(var)
            except:
                pass
            for v2 in fpi[var].keys():
                # FIXME
                raise Exception('I gave up implementing this; use add_profile_tqc or add_profile_wind')
    # FIXME I gave up implementing this


def add_profile_tqc(scl,ndir,dp4dir,addprof=True,addqaqc=True,ivars=None,\
                    overwrite=False,sites=SITES):
    ''' Add profile information '''
    # add profiles from scratch
    _ivars = ['profile_t','profile_q','profile_c']
    if ivars in [None]:
        ivars=_ivars
    outvar={}
    for var in ivars:
        if var in _ivars:
            if addprof:
                outvar[var]={}
            if addqaqc:
                outvar['q'+var]=[]
                outvar['q'+var+'_upper']=[]


    #### RUN ALL (TQC)
    for site in sites:
        ovar=outvar.copy()
        fpo=h5py.File(ndir+site+'_L'+str(scl)+'.h5','r+')
        time=fpo['TIME'][:]
        dlt=int(np.min(time[1:]-time[:-1])/60)
        filelist=os.listdir(dp4dir+site)
        filelist.sort()
        inp={'t':{},'q':{},'c':{},'qq':{},'qt':{},'qc':{},\
                't_t':{},'t_q':{},'t_c':{},'ttop':[],'t_ttop':[],'qttop':[]}

        # identify levels, initiate full input arrays
        fpi=h5py.File(dp4dir+site+'/'+filelist[-1],'r')
        lvltqc0=fpi[site].attrs['DistZaxsLvlMeasTow'][:]
        lvltqc=[]
        for x in lvltqc0:
            lvltqc.append(float(x))
        lvltqc.sort()
        for i in range(len(lvltqc)-1):
            inp['t'][i]=[]
            inp['q'][i]=[]
            inp['c'][i]=[]
            inp['qq'][i]=[]
            inp['qt'][i]=[]
            inp['qc'][i]=[]
            inp['t_t'][i]=[]
            inp['t_q'][i]=[]
            inp['t_c'][i]=[]

        # determine input scale appropriate
        if dlt==30:
            s1='000_0'+str(i+1)+'0_30m'
            s2='000_0'+str(i+1)+'0_2m'
        else:
            s1='000_0'+str(i+1)+'0_1m'
            s2='000_0'+str(i+1)+'0_2m'


        # fill input arrays
        ts='temp'
        qs='rtioMoleDryH2o'
        cs='rtioMoleDryCo2'
        for file in filelist:
            if '2024' in file:
                continue
            if '.h5.gz' in file:
                continue
            fpi=h5py.File(dp4dir+site+'/'+file,'r')
            for i in range(len(lvltqc)-1):
                if dlt==30:
                    s1='000_0'+str(i+1)+'0_30m'
                    s2='000_0'+str(i+1)+'0_02m'
                else:
                    s1='000_0'+str(i+1)+'0_01m'
                    s2='000_0'+str(i+1)+'0_02m'
                inp['t'][i].extend(fpi[site]['dp01/data/tempAirLvl'][s1][ts][:]['mean'])
                inp['q'][i].extend(fpi[site]['dp01/data/h2oStor'][s2][qs][:]['mean'])
                inp['c'][i].extend(fpi[site]['dp01/data/co2Stor'][s2][cs][:]['mean'])
                inp['qq'][i].extend(fpi[site]['dp01/qfqm/h2oStor'][s2][qs][:]['qfFinl'])
                inp['qt'][i].extend(fpi[site]['dp01/qfqm/tempAirLvl'][s1][ts][:]['qfFinl'])
                inp['qc'][i].extend(fpi[site]['dp01/qfqm/co2Stor'][s2][cs][:]['qfFinl'])
                tt1=_dpt2utc(fpi[site]['dp01/data/tempAirLvl'][s1][ts][:]['timeBgn'][:])
                tt2=_dpt2utc(fpi[site]['dp01/data/tempAirLvl'][s1][ts][:]['timeEnd'][:])
                tq1=_dpt2utc(fpi[site]['dp01/data/h2oStor'][s2][qs][:]['timeBgn'][:])
                tq2=_dpt2utc(fpi[site]['dp01/data/h2oStor'][s2][qs][:]['timeEnd'][:])
                tc1=_dpt2utc(fpi[site]['dp01/data/co2Stor'][s2][cs][:]['timeBgn'][:])
                tc2=_dpt2utc(fpi[site]['dp01/data/co2Stor'][s2][cs][:]['timeEnd'][:])
                inp['t_t'][i].extend(list((tt1+tt2)/2))
                inp['t_q'][i].extend(list((tq1+tq2)/2))
                inp['t_c'][i].extend(list((tc1+tc2)/2))
            if dlt==30:
                s1='000_0'+str(len(lvltqc))+'0_30m'
            else:
                s1='000_0'+str(len(lvltqc))+'0_01m'
            inp['ttop'].extend(fpi[site]['dp01/data/tempAirTop'][s1][ts][:]['mean'])
            inp['qttop'].extend(fpi[site]['dp01/qfqm/tempAirTop'][s1][ts][:]['qfFinl'])
            tt1=_dpt2utc(fpi[site]['dp01/data/tempAirTop'][s1][ts][:]['timeBgn'][:])
            tt2=_dpt2utc(fpi[site]['dp01/data/tempAirTop'][s1][ts][:]['timeEnd'][:])
            inp['t_ttop'].extend(list((tt1+tt2)/2))

        # make time middle not begining time
        time2=time+scl/2*60

        # Interpolate
        vs=[]
        for k in ovar.keys():
            a=k[-1]
            if a not in vs:
                vs.append(a)

        for v in vs:
            v_long='profile_'+v
            for i in range(len(lvltqc)-1):
                if addprof:
                    ovar[v_long][v.upper()+str(i)]=nscale(time2,inp['t_'+v][:],inp[v][:])
                if addqaqc:
                    ovar['q'+v_long][v.upper()+str(i)]=nscale(time2,inp['t_'+v][:],inp['q'+v][:])
        if 't' in vs:
            v_long='profile_t'
            if addprof:
                ovar[v_long][v.upper()+str(len(lvltqc)-1)]=\
                        nscale(time2,inp['t_ttop'][:],inp['ttop'][:])
            if addqaqc:
                ovar['q'+v_long][v.upper()+str(len(lvltqc)-1)]=\
                        nscale(time2,inp['t_ttop'][:],inp['qttop'][:])

        # Output
        # need to output both types of qprofiles, create structure etc
        # FIXME need to remove qprofile from ovar as this 2D series is
        #       not what we want to output, rather 2 1D series
        # FIXME write output





def add_profile_wind(scl,ndir,wndir,addprof=True,addqaqc=True,ivars=None,\
                    overwrite=False,sites=SITES):


##################################################################
def add_radiation():
    ''' Add radiation information '''


##################################################################
def add_qaqcflags():
    ''' Add qaqc flags from dp04 files '''

##################################################################
def add_pheno():
    ''' Add information from phenocam '''


##################################################################
def remove_variable():
    ''' Remove a given variable '''


##################################################################
def update_variable():
    ''' Update a variable name, add description or units'''

##################################################################
def add_precip():
    ''' Add precipitation information '''

##################################################################
def add_aop():
    ''' Add spatial variables that derives from flights '''




# Notes:

# Divide...
# attrs:
# qualityflags
# radiation
# vegetation
# secondary

# Need to Add
# ANI_XB, ANI_YB, G, RCC90, GCC90, H, LE, SW_IN, SW_OUT LW_IN, LW_OUT, NETRAD, NDVI90,
# P, RH, RHO, SOLAR_ALTITUDE, TIME, USTAR, VPT, VPD

# qCO2, qCO2FX, qH, qH2O, qLE, qT_SONIC, qU, qUSTAR, qV, qW, qprofile_c_upper, qprofile_c, ... qsT_SONIC, qsU, qsV, qsW, qsCO2, qsH2O,

# attrs: nlcdXX, nlcd_dom, tow_height, utc_off, zd, lvls, lvls_u, lat, lon, canopy_height, elevation,

