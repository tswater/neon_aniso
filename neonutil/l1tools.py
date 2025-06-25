# Tools for L1 processing, mainly used by the L1 driver script
import numpy as np
import h5py
import datetime
import os
from subprocess import run
try:
    from nutil import SITES,nscale,sort_together
except:
    from neonutil.nutil import SITES,nscale,sort_together
import pytz
import csv

############# "PRIVATE" FUNCTIONS ###################
def _confirm_user(msg):
    while True:
        user_input = input(f"{msg} (Y/N): ").strip().lower()
        if user_input in ('y', 'yes'):
            return True
        elif user_input in ('n', 'no'):
            return False
        else:
            print("Invalid input. Please enter 'Y' or 'N'.")


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

# convert NEON timestamp to seconds since 1970 utc
def _dpt2utc(tm,source='neon_h5'):
    # source of data should be neon or pheno (for phenocam network)
    tm2=[]
    t0=datetime.datetime(1970,1,1,0,0)
    t0=pytz.utc.localize(t0)
    for t in tm:
        if source=='neon_h5':
            strt=str(t)[2:-1]
            strt=strt[0:20]+'000'+strt[23:]
            dt=datetime.datetime.strptime(strt,"%Y-%m-%dT%H:%M:%S.000Z")
        elif source=='neon_csv':
            strt=str(t)
            dt=datetime.datetime.strptime(strt,"%Y-%m-%dT%H:%M:%SZ")
        elif source=='pheno':
            dt=datetime.datetime.strptime(str(t),"%Y-%m-%d")
        dt=pytz.utc.localize(dt)
        tm2.append((dt-t0).total_seconds())
    return np.array(tm2)

def _load_csv_data(innames,inp,req=None):
    if (req!=None) & (os.path.isdir(inp)):
        return _load_csv_data_neon(innames,inp,req)
    elif (not os.path.isdir(inp)) & (req==None):
        return _load_csv_data_pheno(innames,inp)

def _load_csv_data_pheno(innames,ifile):
    # find header (first row without "#" in beginning)
    # find values (filtering NA)
    # convert what you can to floats
    # adjust time by utcoffset (if its daily, we want the time to be middle of day local time)
    out={}
    pos={}
    tmp={}
    for i in innames:
        out[i]=[]
    out['time']=[]
    with open(ifile) as read_r:
        read_r = csv.reader(read_r)
        head=True
        utcoff=0
        for row in read_r:
            if head:
                if len(row)<3:
                    if 'UTC' in row[0]:
                        try:
                            utcoff=float(row[0][-3:])
                        except Exception:
                            pass
                else:
                    head=False
                    header=row.copy()
                    # get positions in row for variables and get time type
                    for k in out.keys():
                        if k=='time':
                            # 3 options for time: transition_10 [2], date [0], (local_std_time + date) [1]
                            try:
                                pos['date']=header.index('date')
                                tmp['date']=[]
                                timetype=0
                            except Exception:
                                pos['transition_10']=header.index('transition_10')
                                tmp['transition_10']=[]
                                timetype=2
                            try:
                                pos['local_std_time']=header.index('local_std_time')
                                tmp['local_std_time']=[]
                                timetype=1
                            except Exception:
                                pass
                        else:
                            try:
                                pos[k]=header.index(k)
                            except Exception as e:
                                print(e)
                                print('header: '+str(header))
                                raise e
            # we are past the header
            else:
                for k in pos.keys():
                    # load in the data
                    val=row[pos[k]]
                    if val in ['NA',' NA','NA ','']:
                        val=float('nan')
                    if k not in ['direction','date','transition_10','local_std_time']:
                        val=float(val)

                    # place the data
                    if k in tmp.keys():
                        tmp[k].append(val)
                    if k in out.keys():
                        out[k].append(val)
    # now deal with time shit
    # timetype 0: convert to datetime, then offset by utc
    # timetype 1: convert date to datetime, add local time, then offset by utc
    # timetype 2: same as timetype 0 but with a different name
    # ALL: load into time
    if timetype in [0]:
        tme = _dpt2utc(tmp['date'],source='pheno') - utcoff*60*60
        tme = tme + 12*60*60
    if timetype in [1]:
        tme = _dpt2utc(tmp['date'],source='pheno') - utcoff*60*60
        hrs=[]
        mins=[]
        for i in range(len(tmp['local_std_time'])):
            hrs.append(int(tmp['local_std_time'][i][0:2]))
            mins.append(int(tmp['local_std_time'][i][3:5]))
        tme = tme+np.array(hrs)*60*60
        tme = tme+np.array(mins)*60
    if timetype in [2]:
        tme = _dpt2utc(tmp['transition_10'],source='pheno') - utcoff*60*60
        tme = tme + 12*60*60
    out['time']=tme
    return out


def _load_csv_data_neon(innames,idir,req):
    # innames is a list of names to input
    # idir is directory of input
    # req is a list of file name requirements; i.e. req=['_30min'] will only
    #   load files that contain that _30min in file name
    # WARNING: will not work for phenocam
    out={}
    pos={}
    for i in innames:
        out[i]=[]
    out['startDateTime']=[]
    out['endDateTime']=[]
    flist=os.listdir(idir)
    flist.sort()
    filelist=[]
    for file in flist:
        good=True
        for r in req:
            good=good&(r in file)
        if good:
            filelist.append(file)
    for file in filelist:
        with open(idir+'/'+file,encoding='latin-1') as read_r:
            read_r = csv.reader(read_r)
            head=True
            for row in read_r:
                if head:
                    if len(row)>2:
                        header=row.copy()
                        head=False
                        for k in out.keys():
                            try:
                                pos[k]=header.index(k)
                            except Exception as e:
                                print('ERROR in CSV reading with '+k+\
                                      ' variable not in csv for '+idir+'/'+file)
                                raise e

                    else:
                        pass
                    continue
                for k in out.keys():
                    try:
                        a=row[pos[k]]
                    except Exception:
                        a='nan'
                    try:
                        b=float(a)
                    except Exception:
                        if 'Time' in k:
                            b=a
                        else:
                            b=float('nan')
                    out[k].append(b)

    # final checks
    for k in out.keys():
        if 'Time' in k:
            out[k]=_dpt2utc(out[k],source='neon_csv')
    return out

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

def _out_to_h5(_fp,_ov,overwrite,desc={}):
    for k in _ov.keys():
        if type(_ov[k]) is dict:
            ov2=_ov[k]
            if k not in _fp.keys():
                _f=_fp.create_group(k)
            else:
                _f=_fp[k]
            _out_to_h5(_f,ov2,overwrite)

        else:
            try:
                _fp.create_dataset(k,data=np.array(_ov[k][:]))
            except:
                if overwrite:
                    _fp[k][:]=np.array(_ov[k][:])
                else:
                    print('Skipping output of '+str(k))
            _fp[k].attrs['missing_value']=-9999
            if k in desc.keys():
                _fp[k].attrs['description']=desc[k]
    _fp.attrs['last_updated_utc']=str(datetime.datetime.utcnow())
    return None

def _get_qaqc_sci(fp,nm,n):
    try:
        return fp[nm]['qfSci'][:]
    except Exception:
        try:
            return fp[nm]['qfSciRevw'][:]
        except Exception:
            return np.ones((n,))*-1

######################### MAKE BASE ##############################
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
        fname=site+'_'+str(scl)+'m.h5'
        if fname in os.listdir(odir):
            if overwrite:
                print(site+': Replacing '+fname)
                try:
                    run('rm '+odir+fname,shell=True)
                except:
                    pass
            else:
                print(site+': Skipping '+fname)
        else:
            print(site+': Creating '+fname)
        fp=h5py.File(odir+fname,'w')

        fp.attrs['creation_time_utc']=str(datetime.datetime.utcnow())
        fp.attrs['last_updated_utc']=str(datetime.datetime.utcnow())
        fp.attrs['description']='NEON tower data and colocated data from '+\
                'other sources with a '+str(scl)+'min averaging period'

        # add time
        tzutc=datetime.timezone.utc
        times=[]
        if d0 in [None]:
            d0=datetime.datetime(2017,1,1,0,0,tzinfo=tzutc)
        if df in [None]:
            df=datetime.datetime(2023,12,31,23,30,tzinfo=tzutc)

        dt=datetime.timedelta(minutes=dlt)
        time=d0
        while time<=df:
            times.append(time.timestamp())
            time=time+dt
        fp.create_dataset('TIME',data=np.array(times))
        fp['TIME'].attrs['units']='seconds since 1970,UTC'

####################### ADD TURB WRAPPER ########################
# Wraper for adding turbulence information
def add_turb(scl,ndir,tdir,ivars=None,overwrite=False,dlt=None,sites=SITES,debug=False):
    if (scl==30)|(scl==1):
        return add_turb24(scl,ndir,tdir,ivars,overwrite,dlt,sites)
    else:
        return add_turb25(scl,ndir,tdir,ivars,overwrite,dlt,sites,debug=debug)

def add_turb24(scl,ndir,tdir,ivars=None,overwrite=False,dlt=None,sites=SITES):
    print('TURB NOT ADDED because Tyler did not write this code yet')

###################### ADD TURB25 ###############################
# add turbulence from 2025 files
def add_turb25(scl,ndir,tdir,ivars=None,overwrite=False,dlt=None,sites=SITES,debug=False):
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
        fpo=h5py.File(ndir+site+'_'+str(scl)+'m.h5','r+')
        time=fpo['TIME'][:]
        for k in ovar.keys():
            ovar[k]=np.ones((len(time),))*-9999

        flist=os.listdir(tdir+site)
        flist.sort()

        # loop through time
        for file in flist:
            fp_in=h5py.File(tdir+site+'/'+file,'r')
            stdt=datetime.datetime(int(file[8:12]),int(file[13:15]),1,0,tzinfo=datetime.timezone.utc)
            try:
                a=np.where(time==stdt.timestamp())[0][0]
            except Exception as e:
                print(site+'::'+file+'::'+str(e))
                continue

            add='_'+str(scl)+'m'
            N=len(fp_in['UU'+add][:])
            # fix timing for december
            if (dlt<30) and (stdt.year==2023) and (stdt.month==12):
                N=N-int(30/dlt)+1

            for var in ovar.keys():
                ovar[var][a:a+N]=fp_in[var+add][0:N]
        if debug:
            dbgout=':::::::::::::::::::DEBUG:::::::::::::::::::::\n'
            dbgout=dbgout+'Printing turb variables and their means for output at '+str(site)+'\n'
            for var in ovar.keys():
                cnt=np.nanmedian(ovar[var][ovar[var]!=-9999])
                dbgout=dbgout+var+':   '+str(cnt)+'\n'
            print(dbgout+':::::::::::::::::::DEBUG::::::::::::::::::::::',flush=True)
        _out_to_h5(fpo,ovar,overwrite)

##################### ADD STATIONARITY #############################
# Stationarity metric for variances based on Foken 2017 Micrometerology
# as found in Zahn 2023 Relaxed eddy accumulation...
def add_stationarity_zahn23(ndir5,ndir30,ivars=None,overwrite=False,sites=SITES):
    _ivars = ['QQ', 'THETATHETA', 'CC', 'UU', 'VV','WW']
    if ivars in [None]:
        ivars=_ivars
    outvar={}
    for var in ivars:
        if var in _ivars:
            outvar['ST_ZH23_'+var]=[]
    if len(outvar.keys())==0:
        print('add_stationarity_zahn23: No valid variables in ivars')
        return None

    for site in sites:
        fp5=h5py.File(ndir5+site+'_5m.h5','r+')
        fp30=h5py.File(ndir30+site+'_30m.h5','r+')
        time5=fp5['TIME'][:]+2.5*60
        time30=fp30['TIME'][:]+15*60
        idx5in=[]

        for i in range(len(time5)):
            t=time5[i]
            if t in time30:
                idx5in.append(i)
                time30=time30[1:]
        for var in ivars:
            xx5=fp5[var][:][idx5in]
            xx30=fp30[var][:][idx5in]
            outvar['ST_ZH23_'+var]=np.abs((xx5-xx30)/(xx30))*100
        _out_to_h5(fp30,ovar,overwrite)


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
        fpo=h5py.File(ndir+site+'_'+str(scl)+'m.h5','r+')
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
                        ts=fpo['THETA'][:]-273.15
                        svp=611.21*np.exp((18.678-(ts)/234.5)*((ts)/(257.14+(ts))))
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
                        ts=fpo['THETA'][:]-273.14
                        svp=611.21*np.exp((18.678-(ts)/234.5)*((ts)/(257.14+(ts))))
                        svp[ts==-9999]=-9999
                        tmp['svp']=svp
                    e=tmp['e']
                    svp=tmp['svp']
                    rh=e/svp*100
                    ta=fpo['T'][:]
                    gam=np.log(rh/100)+(a*ta)/(b+ta)
                    td=b*gam/(a-gam)
                    td[e==-9999]=-9999
                    td[svp==-9999]=-9999
                    ovar['TD']=td
                case 'RHO':
                    if 'e' not in tmp.keys():
                        e=fpo['PA'][:]*fpo['Q'][:]/(1+fpo['Q'][:])
                        e[fpo['PA'][:]==-9999]=-9999
                        e[fpo['Q'][:]<=0]=-9999
                        tmp['e']=e
                    if 'svp' not in tmp.keys():
                        ts=fpo['THETA'][:]-273.15
                        svp=611.21*np.exp((18.678-(ts)/234.5)*((ts)/(257.14+(ts))))
                        svp[ts==-9999]=-9999
                        tmp['svp']=svp
                    e=tmp['e']
                    svp=tmp['svp']
                    Ra = 286.9
                    Rw = 461.5
                    ts = fpo['THETA'][:]-273.15
                    rho = ((fpo['PA'][:]-e)/(Ra*(ts+273.15))+(e)/(Rw*(ts+273.15)))*1000
                    rho[fpo['PA'][:]<=0]=-9999
                    rho[ts<=-25]=-9999
                    rho[e==-9999]=-9999
                    rho[svp==-9999]=-9999
                    ovar['RHO']=rho
                case 'USTAR':
                    if 'ustar' not in tmp.keys():
                        ustar=(fpo['UW'][:]**2+fpo['VW'][:]**2)**(1/4)
                        ustar[fpo['UW'][:]==-9999]=-9999
                        ustar[fpo['VW'][:]==-9999]=-9999
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
                    H=cp_*wth
                    H[wth==-9999]=-9999
                    H[dmair<0]=-9999
                    H[fpo['Q'][:]==-9999]=-9999
                    ovar['H']=H
                case 'LE':
                    lhv = 2500827 - 2360*(fpo['T'][:]-273)
                    molh2o=18.02*10**(-3)
                    dmair=fpo['DMOLAIRDRY'][:]
                    le=dmair*lhv*molh2o*fpo['WQ'][:]
                    le[fpo['WQ'][:]==-9999]=-9999
                    le[dmair<0]=-9999
                    le[fpo['T'][:]<0]=-9999
                    ovar['LE']=le
                case 'L_MOST':
                    if 'ustar' not in tmp.keys():
                        ustar=(fpo['UW'][:]**2+fpo['VW'][:]**2)**(1/4)
                        ustar[fpo['UW'][:]==-9999]=-9999
                        ustar[fpo['VW'][:]==-9999]=-9999
                        tmp['ustar']=ustar
                    ustar=tmp['ustar']
                    L=-ustar**3*fpo['THETA'][:]/(.4*9.81*fpo['WTHETA'][:])
                    L[ustar==-9999]=-9999
                    L[fpo['WTHETA'][:]==-9999]=-9999
                    L[fpo['THETA'][:]==-9999]=-9999
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
        fpo=h5py.File(ndir+site+'_'+str(scl)+'m.h5','r+')
        fpi=h5py.File(nbdir+site+'_'+str(bscl)+'m.h5','r')
        for k in outvar.keys():
            fpo.attrs[k]=fpi.attrs[k]
        fpo.attrs['last_updated_utc']=str(datetime.datetime.utcnow())



##################################################################
################### ADD PROFILE WRAPPER ##########################
# Wrapper that calls appropriate add profile based on desired method
def add_profile(scl,ndir,idir1,idir2=None,new=True,addprof=True,addqaqc=True,\
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
    raise Exception('To Be Implemented; use add_profile_tqc or add_profile_wind')

    #### RUN ALL
    for site in sites:
        ovar=outvar.copy()
        fpo=h5py.File(ndir+site+'_'+str(scl)+'m.h5','r+')
        fpi=h5py.File(idir+site+'_1m.h5','r')
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
                pass
    # FIXME I gave up implementing this

########################################################################
################## ADD PROFILE TQC ####################################
# Adds the vertical profiles of temperature, water vapor and co2
def add_profile_tqc(scl,ndir,dp4dir,addprof=True,addqaqc=True,ivars=None,\
                    overwrite=False,debug=False,sites=SITES):
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

    if len(outvar.keys())==0:
        print('add_profile_tqc: No valid variables in ivars')
        return None

    #### RUN ALL (TQC)
    for site in sites:
        if debug:
            dbg='::::::::::::::::DEBUG:::::::::::::::::::\n'
            dbg=dbg+'Loading in tqc_profile data for '+site+'\n'
            print(dbg+':::::::::::::::::DEBUG::::::::::::::::::',flush=True)
        ovar=outvar.copy()
        fpo=h5py.File(ndir+site+'_'+str(scl)+'m.h5','r+')
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
                tt2[tt2<tt1]=tt2[0]-tt1[0]+tt1[tt2<tt1]
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
            tt2[tt2<tt1]=tt2[0]-tt1[0]+tt1[tt2<tt1]
            inp['t_ttop'].extend(list((tt1+tt2)/2))

        # make time middle not begining time
        time2=time+scl/2*60

        # Interpolate
        vs=[]
        for k in ovar.keys():
            a=k[-1]
            if 'qprofile' in k:
                ovar[k]=np.zeros((len(time),))
            elif a not in vs:
                vs.append(a)
        if debug:
            print('::::::::::::::::::DEBUG:::::::::::::::')
            print('vs: '+str(vs))
            print('inp: '+str(inp.keys()))
            for k in inp.keys():
                if type(inp[k]) is dict:
                    print('    '+str(k)+': '+str(inp[k].keys()))
                    for kk in inp[k]:
                        print('       '+str(kk)+':'+str(len(inp[k][kk])))
                else:
                    print('    '+str(k)+': '+str(len(inp[k])))
            print('::::::::::::::::::DEBUG:::::::::::::::',flush=True)
        for v in vs:
            v_long='profile_'+v
            for i in range(len(lvltqc)-1):
                if addprof:
                    if debug:
                        print('addprof for '+v_long)
                    ovar[v_long][v.upper()+str(i)]=\
                            nscale(time2,inp['t_'+v][i][:],inp[v][i][:],nearest=False,debug=debug)
                if addqaqc:
                    if debug:
                        print('addqaqc for '+v_long)
                    tmp=nscale(time2,inp['t_'+v][i][:],inp['q'+v][i][:],nearest=False,debug=debug)
                    ovar['q'+v_long]=ovar['q'+v_long][:]+tmp[:]
                    if i in [len(lvltqc)-2,len(lvltqc)-3]:
                        ovar['q'+v_long+'_upper']=ovar['q'+v_long+'_upper'][:]+tmp[:]
            if addqaqc:
                ovar['q'+v_long][ovar['q'+v_long]<.2]=0
                ovar['q'+v_long+'_upper'][ovar['q'+v_long+'_upper']<.2]=0
        # Interpolate top point in temperature profile
        if 't' in vs:
            v_long='profile_t'
            if addprof:
                ovar[v_long][v.upper()+str(len(lvltqc)-1)]=\
                        nscale(time2,inp['t_ttop'][:],inp['ttop'][:],nearest=False,debug=debug)
            if addqaqc:
                tmp=nscale(time2,inp['t_ttop'][:],inp['qttop'][:],nearest=False,debug=debug)
                ovar['q'+v_long]=ovar['q'+v_long]+tmp[:]
                ovar['q'+v_long][ovar['q'+v_long]<.2]=0
                ovar['q'+v_long+'_upper']=ovar['q'+v_long+'_upper'][:]+tmp[:]
                ovar['q'+v_long+'_upper'][ovar['q'+v_long+'_upper']<.2]=0

        # Output
        _out_to_h5(fpo,ovar,overwrite)

#############################################################################
########################### ADD PROFILE WIND ################################
# Adds the profile of windspeed
def add_profile_wind(scl,ndir,wndir,addprof=True,addqaqc=True,ivars=None,\
                    overwrite=False,sites=SITES):
    return None


#############################################################################
######################## ADD RADIATION ######################################
# Adds radiation information (longwave, shortwave, etc.)
def add_radiation(scl,ndir,idir,adddata=True,addqaqc=True,ivars=None,overwrite=False,sites=SITES,debug=False):
    ''' Add radiation information
        scl   : averaging scale in minutes
        ndir  : directory of L1 base files
        ivars : variables to process; None will add all
    '''
    #### SETUP
    _ivars = ['SW_IN', 'SW_OUT', 'LW_IN', 'LW_OUT', 'NETRAD']
    if ivars in [None]:
        ivars=_ivars
    outvar={}
    for var in ivars:
        if var in _ivars:
            outvar[var]=[]
    if len(outvar.keys())==0:
        print('add_radiation No valid variables in ivars')
        return None
    readlist=[]
    if adddata:
        readlist.extend(['inSWMean','inLWMean','outSWMean','outLWMean'])
    if addqaqc:
        readlist.extend(['inSWFinalQF','inLWFinalQF','outSWFinalQF','outLWFinalQF'])

    #### SITE LOOP
    for site in sites:
        if debug:
            dbg='::::::::::::::::DEBUG:::::::::::::::::::\n'
            dbg=dbg+'Loading in radiation data for '+site+'\n'
            print(dbg+':::::::::::::::::DEBUG::::::::::::::::::',flush=True)
        # Setup
        fpo=h5py.File(ndir+site+'_'+str(scl)+'m.h5','r+')
        time=fpo['TIME'][:]
        time2=time+scl/2*60
        ovar=outvar.copy()

        # Load Data
        N=len(readlist)+2
        d=_load_csv_data(readlist,idir+site,['_1min'])
        tm=(d['startDateTime'][:]+d['endDateTime'][:])/2


        # Interpolate Data
        if adddata:
            swin=nscale(time2,tm,d['inSWMean'],debug=debug)
            swout=nscale(time2,tm,d['outSWMean'],debug=debug)
            lwin=nscale(time2,tm,d['inLWMean'],debug=debug)
            lwout=nscale(time2,tm,d['outLWMean'],debug=debug)
            for v in ovar.keys():
                match v:
                    case 'SW_IN':
                        ovar[v]=swin
                    case 'SW_OUT':
                        ovar[v]=swout
                    case 'LW_IN':
                        ovar[v]=lwin
                    case 'LW_OUT':
                        ovar[v]=lwout
                    case 'NETRAD':
                        ovar[v]=swin-swout+lwin-lwout
        if addqaqc:
            radq=nscale(time2,tm,d['inSWFinalQF'],debug=debug)
            radq=radq+nscale(time2,tm,d['outSWFinalQF'],debug=debug)
            radq=radq+nscale(time2,tm,d['inLWFinalQF'],debug=debug)
            radq=radq+nscale(time2,tm,d['outLWFinalQF'],debug=debug)
            ovar['qNETRAD']=radq

        _out_to_h5(fpo,ovar,overwrite)

#############################################################################
####################### ADD GROUND HEAT FLUX ################################
# Add Ground heat flux
def add_ghflx(scl,ndir,idir,adddata=True,addqaqc=True,ivars=None,overwrite=False,sites=SITES):
    ''' Add radiation information
        scl   : averaging scale in minutes
        ndir  : directory of L1 base files
        ivars : variables to process; None will add all
    '''
    #### SETUP
    _ivars = ['G']
    if ivars in [None]:
        ivars=_ivars
    outvar={}
    for var in ivars:
        if var in _ivars:
            outvar[var]=[]

    if len(outvar.keys())==0:
        print('add_ghflx No valid variables in ivars')
        return None

    readlist=[]
    if adddata:
        readlist.extend(['SHFMean'])
    readlist.extend(['finalQF'])

    #### SITE LOOP
    for site in sites:
        # Setup
        fpo=h5py.File(ndir+site+'_'+str(scl)+'m.h5','r+')
        time=fpo['TIME'][:]
        time2=time+scl/2*60
        ovar=outvar.copy()

        if scl==30:
            st1='_30min'
        else:
            st1='_1min'

        # Load Data
        N=len(readlist)+2
        d1=_load_csv_data(readlist,idir+site,[st1,'001.001']) #001.001, 001.003, 001.005
        tm1=(d1['startDateTime'][:]+d1['endDateTime'][:])/2

        d2=_load_csv_data(readlist,idir+site,[st1,'001.003']) #001.001, 001.003, 001.005
        tm2=(d2['startDateTime'][:]+d2['endDateTime'][:])/2

        d3=_load_csv_data(readlist,idir+site,[st1,'001.005']) #001.001, 001.003, 001.005
        tm3=(d3['startDateTime'][:]+d3['endDateTime'][:])/2

        # Interpolate Data
        qgg=np.zeros((3,len(time2)))
        qgg[0,:]=nscale(time2,tm1,d1['finalQF'])
        qgg[1,:]=nscale(time2,tm2,d2['finalQF'])
        qgg[2,:]=nscale(time2,tm3,d3['finalQF'])

        if adddata:
            gg=np.zeros((3,len(time2)))
            gg[0,:]=nscale(time2,tm1,d1['SHFMean'])
            gg[1,:]=nscale(time2,tm2,d2['SHFMean'])
            gg[2,:]=nscale(time2,tm3,d3['SHFMean'])
            gg[gg<=-999]=float('nan')
            gcnt=np.sum(~np.isnan(gg),axis=0)+.00001
            gout=np.nansum(gg,axis=0)/gcnt
            gout[gcnt==.00001]=float('nan')
            ovar['G_full']=gout
            gg[qgg==1]=float('nan')
            gcnt=np.sum(~np.isnan(gg),axis=0)+.00001
            gout=np.nansum(gg,axis=0)/gcnt
            gout[gcnt==.00001]=float('nan')
            ovar['G']=gout

        if addqaqc:
            qgg[np.isnan(qgg)]=1
            gsum=np.sum(qgg,axis=0)
            ovar['qG']=gsum

        _out_to_h5(fpo,ovar,overwrite)

#############################################################################
####################### ADD PRECIPITATION ###################################
# Add Precipitation
def add_precip(scl,ndir,idir1,idir2,adddata=True,addqaqc=False,ivars=None,overwrite=False,sites=SITES):
    ''' Add radiation information
        scl   : averaging scale in minutes
        ndir  : directory of L1 base files
        ivars : variables to process; None will add all
        idir1 : input directory primary precip
        idir2 : input directory secondary precip
    '''
    #### SETUP
    _ivars = ['P']
    if ivars in [None]:
        ivars=_ivars
    outvar={}
    for var in ivars:
        if var in _ivars:
            outvar[var]=[]
    if len(outvar.keys())==0:
        print('add_precip No valid variables in ivars')
        return None

    if addqaqc:
        print('add_precip No QF currently implemented; skipping adding qaqc for precipitation')

    # precipBulk (primary)
    # secPrecipBulk

    #### SITE LOOP
    for site in sites:
        # Setup
        fpo=h5py.File(ndir+site+'_'+str(scl)+'m.h5','r+')
        time=fpo['TIME'][:]
        time2=time+scl/2*60
        ovar=outvar.copy()

        # check if primary or secondary or both
        prime=False
        secnd=False
        if len(os.listdir(idir1+'/'+site))>0:
            prime=True
        if len(os.listdir(idir2+'/'+site))>0:
            secnd=True

        # Load Data and Interpolate
        if prime:
            dp=_load_csv_data(['precipBulk'],idir1+site,['_60min'])
            tmp=(dp['startDateTime'][:]+dp['endDateTime'][:])/2
            if adddata:
                p1=nscale(time2,tmp,dp['precipBulk'],nearest=True)/60
        if secnd:
            ds=_load_csv_data(['secPrecipBulk'],idir2+site,['_1min'])
            tmp=(ds['startDateTime'][:]+ds['endDateTime'][:])/2
            if adddata:
                p2=nscale(time2,tmp,ds['secPrecipBulk'])

        # if we have both, use p1 to set ammount of rain and p2 to set timing
        if prime and secnd:
            ovar['P']=p1.copy()
            ovar['P'][np.isnan(p1)]=p2[np.isnan(p1)]
            ovar['P'][p2==0]=0
        elif prime:
            ovar['P']=p1
        elif secnd:
            ovar['P']=p2


        _out_to_h5(fpo,ovar,overwrite)


##################################################################
###################### ADD QAQC ##################################
# Adds qaqc flags from dp04; does not add qaqc flags for other variables
def add_qaqc(scl,ndir,idir,ivars=None,qsci=False,overwrite=False,sites=SITES):
    ''' Add generic quality control
        scl   : averaging scale in minutes
        ndir  : directory of L1 base files
        ivars : variables to process; None will add all
    '''
    #### SETUP
    _ivars = ['Q','U','V','W','UVW','C','THETA','USTAR','WC','LE','H']
    if ivars in [None]:
        ivars=_ivars
    outvar={}
    for var in ivars:
        if var in _ivars:
            outvar['q'+var]=[]
            if (qsci) & (var in ['Q','U','V','W','UVW','THETA','C']):
                outvar['qs'+var]=[]
    if len(outvar.keys())==0:
        print('add_qaqc: No valid variables in ivars')
        return None

    if scl==30:
        ds_='30m'
    else:
        ds_='01m'

    #### SITE LOOP
    for site in sites:
        # Setup
        fpo=h5py.File(ndir+site+'_'+str(scl)+'m.h5','r+')
        time=fpo['TIME'][:]
        time2=time+scl/2*60
        ovar=outvar.copy()
        for var in ovar.keys():
            ovar[var]=np.ones((len(time2),))*float('nan')
        tmp={}
        tmp['sonitime']=[]
        tmp['qctime']=[]
        tmp['flxtime']=[]
        for var in ovar.keys():
            tmp[var]=[]
        filelist=os.listdir(idir+site)
        filelist.sort()
        fpi=h5py.File(idir+site+'/'+filelist[-1],'r')
        for i in range(10):
            th=str(i)
            try:
                fpi['/'+site+'/dp01/data/soni/000_0'+th+\
                    '0_'+ds_+'/tempSoni/']['timeBgn'][:]
                break
            except:
                pass
        for file in filelist:
            bgn1='/'+site+'/dp01/qfqm/'
            bgn4='/'+site+'/dp04/qfqm/'
            fpi=h5py.File(idir+site+'/'+file,'r')
            nqc=0 # length of qc timeseries in file
            nqs=0 # length of soni timeseries in file
            nqf=0 # length of flux timeseries in file
            for var in tmp.keys():
                match var:
                    # time
                    case 'sonitime':
                        nm='/'+site+'/dp01/data/soni/000_0'+th+\
                                '0_'+ds_+'/tempSoni'
                        data=_dpt2utc(fpi[nm]['timeBgn'][:])+_dpt2utc(fpi[nm]['timeEnd'][:])
                        data=data/2
                        nqs=len(data)
                        tmp[var].extend(data)
                    case 'qctime':
                        nm='/'+site+'/dp01/data/h2oTurb/000_0'+th+\
                                '0_'+ds_+'/rtioMoleDryH2o'
                        data=_dpt2utc(fpi[nm]['timeBgn'][:])+_dpt2utc(fpi[nm]['timeEnd'][:])
                        data=data/2
                        nqc=len(data)
                        tmp[var].extend(data)

                    case 'flxtime':
                        nm='/'+site+'/dp04/data/fluxTemp/turb'
                        data=_dpt2utc(fpi[nm]['timeBgn'][:])+_dpt2utc(fpi[nm]['timeEnd'][:])
                        nqf=len(data)
                        data=data/2
                        tmp[var].extend(data)

                    # basic quality flag
                    case 'qQ':
                        nm=bgn1+'h2oTurb/000_0'+th+'0_'+ds_
                        tmp[var].extend(fpi[nm+'/rtioMoleDryH2o']['qfFinl'][:])
                    case 'qC':
                        nm=bgn1+'co2Turb/000_0'+th+'0_'+ds_
                        tmp[var].extend(fpi[nm+'/rtioMoleDryCo2']['qfFinl'][:])
                    case 'qTHETA':
                        nm=bgn1+'soni/000_0'+th+'0_'+ds_
                        tmp[var].extend(fpi[nm+'/tempSoni']['qfFinl'][:])
                    case 'qU':
                        nm=bgn1+'soni/000_0'+th+'0_'+ds_
                        tmp[var].extend(fpi[nm+'/veloXaxsErth']['qfFinl'][:])
                    case 'qV':
                        nm=bgn1+'soni/000_0'+th+'0_'+ds_
                        tmp[var].extend(fpi[nm+'/veloYaxsErth']['qfFinl'][:])
                    case 'qW':
                        nm=bgn1+'soni/000_0'+th+'0_'+ds_
                        tmp[var].extend(fpi[nm+'/veloZaxsErth']['qfFinl'][:])
                    case 'qUVW':
                        nm=bgn1+'soni/000_0'+th+'0_'+ds_
                        data=fpi[nm+'/veloZaxsErth']['qfFinl'][:]
                        data=data+fpi[nm+'/veloYaxsErth']['qfFinl'][:]
                        data=data+fpi[nm+'/veloXaxsErth']['qfFinl'][:]
                        tmp[var].extend(data)
                    case 'qUSTAR':
                        tmp[var].extend(fpi[bgn4+'fluxMome/turb']['qfFinl'][:])
                    case 'qH':
                        tmp[var].extend(fpi[bgn4+'fluxTemp/turb']['qfFinl'][:])
                    case 'qWC':
                        tmp[var].extend(fpi[bgn4+'fluxCo2/turb']['qfFinl'][:])
                    case 'qLE':
                        tmp[var].extend(fpi[bgn4+'fluxH2o/turb']['qfFinl'][:])

                    # sci reviews
                    case 'qsQ':
                        nm=bgn1+'h2oTurb/000_0'+th+'0_'+ds_+'/rtioMoleDryH2o'
                        tmp[var].extend(_get_qaqc_sci(fpi,nm,nqc))
                    case 'qsC':
                        nm=bgn1+'co2Turb/000_0'+th+'0_'+ds_+'/rtioMoleDryCo2'
                        tmp[var].extend(_get_qaqc_sci(fpi,nm,nqc))
                    case 'qsTHETA':
                        nm=bgn1+'soni/000_0'+th+'0_'+ds_+'/tempSoni'
                        tmp[var].extend(_get_qaqc_sci(fpi,nm,nqs))
                    case 'qsU':
                        nm=bgn1+'soni/000_0'+th+'0_'+ds_+'/veloXaxsErth'
                        tmp[var].extend(_get_qaqc_sci(fpi,nm,nqs))
                    case 'qsV':
                        nm=bgn1+'soni/000_0'+th+'0_'+ds_+'/veloYaxsErth'
                        tmp[var].extend(_get_qaqc_sci(fpi,nm,nqs))
                    case 'qsW':
                        nm=bgn1+'soni/000_0'+th+'0_'+ds_+'/veloZaxsErth'
                        tmp[var].extend(_get_qaqc_sci(fpi,nm,nqs))
                    case 'qsUVW':
                        nm=bgn1+'soni/000_0'+th+'0_'+ds_
                        data=_get_qaqc_sci(fpi,nm+'/veloXaxsErth',nqs)
                        data=data+_get_qaqc_sci(fpi,nm+'/veloYaxsErth',nqs)
                        data=data+_get_qaqc_sci(fpi,nm+'/veloZaxsErth',nqs)
                        tmp[var].extend(data)
        # interpolate
        for var in ovar.keys():
            tmin=[]
            if var in ['qsU','qsV','qsW','qsTHETA','qsUVW','qU','qV','qW','qTHETA','qUVW']:
                tmin=tmp['sonitime']
            elif var in ['qsQ','qsC','qQ','qC']:
                tmin=tmp['qctime']
            elif var in ['qsH','qsUSTAR','qsLE','qsWC','qH','qUSTAR','qLE','qWC']:
                tmin=tmp['flxtime']
            ovar[var]=nscale(time2,tmin,tmp[var])

        _out_to_h5(fpo,ovar,overwrite)




##################################################################
def add_pheno(scl,ndir,idir,ivars=None,overwrite=False,sites=SITES,debug=False):
    ''' Add radiation information
        scl   : averaging scale in minutes
        ndir  : directory of L1 base files
        ivars : variables to process; None will add all
    '''
    #### SETUP
    _ivars = ['GCC90_E','GCC90_D','GCC90_C','NDVI90','SOLAR_ALTITUDE','GROWING']
    if ivars in [None]:
        ivars=_ivars
    outvar={}
    for var in ivars:
        if var in _ivars:
            outvar[var]=[]

    if len(outvar.keys())==0:
        print('add_pheno No valid variables in ivars')
        return None

    #### SITE LOOP
    for site in sites:
        if debug:
            dbg='::::::::::::::::DEBUG:::::::::::::::::::\n'
            dbg=dbg+'Loading phenocam data for '+site+'\n'
            print(dbg+':::::::::::::::::DEBUG::::::::::::::::::',flush=True)
        # Setup
        fpo=h5py.File(ndir+site+'_'+str(scl)+'m.h5','r+')
        time=fpo['TIME'][:]
        time2=time+scl/2*60
        ovar=outvar.copy()
        for va in ovar.keys():
            ovar[va]=np.ones((len(time2),))*float('nan')

        # solar angle
        if 'SOLAR_ALTITUDE' in ovar.keys():
            if debug:
                dbg='::::::::::::::::DEBUG:::::::::::::::::::\n'
                dbg=dbg+'Loading solar altitude data for '+site+'\n'
                print(dbg+':::::::::::::::::DEBUG::::::::::::::::::',flush=True)
            files=os.listdir(idir+'data_record_3')
            sfiles=[]
            inp=[]
            for file in files:
                if (site in file) and ('00033' in file):
                    dm=file[5:8]
                    fnm=idir+'data_record_3'+'/'+file
                    inp.append(_load_csv_data(['solar_elev'],fnm,req=None))
            N=len(inp)
            sangles=np.zeros((N,len(time)))
            for i in range(N):
                sangles[i,:]=nscale(time2,inp[i]['time'],inp[i]['solar_elev'],debug=debug,nearest=False)
            ovar['SOLAR_ALTITUDE']=np.nanmean(sangles,axis=0)

        # GCC
        if ('GCC90_C' in ovar.keys())|('GCC90_D' in ovar.keys()):
            if debug:
                dbg='::::::::::::::::DEBUG:::::::::::::::::::\n'
                dbg=dbg+'Loading gcc data for '+site+'\n'
                print(dbg+':::::::::::::::::DEBUG::::::::::::::::::',flush=True)
            dlist=['GR','AG','SH','TN']
            elist=['EN','EB']
            files=os.listdir(idir+'data_record_4')
            dinp=[]
            einp=[]
            for file in files:
                if '3day' in file:
                    continue
                if site in file:
                    typ=file[24:26]
                    fnm=idir+'data_record_4'+'/'+file
                    if typ in dlist:
                        dinp.append(_load_csv_data(['gcc_90'],fnm))
                    elif typ in elist:
                        einp.append(_load_csv_data(['gcc_90'],fnm))
                    else:
                        continue
            Nd=len(dinp)
            Ne=len(einp)
            dgcc=np.zeros((Nd,len(time)))
            egcc=np.zeros((Ne,len(time)))
            for i in range(Nd):
                dgcc[i,:]=nscale(time2,dinp[i]['time'],dinp[i]['gcc_90'],debug=debug,nearest=False)
            for i in range(Ne):
                egcc[i,:]=nscale(time2,einp[i]['time'],einp[i]['gcc_90'],debug=debug,nearest=False)
            cgcc=np.concatenate((dgcc,egcc))
            ovar['GCC90_C']=np.nanmean(cgcc,axis=0)
            if (Nd>0) & ('GCC90_D' in ovar.keys()):
                ovar['GCC90_D'][:]=np.nanmean(dgcc,axis=0)
            if (Ne>0) and ('GCC90_E' in ovar.keys()):
                ovar['GCC90_E'][:]=np.nanmean(egcc,axis=0)

        # growing period
        if 'GROWING' in ovar.keys():
            if debug:
                dbg='::::::::::::::::DEBUG:::::::::::::::::::\n'
                dbg=dbg+'Loading growing period data for '+site+'\n'
                print(dbg+':::::::::::::::::DEBUG::::::::::::::::::',flush=True)
            dlist=['GR','AG','SH','TN']
            elist=['EN','EB']
            files=os.listdir(idir+'data_record_5')
            dinp=[]
            einp=[]
            for file in files:
                if '3day' in file:
                    continue
                if site in file:
                    typ=file[24:26]
                    fnm=idir+'data_record_5'+'/'+file
                    if typ in dlist:
                        dinp.append(_load_csv_data(['direction'],fnm))
                    elif typ in elist:
                        einp.append(_load_csv_data(['direction'],fnm))
                    else:
                        continue
            Nd=len(dinp)
            Ne=len(einp)
            grd=np.ones((Nd,len(time2)))*float('nan')
            gre=np.ones((Ne,len(time2)))*float('nan')
            for i in range(len(dinp)):
                dii=np.zeros((len(dinp[i]['time']),))
                for indx in range(len(dinp[i]['time'])):
                    val=dinp[i]['direction'][indx]
                    if val=='rising':
                        dii[indx]=1
                tm,di=sort_together(dinp[i]['time'],dii)
                for it in range(len(tm)):
                    t=tm[it]
                    grd[i,time2>t]=di[0,it]

            for i in range(len(einp)):
                dii=np.zeros((len(einp[i]['time']),))
                for indx in range(len(einp[i]['time'])):
                    val=einp[i]['direction'][indx]
                    if val=='rising':
                        dii[indx]=1
                tm,di=sort_together(einp[i]['time'],dii)
                for it in range(len(tm)):
                    t=tm[it]
                    gre[i,time2>t]=di[0,it]
            if Nd>0:
                grd=np.round(np.nanmean(grd,axis=0))
            if Ne>0:
                gre=np.round(np.nanmean(gre,axis=0))

            gr=np.zeros((len(time2),))
            if (Ne>0)&(Nd>0):
                gr[(grd==1)&(gre==0)]=1
                gr[(grd==0)&(gre==1)]=2
                gr[(grd==1)&(gre==1)]=3
            elif (Ne>0):
                gr[(gre==1)]=3
            else:
                gr[(grd==1)]=3
            ovar['GROWING'][:]=gr[:]

        # NDVI
        if 'NDVI90' in ovar.keys():
            if debug:
                dbg='::::::::::::::::DEBUG:::::::::::::::::::\n'
                dbg=dbg+'Loading ndvi data for '+site+'\n'
                print(dbg+':::::::::::::::::DEBUG::::::::::::::::::',flush=True)
            files=os.listdir(idir+'data_record_6')
            inp=[]
            for file in files:
                if (site in file):
                    typ=file[24:26]
                    if typ=='UN':
                        continue
                    fnm=idir+'data_record_6'+'/'+file
                    inp.append(_load_csv_data(['ndvi_90'],fnm,req=None))
            N=len(inp)
            sangles=np.zeros((N,len(time)))
            for i in range(N):
                sangles[i,:]=nscale(time2,inp[i]['time'],inp[i]['ndvi_90'],nearest=False,debug=debug)
            ovar['NDVI90']=np.nanmean(sangles,axis=0)

        desc={'GROWING':'0: not growing period, 1: growing period deciduous, 2: growing period evergreen, 3: growing period all vegetation (3 is also the value for growing season if there is only evergreen or only deciduous)'}

        _out_to_h5(fpo,ovar,overwrite,desc)

##################################################################
def remove_var(scl,ndir,delvar=[],confirm=True,sites=SITES):
    ''' Remove a given variable
        delvar  : list of variables to delete
        confirm : if true, will require user confirmation
    '''
    for var in delvar:
        doit=False
        if confirm:
            msg='Are you sure you would like to delete: '+var+' for the following sites: \n    '
            if len(sites)==47:
                msg=msg+'ALL SITES!\n\n'
            else:
                msg=msg+str(sites)+'\n\n'
            msg=msg+'This would be permenant and could not be undone!!!!\n'
            msg=msg+'Are you sure (respond y/n)?\n'
            doit=_confirm_user(msg)
        else:
            doit=True
        if doit:
            for site in sites:
                fp=h5py.File(ndir+site+'_'+str(scl)+'m.h5','r+')
                del fp[var]



##################################################################
def update_var(scl,ndir,var,rename=None,desc=None,units=None,\
        attr=None,factor=1,confirm=True,sites=SITES):
    ''' Update a variable name, add description or units or factor'''

    doele={}
    doele['rename']= (rename not in [None,''])
    doele['factor']= (factor not in [None,1])
    doele['desc']= (desc not in [None,'','{}'])
    doele['units']= (units not in [None,''])
    doele['attr']= (attr not in [None,'{}'])

    doit=False
    if confirm:
        msg='Are you sure you would like to change '+var+' in the following way: \n'
        if doele['rename']:
            msg=msg+'    rename to '+rename+'\n'
        if doele['factor']:
            msg=msg+'    adjust (multiply) data by factor of '+str(factor)+'\n'
        if doele['desc']:
            dout=''
            for i in range(int(np.floor(len(desc)/50))+1):
                dout=dout+'      '+desc[i*50:(i+1)*50]+'\n'
            msg=msg+'    add a description::\n'
            msg=msg+dout
        if doele['units']:
            msg=msg+'    add units of '+units+'\n'
        if doele['attr']:
            msg=msg+'    add the following attributes:\n'
            for k in attr.keys():
                msg=msg+'      '+k+':'+str(attr[k])+'\n'
        msg=msg+'For the following sites: \n'
        if len(sites)==47:
            msg=msg+'ALL SITES!\n\n'
        else:
            msg=msg+str(sites)+'\n\n'
        msg=msg+'Do you approve (respond y/n)?\n'
        doit=_confirm_user(msg)
    else:
        doit=True
    if doit:
        for site in sites:
            fp=h5py.File(ndir+site+'_'+str(scl)+'m.h5','r+')
            if doele['factor']:
                fp[var][:]=fp[var][:]*factor
            if doele['desc']:
                fp[var].attrs['description']=desc
            if doele['units']:
                fp[var].attrs['units']=units
            if doele['attr']:
                for k in attr.keys():
                    fp[var].attrs[k]=attr[k]
            if doele['rename']:
                fp.move(var,rename)

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

