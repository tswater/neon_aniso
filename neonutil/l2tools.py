# tools for L2 processing, mostly used by L2 driver script


# enable the choice of streamwise or earth coordinates and change
#   variable names. (or both)
# should probably make a "case" class
####################################################################
####################### IMPORT #####################################
import os
import numpy as np
import h5py
from subprocess import run
from types import SimpleNamespace
try:
    from nutil import SITES,nscale,sort_together,out_to_h5
except:
    from neonutil.nutil import SITES,nscale,sort_together,out_to_h5

#####################################################################
#####################################################################
######################## CASE DETAILS ###############################


#####################################################################
#####################################################################
##################### INTERNAL FUNCTIONS ############################
# Functions internal to L2 tools

#### CONVERT VARLIST
# iterate through varlist to "remove" groups (i.e. profiles)
def _convert_varlist(fp,varlist):
    varout=[]
    for v in varlist:
        try:
            fp[v].keys()
        except Exception:
            varout.append(v)
        for k in fp[v].keys():
            varout.append(v+'/'+str(k))
    return varout

#### GET USER CONFIRMATION
def _confirm_user(msg):
    while True:
        user_input = input(f"{msg} (Y/N): ").strip().lower()
        if user_input in ('y', 'yes'):
            return True
        elif user_input in ('n', 'no'):
            return False
        else:
            print("Invalid input. Please enter 'Y' or 'N'.")

#### CASEWRITE
def _h5_casewrite(fp,case,casek):
    exclude=['fpath','l1dir']
    if casek not in fp.keys():
        raise ValueError('This case does not exist in this h5 file')
    for k in case.keys():
        if k in exclude:
            continue
        else:
            fp[casek].attrs[k]=case[k]
    return fp



#####################################################################
#####################################################################
###################### CONSTRUCTION FUNCTIONS #######################
# Functions for building and manipulating L2 file and mask
def maskgen(fp,mask,cvar=None,flags=None,precip=None,stb=None,limvars=None,\
            counter=None,months=None,years=None):
    ''' Generate a Mask '''
    if flags not in [None,[]]:
        for flag in flags:
            mask=mask&(~np.isnan(fp[flag][:]))
            mask=mask&(fp[flag][:]==0)
    if cvar not in [None,[]]:
        for var in cvars:
            n0=np.sum(mask)/len(mask)*100
            mask=mask&(~np.isnan(fp[var][:]))
            mask=mask&(fp[var][:]!=-9999)
    if limvars not in [None,{},[]]:
        for var in limvars:
            mn=limvars[var][0]
            mx=limvars[var][0]
            if mn not in [float('nan'),None]:
                mask=mask&(fp[var][:]>=mn)
            if mx not in [float('nan'),None]:
                mask=mask&(fp[var][:]<=mx)
    if stb not in [None]:
        if stb:
            mask=mask&(fp['L_MOST'][:]>0)
        if not stb:
            mask=mask&(fp['L_MOST'][:]<0)
    if precip not in [None,False]:
        mask=mask&(fp['P']<=0)
    if counter not in [None,False]:
        raise NotImplementedError('Masking countergradient fluxes TBI')
    yrbool= (year not in [None,[]])
    mnbool= (month not in [None,[]])
    if yrbool | mnbool:
        time = fp['TIME'][:]
        yrmsk=[]
        mnmsk=[]
        d0=datetime.datetime(1970,1,1,0,0)
        for t in time:
            dt = d0+datetime.timedelta(seconds=t)
            if yrbool:
                yrmsk.append(dt.year in years)
            else:
                yrmsk.append(True)
            if mnbool:
                mnmsk.append(dt.month in months)
            else:
                mnmsk.append(True)
        mask=mask&np.array(yrmsk)
        mask=mask&np.array(mnmsk)


    return mask

####################################################
########### SUB FUNCTIONS #########################

def build_sub(fp):
    if type(fp)==str:
        fp=h5py.File(fp,'r+')

    fkeys=fp.keys()
    a=-1
    for k in fkeys:
        if k=='main':
            continue
        else:
            try:
                ik=int(k)
            except Exception:
                continue
            if ik>a:
                a=ik
    a=a+1
    fp.create_group(str(a))
    fp[str(a)].create_group('mask')
    fp[str(a)].create_group('L3')
    fp[str(a)].create_group('static')
    return fp,str(a)

def remove_sub(fp,sub,confirm=True):
    if type(fp)==str:
        fp=h5py.File(fp,'r+')
    casenm=fp[sub]['name']
    if confirm:
        if _user_confirm('Remove submask '+sub+' named '+casenm+'?'):
            del fp[sub]
        else:
            print('doing nothing')
    else:
        del fp[sub]
    return

#############################################################
def pull_case(fp,casek,combo=False):
    ''' Pull a case dictionary from an h5 file;
        fp    : filepath OR open h5 file
        casek : the case key (main, 0, 1)
        combo : if combo is True, and casek is a subcase, will generate
                a combined case that includes the basecase information
    '''
    is_fpath=(type(fp)==str)
    if is_fpath:
        fp=h5py.File(fp,'r')
    case={}
    for k in fp[casek].attrs.keys():
        case[k]=fp[casek].attrs[k]
    if is_fpath:
        fp.close()

    return case

#############################################################
def build_L2_file(fpath,case,overwrite=True):
    ''' Generate an empty L2 file '''

    # deal with existing file if it is there
    if os.path.exists(fpath):
        if overwrite:
            if _user_confirm(fpath+' already exists; replace?'):
                run('rm '+fpath,shell=True)
            else:
                return None
        else:
            print(fpath+' already exists; enable overwrite if you wish')
    fp=h5py.File(fpath,'w')
    fp.create_group('main')
    fp['main'].create_group('mask')
    fp['main'].create_group('data')
    fp['main'].create_group('static')
    fp['main'].create_group('L3')
    return fp,'main'

#############################################################
def casegen(case):
    ''' Essentially a driver for constructing a case, including
    '''
    cs=SimpleNamespace(**case)

    isnew= (cs.basecase in [None,''])

    if isnew:
        #  new file
        fp,k=build_L2_file(cs.fpath,True)

        # get mask length
        if cs.sites in [[],None]:
            site='ABBY'
        else:
            site=cs.sites[0]
        fpi=h5py.File(cs.l1dir+site+'_'+str(cs.scale)+'m.h5','r')
        N=len(fpi['TIME'][:])
    else:
        # new sub-case
        fp,k=build_sub(cs.fpath)
        N=0
        for site in fp[basecase]['mask'].keys():
            N=N+np.sum(fp[basecase]['mask'][site][:])

    # get list of sites
    if cs.sites in [[],None]:
        sites=SITES
    else:
        sites=cs.sites

    if not isnew:
        m0=np.ones((N,)).astype(bool)
        m = maskgen(fp[cs.basecase]['data'],m0,cvar=cs.core_vars,flags=cs.core_q,\
                precip=cs.precip,stb=cs.stab,limvars=cs.limvars,\
                counter=cs.counter,months=cs.months,years=cs.years)
        fp[k]['mask'].create_dataset('ALL',data=m,dtype=bool)
    else:
        for site in sites:
            m0=np.ones((N,)).astype(bool)
            fpi=h5py.File(cs.l1dir+site+'_'+str(cs.scale)+'m.h5','r')
            m = maskgen(fpi,m0,cvar=cs.core_vars,flags=cs.core_q,\
                precip=cs.precip,stb=cs.stab,limvars=cs.limvars,\
                counter=cs.counter,months=cs.months,years=cs.years)
            fp[k]['mask'].create_dataset(site,data=m,dtype=bool)

    # save case information to file
    fp=_h5_casewrite(fp,case,k)
    fp.close()
    return k


#####################################################################
#####################################################################
###################### DATA FUNCTIONS ###############################
# Functions for adding new data

#####################################################################
def staticgen(fp,idir,casek='main',case=None,static=None):
    is_fpath=(type(fp)==str)
    if is_fpath:
        fp=h5py.File(fp,'r+')
    if case in [None]:
        cs=SimpleNamespace(**pull_case(fp,'main'))
    else:
        cs=SimpleNamespace(**case)
    if static in [None,[]]:
        static=[]
        fpi=h5py.File(idir+cs.sites[0]+'_'+str(cs.scale)+'m.h5','r')
        for v in fpi.attrs.keys():
            static.append(v)
        fpi.close()
    sites=cs.sites
    sites.sort()
    out={}
    for v in static:
        out[v]=[]
    for site in sites:
        fpi=h5py.File(idir+site+'_'+str(cs.scale)+'m.h5','r')
        for v in static:
            if v in fpi.attrs.keys():
                out[v].extend(fpi.attrs[k])
        fpi.close()
    out['SITE']=sites
    for v in out.keys():
        fp[casek+'/static'].create_dataset(v,data=out[v])




##############################################################
def datagen(outfile,idir,include=None,exclude=None,static=None,zeta=['zL'],\
        conv_nan=True,overwrite=False):

    # load case
    fpo=h5py.File(outfile,'r+')
    cs=SimpleNamespace(**pull_case(fpo,'main'))
    sites=cs.sites
    sites.sort()

    # generate a list of variables to include
    is_include=~(include in [None,[]])
    is_exclude=~(exclude in [None,[]])
    if is_include and is_exclude:
        raise ValueError('Cannot have both include and exclude; must pick one')
    vlist=[]
    site=sites[0]
    fpi=h5py.File(idir+site+'_'+str(cs.scale)+'m.h5','r')
    if is_include:
        for v in include:
            if v not in vlist:
                vlist.append(v)
    if is_exclude:
        for v in fpi.keys():
            if (v not in exclude) and (v[0]!='q'):
                vlist.append(v)

    # convert list to include profile variables fully
    vlist=_convert_varlist(fp,vlist)

    # initialize output
    ovar={}
    ovar['main/data']={}
    for v in vlist:
        ovar['main/data'][v]=[]
    for v in zeta:
        ovar['main/data'][v]=[]
    ovar['main/data']['SITE']=[]

    # build up output for non static variables
    for site in sites:
        fpi=h5py.File(idir+site+'_'+str(cs.scale)+'m.h5','r')
        msk=fpo['main/mask'][site]
        n=np.sum(msk)
        if n==0:
            continue

        # newly generated variables
        ovar['main/data']['SITE'].extend([site]*n)
        z=fpi.attrs['tow_height']
        zd=fpi.attrs['zd']
        lmost=fpi['L_MOST'][:][msk]
        if 'zL' in zeta:
            ovar['main/data']['zL'].extend((z-zd)/lmost)
        if 'zd' in zeta:
            ovar['main/data']['zd'].extend([zd]*n)
        if 'z' in zeta:
            ovar['main/data']['z'].extend([z]*n)
        if 'zzd' in zeta:
            ovar['main/data']['zzd'].extend([z-zd]*n)
        if ('L_MOST' in zeta) and ((is_include and ('L_MOST' not in include)) or\
                (is_exclude and ('L_MOST' in exclude))):
            ovar['main/data']['L_MOST'].extend(lmost)

        # other variables
        for v in vlist:
            ovar['main/data'][v].extend(fpi[v][:][msk])

    # conv nan
    if conv_nan:
        for v in ovar['main/data'].keys():
            arr=np.array(ovar['main/data'][v][:])
            arr[arr==-9999]=float('nan')
            ovar['main/data'][v]=arr

    # static variables:
    staticgen(fpo,static,case=case)

    # output
    out_to_h5(fpo,ovar,overwrite)


##############################################################
def add_foot():
    ''' Add footprint statistics/information. SLOW! '''




#############################################################
def pull_var(l2dir,l2file,var,frmt='new'):
    ''' Tries to pull an L2 variable from existing L2 file
        frmt : if new, will assume its a L2 file from July25 or later,
               if old, will assume its an old style L2 file
    '''

    new=frmt=='new'

    try:
        fpo=h5py.File(l2file,'r+')
    except Exception as e:
        fpo=h5py.File(l2dir+l2file,'r+')

    # assemble list of files to check
    filelist=[]

    for file in os.listdir(l2dir):
        if (file==l2file) or ((l2dir+file)==l2file):
            continue
        if os.path.isdir(file):
            for file2 in os.listdir(l2dir+file):
                if (file2==l2file) or ((l2dir+file+'/'+file2)==l2file):
                    continue
                elif '.h5' in file2:
                    filelist.append(l2dir+file+'/'+file2)
        elif '.h5' in file:
            filelist.append(l2dir+file)

    oscale=fpo['main'].attrs['scale']

    timeo=fpo['main/data']['TIME']

    for file in filelist:
        fpi=h5py.File(file,'r')
        iscale=0
        if new:
            iscale=fpi['main'].attrs['scale']
            stb=fpi['main'].attrs['stb']
        elif '_U_' in file:
            iscale=30
            stb=False
        elif '_S_' in file:
            iscale=1
            stb=True
        if iscale!=oscale:
            continue
        if stb!=fpo['main'].attrs['stb']:
            continue
        if new:
            f=fpi['main/data']
        else:
            f=fpi
        try:
            data=f[var][:]
        except KeyError:
            continue
        timei=f['TIME'][:]
        sitei=f['SITE'][:]

        return None



##############################################################
def add_grad():
    ''' Add gradients from profiles to L2 '''

###############################################################
def add_from_l1():
    ''' Add a new L1 variable to an existing L2 file'''

##############################################################
def remove_var():
    ''' Remove variable from L2 file'''

