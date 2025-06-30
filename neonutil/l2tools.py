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

#####################################################################
#####################################################################
######################## CASE DETAILS ###############################


#####################################################################
#####################################################################
##################### INTERNAL FUNCTIONS ############################
# Functions internal to L2 tools

#### OUTPUT to L2 H5
def _out_to_h5():
    return

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
    return {}

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
        m = maskgen(fp[basecase]['data'],m0,cvar=cs.core_vars,flags=cs.core_q,\
                precip=cs.precip,stb=cs.stab,limvars=cs.limvars,\
                counter=cs.counter,months=cs.months,years=cs.years)
        # FIXME output mask
    else:
        for site in sites:
            m0=np.ones((N,)).astype(bool)
            fpi=h5py.File(cs.l1dir+site+'_'+str(cs.scale)+'m.h5','r')
            m = maskgen(fpi,m0,cvar=cs.core_vars,flags=cs.core_q,\
                precip=cs.precip,stb=cs.stab,limvars=cs.limvars,\
                counter=cs.counter,months=cs.months,years=cs.years)
            # FIXME output mask

    # save case information to file
    fp=_h5_casewrite(fp,case,k)
    fp.close()
    return k


#####################################################################
#####################################################################
###################### DATA FUNCTIONS ###############################
# Functions for adding new data

##############################################################
def datagen(fp,include=None,exclude=None,static=None,zeta=['zL'],conv_nan=True,counter=False):


##############################################################
def add_foot():
    ''' Add footprint statistics/information. SLOW! '''

#############################################################
def pull_var():
    ''' Tries to pull an L2 variable from existing L2 file '''

##############################################################
def add_grad():
    ''' Add gradients from profiles to L2 '''

###############################################################
def add_from_l1():
    ''' Add a new L1 variable to an existing L2 file'''

##############################################################
def remove_var():
    ''' Remove variable from L2 file'''

