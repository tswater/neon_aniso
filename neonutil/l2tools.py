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
import datetime
try:
    from nutil import SITES,nscale,sort_together,out_to_h5,homogenize_list
except:
    from neonutil.nutil import SITES,nscale,sort_together,out_to_h5,homogenize_list

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
            continue
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
            try:
                val = case[k]
                if val in [[],None,float('nan')]:
                    val='NA'
                if k=='limvars':
                    for kk in case[k].keys():
                        fp[casek].attrs['limvars_'+kk]=case[k][kk]
                else:
                    fp[casek].attrs[k]=val
            except Exception as e:
                print(casek)
                print(k)
                print(case[k])
                raise e
    return fp



#####################################################################
#####################################################################
###################### CONSTRUCTION FUNCTIONS #######################
# Functions for building and manipulating L2 file and mask
def maskgen(fp,mask,cvar=None,flags=None,precip=None,stb=None,limvars=None,\
            zdtyp=['S','C','D'][2],counter=None,months=None,years=None,debug=False):
    ''' Generate a Mask '''
    nlist=[]
    slist=['START']
    nlist.append(np.sum(mask))
    if flags not in [None,[]]:
        for flag in flags:
            try:
                mask=mask&(~np.isnan(fp[flag][:]))
                mask=mask&(fp[flag][:]==0)
            except Exception as e:
                if flag=='qUVW':
                    try:
                        for v in ['U','V','W']:
                            flg='q'+v
                            mask=mask&(~np.isnan(fp[flg][:]))
                            mask=mask&(fp[flg][:]==0)
                    except Exception as e2:
                        print(e)
                        print(e2)
                        raise(e)
            slist.append(flag)
            nlist.append(np.sum(mask))
    if cvar not in [None,[]]:
        for var in cvar:
            n0=np.sum(mask)/len(mask)*100
            mask=mask&(~np.isnan(fp[var][:]))
            mask=mask&(fp[var][:]!=-9999)
            slist.append(var)
            nlist.append(np.sum(mask))
    if limvars not in [None,{},[],'NA']:
        for var in limvars:
            mn=limvars[var][0]
            mx=limvars[var][1]

            if var == 'zL':
                z=fp.attrs['tow_height']
                if zdtyp=='C':
                    zd=[fpi.attrs['zd_comp']]*len(mask)
                elif zdtyp=='S':
                    zd=_s2full(fp.attrs['zd_S'][:],fpi['TIME'][:])
                else:
                    zd=[fp.attrs['zd']]*len(mask)

                data=(z-zd)/fp['L_MOST'][:]
            elif '/' in var:
                vsplt=var.split('/')
                data=fp[vsplt[0]][:]/fp[vsplt[1]][:]
            elif '*' in var:
                vsplt=var.split('*')
                data=fp[vsplt[0]][:]*fp[vsplt[1]][:]
            else:
                data=fp[var][:]
            if mn not in [float('nan'),None,'NA']:
                mask=mask&(data>=mn)
            if mx not in [float('nan'),None,'NA']:
                mask=mask&(data<=mx)
            slist.append('LIMIT_'+var)
            nlist.append(np.sum(mask))
    if stb not in [None]:
        if stb:
            mask=mask&(fp['L_MOST'][:]>0)
        if not stb:
            mask=mask&(fp['L_MOST'][:]<0)
        slist.append('Stability')
        nlist.append(np.sum(mask))
    if precip not in [None,False,'NA']:
        mask=mask&(fp['P'][:]<=0)
        slist.append('Precip')
        nlist.append(np.sum(mask))
    if hasattr(counter,"__len__") or (counter==True):
        if not hasattr(counter,"__len__"):
            counter=['profile_t','profile_u','profile_q','profile_c']
        for k in counter:
            v=k[-1]
            if k in ['profile_u','profile_q','profile_c']:
                lk=len(fp[k].keys())
                top=fp[v.upper()][:]
                top2=fp[k][v.upper()+str(lk-1)][:]
                delta=top-top2
                if v=='u':
                    flux=fp['UsW'][:]
                else:
                    flux=fp['W'+v.upper()][:]
                mask=mask&(delta*flux<0)
            elif k in ['profile_t']:
                lk=len(fp[k].keys())
                top=fp[k][v.upper()+str(lk-1)][:]
                top2=fp[k][v.upper()+str(lk-2)][:]
                delta=top-top2
                flux=fp['WTHETA'][:]
                mask=mask&(delta*flux<0)

        slist.append('Counter')
        nlist.append(np.sum(mask))

    yrbool= (years not in [None,[],'NA'])
    mnbool= (months not in [None,[],'NA'])
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
        slist.append('Months/years')
        nlist.append(np.sum(mask))

    if debug:
        msg='DEBUG maskgen: \n'
        for i in range(len(slist)):
            msg=msg+'    '+slist[i]+': '+str(nlist[i])+'\n'
        print(msg)


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
        if _confirm_user('Remove submask '+sub+' named '+casenm+'?'):
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
            if _confirm_user(fpath+' already exists; replace?'):
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
def casegen(case,debug=False):
    ''' Essentially a driver for constructing a case, including
    '''
    cs=SimpleNamespace(**case)

    isnew= (cs.basecase in [None,'NA',''])

    if isnew:
        #  new file
        fp,k=build_L2_file(cs.fpath,True)

        # get mask length
        if cs.sites in [[],None,'NA']:
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
    if cs.sites in [[],None,'NA']:
        sites=SITES
    else:
        sites=cs.sites

    if not isnew:
        m0=np.ones((N,)).astype(bool)
        m = maskgen(fp[cs.basecase]['data'],m0,cvar=cs.core_vars,flags=cs.core_q,\
                precip=cs.precip,stb=cs.stab,limvars=cs.limvars,\
                counter=cs.counter,months=cs.months,years=cs.years,debug=debug)
        fp[k]['mask'].create_dataset('ALL',data=m,dtype=bool)
    else:
        for site in sites:
            m0=np.ones((N,)).astype(bool)
            fpi=h5py.File(cs.l1dir+site+'_'+str(cs.scale)+'m.h5','r')
            m = maskgen(fpi,m0,cvar=cs.core_vars,flags=cs.core_q,\
                precip=cs.precip,stb=cs.stab,limvars=cs.limvars,\
                counter=cs.counter,months=cs.months,years=cs.years,debug=debug)
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
    sites=cs.sites
    if hasattr(sites, "__len__"):
        try:
            if sites=='NA':
                sites=SITES
        except Exception as e:
            pass
        sites=sites
    elif sites in [None,'NA',[]]:
        sites=SITES
    if static is None:
        static=[]
        fpi=h5py.File(idir+sites[0]+'_'+str(cs.scale)+'m.h5','r')
        for v in fpi.attrs.keys():
            static.append(v)
        fpi.close()
    if len(static)==0:
        return
    sites.sort()
    out={}
    for v in static:
        out[v]=[]
    for site in sites:
        fpi=h5py.File(idir+site+'_'+str(cs.scale)+'m.h5','r')
        for v in static:
            if v in fpi.attrs.keys():
                out[v].append(fpi.attrs[v])
        fpi.close()
    out['SITE']=sites
    for v in out.keys():
        try:
            l2=out[v]
            fp[casek+'/static'].create_dataset(v,data=l2)
        except ValueError as e:
            try:
                l2=homogenize_list(out[v])
                fp[casek+'/static'].create_dataset(v,data=l2)
            except Exception as e2:
                if 'name already exists' in e:
                    fp[casek+'/static'][v]=l2
                else:
                    print(e)
                    raise(e2)

def _s2full(zds,time):
    # Make zd seasonal into a full zd situation
    month=[]
    for t in time:
        dt=datetime.datetime(1970,1,1,0,0)+datetime.timedelta(seconds=t)
        month.append(dt.month)
    month=np.array(month)
    timez=np.linspace(1,12,12)
    timi=[]
    zdsi=[zds[10],zds[11]]
    zdsi.extend(zds)
    zdsi.extend([zds[0],zds[1]])
    zout=np.zeros((len(time),))
    for i in range(12):
        n=np.sum(month==(i+1))
        if n==0:
            continue
        if np.isnan(zds[i]):
            a=np.nanmean(zdsi[i:i+5])
        else:
            a=zdsi[i+2]
        if np.isnan(a):
            a=np.nanmean(zds)

        zout[month==(i+1)]=a
    return zout

##############################################################
def datagen(outfile,idir,include=None,exclude=None,static=None,zdtyp=['S','C','D'][2],zeta=['zL'],\
        conv_nan=True,overwrite=False,debug=False):

    # zdtyp: S is for seasonal, C is for constant computed, D is for default from NEON

    # Streamwise List and Earth List
    slist=['Us','Vs','UsUs','VsVs','UsVs','UsW','VsW',\
            'ANI_XBs','ANI_YBs','ANID_YBs','ANID_XBs',\
            'ST_UsUs_1','ST_UsUs_5','ST_VsVs_1','ST_VsVs_5']
    elist=['U','V','UU','VV','UV','UW','VW',\
            'ANI_XB','ANI_YB','ANID_YB','ANID_XB',\
            'ST_UU_1','ST_UU_5','ST_VV_1','ST_VV_5']

    # load case

    if type(outfile)==str:
        fpo=h5py.File(outfile,'r+')
    else:
        fpo=outfile
    cs=SimpleNamespace(**pull_case(fpo,'main'))
    sites=cs.sites
    if hasattr(sites, "__len__"):
        try:
            if sites=='NA':
                sites=SITES
        except Exception as e:
            sites=sites
    elif sites is None:
        sites=SITES
    elif sites in [None,'NA',[]]:
        sites=SITES
    sites.sort()
    wind_sys=cs.wind_sys

    # generate a list of variables to include
    is_include= not (include in ['NA',None,[],'[]'])
    is_exclude= not (exclude in ['NA',None,[],'[]'])
    if is_include and is_exclude:
        msg='Cannot have both include and exclude; must pick one'
        msg=msg+'\nINCLUDE: '+str(include)+'\nEXCLUDE: '+str(exclude)
        raise ValueError(msg)
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

    proflist=[]
    vlist2=[]
    for v in vlist:
        if 'profile' in v:
            proflist.append(v)
        else:
            vlist2.append(v)
    vlist=vlist2

    # initialize output
    ovar={}
    ovar['main/data']={}
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
        if zdtyp=='C':
            zd=[fpi.attrs['zd_comp']]*n
            z0=[fpi.attrs['z0']]*n
        elif zdtyp=='S':
            z0=_s2full(fpi.attrs['z0_S'][:],fpi['TIME'][:][msk])
            zd=_s2full(fpi.attrs['zd_S'][:],fpi['TIME'][:][msk])
        else:
            z0=[fpi.attrs['z0']]*n
            zd=[fpi.attrs['zd']]*n
        lmost=fpi['L_MOST'][:][msk]
        if 'z0' in zeta:
            ovar['main/data']['z0'].extend(z0)
        if 'zL' in zeta:
            ovar['main/data']['zL'].extend((z-zd)/lmost)
        if 'zd' in zeta:
            ovar['main/data']['zd'].extend(zd)
        if 'z' in zeta:
            ovar['main/data']['z'].extend([z]*n)
        if 'zzd' in zeta:
            ovar['main/data']['zzd'].extend(np.array([z]*n)-zd)
        if ('L_MOST' in zeta) and ((is_include and ('L_MOST' not in include)) or\
                (is_exclude and ('L_MOST' in exclude))):
            ovar['main/data']['L_MOST'].extend(lmost)

    # conv nan
    if conv_nan:
        for v in ovar['main/data'].keys():
            try:
                arr=np.array(ovar['main/data'][v][:])
            except Exception as e:
                print(v)
                print(ovar['main/data'][v].shape)
                print(len(ovar['main/data'][v]))
                raise(e)
            arr[arr==-9999]=float('nan')
            ovar['main/data'][v]=arr

    # fix site array
    strlist=ovar['main/data']['SITE']
    asciiList = [n.encode("ascii", "ignore") for n in strlist]
    ovar['main/data']['SITE']=asciiList

    # static variables:
    staticgen(fpo,idir,static=static)

    # output
    out_to_h5(fpo,ovar,overwrite)

    # now deal with other variables:
    # other variables
    for v in vlist:
        ovar={'main/data':{}}
        if debug:
            print('Reading in '+str(v)+' for '+site)
        if (wind_sys=='streamwise')&(v in slist):
            v2=v.replace('s','')
        elif (wind_sys=='streamwise')&(v in elist):
            v2=v+'e'
        else:
            v2=v
        ovar['main/data'][v2]=[]
        for site in sites:
            fpi=h5py.File(idir+site+'_'+str(cs.scale)+'m.h5','r')
            msk=fpo['main/mask'][site]
            n=np.sum(msk)
            if n==0:
                continue
            ovar['main/data'][v2].extend(fpi[v][:][msk])

        if conv_nan:
            for v in ovar['main/data'].keys():
                arr=np.array(ovar['main/data'][v][:])
                arr[arr==-9999]=float('nan')
                ovar['main/data'][v]=arr

        out_to_h5(fpo,ovar,overwrite)

    # now handle profile stuff
    for v in proflist:
        ovar={'main/data':{}}
        if v[-1]=='t':
            for i in range(4):
                ovar['main/data'][v[-1].upper()+str(i)]=[]
        else:
            for i in range(3):
                ovar['main/data'][v[-1].upper()+str(i)]=[]
        for site in sites:
            fpi=h5py.File(idir+site+'_'+str(cs.scale)+'m.h5','r')
            msk=fpo['main/mask'][site]
            n=np.sum(msk)
            if n==0:
                continue
            a=fpi[v].keys()
            la=len(a)
            if v[-1]=='t':
                for i in range(4):
                    nmo=v[-1].upper()+str(i)
                    nmi=v[-1].upper()+str(la-4+i)
                    ovar['main/data'][nmo].extend(fpi[v][nmi][:][msk])
            else:
                for i in range(3):
                    nmo=v[-1].upper()+str(i)
                    nmi=v[-1].upper()+str(la-3+i)
                    ovar['main/data'][nmo].extend(fpi[v][nmi][:][msk])
        out_to_h5(fpo,ovar,overwrite)

        if not v[-1]=='t':
            fpo['main/data/'+v[-1].upper()+'3']=h5py.SoftLink('/main/data/'+v[-1].upper())
    fpo.close()


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
            stb=fpi['main'].attrs['stab']
        elif '_U_' in file:
            iscale=30
            stb=False
        elif '_S_' in file:
            iscale=1
            stb=True
        if iscale!=oscale:
            continue
        if stb!=fpo['main'].attrs['stab']:
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
def add_from_l1(fpath,casek,ivars,scl=None,l1dir=None,conv_nan=True,sites=SITES):
    ''' Add a new L1 variable to an existing L2 file'''
    fpo=h5py.File(fpath,'r+')

    case=pull_case(fpo,casek)
    if l1dir is None:
        l1dir=case['l1dir']
    if scl is None:
        scl=case['scale']

    ovar={}
    ovar[casek+'/data']={}
    for var in ivars:
        ovar[casek+'/data'][var]=[]

    for site in sites:
        fpi=h5py.File(l1dir+site+'_'+str(scl)+'m.h5','r')
        m=fpo[casek]['mask'][site]

        for var in ovar[casek+'/data'].keys():
            ovar[casek+'/data'][var].extend(fpi[var][:][m])

    if conv_nan:
        for v in ovar['main/data'].keys():
            arr=np.array(ovar['main/data'][v][:])
            arr[arr==-9999]=float('nan')
            ovar['main/data'][v]=arr


    out_to_h5(fpo,ovar,True)




##############################################################
def remove_var():
    ''' Remove variable from L2 file'''

