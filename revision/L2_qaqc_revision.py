import os
import h5py
import numpy as np

exclude=[]

idir='/home/tswater/Documents/Elements_Temp/NEON/'
odir='/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data/'

exclude=['U_SIGMA','V_SIGMA','W_SIGMA','T_SONIC_SIGMA','H2O_SIGMA','CO2_SIGMA','RHO','CO2']
redo_mask=True

#['H2O','H2O_SIGMA','LE','RHO'],['qLE','qH2O']
# ['CO2','CO2_SIGMA','H2O','RHO'],['qCO2FX','qCO2']

# stb definitions: 0  = unstable
#                  1  = stable (1 min)
#                  30 = stable (30 min)

# kargs options: 'highwind'
#                'leafon'
#                'leafoff'

##################
cases=[{'stb':30,'typ':'S30','core_vars':['Ustr','Vstr','Wstr','UU','VV','WW'],'core_qs':[],'kargs':{}},
       {'stb':1,'typ':'S_high_u','core_vars':['Ustr','Vstr','Wstr','UU','VV','WW'],'core_qs':[],'kargs':{'highwind':1}},
       {'stb':0,'typ':'U_high_u','core_vars':['Ustr','Vstr','Wstr','UU','VV','WW'],'core_qs':[],'kargs':{'highwind':1}},
       {'stb':1,'typ':'S_leafon','core_vars':['Ustr','Vstr','Wstr','UU','VV','WW'],'core_qs':[],'kargs':{'leafon':0}},
       {'stb':0,'typ':'U_leafon','core_vars':['Ustr','Vstr','Wstr','UU','VV','WW'],'core_qs':[],'kargs':{'leafon':0}},
       {'stb':1,'typ':'S_leafoff','core_vars':['Ustr','Vstr','Wstr','UU','VV','WW'],'core_qs':[],'kargs':{'leafoff':0}},
       {'stb':0,'typ':'U_leafoff','core_vars':['Ustr','Vstr','Wstr','UU','VV','WW'],'core_qs':[],'kargs':{'leafoff':0}}]

cases=[{'stb':30,'typ':'S30','core_vars':['Ustr','Vstr','Wstr','UU','VV','WW'],'core_qs':[],'kargs':{}}]
cases=[{'stb':30,'typ':'S30','core_vars':['Ustr','Vstr','Wstr','UU','VV','WW'],'core_qs':[],'kargs':{'epsi_uvw':50}}]
##################
def core_q(mask,fp,addvar=[],addq=[]):
    flags=['qU','qV','qW','qUSTAR','qH','qT_SONIC']
    flags.extend(addq)
    for flag in flags:
        mask=mask&(~np.isnan(fp[flag][:]))
        mask=mask&(fp[flag][:]==0)

    cvars=['ANI_YB','L_MOST','USTAR','WTHETA','T_SONIC']
    cvars.extend(addvar)
    for var in cvars:
        n0=np.sum(mask)/len(mask)*100
        mask=mask&(~np.isnan(fp[var][:]))
        mask=mask&(fp[var][:]!=-9999)
        n1=np.sum(mask)/len(mask)*100
        #print(var+': '+str(n0)+' -> '+str(n1))

    return mask


##################
sites=[]
for file in os.listdir(idir+'neon_1m/'):
    if 'L1' in file:
        sites.append(file[0:4])
sites.sort()

for case in cases:
    print('STARTING CASE '+str(case['typ']))
    start1=True
    start2=True
    typ=case['typ']
    out={}
    print('    ',end='',flush=True)
    if redo_mask:
        fpm=h5py.File(odir+'L2mask_'+str(typ)+'.h5','w')
        fpm.close()
        for site in sites:
            fpm=h5py.File(odir+'L2mask_'+str(typ)+'.h5','r+')
            print('.',end='',flush=True)
            if case['stb']==1:
                fp=h5py.File(idir+'neon_1m/'+site+'_L1.h5','r')
            elif case['stb'] in [0,30]:
                fp=h5py.File(idir+'neon_30m/'+site+'_L30.h5','r')
            time=fp['TIME'][:]
            N=len(time)
            m=np.ones((N,),dtype=bool)
            if case['stb'] in [1,30]:
                m=m&(fp['L_MOST'][:]>0)
            else:
                m=m&(fp['L_MOST'][:]<0)
            if 'Q' in case['typ']:
                m=m&(fp['P'][:]<=0)
            for k in case['kargs'].keys():
                match k:
                    case 'highwind':
                        m=m&(fp['Ustr'][:]>case['kargs'][k])
                    case 'leafon':
                        m=m
                    case 'leafoff':
                        m=m
                    case 'epsi_uvw':
                        m=m&getEpsi(fp,['U','V','W'])
            m=core_q(m,fp,case['core_vars'],case['core_qs'])

            fpkeys=list(fp.keys())
            fpm.create_dataset(site,data=m)
            fp.close()
            fpm.close()

    site='ABBY'
    if case['stb']==1:
        fp=h5py.File(idir+'neon_1m/'+site+'_L1.h5','r')
    elif case['stb'] in [0,30]:
        fp=h5py.File(idir+'neon_30m/'+site+'_L30.h5','r')

    fpkeys=list(fp.keys())
    fp.close()

    # create output
    outname='NEON_TW_'+str(typ)+'.h5'
    fpo=h5py.File(odir+outname,'w')
    fpo.close()
    # add data
    fpkeys.extend(['SITE','zzd'])
    print('')
    print('    PREPROCESSING DONE',flush=True)
    print('    Begining Variable Processing')
    print('    ',end='')
    for k in fpkeys:
        out=[]
        print('.',end='',flush=True)
        fpo=h5py.File(odir+outname,'r+')
        processed=False
        for site in sites:
            fpm=h5py.File(odir+'L2mask_'+str(typ)+'.h5','r')
            m2=fpm[site][:]
            if case['stb']==1:
                fp=h5py.File(idir+'neon_1m/'+site+'_L1.h5','r')
            elif case['stb'] in [0,30]:
                fp=h5py.File(idir+'neon_30m/'+site+'_L30.h5','r')
            if 'vertical' in k:
                continue
            if 'q' in k:
                continue
            elif 'profile' in k:
                continue
            if k in exclude:
                continue
            if k == 'zzd':
                out.extend([fp.attrs['tow_height']-fp.attrs['zd']]*int(np.sum(m2)))
            elif k == 'SITE':
                out.extend([site]*int(np.sum(m2)))
            else:
                out.extend(fp[k][m2])
            processed=True
            fp.close()
            fpm.close()
        if not processed:
            continue
        if k=='SITE':
            fpo.create_dataset(k,data=np.array(out[:],dtype='S'))
        else:
            fpo.create_dataset(k,data=np.array(out[:]))
        fpo.close()
    print()

