import os
import h5py
import numpy as np

out={}

uvw_only=True
include_static=False

exclude=['U_SIGMA','V_SIGMA','W_SIGMA','T_SONIC_SIGMA','H2O_SIGMA','CO2_SIGMA','RHO','CO2','U','V','W']

idir='/home/tswater/Documents/Elements_Temp/NEON/'
odir='/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data_v2/'

if not uvw_only:
    out['U_H2O']={}
    out['U_CO2']={}
    out['S_H2O']={}
    out['S_CO2']={}

out['U_UVWT']={}
out['S_UVWT']={}
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
start1=True
start2=True
for site in sites:
    print(site)

    fp1=h5py.File(idir+'neon_1m/'+site+'_L1.h5','r')
    fp30=h5py.File(idir+'neon_30m/'+site+'_L30.h5','r')

    ##################
    #### UNSTABLE ####
    time=fp30['TIME'][:]
    N30=len(time)

    mu=np.ones((N30,),dtype=bool)
    mu=mu&(fp30['L_MOST'][:]<=0)

    mu_t=core_q(mu.copy(),fp30,['Ustr','Vstr','Wstr','UU','VV','WW','T_SONIC_SIGMA','VPT'])
    if not uvw_only:
        mu_h=core_q(mu.copy(),fp30,['H2O','H2O_SIGMA','LE','RHO'],['qLE','qH2O'])
        mu_c=core_q(mu.copy(),fp30,['CO2','CO2_SIGMA','H2O','RHO'],['qCO2FX','qCO2'])

    ################
    #### STABLE ####
    time=fp1['TIME'][:]
    N1=len(time)

    ms=np.ones((N1,),dtype=bool)
    ms=ms&(fp1['L_MOST'][:]>0)

    ms_t=core_q(ms.copy(),fp1,['Ustr','Vstr','Wstr','UU','VV','WW','T_SONIC_SIGMA','VPT'])
    if not uvw_only:
        ms_h=core_q(ms.copy(),fp1,['H2O','H2O_SIGMA','LE','RHO'],['qLE','qH2O'])
        ms_c=core_q(ms.copy(),fp1,['CO2','CO2_SIGMA','H2O','RHO'],['qCO2FX','qCO2'])


    if not uvw_only:
        print('   |     |UUUUU|SSSSS|UH2O |UCO2 |UUVWT|SH2O |SCO2 |SUVWT|')
        print('   |PERC |'+str(np.sum(mu)/N30*100)[0:5]+'|'+str(np.sum(ms)/N1*100)[0:5]+'|'+str(np.sum(mu_h)/N30*100)[0:5]+'|'+str(np.sum(mu_c)/N30*100)[0:5]+'|'+str(np.sum(mu_t)/N30*100)[0:5]+'|'+\
                str(np.sum(ms_h)/N1*100)[0:5]+'|'+str(np.sum(ms_c)/N1*100)[0:5]+'|'+str(np.sum(ms_t)/N1*100)[0:5]+'|',flush=True)
    for k in fp30.keys():
        if 'vertical' in k:
            continue
        if 'q' in k:
            continue
        if k in exclude:
            continue
        if start1:
            if not uvw_only:
                out['U_H2O'][k]=[]
                out['U_CO2'][k]=[]
                out['S_H2O'][k]=[]
                out['S_CO2'][k]=[]
            out['U_UVWT'][k]=[]
            out['S_UVWT'][k]=[]
        if not uvw_only:
            out['U_H2O'][k].extend(fp30[k][mu_h])
            out['U_CO2'][k].extend(fp30[k][mu_c])
            out['S_H2O'][k].extend(fp1[k][ms_h])
            out['S_CO2'][k].extend(fp1[k][ms_c])

        out['U_UVWT'][k].extend(fp30[k][mu_t])
        out['S_UVWT'][k].extend(fp1[k][ms_t])
    start1=False

    if include_static:
        for k in fp1.attrs.keys():
            if start2:
                if not uvw_only:
                    out['U_H2O'][k]=[]
                    out['U_CO2'][k]=[]
                    out['S_H2O'][k]=[]
                    out['S_CO2'][k]=[]
                out['U_UVWT'][k]=[]
                out['S_UVWT'][k]=[]
            if not uvw_only:
                out['U_H2O'][k].extend([fp30.attrs[k]]*int(np.sum(mu_h)))
                out['U_CO2'][k].extend([fp30.attrs[k]]*int(np.sum(mu_c)))
                out['S_H2O'][k].extend([fp1.attrs[k]]*int(np.sum(ms_h)))
                out['S_CO2'][k].extend([fp1.attrs[k]]*int(np.sum(ms_c)))
            out['U_UVWT'][k].extend([fp30.attrs[k]]*int(np.sum(mu_t)))
            out['S_UVWT'][k].extend([fp1.attrs[k]]*int(np.sum(ms_t)))

    if start2:
        k='SITE'
        if not uvw_only:
            out['U_H2O'][k]=[]
            out['U_CO2'][k]=[]
            out['S_H2O'][k]=[]
            out['S_CO2'][k]=[]

        out['U_UVWT'][k]=[]
        out['S_UVWT'][k]=[]

    start2=False
    if not uvw_only:
        out['U_H2O']['SITE'].extend([site]*int(np.sum(mu_h)))
        out['U_CO2']['SITE'].extend([site]*int(np.sum(mu_c)))
        out['S_H2O']['SITE'].extend([site]*int(np.sum(ms_h)))
        out['S_CO2']['SITE'].extend([site]*int(np.sum(ms_c)))
    out['U_UVWT']['SITE'].extend([site]*int(np.sum(mu_t)))
    out['S_UVWT']['SITE'].extend([site]*int(np.sum(ms_t)))
    print('     DONE',flush=True)
print()
print('SAVING...')
print()
for typ in out.keys():
    print('Saving '+str(typ))
    outname='NEON_TW_'+str(typ)+'.h5'
    fp=h5py.File(odir+outname,'w')
    for k in out[typ].keys():
        print('    '+k)
        if k in ['description','creation_time_utc','last_updated_utc']:
            continue
        if k=='SITE':
            fp.create_dataset(k,data=np.array(out[typ][k][:],dtype='S'))
        else:
            fp.create_dataset(k,data=np.array(out[typ][k][:]))
    fp.close()












