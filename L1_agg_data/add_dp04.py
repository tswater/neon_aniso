# ---------------------------------------------------------------- #
# ----------- CODE DESCRIPTION FOR ADD_DP04.py ------------------- #
# ---------------------------------------------------------------- #
# Add the important contents of the main product files to the base #
# h5 including stuff

import os
import netCDF4 as nc
import numpy as np
import h5py
import datetime
import ephem
import rasterio
import csv
import sys
import subprocess
from mpi4py import MPI
# MPI4PY Stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

is1m=bool(int(sys.argv[1]))
if is1m:
    dt_=1
    ds_='01m'
else:
    dt_=30
    ds_='30m'
base_dir='/home/tsw35/tyche/neon_'+str(dt_)+'m/'
index=0

# ------------------------- #
# USER INPUTS AND CONSTANTS #
# ------------------------- #
neon_dir = '/home/tsw35/soteria/data/NEON/dp04/'

outvar = {'T_SONIC_SIGMA':[],
'H':[],'LE':[],'PA':[],'VPD':[],'TA':[],'USTAR':[],'WS':[],
'RH':[],'T_SONIC':[],'VPT':[],'RHO':[],'H2O':[],'W':[],'CO2_SIGMA':[],
'H2O_SIGMA':[],'U':[],'V':[],'qT_SONIC':[],'qH2O':[],'qCO2':[],
'CO2':[],'CO2FX':[],'qH':[],'qLE':[],'qUSTAR':[],'qCO2FX':[],
'U_SIGMA':[],'V_SIGMA':[],'W_SIGMA':[],'qU':[],'qV':[],'qW':[],
'qsT_SONIC':[],'qsH2O':[],'qsCO2':[],'qsU':[],'qsV':[],'qsW':[]}

units = {'U_SIGMA':'m s-1','V_SIGMA':'m s-1','W_SIGMA':'m s-1',
'T_SONIC_SIGMA':'deg C','H':'W m-2','LE':'W m-2','PA':'kPa',
'VPD':'hPa','TA':'deg C','USTAR':'m s-1','WS':'m s-1',
'H2O_SIGMA':'mmolH2O mol-1','H2O':'mmolH2O mol-1',
'RH':'%','T_SONIC':'deg C','U':'m s-1','V':'m s-1',
'CO2_FLUX':'umolCo2 m-2 s-1','CO2':'umolCo2 mol-1'}

desc = {}
# -------------- #
# MAIN CODE LOOP #
# -------------- #
sites=os.listdir(neon_dir)
#sites=['TEAK','PUUM','NOGP','LAJA','JORN','BONA']
for site in sites[rank::size]:
    if len(site)>4:
        continue
    print(site+': ',end='',flush=True)
    # dictionary for storing data before loading to h5
    ovar=outvar.copy()
    for k in ovar.keys():
        ovar[k]=[]
    
    # Identify the base filename
    site_files=os.listdir(neon_dir+site)
    dp04_base=site_files[0][0:33]
    
    # Load in the base file
    fp_out=h5py.File(base_dir+site+'_L'+str(dt_)+'.h5','r+')
    time=fp_out['TIME'][:] 
    
    # Initialize site-level constants
    site_cnst={'lat':0,'lon':0,'elev':0,'zd':0,'towH':0,'canH':0}
    lat =0
    lon =0
    elev=0
    zd  =0
    towH=0
    canH=0
    utc_off=0
    
    # ----------------- #
    # LOOP THROUGH TIME #
    # ----------------- #
    old_month=0
    for t in time:
        tdt=datetime.datetime.fromtimestamp(t, datetime.timezone.utc)
        if tdt.month!=old_month:
            # Check to see if file exists for this month
            old_month=tdt.month
            try:
                fp_in=h5py.File(neon_dir+site+'/'+dp04_base+\
                  str(tdt.year)+'-'+f'{tdt.month:02}'+'.basic.h5','r')
                _load=True
                print('.',end='',flush=True)
                index=0
            except Exception:
                _load=False
                dp04_times=[]
                print("'",end='',flush=True)
        else:
            _load=False
        if _load:
            # ----------------------- #
            # LOAD IN DP04 MONTH FILE #
            # ----------------------- #
            for i in range(10):
                try:
                    th=str(i)
                    fp_in['/'+site+'/dp01/data/soni/000_0'+th+\
                          '0_'+ds_+'/tempSoni/']['timeBgn'][:]
                    break;
                except:
                    pass
            tst=fp_in['/'+site+'/dp01/data/soni/000_0'+th+\
                      '0_'+ds_+'/tempSoni/']['timeBgn'][:]
            
            # Create an array of timestamps
            dp04_times=[]
            for ts in tst:
                dt = datetime.datetime(int(ts[0:4]),int(ts[5:7]),\
                     int(ts[8:10]),int(ts[11:13]),int(ts[14:16]))
                dp04_times.append(dt.replace\
                     (tzinfo=datetime.timezone.utc).timestamp())
            
            # load in other arrays of interest
            TS=fp_in['/'+site+'/dp01/data/soni/000_0'+th+\
                          '0_'+ds_+'/tempSoni/']['mean'][:]
            TSS=np.sqrt(fp_in['/'+site+'/dp01/data/soni/000_0'+\
                             th+'0_'+ds_+'/tempSoni/']['vari'][:])
            TA= fp_in['/'+site+'/dp01/data/soni/000_0'+th+\
                      '0_'+ds_+'/tempAir/']['mean'][:]
            WS= fp_in['/'+site+'/dp01/data/soni/000_0'+th+'0_'+ds_+'/'+\
                      'veloXaxsYaxsErth/']['mean'][:]
            U= fp_in['/'+site+'/dp01/data/soni/000_0'+th+'0_'+ds_+'/'+\
                      'veloXaxsErth/']['mean'][:]
            V= fp_in['/'+site+'/dp01/data/soni/000_0'+th+'0_'+ds_+'/'+\
                      'veloYaxsErth/']['mean'][:]
            W= fp_in['/'+site+'/dp01/data/soni/000_0'+th+'0_'+ds_+'/'+\
                      'veloZaxsErth/']['mean'][:]
            U_SIGMA= np.sqrt(fp_in['/'+site+'/dp01/data/soni/000_0'+th+'0_'+ds_+'/'+\
                      'veloXaxsErth/']['vari'][:])
            V_SIGMA= np.sqrt(fp_in['/'+site+'/dp01/data/soni/000_0'+th+'0_'+ds_+'/'+\
                      'veloYaxsErth/']['vari'][:])
            W_SIGMA= np.sqrt(fp_in['/'+site+'/dp01/data/soni/000_0'+th+'0_'+ds_+'/'+\
                      'veloZaxsErth/']['vari'][:])
            PA= fp_in['/'+site+'/dp01/data/h2oTurb/000_0'+th+\
                      '0_'+ds_+'/presAtm']['mean'][:]
            TD = fp_in['/'+site+'/dp01/data/h2oTurb/000_0'+th+\
                       '0_'+ds_+'/tempDew']['mean'][:]
            RH = 100*(np.exp((17.625*TD)/(243.04+TD))/\
                      np.exp((17.625*TA)/(243.04+TA)))
            RH[RH>100]=100
            RH[RH<0]=-9999
            H2O= fp_in['/'+site+'/dp01/data/h2oTurb/000_0'+th+\
                       '0_'+ds_+'/rtioMoleDryH2o']['mean'][:]
            H2OS =np.sqrt(np.array(fp_in['/'+site+\
                        '/dp01/data/h2oTurb/000_0'+th+\
                        '0_'+ds_+'/rtioMoleDryH2o']['vari'][:]))
            USTAR=fp_in['/'+site+'/dp04/data/fluxMome/turb']\
                        ['veloFric'][:]
            LE=fp_in['/'+site+'/dp04/data/fluxH2o/turb']['flux'][:]
            H=fp_in['/'+site+'/dp04/data/fluxTemp/turb']['flux'][:]
            CO2FX=fp_in['/'+site+'/dp04/data/fluxCo2/turb']['flux'][:]
            CO2=fp_in['/'+site+'/dp01/data/co2Turb/000_0'+th+\
                       '0_'+ds_+'/rtioMoleDryCo2']['mean'][:]
            CO2_SIGMA=np.sqrt(fp_in['/'+site+'/dp01/data/co2Turb/000_0'+th+\
                       '0_'+ds_+'/rtioMoleDryCo2']['vari'][:])
            
            qCO2= fp_in['/'+site+'/dp01/qfqm/co2Turb/000_0'+th+\
                        '0_'+ds_+'/rtioMoleDryCo2']['qfFinl'][:]
            qTS=fp_in['/'+site+'/dp01/qfqm/soni/000_0'+th+\
                          '0_'+ds_+'/tempSoni/']['qfFinl'][:]
            qU=fp_in['/'+site+'/dp01/qfqm/soni/000_0'+th+\
                          '0_'+ds_+'/veloXaxsErth/']['qfFinl'][:]
            qV=fp_in['/'+site+'/dp01/qfqm/soni/000_0'+th+\
                          '0_'+ds_+'/veloYaxsErth/']['qfFinl'][:]
            qW=fp_in['/'+site+'/dp01/qfqm/soni/000_0'+th+\
                          '0_'+ds_+'/veloZaxsErth/']['qfFinl'][:]

            qH2O= fp_in['/'+site+'/dp01/qfqm/h2oTurb/000_0'+th+\
                        '0_'+ds_+'/rtioMoleDryH2o']['qfFinl'][:]
            try:
                qLE=fp_in['/'+site+'/dp04/qfqm/fluxH2o/turb']['qfFinl'][:]
                qH=fp_in['/'+site+'/dp04/qfqm/fluxTemp/turb']['qfFinl'][:]
                qCO2FX=fp_in['/'+site+'/dp04/qfqm/fluxCo2/turb']['qfFinl'][:]
                qUSTAR=fp_in['/'+site+'/dp04/qfqm/fluxMome/turb']['qfFinl'][:]
                if is1m:
                    qLE=np.repeat(qLE,30)
                    qH=np.repeat(qH,30)
                    qCO2FX=np.repeat(qCO2FX,30)
                    qUSTAR=np.repeat(qUSTAR,30)
            except Exception as e:
                Nn=len(qCO2)
                qLE=np.ones((Nn,))*-1
                qH=np.ones((Nn,))*-1
                qCO2FX=np.ones((Nn,))*-1
                qUSTAR=np.ones((Nn,))*-1

            ### TRY TO ADD SCIENCE FLAGS ###
            try:
                qsCO2= fp_in['/'+site+'/dp01/qfqm/co2Turb/000_0'+th+\
                        '0_'+ds_+'/rtioMoleDryCo2']['qfSci'][:]
            except Exception as e:
                try:
                    qsCO2= fp_in['/'+site+'/dp01/qfqm/co2Turb/000_0'+th+\
                        '0_'+ds_+'/rtioMoleDryCo2']['qfSciRevw'][:]
                except Exception as e:
                    qsCO2=np.ones((len(qCO2),))*-1
            
            try:    
                qsTS=fp_in['/'+site+'/dp01/qfqm/soni/000_0'+th+\
                          '0_'+ds_+'/tempSoni/']['qfSciRevw'][:]
            except Exception as e:
                try: 
                    qsTS=fp_in['/'+site+'/dp01/qfqm/soni/000_0'+th+\
                          '0_'+ds_+'/tempSoni/']['qfSci'][:]
                except Exception as e:
                    qsTS=np.ones((len(qCO2),))*-1
            
            try:    
                qsU=fp_in['/'+site+'/dp01/qfqm/soni/000_0'+th+\
                          '0_'+ds_+'/veloXaxsErth/']['qfSci'][:]
            except Exception as e:
                try: 
                    qsU=fp_in['/'+site+'/dp01/qfqm/soni/000_0'+th+\
                          '0_'+ds_+'/veloXaxsErth/']['qfSciRevw'][:]
                except Exception as e:
                    qsU=np.ones((len(qCO2),))*-1
            
            try:
                qsV=fp_in['/'+site+'/dp01/qfqm/soni/000_0'+th+\
                          '0_'+ds_+'/veloYaxsErth/']['qfSci'][:]
            except Exception as e:
                try:
                    qsV=fp_in['/'+site+'/dp01/qfqm/soni/000_0'+th+\
                          '0_'+ds_+'/veloYaxsErth/']['qfSciRevw'][:]
                except Exception as e:
                    qsV=np.ones((len(qCO2),))*-1

            try:
                qsW=fp_in['/'+site+'/dp01/qfqm/soni/000_0'+th+\
                          '0_'+ds_+'/veloZaxsErth/']['qfSci'][:]
            except Exception as e:
                try:
                    qsW=fp_in['/'+site+'/dp01/qfqm/soni/000_0'+th+\
                          '0_'+ds_+'/veloZaxsErth/']['qfSciRevw'][:]
                except Exception as e:
                    qsW=np.ones((len(qCO2),))*-1
            
            
            
            try:    
                qsH2O= fp_in['/'+site+'/dp01/qfqm/h2oTurb/000_0'+th+\
                        '0_'+ds_+'/rtioMoleDryH2o']['qfSci'][:]
            except Exception as e:
                try:
                    qsH2O= fp_in['/'+site+'/dp01/qfqm/h2oTurb/000_0'+th+\
                        '0_'+ds_+'/rtioMoleDryH2o']['qfSciRevw'][:]
                except Exception as e:
                    qsH2O=np.ones((len(qCO2),))*-1

            
            ### SECOND LEVEL COMPUTATIONS ###
            svp = .61121*np.exp((18.678-(TS)/234.5)*((TS)/(257.14+(TS))))
            vpd = svp*(1-RH/100)*10
            vpd[RH<0]=-9999
            e  = svp*RH/100
            r  = .622*e/(PA-e)
            # potential temperature and virtual potential temperature
            pt = (TS+273)*(100/PA)**(2/7)
            vpt= pt*(1+.61*r)
            vpt[PA==-9999]=-9999
            vpt[RH==-9999]=-9999
            vpt[TS==-9999]=-9999
            # air density calculations
            Ra = 286.9
            Rw = 461.5
            rho = ((PA-e)/(Ra*(TS+273))+(e)/(Rw*(TS+273)))*1000
            rho[PA==-9999]=-9999
            rho[RH==-9999]=-9999
            rho[TS==-9999]=-9999
            
            # Load in monthly values (planar fit axis coeff)
            pfangx=fp_in[site].attrs['Pf$AngEnuXaxs'][0]
            pfangy=fp_in[site].attrs['Pf$AngEnuYaxs'][0]
            pfofst=fp_in[site].attrs['Pf$Ofst'][0]
            try:
                pfangx=float(pfangx)
            except Exception:
                pfangx=-9999
            try:
                pfangy=float(pfangy)
            except Exception:
                pfangy=-9999
            try:
                pfofst=float(pfofst)
            except Exception:
                pfofst=-9999
    
            # Load in Constants
            if lat==0:
                try: lat=float(fp_in[site].attrs['LatTow'][0])
                except Exception:pass
            if lon==0:
                try: lon=float(fp_in[site].attrs['LonTow'][0])
                except Exception:pass
            if elev==0:
                try: elev=float(fp_in[site].attrs['ElevRefeTow'][0])
                except Exception:pass
            if zd==0:
                try: zd=float(fp_in[site].attrs['DistZaxsDisp'][0])
                except Exception:pass
            if towH==0:
                try: towH=float(fp_in[site].attrs['DistZaxsTow'][0])
                except Exception:pass
            if canH==0:
                try: canH=float(fp_in[site].attrs['DistZaxsCnpy'][0])
                except Exception:pass
            if utc_off ==0:
                timezones={'HST':-10,'MST':-7,'PST/MST':-7.5,'PST':-8,'EST':-5,
                           'CST':-6,'AKST':-9,'AST':-4}
                tz_site = str(fp_in[site].attrs['ZoneTime'][0])[2:-1]
                utc_off = timezones[tz_site]
            
    
        # --------------------- #
        # PERFORM EACH TIMESTEP #
        # --------------------- #
        #get dp04 index
        if (len(dp04_times)>index+1) and (dp04_times[index+1]==t):
            index=index+1
        else:
            a=np.where(dp04_times[index:]==t)[0]
            if len(a)==0:
                # if this time is not present in dp04 or file does not
                # exist, we fill with empty
                for var in ovar.keys():
                    ovar[var].append(-9999)
                continue
            else:
                index=a[0]
        
        # now that we have an index, load all the data
        ovar['T_SONIC_SIGMA'].append(TSS[index])
        ovar['T_SONIC'].append(TS[index])
        ovar['WS'].append(WS[index])
        ovar['RH'].append(RH[index])
        ovar['TA'].append(TA[index])
        ovar['PA'].append(PA[index])
        ovar['H2O'].append(H2O[index])
        ovar['H2O_SIGMA'].append(H2OS[index])
        ovar['qH2O'].append(qH2O[index])
        ovar['qT_SONIC'].append(qTS[index])
        ovar['VPD'].append(vpd[index])
        ovar['VPT'].append(vpt[index])
        ovar['RHO'].append(rho[index])
        ovar['U'].append(U[index])
        ovar['V'].append(V[index])
        ovar['W'].append(W[index])
        ovar['CO2_SIGMA'].append(CO2_SIGMA[index])
        ovar['CO2'].append(CO2[index])
        ovar['U_SIGMA'].append(U_SIGMA[index])
        ovar['V_SIGMA'].append(V_SIGMA[index])
        ovar['W_SIGMA'].append(W_SIGMA[index])
        ovar['qCO2'].append(qCO2[index])
        ovar['qCO2FX'].append(qCO2FX[index])
        ovar['qUSTAR'].append(qUSTAR[index])
        ovar['qLE'].append(qLE[index])
        ovar['qH'].append(qH[index])
        ovar['qU'].append(qU[index])
        ovar['qV'].append(qV[index])
        ovar['qW'].append(qW[index])
        ovar['qsU'].append(qsU[index])
        ovar['qsV'].append(qsV[index])
        ovar['qsW'].append(qsW[index])
        ovar['qsH2O'].append(qsH2O[index])
        ovar['qsT_SONIC'].append(qsTS[index])
        ovar['qsCO2'].append(qsCO2[index])
        if not is1m:
            ovar['CO2FX'].append(CO2FX[index])
            ovar['H'].append(H[index])
            ovar['LE'].append(LE[index])
            ovar['USTAR'].append(USTAR[index])
        
    # -------------------- #
    # CREATE HDF5 DATASETS #
    # -------------------- #
    for key in ovar.keys():
        if is1m and (key in ['LE','H','CO2FX','USTAR']):
            continue
        try:
            fp_out.create_dataset(key,data=np.array(ovar[key][:]))
        except:
            fp_out[key][:]=np.array(ovar[key][:])
        if 'q' in key:
            fp_out[key].attrs['missing_value']=-1
        else:
            fp_out[key].attrs['missing_value']=-9999
        fp_out[key].attrs['source']='NEON_dp04'
        if key in units.keys():
            fp_out[key].attrs['units']=units[key]
        if key in desc.keys():
             fp_out[key].attrs['description']=desc[key]
    fp_out.attrs['utc_off']=utc_off
    fp_out.attrs['elevation']=elev
    fp_out.attrs['tow_height']=towH
    fp_out.attrs['canopy_height']=canH
    fp_out.attrs['zd']=zd
    fp_out.attrs['lat']=lat
    fp_out.attrs['lon']=lon
    fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())
    print('*',flush=True)
