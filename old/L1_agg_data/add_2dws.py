# ---------------------------------------------------------------- #
# ----------- CODE DESCRIPTION FOR ADD_RAD.py ------------------- #
# ---------------------------------------------------------------- #
# Add radiaton (netrad) and determine night vs day

import numpy as np
import h5py
import datetime
import os
import csv
import sys
from mpi4py import MPI
# MPI4PY Stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#### INPUT ####
is1m=bool(int(sys.argv[1]))
if is1m:
    dt_=1
else:
    dt_=30
base_dir='/home/tsw35/tyche/neon_'+str(dt_)+'m/'
index=np.zeros((10,))
# -------------------- #
# CONSTANTS AND INPUTS #
# -------------------- #
neon_dir   = '/home/tsw35/soteria/data/NEON/wind_2d/'

# -------------- #
# MAIN CODE LOOP #
# -------------- #
sites=os.listdir(neon_dir)
#sites=['ABBY']
for site in sites[rank::size]:
    if len(site)>4:
        continue
    if site == 'zips':
        continue
    print(site+': ',end='',flush=True)
    
    # Load in the base file
    fp_out=h5py.File(base_dir+site+'_L'+str(dt_)+'.h5','r+')
    time=fp_out['TIME'][:]
    print(time.shape)
    out_wind={}
    old_month=0
    count1=0
    count2={}
    
    # Find the heights
    num_levels=0
    for file in os.listdir(neon_dir+site):
        level=int(file[32:34])
        if level>num_levels:
            num_levels=level
    
    for t in time:
        count1=count1+1
        tdt=datetime.datetime.fromtimestamp(t, datetime.timezone.utc)
        if tdt.month!=old_month:
            # Check to see if file exists for this month
            old_month=tdt.month
            filelist=[]
            date_str=str(tdt.year)+'-'+f'{tdt.month:02}'
            for file in os.listdir(neon_dir+site):
                if (date_str in file):
                    if is1m and ('2min' in file):
                        filelist.append(file)
                    elif (not is1m) and ('30min' in file):
                        filelist.append(file)
            if len(filelist)>0:
                print('.',end='',flush=True)
                _load=True
            else:
                print("'",end='',flush=True)
                _load=False
        else:
            _load=False
        if _load:
            index=np.zeros((len(filelist),))
            ws_times_all={}
            ws_wind_all={}
            for file in filelist:
                # setup data structure
                if file not in ws_wind_all.keys():
                    ws_wind_all[file]=[]
                    ws_times_all[file]=[]

                # OPEN file and load in info
                with open(neon_dir+site+'/'+file) as read_r:
                    read_r = csv.reader(read_r)
                    for row in read_r:
                        if row[0] == 'startDateTime':
                            continue
                        ts = row[0]
                        dt = datetime.datetime(int(ts[0:4]),int(ts[5:7]),\
                            int(ts[8:10]),int(ts[11:13]),int(ts[14:16]))
                        tp = dt.replace(tzinfo=datetime.timezone.utc).timestamp()
                        ws_times_all[file].append(tp)
                        try:
                            ws_wind_all[file].append(float(row[2]))
                        except:
                            ws_wind_all[file].append(-9999)
                if is1m:
                    ts2_=np.linspace(0,len(ws_wind_all[file])*2-2,len(ws_wind_all[file]))
                    ts1_=np.linspace(0,len(ws_wind_all[file])*2-1,len(ws_wind_all[file]*2))
                    ws_times_all[file]=np.interp(ts1_,ts2_,ws_times_all[file])
                    ws_wind_all[file]=np.array(ws_wind_all[file],dtype=float)
                    ws_wind_all[file][ws_wind_all[file]==-9999]=float('nan')
                    ws_wind_all[file]=np.interp(ts1_,ts2_,ws_wind_all[file])
                    ws_wind_all[file][np.isnan(ws_wind_all[file])]=-9999

        # --------------------- #
        # PERFORM EACH TIMESTEP #
        # --------------------- #
        #add wind for heights without file
        if len(filelist)<num_levels:
            rep_heights=[]
            for file in filelist:
                height = int(file[32:34])
                if height not in rep_heights:
                    rep_heights.append(height)
            for i in range(1,num_levels+1):
                if i not in rep_heights:
                    if i not in out_wind.keys():
                        out_wind[i]=[]
                        count2[i]=0
                    out_wind[i].append(-9999)
                    count2[i]=count2[i]+1
        #
        #if (len(dp04_times)>index+1) and (dp04_times[index+1]==t):
        #    index=index+1
        
        # add wind for heights with a file
        ii=0
        for file in filelist:
            height = int(file[32:34])
            if height not in out_wind.keys():
                out_wind[height]=[]
                count2[height]=0
            if (len(ws_times_all[file][:])>(index[ii]+1)) and (ws_times_all[file][:][int(index[ii]+1)]==t):
                index[ii]=index[ii]+1
            else:
                a=np.where(ws_times_all[file][:]==t)[0]    
                
                if (len(a)==0) or (len(ws_wind_all[file])==0):
                    out_wind[height].append(-9999)
                    ii=ii+1
                    continue
                elif ws_wind_all[file][a[0]]==-9999:
                    out_wind[height].append(-9999)
                    ii=ii+1
                    continue
                else:
                    index[ii]=a[0]
            out_wind[height].append(ws_wind_all[file][int(index[ii])])
            count2[height]=count2[height]+1
            ii=ii+1
    # -------------------- #
    # CREATE HDF5 DATASETS #
    # -------------------- #
    print(count1)
    print(count2)
    try:
        fp_out.create_group('vertical_wind')
    except Exception:
        pass
    # Identify the base filename
    neon_dp4 = '/home/tsw35/soteria/data/NEON/dp04/'
    site_files=os.listdir(neon_dp4+site)
    neon_dp4=neon_dp4+site+'/'+site_files[0]
    
    fp3=h5py.File(neon_dp4,'r')
    heights=np.array(fp3[site].attrs['DistZaxsLvlMeasTow'])
    fp_out['vertical_wind'].attrs['heights']=heights
    for h in out_wind.keys():
        atr_nm='vertical_wind/WIND_'+str(heights[h-1])[2:-1]
        try:
            fp_out.create_dataset(atr_nm,data=np.array(out_wind[h]))
        except:
            fp_out[atr_nm][:]=np.array(out_wind[h])
        fp_out[atr_nm].attrs['source']='NEON 2D Wind'
        fp_out[atr_nm].attrs['missing_value']=-9999
        fp_out[atr_nm].attrs['height']=heights[h-1]
        fp_out[atr_nm].attrs['description']='2D Windspeed at Height'
        print('*',end='',flush=True)
        fp_out.attrs['last_updated_utc']=str(datetime.datetime.utcnow())
    print()


    
    
        
