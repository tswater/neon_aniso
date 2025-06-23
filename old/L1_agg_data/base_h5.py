# ---------------------------------------------------------------- #
# ----------- CODE DESCRIPTION FOR BASE_H5.py -------------------- #
# ---------------------------------------------------------------- #
# Creates empty h5 files for all the files of interest. These      #
# files will have basic structure, time, etc. but nothing else     #

# ------- #
# IMPORTS #
# ------- #
import numpy as np
import h5py
import datetime
import os
import subprocess
import sys
#### INPUT ####
is1m=bool(int(sys.argv[1]))
if is1m:
    dt_=1
else:
    dt_=30
neon_dir='/home/tsw35/xTyc_shared/NEON_raw_data/'
out_dir='/home/tsw35/tyche/neon_'+str(dt_)+'m/'

# ----------------------------- #
# USER PARAMETERS and CONSTANTS #
# ----------------------------- #
start_year=2017
start_month=1
end_year=2023
end_month=12
end_day=31
neon_dir='/home/tsw35/xTyc_shared/NEON_raw_data/'
out_dir='/home/tsw35/tyche/neon_'+str(dt_)+'m/'
replace=True
# ------------------- #
# SETUP and MAIN LOOP #
# ------------------- #
# loop through all NEON files and generate hdf5 files
files=os.listdir(neon_dir)
#files=['ABBY']
for file in files:
    if 'scripts' in file:
        continue
    filename=file+'_L'+str(dt_)+'.h5'
    
    # check if replacing, skipping or creating
    if (filename in os.listdir(out_dir)):
        if replace:
            print('Replacing '+filename)
            try:
                subprocess.run('rm '+out_dir+filename,shell=True)
            except:
                pass
        else:
            print('Skipping '+filename)
            continue
    else:
        print('Creating '+filename)
    fp=h5py.File(out_dir+file+'_L'+str(dt_)+'.h5')
    
    # add attributes
    fp.attrs['creation_time_utc']=str(datetime.datetime.utcnow())
    fp.attrs['last_updated_utc']=str(datetime.datetime.utcnow())
    fp.attrs['description']='NEON tower data and colocated data from other sources'
    
    # add time
    tzutc=datetime.timezone.utc
    times=[]
    start_time=datetime.datetime(start_year,start_month,1,0,0,tzinfo=tzutc)
    end_time=datetime.datetime(end_year,end_month,end_day,23,30,tzinfo=tzutc)
    dt=datetime.timedelta(minutes=dt_)
    time=start_time
    while time<=end_time:
        times.append(time.timestamp())
        time=time+dt
    fp.create_dataset('TIME',data=np.array(times))
    fp['TIME'].attrs['units']='seconds since 1970,UTC'
    
    
    
