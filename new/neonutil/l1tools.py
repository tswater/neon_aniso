# Tools for L1 processing, mainly used by the L1 driver script
import numpy as np
import h5py
import datetime
import os
from subprocess import run
from nutil import SITES

############# MAKE BASE ###############
# make base h5 file to add onto for L1
def make_base(scl,odir,d0=None,df=None,overwrite=False,sites=SITES):
    ''' scl   averaging scale in minutes, int
        odir  directory for output
        d0    datetime for first timestep,
        df    datetime for last timestep '''
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
        if d0 in None:
            d0=datetime.datetime(start_year,start_month,1,0,0,tzinfo=tzutc)
        if df in None:
            df=datetime.datetime(end_year,end_month,end_day,23,30,tzinfo=tzutc)

        dt=datetime.timedelta(minutes=scl)
        time=d0
        while time<=df:
            times.append(time.timestamp())
            time=time+dt
        fp.create_dataset('TIME',data=np.array(times))
        fp['TIME'].attrs['units']='seconds since 1970,UTC'
#################################################################



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

