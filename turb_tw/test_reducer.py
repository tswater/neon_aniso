# script to strip all un-needed information from a raw neon file

import os
import h5py
from subprocess import run
from datetime import datetime
import uuid

def downsize(fp,site):
    del fp[site+'/dp01']
    del fp[site+'/dp0p/qfqm']
    for k in fp[site+'/dp0p/data/'].keys():
        if k not in ['irgaTurb','soni']:
            del fp[site+'/dp0p/data/'+k]
    for k in fp[site+'/dp0p/data/irgaTurb'].keys():
        for kk in fp[site+'/dp0p/data/irgaTurb/'+k].keys():
            if kk not in ['rtioMoleDryCo2','rtioMoleDryH2o']:
                del fp[site+'/dp0p/data/irgaTurb/'+k+'/'+kk]
    for k in fp[site+'/dp0p/data/soni'].keys():
        for kk in fp[site+'/dp0p/data/soni/'+k].keys():
            if kk not in ['veloXaxs','veloYaxs','veloZaxs','veloSoni']:
                del fp[site+'/dp0p/data/soni/'+k+'/'+kk]

file='NEON.D19.BONA.IP0.00200.001.ecte.2023-06-06.l0p.h5'
site='BONA'
fp=h5py.File(file,'r+')
downsize(fp,site)
fp.close()
tmpname=str(uuid.uuid4())
run('mv '+file+' '+tmpname,shell=True)
run('h5repack --filter=GZIP=1 '+tmpname+' '+file,shell=True)
run('rm '+tmpname,shell=True)


