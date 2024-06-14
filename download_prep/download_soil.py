import subprocess
import os
from mpi4py import MPI
import datetime

# MPI4PY Stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

text_dir='/home/tsw35/soteria/data/NEON/download_prep/textfilesoil'

sites = {'YELL':'D12','TREE':'D05','STEI':'D05','WREF':'D16',
         'ABBY':'D16','SCBI':'D02','MLBS':'D07','BLAN':'D02',
         'ONAQ':'D15','MOAB':'D13','CLBJ':'D11','ORNL':'D07',
         'GRSM':'D07','LAJA':'D04','GUAN':'D04','OAES':'D11',
         'WOOD':'D09','NOGP':'D09','DCFS':'D09','JORN':'D14',
         'BART':'D01','UNDE':'D05','HARV':'D01','SERC':'D02',
         'UKFS':'D06','KONZ':'D06','KONA':'D06','PUUM':'D20',
         'JERC':'D03','OSBS':'D03','DSNY':'D03','STER':'D10',
         'RMNP':'D10','NIWO':'D13','CPER':'D10','TEAK':'D17',
         'SOAP':'D17','SJER':'D17','SRER':'D14','TOOL':'D18',
         'HEAL':'D19','DEJU':'D19','BONA':'D19','BARR':'D18',
         'TALL':'D08','LENO':'D08','DELA':'D08'}



for site in list(sites.keys())[rank::size]:
    try:
        subprocess.run('mkdir ../'+'soil/'+site,shell=True)
    except:
        pass
    start_date =datetime.date(2017,1,1)
    end_date   =datetime.date(2024,1,1)
    dt = (end_date-start_date).days
    
    fp = open(text_dir+'/'+site+'.txt','w')
    om=0
    for i in range(dt):
        m_date=start_date+datetime.timedelta(days=i)
        if m_date.month==om:
            continue
        else:
            om=m_date.month
        wstr="https://data.neonscience.org/api/v0/data/package/DP1.00040.001/"
        mstr=str(m_date.month)
        if(m_date.month<10):
            mstr='0'+mstr
            dstr=str(m_date.day)
        if(m_date.day<10):
            dstr='0'+dstr
        datestr = str(m_date.year)+'-'+mstr
        outname = site+'-'+datestr+'.zip'
        wstr=wstr+site+'/'+datestr+'?package=basic'
        fp.write(wstr+'\n')
    print(site+' complete',flush=True)
    fp.close()

#### NOW ACTUALLY DOWNLOAD #### 
for site in list(sites.keys())[rank::size]:
    print(site,flush=True)
    os.chdir('/home/tsw35/soteria/data/NEON/soil/'+site)
    cmd = 'wget -nv -i '+text_dir+'/'+site+'.txt'
    print(cmd)
    subprocess.run(cmd,shell=True)
    



