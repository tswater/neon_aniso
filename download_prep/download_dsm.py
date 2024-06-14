import subprocess
import os
from mpi4py import MPI
import datetime
import json


# MPI4PY Stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

text_dir='/home/tsw35/soteria/data/NEON/download_prep/textfileprecip'

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



start_date =datetime.date(2020,1,1)
end_date   =datetime.date(2024,1,1)
dt = (end_date-start_date).days

sitelist=list(sites.keys())
sitelist.sort()

sitelist=['ORNL','PUUM','UNDE']

for site in sitelist[rank::size]:
    os.chdir('/home/tsw35/soteria/data/NEON/download_prep')
    print(site,flush=True)
    try:
        subprocess.run('mkdir ../'+'d_workspace/'+site,shell=True)
    except:
        pass
    
    try:
        subprocess.run('mkdir ../'+'dsm/'+site,shell=True)
    except:
        pass
    try:
        subprocess.run('mkdir ../'+'dtm/'+site,shell=True)
    except:
        pass
    if site in ['BARR','ORNL','NIWO','PUUM']:
        start_date=datetime.date(2018,1,1)
    else:
        start_date =datetime.date(2021,1,1)
    end_date   =datetime.date(2024,1,1)
    dt = (end_date-start_date).days
    
    # download the list of data
    print(site+' identifying download list',flush=True)
    om=0
    os.chdir('/home/tsw35/soteria/data/NEON/d_workspace/'+site)
    for i in range(dt):
        m_date=start_date+datetime.timedelta(days=i)
        if m_date.month==om:
            continue
        else:
            om=m_date.month
        #https://data.neonscience.org/api/v0/data/DP3.30024.001/ABBY/2022-07?package=basic
        wstr="https://data.neonscience.org/api/v0/data/DP3.30024.001/"
        mstr=str(m_date.month)
        if(m_date.month<10):
            mstr='0'+mstr
            dstr=str(m_date.day)
        if(m_date.day<10):
            dstr='0'+dstr
        datestr = str(m_date.year)+'-'+mstr
        outname = site+'-'+datestr+'.json'
        wstr=wstr+site+'/'+datestr+'?package=basic'
        try:
            subprocess.run('wget -nv -O '+outname+' '+wstr,shell=True)
        except:
            continue

        try:
            with open(outname) as f:
                data = f.read()
            js = json.loads(data)
            files=js['data']['files']
        except:
            continue
        if len(files)<5:
            pass
        else:
            print(site+' download list identified; downloading',flush=True)
            break
    # clear out packages 
    for file in os.listdir('/home/tsw35/soteria/data/NEON/d_workspace/'+site):
        subprocess.run('rm '+file,shell=True)
    
    dwnld_list=[]

    # Start downloading real data
    for file in files:
        a=file['url']
        if '.tif' in a:
            dwnld_list.append(a)

    for a in dwnld_list:
        subprocess.run('wget -nv '+a+' tifs/',shell=True)

    subprocess.run('mv *DSM* ../../dsm/'+site+'/',shell=True)
    subprocess.run('mv *DTM* ../../dtm/'+site+'/',shell=True)

    



