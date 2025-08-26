import subprocess
import os
import datetime
import json

dwnld_dir='/home/tswater/Documents/tyche/data/neon/'

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except:
    rank=0
    size=1

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



start_date =datetime.date(2017,1,1)
end_date   =datetime.date(2024,1,1)
dt = (end_date-start_date).days

sitelist0=list(sites.keys())
sitelist0.sort()
sitelist=[]
for file in sitelist0:
    if (file in os.listdir(dwnld_dir+'h2oidx')):
        if (len(os.listdir(dwnld_dir+'h2oidx/'+file))>0):
            pass
        else:
            sitelist.append(file)
    else:
        sitelist.append(file)
print(sitelist)
#sitelist=['ABBY']

for site in sitelist[rank::size]:
    os.chdir(dwnld_dir)
    print(site,flush=True)
    try:
        subprocess.run('mkdir d_workspace/'+site,shell=True)
    except:
        continue

    try:
        subprocess.run('mkdir h2oidx/'+site,shell=True)
    except:
        pass

    # download the list of data
    print(site+' identifying download list',flush=True)
    om=0
    os.chdir('d_workspace/'+site)
    for i in range(dt):
        m_date=start_date+datetime.timedelta(days=i)
        if m_date.month==om:
            continue
        else:
            om=m_date.month
        for ver in [1,2]:
            #https://data.neonscience.org/api/v0/data/DP3.30024.001/ABBY/2022-07?package=basic
            wstr="https://data.neonscience.org/api/v0/data/DP3.30019.00"+str(ver)+"/"
            mstr=str(m_date.month)
            if(m_date.month<10):
                mstr='0'+mstr
                dstr=str(m_date.day)
            if(m_date.day<10):
                dstr='0'+dstr
            datestr = str(m_date.year)+'-'+mstr
            outname = site+'-'+datestr+'.json'
            wstr=wstr+site+'/'+datestr+'?package=basic'
            subprocess.run('rm '+outname,stderr=subprocess.STDOUT,stdout=subprocess.DEVNULL,shell=True)
            subprocess.run('wget -nv -O '+outname+' '+wstr,shell=True)

            try:
                with open(outname) as f:
                    data = f.read()
                js = json.loads(data)
                files=js['data']['files']
                if len(files)<5:
                    continue
                print(outname+' verified for data',flush=True)
                subprocess.run('mkdir '+datestr,shell=True)
                subprocess.run('mv '+outname+' '+datestr+'/',shell=True)
            except Exception as e:
                subprocess.run('rm '+outname,shell=True)
                print(e)
                continue

    # clear out packages
    for file in os.listdir(dwnld_dir+'d_workspace/'+site):
        if ('package' in file) or ('json' in file):
            subprocess.run('rm '+file,shell=True)

    for folder in os.listdir('./'):
        outname = site+'-'+folder+'.json'
        print('Downloading data from '+outname,flush=True)
        with open(folder+'/'+outname) as f:
            data = f.read()
        js = json.loads(data)
        files=js['data']['files']
        dwnld_list=[]

        # Start downloading real data
        for file in files:
            a=file['url']
            if ('.zip' in a):
                dwnld_list.append(a)

        os.chdir(folder)
        subprocess.run('rm '+outname,shell=True)
        for a in dwnld_list:
            subprocess.run('wget -nv '+a,shell=True)
        os.chdir('../')

    subprocess.run('mv * ../../h2oidx/'+site+'/',shell=True)
    #subprocess.run('mv *DTM* ../../dtm/'+site+'/',shell=True)





