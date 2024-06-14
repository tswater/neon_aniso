import subprocess
import datetime
import os

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

raw1='/home/tsw35/soteria/data/NEON/raw_data/'

end_date=datetime.datetime(2024,1,1,0,0)

#sites = {'RMNP':'D10','STEI':'D05'}
for site in sites.keys():
    try:
        subprocess.run('mkdir ../'+'raw_data/'+site,shell=True)
    except:
        pass
    
    #/home/tsw35/soteria/data/NEON/raw_data/
    #NEON.D04.LAJA.IP0.00200.001.ecte.2020-05-31.l0p.h5

    rawlist=os.listdir(raw1+site)

    rawlist.sort()
    try:
        first=rawlist[-1]
        start_date=datetime.datetime(int(first[33:37]),int(first[38:40]),int(first[41:43]))
    except:
        start_date=datetime.datetime(2017,1,1,0)

    if start_date>end_date:
        continue

    dt = (end_date-start_date).days
    

    fp = open('textfiles/'+site+'.txt','w')
    for i in range(dt):
        m_date=start_date+datetime.timedelta(days=i)
        wstr='https://storage.googleapis.com/neon-sae-files/ods/dataproducts/'
        mstr=str(m_date.month)
        dstr=str(m_date.day)
        if(m_date.month<10):
            mstr='0'+mstr
            dstr=str(m_date.day)
        if(m_date.day<10):
            dstr='0'+dstr
        datestr = str(m_date.year)+'-'+mstr+'-'+dstr
        wstr=wstr+'IP0/'+datestr+'/'+site+'/'+'NEON.'+sites[site]+'.'+\
            site+'.IP0.00200.001.ecte.'+datestr+'.l0p.h5.gz'
        fp.write(wstr+'\n')
    print(site+' complete',flush=True)
    fp.close()

