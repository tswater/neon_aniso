# for unstable periods only, create datasets for anisotropy by timescale
import numpy as np
import os
import h5py
from datetime import datetime,timedelta
import pickle
from mpi4py import MPI

stbctf=.1 #minimum stability
ctf=.005 #maximum acceptable nans as a fraction
tctf=180 #minimum length of time to count in minutes

idir='/home/tswater/tyche/data/neon/raw_streamwise/'
odir='/home/tswater/tyche/data/neon/mrd_30/'
sites=os.listdir(idir)

# MPI4PY Stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def aniso(u,v,w):
    up=u-np.nanmean(u)
    vp=v-np.nanmean(v)
    wp=w-np.nanmean(w)

    uu=np.nanmean(up*up)
    vv=np.nanmean(vp*vp)
    ww=np.nanmean(wp*wp)
    uv=np.nanmean(up*vp)
    uw=np.nanmean(up*wp)
    vw=np.nanmean(vp*wp)

    m=(np.sum(np.abs(u)>40)==0)
    m=m&(np.sum(np.abs(v)>40)==0)
    m=m&(np.sum(np.abs(w)>40)==0)

    for i in [uu,vv,ww,uv,uw,vw]:
        m=m&(~np.isnan(i))
    if not m:
        return float('nan'),float('nan')

    k=uu+vv+ww
    ani=np.ones((3,3))*-9999
    ani[0,0]=uu/k-1/3
    ani[1,1]=vv/k-1/3
    ani[2,2]=ww/k-1/3
    ani[0,1]=uv/k
    ani[1,0]=uv/k
    ani[2,0]=uw/k
    ani[0,2]=uw/k
    ani[1,2]=vw/k
    ani[2,1]=vw/k

    try:
        lams=np.linalg.eig(ani[:,:])[0]
        lams.sort()
        lams=lams[::-1]
        xb=lams[0]-lams[1]+.5*(3*lams[2]+1)
        yb=np.sqrt(3)/2*(3*lams[2]+1)
    except:
        return float('nan'),float('nan')
    return yb,xb

for site in sites[rank::size]:
    out={'yb1':[],'yb2':[],'yb5':[],'yb10':[],'yb15':[],'yb30':[],
         'xb1':[],'xb2':[],'xb5':[],'xb10':[],'xb15':[],'xb30':[]}

    months=os.listdir(idir+site)
    months.sort()
    fps=h5py.File('/home/tswater/Documents/Elements_Temp/NEON/neon_30m/'+site+'_L30.h5','r')
    time=fps['TIME'][:]
    lmost=(fps.attrs['tow_height']-fps.attrs['zd'])/fps['L_MOST']
    n1=len(lmost)
    dn=20*60
    for month in months[5:6]:
        print(month)
        fp=h5py.File(idir+site+'/'+month,'r')
        u=fp['Ustr'][:]
        v=fp['Vstr'][:]
        w=fp['Wstr'][:]
        n=int(len(w)/(20*60))
        t0=datetime(int(month[8:12]),int(month[13:15]),1,0,0)
        try:
            tf=datetime(int(month[8:12]),int(month[13:15])+1,1,0,0)
        except Exception:
            tf=datetime(int(month[8:12])+1,1,1,0,0)
        m=time>=(t0-datetime(1970,1,1,0,0)).total_seconds()
        m=m&(time<(tf-datetime(1970,1,1,0,0)).total_seconds())
        lmost=(fps.attrs['tow_height']-fps.attrs['zd'])/fps['L_MOST'][m]
        for t in range(n):
            print('.',end='',flush=True)
            for scl in [1,2,5,10,15,30]:
                if (t<=(scl+1))|(t>=(n-scl-1)):
                    out['yb'+str(scl)].append(float('nan'))
                    out['xb'+str(scl)].append(float('nan'))
                else:
                    if t/30-np.floor(t/30)>.5:
                        try:
                            unst=(lmost[int(np.floor(t/30))]<0)&(lmost[int(np.floor(t/30))+1]<0)
                        except Exception:
                            unst=False
                    else:
                        try:
                            unst=(lmost[int(np.floor(t/30))]<0)&(lmost[int(np.floor(t/30))-1]<0)
                        except Exception:
                            unst=False
                    if not unst:
                        out['yb'+str(scl)].append(float('nan'))
                        out['xb'+str(scl)].append(float('nan'))
                    else:
                        yb,xb=aniso(u[t*dn-int(dn*scl/2):t*dn+int(dn*scl/2)],v[t*dn-int(dn*scl/2):t*dn+int(dn*scl/2)],w[t*dn-int(dn*scl/2):t*dn+int(dn*scl/2)])
                        out['yb'+str(scl)].append(yb)
                        out['xb'+str(scl)].append(xb)

    pickle.dump(out,open(odir+site+'_aniscale.p','wb'))
