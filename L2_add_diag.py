import numpy as np
import os
import h5py

l2dir='/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data_v2/'

def aniso(uu,vv,ww,uv,uw,vw):
    n=len(uu)
    m=uu<10000
    m=m&(vv<10000)
    m=m&(ww<10000)
    m=m&(uv>-9999)
    m=m&(uw>-9999)
    m=m&(vw>-9999)
    for i in [uu,vv,ww,uv,uw,vw]:
        m=m&(~np.isnan(i))

    k=uu[m]+vv[m]+ww[m]
    ani=np.ones((n,3,3))*-9999
    ani[m,0,0]=uu[m]/k-1/3
    ani[m,1,1]=vv[m]/k-1/3
    ani[m,2,2]=ww[m]/k-1/3
    ani[m,0,1]=uv[m]/k
    ani[m,1,0]=uv[m]/k
    ani[m,2,0]=uw[m]/k
    ani[m,0,2]=uw[m]/k
    ani[m,1,2]=vw[m]/k
    ani[m,2,1]=vw[m]/k
    return ani


for file in os.listdir(l2dir):
    print(file)
    fp=h5py.File(l2dir+file,'r+')
    uu=fp['UU'][:]
    vv=fp['VV'][:]
    ww=fp['WW'][:]

    N=len(uu)
    uv=np.zeros((N,))
    uw=np.zeros((N,))
    vw=np.zeros((N,))

    bij=aniso(uu,vv,ww,uv,uw,vw)
    yb=np.ones((N,))*-9999
    xb=np.ones((N,))*-9999
    counts=np.linspace(0,N,100)
    counts=counts.astype(int)
    for t in range(N):
        if np.sum(bij[t,:,:]==-9999)>0:
            continue
        if t in counts:
            print('.',end='',flush=True)
        lams=np.linalg.eig(bij[t,:,:])[0]
        lams.sort()
        lams=lams[::-1]
        xb[t]=lams[0]-lams[1]+.5*(3*lams[2]+1)
        yb[t]=np.sqrt(3)/2*(3*lams[2]+1)

    # OUTPUT
    fp.create_dataset('ANID_YB',data=yb)
    fp.create_dataset('ANID_XB',data=xb)
    fp.close()
