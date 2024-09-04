# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from calendar import monthrange

# %%
np.sqrt(1440*31)

# %%
fp=h5py.File('/home/tsw35/xTyc_shared/NEON_raw_data/ABBY/NEON.D16.ABBY.IP0.00200.001.ecte.2020-03-24.l0p.h5','r')


# %%
udata=fp['ABBY/dp0p/data/soni/000_050/veloXaxs'][:]
udata[np.abs(udata)>50]=float('nan')
wdata=fp['ABBY/dp0p/data/soni/000_050/veloZaxs'][:]
wdata[np.abs(wdata)>50]=float('nan')

# %%
tdata=fp['ABBY/dp0p/data/soni/000_050/tempSoni'][:]

# %%
hdata=fp['ABBY/dp0p/data/irgaTurb/000_050/rtioMoleDryH2o'][:]
cdata=fp['ABBY/dp0p/data/irgaTurb/000_050/rtioMoleDryCo2'][:]


# %%
def despike(data,wnd=9,numbin=2,thrs=10,frmin=0.025):
    out=np.zeros(data.shape)
    if np.sum(~np.isnan(data))/len(data)<frmin:
        out[:]=data[:]
        return out
    dnorm=data-np.nanmean(data)
    dnorm=dnorm/np.nanstd(dnorm)
    dndiff=dnorm[1:]-dnorm[0:-1]
    if np.sum(~np.isnan(dndiff))==0:
        out[:]=data[:]
        return out
    reso=np.nanmin(np.abs(dndiff))
    thnmdt=max(5,floor(wnd/2))
    dmedflt=


# %%

# %%
smd=udata[275000:278000]

# %%
np.savetxt("test.csv", udata,  
              delimiter = ",")

# %%

# %%
#library(eddy4R.qaqc)
#data <- read.csv("test.csv")
#data <- unlist(data)
#dout <- eddy4R.qaqc::def.dspk.br86(data)
from numpy import genfromtxt

# %%
my_data = genfromtxt('dspkout2.csv', delimiter=',')

# %%
dd=udata-my_data[:,1].T

# %%
np.sum(np.isnan(my_data[:,1]))-np.sum(np.isnan(udata))

# %%

# %%
n=1727999+1

# %%
n/24/60/60

# %%
np.savetxt("test_w.csv", wdata,  
              delimiter = ",")

# %%
pwd

# %%
plt.plot(cdata)

# %%
np.nanmin(tdata)

# %%
data={}
for folder in os.listdir('/home/tsw35/soteria/data/NEON/dp04/'):
    print(folder,end='',flush=True)
    if len(folder)>4:
        continue
    data[folder]=[]
    flist=os.listdir('/home/tsw35/soteria/data/NEON/dp04/'+folder)
    flist.sort()
    for file in flist:
        print('.',end='')
        fp=h5py.File('/home/tsw35/soteria/data/NEON/dp04/'+folder+'/'+file,'r')
        a=fp[folder].attrs['Pf$AngEnuXaxs'][:]
        if len(a)>1:
            data[folder].append(99)
        else:
            data[folder].append(a[0])
    print()

# %%
NAcount=0
longcount=0
singcount=0
for k in data.keys():
    for i in data[k]:
        if i==b'NA':
            NAcount=NAcount+1
        elif len(str(i))<20:
            singcount=singcount+1
        elif len(str(i))>20:
            longcount=longcount+1
print(NAcount)
print(longcount)
print(singcount)

# %%
data['STER']

# %%
fp=h5py.File('/home/tsw35/soteria/data/NEON/dp04/STER/NEON.D10.STER.DP4.00200.001.nsae.2018-03.basic.h5','r')
a=fp['STER'].attrs['Pf$AngEnuXaxs'][:]

# %%
a

# %%
a='NEON.D16.ABBY.DP4.00200.001.nsae.2017-09-15.expanded.20240121T033159Z.h5'

# %%
a[0:52]

# %%
a=np.random.rand(4,4)

# %%
a=np.zeros((4,4))
a[0,:]=np.linspace(.2,.7,4)
a[3,:]=np.linspace(.3,.5,4)
a[1:3,0]=np.random.rand(1,2)/2+.1
a[1:3,3]=np.random.rand(1,2)/2+.4
a[1:3,1:3]=np.random.rand(2,2)/3+.66

# %%
a[3,3]=.33

# %%
b=np.copy(a)
b[0:2,0:2]=np.mean(a[0:2,0:2])
b[0:2,2:]=np.mean(a[0:2,2:])
b[2:,2:]=np.mean(a[2:,2:])
b[2:,0:2]=np.mean(a[2:,0:2])

# %%
a[1,2]=a[2,2]
a[2,2]=a[2,1]

# %%
a[0,2]=a[0,2]+.05

# %%
a[2,2]=a[2,2]+.05

# %%
a[2,1]=a[2,1]-.05

# %%
a[1,3]=a[1,3]+.09
a[1,2]=a[1,2]-.15

# %%
b=np.copy(a)
b[0:2,0:2]=np.mean(a[0:2,0:2])
b[0:2,2:]=np.mean(a[0:2,2:])
b[2:,2:]=np.mean(a[2:,2:])
b[2:,0:2]=np.mean(a[2:,0:2])

# %%
plt.figure(dpi=400)
plt.subplot(1,2,1)
plt.imshow(a,cmap='coolwarm',vmin=0,vmax=1)
plt.xticks([-.5,.5,1.5,2.5],[])
plt.yticks([-.5,.5,1.5,2.5],[])
plt.tick_params(axis='both', which='both',length=0)
plt.grid(color='k')
plt.subplot(1,2,2)
plt.imshow(b,cmap='coolwarm',vmin=0,vmax=1)
plt.xticks([-.5,.5,1.5,2.5],[])
plt.yticks([-.5,.5,1.5,2.5],[])
plt.tick_params(axis='both', which='both',length=0)
plt.grid(color='k')

# %%
c=a.copy()
c=(c-.1)/.77
d=np.copy(c)
d[0:2,0:2]=np.mean(c[0:2,0:2])
d[0:2,2:]=np.mean(c[0:2,2:])
d[2:,2:]=np.mean(c[2:,2:])
d[2:,0:2]=np.mean(c[2:,0:2])

# %%
plt.figure(figsize=(8,4),dpi=500)
plt.subplot(1,2,1)
plt.imshow(c,cmap='coolwarm',vmin=0,vmax=1)
plt.xticks([-.5,.5,1.5,2.5],[])
plt.yticks([-.5,.5,1.5,2.5],[])
plt.tick_params(axis='both', which='both',length=0)
plt.grid(color='k')
plt.subplot(1,2,2)
plt.imshow(d,cmap='coolwarm',vmin=0,vmax=1)
plt.xticks([-.5,.5,1.5,2.5],[])
plt.yticks([-.5,.5,1.5,2.5],[])
plt.tick_params(axis='both', which='both',length=0)
plt.grid(color='k')

# %%
np.min(a)

# %%
fp=h5py.File('/home/tsw35/soteria/data/NEON/aniso_1m/testing/NEON_TW_2017-09.h5','r')
fp4=h5py.File('/home/tsw35/soteria/data/NEON/dp04/ABBY/NEON.D16.ABBY.DP4.00200.001.nsae.2017-09.basic.h5','r')

# %%
fp=h5py.File('/home/tsw35/soteria/data/NEON/aniso_1m/testing/NEON_TW_2017-09_new.h5','r')

# %%
for k in fp.keys():
    data=fp[k][:]
    if len(data)>24000:
        data[24000:26000]=float('nan')
    else:
        data[750:850]=float('nan')
        
    print(k)
    print('   Length : '+str(len(data)))
    print('   NaNs   : '+str(np.sum(np.isnan(data))))
    print('   Mean   : '+str(np.nanmean(data)))
    print('   Vari   : '+str(np.nanvar(data)))
    print('   1stPCT : '+str(np.nanpercentile(data,1)))
    print('   99stPCT: '+str(np.nanpercentile(data,99)))
    plt.figure()
    plt.plot(data)
    plt.title(k)

# %%
for k in fp.keys():
    data=fp[k][:]
    print(k)
    print('   Length : '+str(len(data)))
    print('   NaNs   : '+str(np.sum(np.isnan(data))))
    print('   Mean   : '+str(np.nanmean(data)))
    print('   Vari   : '+str(np.nanvar(data)))
    print('   1stPCT : '+str(np.nanpercentile(data,1)))
    print('   99stPCT: '+str(np.nanpercentile(data,99)))
    plt.figure()
    plt.plot(data)
    plt.title(k)

# %%
fp2=h5py.File('/home/tsw35/soteria/data/eddy_v2/lst/ABBY_L2.h5','r')

# %%
fp2.keys()

# %%
fp2.attrs.keys()

# %%
print(fp2.attrs['zd'])
print(fp2.attrs['tow_height'])

# %%
zz=18.59-3.9

# %%
zLh2=fp2['ZL'][:][48*303:48*333]
zL=zz/fp['L_MOST'][:]
print(len(zLh2))
print(len(zL))
zLh2[zLh2==-9999]=float('nan')
plt.plot(zLh2[675:848])
plt.plot(zL[675:848])
plt.ylim(-10,10)

# %%
zLh2=fp2['ZL'][:][48*303:48*333]
zL=zz/fp['L_MOST'][:]
print(len(zLh2))
print(len(zL))
zLh2[zLh2==-9999]=float('nan')
#plt.plot(zLh2)
#plt.plot(zL)
plt.grid()
plt.plot((zL-zLh2)/zLh2,'o')
plt.ylim(-2,2)

# %%
plt.grid()
plt.semilogy(np.abs(zL-zLh2),'o')

# %%
print(np.sum((zL>0)&(zLh2>0))+np.sum((zL<0)&(zLh2<0)))
print(np.sum((zL<0)&(zLh2>0))+np.sum((zL>0)&(zLh2<0)))

# %%
delta=zL-zLh2
print(np.sum(np.isnan(delta))/len(delta))
plt.hist(delta,bins=np.linspace(-10,10))
plt.title('')

# %%
time=fp2['TIME'][:]

# %%
time[0]

# %%
fp2.attrs['description']

# %%
from datetime import datetime,timedelta

# %%
dt0=datetime(1970,1,1,0)

# %%
dt0+timedelta(seconds=time[48*303])

# %%
np.nanmean(np.abs(zz/zLh2))

# %%
np.nanmean(np.abs(fp['L_MOST'][:]))

# %%
H=fp2['H'][:][48*303:48*333]
H[H==-9999]=float('nan')
np.nanmean(H)

# %%
t30=np.linspace(0,1,48*30)
t1=np.linspace(0,1,48*30*30)
plt.plot(t30,H)
plt.plot(t1,fp['H_1m'][:])

# %%
vpt=fp2['VPT'][:]
vpt[vpt<=-100]=float('nan')
np.nanmean(vpt)

# %%
vpt=fp2['TA'][:]
vpt[vpt<=-100]=float('nan')
np.nanmean(vpt+273)

# %%
fp4

# %%
u=fp4['ABBY/dp01/data/soni/000_050_01m/veloXaxsErth']['mean']
v=fp4['ABBY/dp01/data/soni/000_050_01m/veloYaxsErth']['mean']
w=fp4['ABBY/dp01/data/soni/000_050_01m/veloZaxsErth']['mean']

# %%
t=fp4['ABBY/dp01/data/soni/000_050_01m/tempAir']['mean']

# %%
h2o=fp4['ABBY/dp01/data/h2oTurb/000_050_01m/rtioMoleDryH2o']['mean']

# %%
usig=np.sqrt(fp4['ABBY/dp01/data/soni/000_050_01m/veloXaxsErth']['vari'])
usig2=fp['U_SIGMA_1m'][:]

# %%
wsig=np.sqrt(fp4['ABBY/dp01/data/soni/000_050_01m/veloZaxsErth']['vari'])
wsig2=fp['W_SIGMA_1m'][:]

# %%
plt.plot(fp['U_1m'][:])
plt.plot(u)
plt.xlim(0,1440)

# %%
plt.plot(fp['V_1m'][:])
plt.plot(v)
plt.xlim(0,1440)

# %%

# %%
plt.plot(w-fp['W_1m'][:])
#plt.plot(w)
plt.xlim(0,1440)
plt.ylim(-1,1)

# %%
plt.plot(wsig)
plt.plot(wsig2)
plt.xlim(0,1440)
plt.ylim(-1,1)

# %%
plt.plot(wsig-wsig2)
plt.xlim(0,1440)
plt.ylim(-.1,.1)

# %%
ta=fp['TA_1m'][:]-273.15

# %%
plt.plot(ta)
plt.plot(t)

# %%
plt.hist(ta-t,bins=np.linspace(-.25,.25))

# %%
h2o2=fp['H2O_1m'][:]

# %%
plt.hist(h2o2*1000-h2o,bins=np.linspace(-.05,.05))

# %%
# water is basically perfect
# temperature has some unknown issues, off by like .1
# velocity is MAYBE near perfect
# H and LE are, therefore, probably pretty good
# USTAR is probably good
# virtual potential temperature is weird 

# %%
ustar=(fp['UW_1m'][:]**2+fp['VW_1m'][:]**2)**(1/4)
ustar_o=fp['USTAR_1m'][:]

# %%
plt.plot(ustar)
plt.plot(ustar_o)
plt.ylim(-.05,.5)
plt.xlim(5000,7000)

# %%
ustar=(fp['UW'][:]**2+fp['VW'][:]**2)**(1/4)
plt.plot(ustr)
plt.plot(ustar)
plt.xlim(0,2000/30)
plt.ylim(0,.5)

# %%
ustr=fp4['ABBY/dp04/data/fluxMome/turb']['veloFric']

# %%

# %%
plt.hist(ustar-ustar_o,bins=np.linspace(-.5,.5))

# %%
plt.hist(ustar**3/ustar_o**3,bins=np.linspace(0,1))

# %%
np.nanmean(ustar**3/ustar_o**3)

# %%
LL=fp['L_MOST_1m'][:]
L2=LL*ustar**3/ustar_o**3

# %%
fpTW=h5py.File('/home/tsw35/soteria/data/NEON/aniso_1m/testing/NEON_TW_2017-09_t2.h5','r')
fp4=h5py.File('/home/tsw35/soteria/data/NEON/dp04/ABBY/NEON.D16.ABBY.DP4.00200.001.nsae.2017-09.basic.h5','r')
fpL2=h5py.File('/home/tsw35/soteria/data/eddy_v2/lst/ABBY_L2.h5','r')


# %%
# Check L, H2O, U, V, W, sigU, sigV, sigW

# %%
def checkData(d1,d2,idx1=10000,idx2=13000):
    d2[d2==-9999]=float('nan')
    plt.figure(figsize=(8,4))
    vmax=max(np.nanpercentile(d1,99),np.nanpercentile(d2,99))
    vmin=min(np.nanpercentile(d1,1),np.nanpercentile(d2,1))
    
    plt.subplot(2,3,1)
    plt.hist(d1,np.linspace(vmin,vmax))
    plt.title('TW')
    
    plt.subplot(2,3,2)
    plt.hist(d2,np.linspace(vmin,vmax))
    plt.title('D4')
    
    plt.subplot(2,3,3)
    vv=(vmax-vmin)/2
    plt.hist(d1-d2,np.linspace(-vv/10,vv/10))
    plt.title('DELTA')
    
    plt.subplot(2,3,4)
    plt.plot(d1[:])
    plt.plot(d2[:])
    plt.ylim(vmin,vmax)
    
    plt.subplot(2,3,5)
    plt.plot(d1[idx1:idx2])
    plt.plot(d2[idx1:idx2])
    plt.ylim(vmin,vmax)
    
    plt.subplot(2,3,6)
    plt.plot((d1-d2)[idx1:idx2])
    plt.ylim(-vv/10,vv/10)


# %%
fp=h5py.File('/home/tsw35/soteria/data/NEON/aniso_1m/testing/NEON_TW_2017-09_t1.h5','r')

# %%
u=fp4['ABBY/dp01/data/soni/000_050_01m/veloXaxsErth']['mean']
v=fp4['ABBY/dp01/data/soni/000_050_01m/veloYaxsErth']['mean']
w=fp4['ABBY/dp01/data/soni/000_050_01m/veloZaxsErth']['mean']
t_=fp4['ABBY/dp01/data/soni/000_050_01m/tempAir']['mean']
tsig=np.sqrt(fp4['ABBY/dp01/data/soni/000_050_01m/tempAir']['vari'])
ts_=fp4['ABBY/dp01/data/soni/000_050_01m/tempSoni']['mean']
usig=fp4['ABBY/dp01/data/soni/000_050_01m/veloXaxsErth']['vari']
vsig=fp4['ABBY/dp01/data/soni/000_050_01m/veloYaxsErth']['vari']
wsig=np.sqrt(fp4['ABBY/dp01/data/soni/000_050_01m/veloZaxsErth']['vari'])

# %%
h2o=fp4['ABBY/dp01/data/h2oTurb/000_050_01m/rtioMoleDryH2o']['mean']

# %%
dataTW=fpTW['H2O_1m'][:]*1000  #np.sqrt(fpTW['V_1m'][:]**2+fpTW['U_1m'][:]**2)#-273.15#*1000
data4 =h2o #np.sqrt(vsig) #np.sqrt(u**2+v**2)
checkData(dataTW,data4)

# %%
np.nanmean(fpTW['TA_SIGMA_1m'][:]-tsig)

# %%
plt.hist(dataTW-data4,np.linspace(-.01,0.01))

# %%
plt.hist(fpTW['V_1m'][:],np.linspace(-5,5))

# %%
for i in range(30):
    print(np.sum(np.isnan(v[i*(30*48):(i+1)*48*30])))

# %%
#NEON.D16.ABBY.DP4.00200.001.nsae.2017-09-02.expanded.h5
fpd0=h5py.File('/home/tsw35/xTyc_shared/NEON_raw_data/ABBY/NEON.D16.ABBY.IP0.00200.001.ecte.2017-09-01.l0p.h5','r')

# %%
fp4ex=h5py.File('/home/tsw35/soteria/data/NEON/test/ABBY/NEON.D16.ABBY.DP4.00200.001.nsae.2017-09-01.expanded.h5','r')

# %%
aze=fp4ex['ABBY/dp01/data/soni/000_050_30m/angZaxsErth']['mean'][:]


# %%
u0=fpd0['ABBY/dp0p/data/soni/000_050/veloXaxs'][:]
v0=fpd0['ABBY/dp0p/data/soni/000_050/veloYaxs'][:]
w0=fpd0['ABBY/dp0p/data/soni/000_050/veloZaxs'][:]

# %%
ax=0.022712
ay=0.012124
ofst=0.008146
import math

# %%
np.sin(ax)

# %%
uF,vF,wF=planarFit(u0,v0,w0)

# %%
aa=float(fpd0['ABBY/dp0p/data/soni/'].attrs['AngNedZaxs'][0])

# %%

# %%
uF1,vF1,wF1=planarFit(u0,v0,w0,vers=1)

# %%
uS,vS,wS,aE,_=rotStr(uF,vF,wF)

# %%
uS1,vS1,wS1,_,_=rotStr(uF,vF,wF,vers=1)


# %%
uS11,vS11,wS11,_,_=rotStr(uF1,vF1,wF1,vers=1)

# %%
uS2,vS2,wS2,aE2,_=rotStr(uF,vF,wF,vers=2)

# %%
uI,vI,wI=rotInt(u0,v0,w0,aa)

# %%
np.sin(math.pi/2)

# %%
uiF,viF,wiF=planarFit(uI,vI,wI)
uiS,viS,wiS,aiE,_=rotStr(uiF,viF,wiF)

# %%
uiF1,viF1,wiF1=planarFit(uI,vI,wI,vers=1)


# %%
def rotInt(ld_u,ld_v,ld_w,a_inst):
    ason=np.radians(a_inst)-math.pi
    if ason<0:
        ason=ason+2*math.pi
    amet=math.pi/2-ason
    if amet<0:
        amet=amet+2*math.pi
    print(amet/math.pi)
    ld_uu=ld_u*np.cos(amet)-ld_v*np.sin(amet)
    ld_vv=ld_u*np.sin(amet)+ld_v*np.cos(amet)
    return ld_uu,ld_vv,ld_w


# %%
def rotStr(ld_u,ld_v,ld_w,vers=0):
    rot_ul=[]
    rot_vl=[]
    rot_wl=[]
    a_ertl=[]
    a_rotl=[]
    if vers==0:
        for t in range(1440):
            uin=ld_u[t*20*60:(t+1)*20*60]
            vin=ld_v[t*20*60:(t+1)*20*60]

            a_ert=np.arctan2(vin,uin)
            a_ertm=np.nanmean(a_ert)

            a_rot=(a_ertm+math.pi)%(2*math.pi)
            mat_rot=np.matrix(np.zeros((3,3)))
            mat_rot[0,0]=math.cos(a_rot)
            mat_rot[0,1]=math.sin(a_rot)
            mat_rot[1,0]=-math.sin(a_rot)
            mat_rot[1,1]=math.cos(a_rot)
            mat_rot[2,2]=1
            mat_rot=mat_rot.T

            combo=np.array([ld_v[t*20*60:(t+1)*20*60],ld_u[t*20*60:(t+1)*20*60],ld_w[t*20*60:(t+1)*20*60]])
            [rot_v,rot_u,rot_w]=np.dot(mat_rot,combo)
            rot_u=np.squeeze(np.array(rot_u[0,:]).T)
            rot_v=np.squeeze(np.array(rot_v[0,:]).T)
            rot_w=np.squeeze(np.array(rot_w[0,:]).T)
            
            rot_ul.extend(rot_u)
            rot_vl.extend(rot_v)
            rot_wl.extend(rot_w)
            a_ertl.extend(a_ert)
            a_rotl.append(a_rot)
    elif vers==1:
        for t in range(48):
            uin=ld_u[t*20*60*30:(t+1)*20*60*30]
            vin=ld_v[t*20*60*30:(t+1)*20*60*30]

            a_ert=-np.arctan2(vin,uin)
            a_ertm=np.nanmean(a_ert)

            a_rot=(a_ertm+math.pi)%(2*math.pi)
            mat_rot=np.matrix(np.zeros((3,3)))
            mat_rot[0,0]=math.cos(a_rot)
            mat_rot[0,1]=math.sin(a_rot)
            mat_rot[1,0]=-math.sin(a_rot)
            mat_rot[1,1]=math.cos(a_rot)
            mat_rot[2,2]=1
            mat_rot=mat_rot.T

            combo=np.array([ld_v[t*20*60*30:(t+1)*30*20*60],ld_u[t*20*60*30:(t+1)*20*60*30],ld_w[t*20*60*30:(t+1)*20*30*60]])
            [rot_v,rot_u,rot_w]=np.dot(mat_rot,combo)
            rot_u=np.squeeze(np.array(rot_u[0,:]).T)
            rot_v=np.squeeze(np.array(rot_v[0,:]).T)
            rot_w=np.squeeze(np.array(rot_w[0,:]).T)
            
            rot_ul.extend(rot_u)
            rot_vl.extend(rot_v)
            rot_wl.extend(rot_w)
            a_ertl.extend(a_ert)
            a_rotl.append(a_rot)
    elif vers==2:
        for t in range(48):
            uin=ld_u[t*20*60*30:(t+1)*20*60*30]
            vin=ld_v[t*20*60*30:(t+1)*20*60*30]

            a_ert=np.arctan2(uin,vin)
            a_ertm=np.nanmean(a_ert)

            a_rot=(a_ertm+math.pi)%(2*math.pi)
            mat_rot=np.matrix(np.zeros((3,3)))
            mat_rot[0,0]=math.cos(a_rot)
            mat_rot[0,1]=math.sin(a_rot)
            mat_rot[1,0]=-math.sin(a_rot)
            mat_rot[1,1]=math.cos(a_rot)
            mat_rot[2,2]=1
            mat_rot=mat_rot.T

            combo=np.array([ld_v[t*20*60*30:(t+1)*30*20*60],ld_u[t*20*60*30:(t+1)*20*60*30],ld_w[t*20*60*30:(t+1)*20*30*60]])
            [rot_v,rot_u,rot_w]=np.dot(mat_rot,combo)
            rot_u=np.squeeze(np.array(rot_u[0,:]).T)
            rot_v=np.squeeze(np.array(rot_v[0,:]).T)
            rot_w=np.squeeze(np.array(rot_w[0,:]).T)
            
            rot_ul.extend(rot_u)
            rot_vl.extend(rot_v)
            rot_wl.extend(rot_w)
            a_ertl.extend(a_ert)
            a_rotl.append(a_rot)


    return rot_ul, rot_vl, rot_wl, a_ertl, a_rotl


# %%
def planarFit(ld_u,ld_v,ld_w,vers=0):
    if vers==0:
        ld_w=ld_w-ofst
        mat_pitch=np.matrix([[math.cos(ay),0,-math.sin(ay)],[0,1,0],[math.sin(ay),0,math.cos(ay)]])
        mat_roll=np.matrix([[1,0,0],[0,math.cos(ax),math.sin(ax)],[0,-math.sin(ax),math.cos(ax)]])
        mat_rot=np.dot(mat_pitch,mat_roll)
        print(mat_pitch)
        print(mat_roll)
        print(mat_rot)

        combo=np.array([ld_u,ld_v,ld_w])
        [ld_u,ld_v,ld_w]=np.dot(mat_rot,combo)

        ld_u=np.squeeze(np.array(ld_u[0,:]).T)
        ld_v=np.squeeze(np.array(ld_v[0,:]).T)
        ld_w=np.squeeze(np.array(ld_w[0,:]).T)
    if vers==1:
        ld_w=ld_w-ofst
        mat_pitch=np.matrix([[math.cos(ay),0,-math.sin(ay)],[0,1,0],[math.sin(ay),0,math.cos(ay)]])
        mat_roll=np.matrix([[1,0,0],[0,math.cos(ax),math.sin(ax)],[0,-math.sin(ax),math.cos(ax)]])
        mat_rot=np.dot(mat_pitch,mat_roll)

        combo=np.array([ld_u,ld_v,ld_w])
        [ld_u,ld_v,ld_w]=np.dot(mat_rot.T,combo)

        ld_u=np.squeeze(np.array(ld_u[0,:]).T)
        ld_v=np.squeeze(np.array(ld_v[0,:]).T)
        ld_w=np.squeeze(np.array(ld_w[0,:]).T)
    return ld_u,ld_v,ld_w


# %%
def to1m(d):
    out=np.ones((1440,))*float('nan')
    for i in range(1440):
        out[i]=np.nanmean(d[i*(20*60):(i+1)*(20*60)])
    return out


# %%
def to30m(d):
    out=np.ones((48,))*float('nan')
    for i in range(48):
        out[i]=np.nanmean(d[i*(20*60*30):(i+1)*(20*60*30)])
    return out


# %%
u_4=fp4ex['ABBY/dp01/data/soni/000_050_01m/veloXaxsErth']['mean']#[0:1440]
v_4=fp4ex['ABBY/dp01/data/soni/000_050_01m/veloYaxsErth']['mean']#[0:1440]
w_4=fp4ex['ABBY/dp01/data/soni/000_050_01m/veloZaxsErth']['mean']#[0:1440]

# %%

# %%

# %%
plt.hist(v_4-to1m(viF),bins=np.linspace(-.1,.1))

# %%
plt.plot(to30m(aiE))
plt.plot(np.radians(aze))

# %%
aze

# %%
plt.figure(figsize=(10,3))
plt.subplot(1,3,1)
plt.plot(u_4)
plt.plot(to1m(uiF))
plt.subplot(1,3,2)
plt.plot(v_4)
plt.plot(to1m(viF))
plt.subplot(1,3,3)
plt.plot(w_4)
plt.plot(to1m(wiF))

# %%
plt.plot(u0)
plt.plot(v0)

# %%
a=3
R=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
uR,vR,wR=np.dot(R,np.array([uF,vF,wF]))
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(u_4)
plt.plot(to1m(uR))
plt.subplot(1,2,2)
plt.plot(v_4)
plt.plot(to1m(vR))

# %%
np.nanmean(wF-wR)

# %%
plt.hist(u_4-to1m(uiF),np.linspace(-.01,.01))
plt.title('')

# %%
plt.hist(v_4-to1m(viF),np.linspace(-.01,.01))
plt.title('')

# %%
plt.hist(w_4-to1m(wiF),np.linspace(-.01,.01))
plt.title('')

# %%

# %%
fp=h5py.File('/home/tsw35/soteria/data/NEON/aniso_1m/testing/NEON_TW_2017-09_t1.h5','r')
fp4=h5py.File('/home/tsw35/soteria/data/NEON/dp04/ABBY/NEON.D16.ABBY.DP4.00200.001.nsae.2017-09.basic.h5','r')

# %%

# %%

# %%

# %%

# %%
u=fp4['ABBY/dp01/data/soni/000_050_01m/veloXaxsErth']['mean']
v=fp4['ABBY/dp01/data/soni/000_050_01m/veloYaxsErth']['mean']
w=fp4['ABBY/dp01/data/soni/000_050_01m/veloZaxsErth']['mean']

# %%
300*(101/97)**(2/7)

# %%
t=fpd0['ABBY/dp0p/data/soni/000_050/tempSoni'][:]
vs=fpd0['ABBY/dp0p/data/soni/000_050/veloSoni'][:]

# %%
tt=vs**2/1.4/(8.314462175/28.97/10**(-3))

# %%
1/1.4/(8.314462/28.97/10**(-3))

# %%
fpd0['ABBY/dp0p/data/soni/000_050/veloSoni']

# %%
plt.hist(t-tt,bins=np.linspace(-.1,.1))

# %%
ttt=fp4['ABBY/dp01/data/soni/000_050_01m/tempSoni']['mean']+273.15
ttta=fp4['ABBY/dp01/data/soni/000_050_01m/tempAir']['mean']+273.15

# %%
ttt=ttt[0:1440]
ttta=ttta[0:1440]

# %%
plt.hist(to1m(tt)-ttt,bins=np.linspace(-.1,.1))

# %%
print(np.nanmean(to1m(t)*ttta/ttt-ttta))

# %%
np.nanmean(ttta/ttt)

# %%
np.nanmean(t-tt)

# %%
fp.close()

# %%
fp=h5py.File('/home/tsw35/tyche/neon_1m/ABBY_L1.h5','r')

# %%
fp.keys()

# %%
fp.attrs.keys()

# %%
fp.keys()

# %%
m=fp['ANI_YB'][:]<0
m=m|(fp['ANI_YB'][:]>(np.sqrt(3)/2))
m=m&(~(fp['ANI_YB'][:]==-9999))

# %%
for k in fp.keys():
    print(k,end=',')
    if 'wind' in k:
        for v in fp[k].keys():
            plt.figure()
            data=fp[k][v][:]
            data=np.array(data,dtype=float)
            data[data==-9999]=float('nan')
            plt.figure()
            plt.plot(data[-1440*5:-1440])
            plt.title(k+':'+v)
        continue
    data=fp[k]#[m]
    data=np.array(data,dtype=float)
    if 'q' in k:
        data[data==-1]=float('nan')
    data[data==-9999]=float('nan')
    plt.figure()
    #plt.hist(data,np.linspace(np.nanpercentile(data,1),np.nanpercentile(data,99)))
    plt.plot(data[-1440*5:-1440])
    plt.title(k)

# %%
np.sqrt(3)/2

# %%
(7571260-6275440)/60/60/24

# %%
dirt='/home/tsw35/soteria/data/eddy_v2/lst/ABBY_L2.h5'
fp2=h5py.File(dirt,'r')

# %%
data=fp2['UU'][:]
data[data==-9999]=float('nan')
plt.plot(data)


# %%

# %%
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


# %%
bijt=aniso(fp2['UU'][:],fp2['VV'][:],fp2['WW'][:],fp2['UV'][:],fp2['UW'][:],fp2['VW'][:])

# %%
N=bijt.shape[0]

# %%
bijt=aniso(fp2['UU'][:],fp2['VV'][:],fp2['WW'][:],fp2['UV'][:],fp2['UW'][:],fp2['VW'][:])
N=bijt.shape[0]
yb=np.ones((N,))*-9999
xb=np.ones((N,))*-9999
for t in range(N):
    if np.sum(bijt[t,:,:]==-9999)>0:
        continue
    lams=np.linalg.eig(bijt[t,:,:])[0]
    lams.sort()
    lams=lams[::-1]
    xb[t]=lams[0]-lams[1]+.5*(3*lams[2]+1)
    yb[t]=np.sqrt(3)/2*(3*lams[2]+1)

# %%
yb[yb==-9999]=float('nan')

# %%
yb_=yb

# %%
time2=fp2['TIME'][:]

# %%
plt.plot(yb)

# %%
#m=m&(fp['qU']==0)
uu=fp['U_SIGMA'][m]**2
vv=fp['V_SIGMA'][m]**2
ww=fp['W_SIGMA'][m]**2
uv=fp['UV'][m]
uw=fp['UW'][m]
vw=fp['VW'][m]

# %%
bijt=aniso(uu,vv,ww,uv,uw,vw)
N=bijt.shape[0]
yb=np.ones((N,))*-9999
xb=np.ones((N,))*-9999
for t in range(N):
    if np.sum(bijt[t,:,:]==-9999)>0:
        continue
    lams=np.linalg.eig(bijt[t,:,:])[0]
    lams.sort()
    lams=lams[::-1]
    xb[t]=lams[0]-lams[1]+.5*(3*lams[2]+1)
    yb[t]=np.sqrt(3)/2*(3*lams[2]+1)

# %%
yb[yb==-9999]=float('nan')


# %%

# %%
plt.hist(fp2['VW'][:],np.linspace(0,10))
plt.title('')

# %%
plt.plot(uv)
plt.plot(vw)
plt.plot(uw)

# %%
len(uu)

# %%
len(fp['UV'][:])

# %%
m=fp['ANI_YB'][:]<0
m=m|(fp['ANI_YB'][:]>(np.sqrt(3)/2))
m=m&(~(fp['ANI_YB'][:]==-9999))

# %%
m=m&(fp['qU'][:]==0)

# %%
np.sum(m)

# %%
mm=(fp['U'][:]>-9999)
m2=(fp['UV'][:]>-9999)

# %%
plt.plot(mm,'o')
plt.plot(m2-.05,'o')
plt.ylim(.9,1.1)


# %%
fp['TIME'][-1]

# %%
import datetime

# %%
datetime.datetime(2023,12,31,23,30,tzinfo=datetime.timezone.utc).timestamp()

# %%
yb=fp['ANI_YB'][:]
yb[yb<=-9999]=float('nan')
plt.plot(fp['TIME'][:],yb)
plt.plot(time2,yb_)
plt.ylim([-1,1])
plt.xlim(1.51*10**9,1.5105*10**9)

# %%
yb=fp['ANI_YB'][:]
yb[yb<=-9999]=float('nan')
plt.plot(fp['TIME'][:],yb)
plt.plot(time2,yb_)
plt.ylim([-1,1])
plt.xlim(1.5102*10**9,1.5105*10**9)
plt.grid(True)

plt.figure()
plt.plot(fp['TIME'][:],fp['U_SIGMA'][:]**2)
plt.plot(time2,fp2['VV'][:])
plt.xlim(1.5102*10**9,1.5105*10**9)
plt.ylim([0,5])
plt.grid(True)

plt.figure()
plt.plot(fp['TIME'][:],fp['V_SIGMA'][:]**2)
plt.plot(time2,fp2['UU'][:])
plt.xlim(1.5102*10**9,1.5105*10**9)
plt.ylim([0,5])
plt.grid(True)

plt.figure()
plt.plot(fp['TIME'][:],fp['W_SIGMA'][:]**2)
plt.plot(time2,fp2['WW'][:])
plt.xlim(1.5102*10**9,1.5105*10**9)
plt.ylim([0,5])
plt.grid(True)


plt.figure()
plt.plot(fp['TIME'][:],np.sqrt(fp['UV'][:]**2))
plt.plot(time2,np.sqrt(fp2['UV'][:]**2))
plt.xlim(1.5102*10**9,1.5105*10**9)
plt.ylim([0,1])
plt.grid(True)


plt.figure()
plt.plot(fp['TIME'][:],np.sqrt(fp['UW'][:]**2))
plt.plot(time2,np.sqrt(fp2['VW'][:]**2))
plt.xlim(1.5102*10**9,1.5105*10**9)
plt.ylim([0,1])
plt.grid(True)

plt.figure()
plt.plot(fp['TIME'][:],np.sqrt(fp['VW'][:]**2))
plt.plot(time2,np.sqrt(fp2['UW'][:]**2))
plt.xlim(1.5102*10**9,1.5105*10**9)
plt.ylim([0,1])
plt.grid(True)

# %%
fp3=h5py.File('/home/tsw35/soteria/data/NEON/aniso_1m/ABBY/NEON_TW_2017-08.h5','r')

# %%
fp3.close()

# %%
plt.plot(fp3['WTHETA'][:])

# %%
for k in fp3.keys():
    data=fp3[k][:]
    data=np.array(data)
    data[data==-9999]=float('nan')
    plt.figure()
    print(k)
    plt.plot(data)
    plt.title(k)

# %%
fp3.close()

# %%
plt.plot(fp3['H_1m'][:])
plt.plot(fp3['L_MOST_1m'][:])
plt.ylim(-100,500)

# %%
plt.plot(fp3['L_MOST_1m'][:])
plt.ylim(-100,100)

# %%
plt.scatter(fp3['H_1m'][:],fp3['L_MOST_1m'][:])
plt.xlim(-200,600)
plt.ylim(-500,500)

# %%
a=fp3['H_1m'][:]

# %%
b=a.copy()

# %%
a[a<100]=9

# %%
np.nanmean(b)

# %%
np.nanmean(a)

# %%
[3]*5

# %%
site='ABBY'

# %%
fp1=h5py.File('/home/tsw35/tyche/neon_1m/'+site+'_L1.h5','r')

# %%
a=fp1['Ustr'][:]
b=fp1['Vstr'][:]
c=fp1['Ustr'][:]

# %%
print(len(a))
print(np.sum(np.isnan(a)))
print(np.sum(a==-9999))
print((np.sum(np.isnan(a))+np.sum(a==-9999))/len(a))

# %%
print(len(b))
print(np.sum(np.isnan(b)))
print(np.sum(b==-9999))
print((np.sum(np.isnan(b))+np.sum(b==-9999))/len(b))

# %%
print(len(c))
print(np.sum(np.isnan(c)))
print(np.sum(c==-9999))
print((np.sum(np.isnan(c))+np.sum(c==-9999))/len(c))

# %%
1440*7*365

# %%
plt.plot(np.isnan(a))
plt.xlim(1*10**6,1.000*10**6+1440)

# %%
fp2=h5py.File('/home/tsw35/soteria/data/NEON/aniso_1m/ABBY/NEON_TW_2017-10.h5','r')

# %%
aa=fp2['Ustr_1m'][:]
print(np.sum(np.isnan(aa))/(1440*48)*100)

# %%
for file in os.listdir('/home/tsw35/soteria/data/NEON/aniso_1m/ABBY/'):
    fp2=h5py.File('/home/tsw35/soteria/data/NEON/aniso_1m/ABBY/'+file,'r')
    print(file)
    aa=fp2['UU_1m'][:]
    print('   '+str(np.sum(np.isnan(aa))/(1440*48)*100))

# %%
fp2.close()

# %%
10*32

# %%
