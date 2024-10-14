# run MRD over stable and unstable periods separately
import numpy as np
import os
import h5py
from datetime import datetime,timedelta
import pickle

from mpi4py import MPI

# MPI4PY Stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

stbctf=.1 #minimum stability
ctf=.005 #maximum acceptable nans as a fraction
tctf=180 #minimum length of time to count in minutes

idir='/home/tswater/tyche/data/neon/raw_streamwise/'
odir='/home/tswater/tyche/data/neon/mrd_30/'
sites=os.listdir(idir)

####### MRD ###########
def MRD(u):
    testNan = np.isnan(u)
    ind = np.where(testNan == True)
    u[ind] = 0

    Number_of_Nans = np.size(ind) #Determines how many NaNs have been found.
    # Next, we define a couple parameters related to the signal.print(f"The total number of NaNs found is of {Number_of_Nans} points")

    #Make the Signal a power of 2.
    M = np.int64(np.floor(np.log2(len(u)))) #Maximum Power of 2 within the length of the signal measure at 20Hz.
    u_short = u[0:int(2**M)]

    #-----------------------------------------------------------------------------------
    var1 = u_short
    var2 = u_short

    a = np.array(var1)
    b = np.array(var2)

    D = np.zeros(M+1)
    Mx = 0
    for ims in range(0,M-Mx+1):
        ms = M-ims  # Scale
        l = 2**ms    # Number of points (width) of the averaging segments "nw" at a given scale "m".
        nw = np.int64((2**M)/l)  # Number of segments, each with "l" number of points.

        sumab = 0


        for i in range(1,nw+1):  #Loop through the different averaging segments "nw"
            k = (i-1)*l
            za = a[k]
            zb = b[k]

            for j in range(k+1,k+l):  #Loop within the datapoints inside one specific [i] segment (ot of the total "nw").
                za = za + a[j]  #Cumulative sum of subsegment "i" in time series "a"
                zb = zb + b[j]  #Cumulative sum of subsegment "i" in time series "b"

            za = za/l
            zb = zb/l
            sumab = sumab + (za*zb)

            for j in range(k,i*l): #Subtract the mean from the time series to form the residual to be reused in next iteration.
                tmpa = a[j] - za
                tmpb = b[j] - zb
                a[j] = tmpa
                b[j] = tmpb


        if nw>1: #Computing the MR spectra at a given scale[m]. For scale ms = M is the largest scale.
            D[ms] = (sumab/nw)
    return D,M



for site in sites[rank::size]:
    months=os.listdir(idir+site)
    months.sort()
    t0=datetime(int(months[0][8:12]),int(months[0][13:15]),1,0,0)
    fps=h5py.File('/home/tswater/Documents/Elements_Temp/NEON/neon_30m/'+site+'_L30.h5','r')
    time=fps['TIME'][:]
    m=time>=(t0-datetime(1970,1,1,0,0)).total_seconds()
    lmost=(fps.attrs['tow_height']-fps.attrs['zd'])/fps['L_MOST'][m]
    time=time[m]
    n1=len(lmost)
    dn=20*60*30
    tcount=0
    # export lists
    out={'stable':{'Du':[],'Dv':[],'Dw':[],'Mu':[],'Mv':[],'Mw':[]},
         'unstable':{'Du':[],'Dv':[],'Dw':[],'Mu':[],'Mv':[],'Mw':[]}}

    # loop changing variables
    stab=0
    tcount=0
    month=0
    u_=[]
    v_=[]
    w_=[]
    print(site,flush=True)
    looptimes1=np.linspace(0,n1-1,100).astype(int)
    dt=t0
    tfp=0
    for t in range(n1):
        # load u,v,w
        dt=dt+timedelta(minutes=1)
        if month!=dt.month:
            month=dt.month
            fp=h5py.File(idir+site+'/'+months[month-1],'r')
            tfp=0
        else:
            tfp=tfp+dn
        u=fp['Ustr'][tfp:tfp+dn]
        v=fp['Vstr'][tfp:tfp+dn]
        w=fp['Wstr'][tfp:tfp+dn]

        if t in looptimes1:
            print('.',end='',flush=True)
        stab_=lmost[t]
        if np.isnan(stab_):
            # stop; try to export
            # create nanpct
            nanpct=np.sum(np.isnan(u_)|(np.abs(u_)>50))/len(u_)
            if (nanpct>ctf) or (tcount<tctf) or (np.sum(np.isnan(nanpct))>0):
                pass
            elif np.abs(stab)>=.1:
                Du,Mu=MRD(np.array(u_))
                Dv,Mv=MRD(np.array(v_))
                Dw,Mw=MRD(np.array(w_))
                # export
                if stab<=-stbctf:
                    ss='unstable'
                elif stab>=stbctf:
                    ss='stable'
                out[ss]['Du'].append(Du)
                out[ss]['Dv'].append(Dv)
                out[ss]['Dw'].append(Dw)
                out[ss]['Mu'].append(Mu)
                out[ss]['Mv'].append(Mv)
                out[ss]['Mw'].append(Mw)
            else:
                pass

            #start new
            u_=[]
            v_=[]
            w_=[]
            tcount=0
            stab=0

        elif ((stab_>=stbctf)&(stab>=stbctf))|((stab_<=-stbctf)&(stab<=-stbctf)):
            # grow u,v,w list
            tcount=tcount+30
            u_.extend(u)
            v_.extend(v)
            w_.extend(w)
            stab=stab_
        elif tcount>=tctf:
            # create nanpct
            nanpct=np.sum(np.isnan(u_)|(np.abs(u_)>50))/len(u_)
            if (nanpct>ctf) or (np.sum(np.isnan(nanpct))>0):
                pass
            else:
                Du,Mu=MRD(np.array(u_))
                Dv,Mv=MRD(np.array(v_))
                Dw,Mw=MRD(np.array(w_))
                # export
                if stab<=-stbctf:
                    ss='unstable'
                elif stab>=stbctf:
                    ss='stable'
                out[ss]['Du'].append(Du)
                out[ss]['Dv'].append(Dv)
                out[ss]['Dw'].append(Dw)
                out[ss]['Mu'].append(Mu)
                out[ss]['Mv'].append(Mv)
                out[ss]['Mw'].append(Mw)

            #start new
            u_=[]
            v_=[]
            w_=[]
            tcount=0
            # grow u,v,w list
            tcount=tcount+30
            u_.extend(u)
            v_.extend(v)
            w_.extend(w)
            stab=stab_
        else:
            # do not export, start new, grow u,v,w list
            #start new
            u_=[]
            v_=[]
            w_=[]
            tcount=0
            # grow u,v,w list
            tcount=tcount+30
            u_.extend(u)
            v_.extend(v)
            w_.extend(w)
            stab=stab_
    pickle.dump(out,open(odir+site+'_mrd_v2.p','wb'))
