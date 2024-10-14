import numpy as np
import scipy as sci
import os
import h5py
import pickle
import inspect
from scipy import optimize

vmx_a=.7
vmn_a=.1

##### CURVE FUNCTIONS #####
def old(var,zL):
    if var=='Uu':
        return 2.55*(1-3*zL)**(1/3)
    elif var=='Vu':
        return 2.05*(1-3*zL)**(1/3)
    elif var=='Wu':
        return 1.35*(1-3*zL)**(1/3)
    elif var=='Tu':
        phi_old=.99*(.067-zL)**(-1/3)
        phi_old[zL>-0.05]=.015*(-zL[zL>-0.05])**(-1)+1.76
        return phi_old
    elif var=='H2Ou':
        return np.sqrt(30)*(1-25*zL)**(-1/3)
    elif var=='CO2u':
        return np.sqrt(30)*(1-25*zL)**(-1/3)
    elif var=='Us':
        return 2.06*np.ones((len(zL)))
    elif var=='Vs':
        return 2.06*np.ones((len(zL)))
    elif var=='Ws':
        return 1.6*np.ones((len(zL)))
    elif var=='Ts':
        return 0.00087*(zL)**(-1.4)+2.03
    elif var=='H2Os':
        return 2.74*np.ones((len(zL)))
    elif var=='CO2s':
        return 2.74*np.ones((len(zL)))
    else:
        return float('nan')

def T_stb(zL,a,b,c,d):
    logphi=a+b*np.log10(zL)+c*np.log10(zL)**2+d*np.log10(zL)**3
    phi=10**(logphi)
    return phi

def T_ust(zL,a):
    phi=1.07*(0.05+np.abs(zL))**(-1/3)+\
        (-1.14+a*(np.abs(zL)**(-9/10)))*(1-np.tanh(10*(np.abs(zL))**(2/3)))
    return phi

def T_ust2(zL,a):
    phi=1.07*(0.05+np.abs(zL))**(-1/3)+\
        (-1.14+a*(np.abs(zL)**(-9/10)))*(1-np.tanh(10*(np.abs(zL))**(2/3)))
    return phi

def U_stb(zL,a,c):
    phi=a*(1+3*zL)**(c)
    return phi

def U_ust(zL,a):
    return a*(1-3*zL)**(1/3)

def W_stb(zL,a,c):
    return U_stb(zL,a,c)

def W_ust(zL,a):
    return U_ust(zL,a)

def C_stb(zL,a,b,c,d):
    logphi=a+b*np.log10(zL)+c*np.log10(zL)**2+d*np.log10(zL)**3
    phi=10**(logphi)
    return phi

def C_ust(zL,a):
    #return a*(1+b*zL)**(-1/3)
    return a*(1-25*zL)**(-1/3)
    #return a*(1+b*zL)**(c)
    #return (-zL)**a


#########################################
# vars/stabs
# |---> sites
# |   |---> [all sites]
# |       |---> binvals, counts, params, param_var,SS,SSlo,SShi
# |---> equation = fxn
# |---> params = []
# |---> std_params = []


def get_phi(fp,var):
    zL_=(fp['zzd'][:])/fp['L_MOST'][:]
    ani=fp['ANI_YB'][:]
    if 'U' in var:
        phi=np.sqrt(fp['UU'][:])/fp['USTAR'][:]
    elif 'V' in var:
        phi=np.sqrt(fp['VV'][:])/fp['USTAR'][:]
    elif 'W' in var:
        phi=np.sqrt(fp['WW'][:])/fp['USTAR'][:]
    elif 'T' in var:
        phi=np.abs(fp['T_SONIC_SIGMA'][:]/(fp['WTHETA'][:]/fp['USTAR'][:]))
    elif 'H2O' in var:
        molh2o=18.02*10**(-3)
        moldry=28.97*10**(-3)
        kgdry_m3=fp['RHO'][:]/(fp['H2O'][:]/1000+1)
        rr = fp['H2O_SIGMA'][:]*molh2o/moldry
        lhv=2500827 - 2360*(fp['T_SONIC'][:]-273)
        phi=np.abs(rr/(fp['LE'][:]/lhv/fp['USTAR'][:]))*kgdry_m3/10**3
    elif 'CO2' in var:
        molh2o=18.02*10**(-3)
        moldry=28.97*10**(-3)
        co2 = fp['CO2_SIGMA'][:]
        if 's' in var:
            co2=co2/10**6
        kgdry_m3=fp['RHO'][:]/(fp['H2O'][:]/1000+1)
        moldry_m3=kgdry_m3/moldry
        phi=np.abs(co2/(fp['CO2FX'][:]/fp['USTAR'][:]))*moldry_m3
    return phi, ani, zL_


def binit(ani,binsize=float('nan'),n=100,vmn_a=vmn_a,vmx_a=vmx_a):
    mmm=(ani>=vmn_a)&(ani<=vmx_a)
    N=np.sum(mmm)
    if np.isnan(binsize):
        pass
    else:
        n=np.floor(N/binsize)
    anibins=np.zeros((n+1,))
    for i in range(n+1):
        anibins[i]=np.nanpercentile(ani[mmm],i/n*100)
    return anibins,mmm





###########################################################

d_fit={}

fxns_={'Uu':U_ust,'Vu':U_ust,'Wu':W_ust,'Tu':T_ust,'H2Ou':C_ust,'CO2u':C_ust,\
       'Us':U_stb,'Vs':U_stb,'Ws':W_stb,'Ts':T_stb,'H2Os':C_stb,'CO2s':C_stb}

p0s={'Uu':[2.5],'Vu':[2],'Wu':[1.3],'Tu':[.1],'H2Ou':[-1/3],'CO2u':[-.25],\
#p0s={'Uu':[2.5],'Vu':[2],'Wu':[1.3],'Tu':[.1],'H2Ou':[4,-25,-1/3],'CO2u':[4,-80,-.25],\
     'Us':[2,.08],'Vs':[2,.08],'Ws':[1.4,.03],'Ts':[.35,0.025,-.03,-.025],\
     'H2Os':[.5,-.1,.05,.1],'CO2s':[.5,-.1,.05,.1]}

bounds={'Uu':([0],[10]),\
        'Vu':([0],[10]),\
        'Wu':([0],[10]),\
        'Tu':([0],[10]),\
        'H2Ou':([-1],[-.1]),\
        'CO2u':([-1],[-.1]),\
        'Us':([0,0],[5,1]),\
        'Vs':([0,0],[5,1]),\
        'Ws':([0,0],[5,1]),\
        'Ts':([.05,-.1,-.1,-.1],[.6,.1,.1,0]),\
        'H2Os':([0,-.1,-.1,-.1],[1,0,.1,.1]),\
        'CO2s':([0,-.1,-.1,-.1],[1,0,.1,.1])}

sites=[]
for file in os.listdir('/home/tswater/Documents/Elements_Temp/NEON/neon_1m/'):
    if 'L1' in file:
        sites.append(file[0:4])
sites.sort()
sites.append('ALL')

fp=h5py.File('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data_v2/NEON_TW_U_UVWT.h5','r')

varlist=list(fxns_.keys())
#varlist=['Uu','Vu','Wu','Tu','H2Ou','CO2u','Us','Vs','Ws']
varlist=['Uu','Vu','Wu','Us','Vs','Ws']

for var in varlist:
    print(var)
    # Identify the filename to load in the data
    fname='/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data_v2/NEON_TW_'
    if 'u' in var:
        fname=fname+'U_'
        zLbins=-np.logspace(-4,2,40)[-1:0:-1]
    else:
        fname=fname+'S_'
        zLbins=np.logspace(-4,2,40)
    if ('U' in var) or ('V' in var) or ('T' in var) or ('W' in var):
        fname=fname+'UVWT.h5'
    elif ('H2O' in var):
        fname=fname+'H2O.h5'
    else:
        fname=fname+'CO2.h5'
    fp=h5py.File(fname,'r')

    phi_,ani_,zL_ = get_phi(fp,var)

    anibins,m=binit(ani_)

    # Initialize d_fit structure
    d_fit[var]={'param':[],'param_var':[],\
                'SS':float('nan'),'SSlo':float('nan'),'SShi':float('nan'),\
                'anibins':anibins}

    Np=len(inspect.signature(fxns_[var]).parameters)-1
    params=np.ones((len(anibins)-1,Np))*float('nan')
    p_vars=np.ones((len(anibins)-1,Np))*float('nan')
    print('    ')
    for i in range(len(anibins)-1):
        print('.',end='',flush=True)
        m=(ani_>anibins[i])&(ani_<anibins[i+1])&(~np.isnan(phi_))
        try:
            if var in bounds.keys():
                params[i,:],pcov=optimize.curve_fit(fxns_[var],zL_[m],phi_[m],p0s[var],bounds=bounds[var],loss='cauchy')
            else:
                params[i,:],pcov=optimize.curve_fit(fxns_[var],zL_[m],phi_[m],p0s[var])
            for p in range(Np):
                p_vars[i,p]=pcov[p,p]
        except Exception as e:
            print(e)
    print()
    d_fit[var]['param']=params[:]
    d_fit[var]['param_var']=p_vars

pickle.dump(d_fit,open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_fit_v2.p','wb'))
