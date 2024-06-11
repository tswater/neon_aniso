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

#########################################
# vars/stabs
# |---> sites
# |   |---> [all sites]
# |       |---> binvals, counts, params, param_var,SS,SSlo,SShi
# |---> equation = fxn
# |---> params = []
# |---> std_params = []


def get_phi(fp,var):
    zL_=(fp['tow_height'][:]-fp['zd'][:])/fp['L_MOST'][:]
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
        co2=co2/10**6
        kgdry_m3=fp['RHO'][:]/(fp['H2O'][:]/1000+1)
        moldry_m3=kgdry_m3/moldry
        phi=np.abs(co2/(fp['CO2FX'][:]/fp['USTAR'][:]))*moldry_m3
    return phi, ani, zL_


def binit(ani,binsize=float('nan'),n=100):
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
    
        
def T_stb(zL,a,b,c,d):
    logphi=a+b*np.log10(zL)+c*np.log10(zL)**2+d*np.log10(zL)**3
    phi=10**(logphi)
    return phi

def T_stb2(zL,a,b):
    return a*(zL)**(-1.4)+b #sfyri reference

def T_stb3(zL,a,b):
    return a*(1+b*zL)**(-1) #Nadeau 2013a

def T_stb4(zL,a,b):
    return a*(zL)**(-1)+b #Pahlow 2001

def T_stb5(zL,a,b,c):
    return a*(zL)**(b)+c #Pahlow 2001

def T_stb6(zL,a,b):
    return a*(zL)**(-1/3)+b # Quan and Hu 2009

def T_stb7(zL,a,b):
    return a*(1+b*zL)**(-1/3) #Nadeau 2013a

def C_stb7(zL,a,b):
    return a*np.log10(zL)+b

def C_stb2(zL,a,b): # ramana 2004
    return a*(1+b*zL)**(-.33)

def C_stb3(zL,a,b): # ramana 2004 another degree of freedom
    return a*(1+b*zL)**(-.05)

def C_stb4(zL,a,b):
    return a*(zL)**(-.2)+b

def C_stb5(zL,a,b):
    return a*(zL)**(-.05)+b #Pahlow 2001

def C_stb6(zL,a,b):
    return a*(zL)**(-.1)+b #Pahlow 2001

###########################################################

d_fit={}

fxns_={'T2':T_stb2,'T3':T_stb3,'T4':T_stb4,'T5':T_stb5,'T6':T_stb6,'T7':T_stb7,\
       'H2O2':C_stb2,'H2O3':C_stb3,'H2O4':C_stb4,'H2O5':C_stb5,'H2O6':C_stb6,'H2O7':C_stb7,\
       'CO22':C_stb2,'CO23':C_stb3,'CO24':C_stb4,'CO25':C_stb5,'CO26':C_stb6,'CO27':C_stb7}
p0s={'T2':[.00087,2.03],'T3':[3.22,.83],'T4':[0.05,3],\
     'T5': [.05,-1,3],'T6':[3,0],'T7':[3.22,.83],\
     'H2O2':[6.45,.25],'H2O3':[6.45,.25],'H2O4':[2,0],'H2O5':[2,0],'H2O6':[2,0],'H2O7':[-1,0],\
     'CO22':[6.45,.25],'CO23':[6.45,.25],'CO24':[2,0],'CO25':[2,0],'CO26':[2,0],'CO27':[-1,0]}

bounds={'T2':([0.00001,.1],[1,25]),'T3':([.1,.01],[25,10]),'T4':([.00001,.01],[8,25]),\
        'T5':([.00001,-2,.01],[8,-.2,25]),'T6':([.001,-1],[30,8]),'T7':([.01,.001],[50,25])}
for i in range(2,8):
    if i<4:
        bd=([0,0],[100,1])
    elif i==7:
        bd=([-20,-20],[0,20])
    else:
        bd=([0,-20],[100,20])
    bounds['H2O'+str(i)]=bd
    bounds['CO2'+str(i)]=bd

#fxns_={'Uu':U_ust,'Vu':U_ust,'Wu':W_ust,'Tu':T_ust,'H2Ou':C_ust,'CO2u':C_ust,\
#       'Us':U_stb,'Vs':U_stb,'Ws':W_stb,'Ts':T_stb,'H2Os':C_stb,'CO2s':C_stb}

#p0s={'Uu':[2.5],'Vu':[2],'Wu':[1.3],'Tu':[.1],'H2Ou':[-1/3],'CO2u':[-.25],\
#p0s={'Uu':[2.5],'Vu':[2],'Wu':[1.3],'Tu':[.1],'H2Ou':[4,-25,-1/3],'CO2u':[4,-80,-.25],\
#     'Us':[2,.08],'Vs':[2,.08],'Ws':[1.4,.03],'Ts':[.35,0.025,-.03,-.025],\
#     'H2Os':[.5,-.1,.05,.1],'CO2s':[.5,-.1,.05,.1]}

#bounds={'Uu':([0],[10]),\
#        'Vu':([0],[10]),\
#        'Wu':([0],[10]),\
#        'Tu':([0],[10]),\
#        'H2Ou':([-1],[-.1]),\
#        'CO2u':([-1],[-.1]),\
#        'Us':([0,0],[5,1]),\
#        'Vs':([0,0],[5,1]),\
#        'Ws':([0,0],[5,1]),\
#        'Ts':([.05,-.1,-.1,-.1],[.6,.1,.1,0]),\
#        'H2Os':([0,-.1,-.1,-.1],[1,0,.1,.1]),\
#        'CO2s':([0,-.1,-.1,-.1],[1,0,.1,.1])}

sites=[]
for file in os.listdir('/home/tsw35/tyche/neon_1m/'):
    if 'L1' in file:
        sites.append(file[0:4])
sites.sort()
sites.append('ALL')

fp=h5py.File('/home/tsw35/soteria/neon_advanced/qaqc_data/NEON_TW_U_UVWT.h5','r')

varlist=list(fxns_.keys())
#varlist=['Uu','Vu','Wu','Tu','H2Ou','CO2u','Us','Vs','Ws']
#varlist=['CO2u','H2Ou']
varlist=['CO26']

for var in varlist:
    print(var)
    # Identify the filename to load in the data
    fname='/home/tsw35/soteria/neon_advanced/qaqc_data/NEON_TW_'
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

pickle.dump(d_fit,open('/home/tsw35/soteria/neon_advanced/data/d_fit_scalarC.p','wb'))
