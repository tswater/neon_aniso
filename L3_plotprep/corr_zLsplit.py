import numpy as np
import h5py
import pickle
import os
from scipy import stats

sites=[]
for file in os.listdir('/home/tswater/Documents/Elements_Temp/NEON/neon_1m'):
    if 'L1' in file:
        sites.append(file[0:4])
sites.sort()

fps=h5py.File('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data_v2/NEON_TW_S_UVWT.h5','r')
fpu=h5py.File('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data_v2/NEON_TW_U_UVWT.h5','r')

xvars=['ANI_XB','ANI_YB','D_MOST_U','D_MOST_V','D_MOST_W','D_SC23_U',\
       'D_SC23_V','D_SC23_W']
yvars=['ff_cov_chm','ff_cov_dsmchm','tke','zL']

for k in fpu.keys():
    if 'nlcd' in k:
        pass
    elif 'aspect' in k:
        pass
    elif k=='TIME':
        pass
    elif k=='USTAR2':
        pass
    elif k in ['WTHETA','VPD','T_SONIC','SW_OUT','SW_IN','LW_OUT','Vstr','WS','DAY']:
        pass
    elif 'SIGMA' in k:
        pass
    elif 'slope' in k:
        pass
    elif k in ['LW_IN','VPT']:
        pass
    elif 'CO2' in k:
        pass
    elif k=='SITE':
        pass
    else:
        yvars.append(k)

def fix(d1,d2):
    new1=[]
    new2=[]
    for i in range(len(d1)):
        if (np.isnan(d2[i])) or (d2[i]==float('Inf')):
            continue
        else:
            new1.append(d1[i])
            new2.append(d2[i])
    return np.array(new1),np.array(new2)


#### FXNS ####
def u_most_u(zL,ani):
    return 2.55*(1-3*zL)**(1/3)

def u_most_v(zL,ani):
    return 2.05*(1-3*zL)**(1/3)

def u_most_w(zL,ani):
    return 1.35*(1-3*zL)**(1/3)

def u_sc23_u(zL,ani):
    a=.784-2.582*np.log10(ani)
    u_u_stp=a*(1-3*zL)**(1/3)
    return u_u_stp

def u_sc23_v(zL,ani):
    a=.725-2.702*np.log10(ani)
    return a*(1-3*zL)**(1/3)

def u_sc23_w(zL,ani):
    a=1.119-0.019*ani-.065*ani**2+0.028*ani**3
    return a*(1-3*zL)**(1/3)

def u_tsw_u(zL,ani):
    a=4.123621495800058-9.019103428856788*ani+7.031233766838203*ani**2
    u_u_stp=a*(1-3*zL)**(1/3)
    return u_u_stp

def u_tsw_v(zL,ani):
    a=4.287424385993223-11.432206917682109*ani+10.714434337994307*ani**2
    return a*(1-3*zL)**(1/3)

def u_tsw_w(zL,ani):
    a=0.8322246683633081+ani*2.2982094340935624-ani**2*5.9245962466602675+ani**3*4.359932190370337
    return a*(1-3*zL)**(1/3)

def s_most_u(zL,ani):
    return 2.06*zL
def s_most_v(zL,ani):
    return 2.06*zL
def s_most_w(zL,ani):
    return 1.6*zL

def s_sc23_u(zL,ani):
    a_=np.array([2.332,-2.047,2.672])
    c_=np.array([.255,-1.76,5.6,-6.8,2.65])
    a=0
    c=0
    for i in range(3):
        a=a+a_[i]*ani**i
    for i in range(5):
        c=c+c_[i]*ani**i
    return a*(1+3*zL)**(c)

def s_sc23_v(zL,ani):
    a_=np.array([2.385,-2.781,3.771])
    c_=np.array([.654,-6.282,21.975,-31.634,16.251])
    a=0
    c=0
    for i in range(3):
        a=a+a_[i]*ani**i
    for i in range(5):
        c=c+c_[i]*ani**i
    return a*(1+3*zL)**(c)

def s_sc23_w(zL,ani):
    a_=np.array([.953,.188,2.243])
    c_=np.array([.208,-1.935,6.183,-7.485,3.077])
    a=0
    c=0
    for i in range(3):
        a=a+a_[i]*ani**i
    for i in range(5):
        c=c+c_[i]*ani**i
    return a*(1+3*zL)**(c)


def s_tsw_u(zL,ani):
    a_=np.array([2.48999325402708,-2.9309894919097794,3.9665235021601855])
    c_=np.array([-0.014091947587046363,0.4413024489123778,-0.4734026436270551])
    a=0
    c=0
    for i in range(3):
        a=a+a_[i]*ani**i
    for i in range(3):
        c=c+c_[i]*ani**i
    return a*(1+3*zL)**(c)

def s_tsw_v(zL,ani):
    a_=np.array([1.3330870632847518,0.5841790794590472,1.3070551206541443])
    c_=np.array([0.22831158178720407,-0.48384488258397285,0.48954414345477587,-0.14416529116966556])
    a=0
    c=0
    for i in range(3):
        a=a+a_[i]*ani**i
    for i in range(4):
        c=c+c_[i]*ani**i
    return a*(1+3*zL)**(c)

def s_tsw_w(zL,ani):
    a_=np.array([0.8568248498732863,0.3850311195961679,2.398970306113473])
    c_=np.array([0.08589790229105511,0.09024112195753205,-0.3132154689043823,0.24661181960543574])
    a=0
    c=0
    for i in range(3):
        a=a+a_[i]*ani**i
    for i in range(4):
        c=c+c_[i]*ani**i
    return a*(1+3*zL)**(c)

fxns={0:{'D_MOST_U':u_most_u,'D_MOST_V':u_most_v,'D_MOST_W':u_most_w,\
        'D_SC23_U':u_sc23_u,'D_SC23_V':u_sc23_v,'D_SC23_W':u_sc23_w,\
        'D_TSW_U':u_tsw_u,'AD_TSW_V':u_tsw_v,'AD_TSW_W':u_tsw_w,},
      1:{'D_MOST_U':s_most_u,'D_MOST_V':s_most_v,'D_MOST_W':s_most_w,\
        'D_SC23_U':s_sc23_u,'D_SC23_V':s_sc23_v,'D_SC23_W':s_sc23_w,\
        'AD_TSW_U':s_tsw_u,'AD_TSW_V':s_tsw_v,'AD_TSW_W':s_tsw_w,}}



Nx=len(xvars)
Ny=len(yvars)

d_unst={'xvars':xvars,'yvars':yvars,'spearmanr':np.ones((2,Nx,Ny))*float('nan'),'sitelevel':{}}
d_stbl={'xvars':xvars,'yvars':yvars,'spearmanr':np.ones((2,Nx,Ny))*float('nan'),'sitelevel':{}}

for site in sites:
    d_unst['sitelevel'][site]={'spearmanr':np.ones((2,Nx,Ny))*float('nan')}
    d_stbl['sitelevel'][site]={'spearmanr':np.ones((2,Nx,Ny))*float('nan')}


# need to compute 'ff_cov_chm','ff_cov_dsmchm','tke','z_zd','zL'
for i in [0]:
    for st in [0,1]:
        if i==0:
            d_=d_unst
            fp=fpu
        else:
            d_=d_stbl
            fp=fps
        for j in range(Nx):
            varx=xvars[j]
            zL=fp['zzd'][:]/fp['L_MOST'][:]
            if st==0:
                mm=np.abs(zL)<.1
            elif st==1:
                mm=np.abs(zL)>=.1
            if 'ANI' in varx:
                xdata=fp[varx][:]
            else:
                zL=fp['zzd'][:]/fp['L_MOST'][:]
                ani=fp['ANI_YB'][:]
                if '_U' in varx:
                    phi=np.sqrt(fp['UU'][:])/fp['USTAR'][:]
                elif '_V' in varx:
                    phi=np.sqrt(fp['VV'][:])/fp['USTAR'][:]
                elif '_W' in varx:
                    phi=np.sqrt(fp['WW'][:])/fp['USTAR'][:]
                phix=fxns[i][varx](zL,ani)
                xdata=phix-phi
            for k in range(Ny):
                vary=yvars[k]
                if vary=='ff_cov_chm':
                    ydata=fp['std_chm'][:]/(fp['mean_chm'][:]+.001)
                elif vary=='ff_cov_dsmchm':
                    ydata=fp['std_dsm'][:]/(fp['mean_chm'][:]+.001)
                elif vary=='tke':
                    ydata=.5*(fp['UU'][:]+fp['VV'][:]+fp['WW'][:])
                elif vary=='zL':
                    ydata=fp['zzd'][:]-fp['L_MOST'][:]
                else:
                    ydata=fp[vary][:]
                print(varx+' '+vary,flush=True)
                if np.sum(np.isnan(ydata))>0:
                    xx,yy=fix(xdata[mm],ydata[mm])
                else:
                    xx=xdata[mm]
                    yy=ydata[mm]
                d_['spearmanr'][st,j,k]=stats.spearmanr(xx,yy)[0]
                for site in np.unique(fp['SITE'][:]):
                    print('.',end='',flush=True)
                    m=fp['SITE'][:]==site
                    if np.sum(np.isnan(ydata[m]))>0:
                        xx,yy=fix(xdata[m&mm],ydata[m&mm])
                    else:
                        xx=xdata[m&mm]
                        yy=ydata[m&mm]
                    d_['sitelevel'][str(site)[2:-1]]['spearmanr'][st,j,k]=stats.spearmanr(xx,yy)[0]
    if i==0:
        pickle.dump(d_unst,open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_corr_ust_zL.p','wb'))
    else:
        pickle.dump(d_stbl,open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_corr_stb_zL.p','wb'))

####################################
######## PICKLE ####################
####################################

