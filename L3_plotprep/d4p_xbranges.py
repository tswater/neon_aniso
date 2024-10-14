import numpy as np
import h5py
import pickle
import os

# Same as data4plots but compute errors for certain xb ranges

sites=[]
for file in os.listdir('/home/tswater/Documents/Elements_Temp/NEON/neon_1m'):
    if 'L1' in file:
        sites.append(file[0:4])
sites.sort()

vmx_a=.7
vmn_a=.1
procdir='/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data_v2/'

anibins=np.linspace(vmn_a,vmx_a,11)
zLbins=np.logspace(-4,2,40)[-1:0:-1]

##### FUNCITONS #####
def binplot1d(xx,yy,ani,stb,xbins=-zLbins,anibins=anibins):
    if stb:
        xbins=-xbins[-1:0:-1]
    xplot=(xbins[0:-1]+xbins[1:])/2
    yplot=np.zeros((len(anibins)-1,len(xplot)))
    count=np.zeros((len(anibins)-1,len(xplot)))
    aniplot=(anibins[0:-1]+anibins[1:])/2
    for i in range(len(anibins)-1):
        #print('ANI: '+str(anibins[i]))
        for j in range(len(xbins)-1):
            #print('  ZL: '+str(xbins[j]))
            pnts=yy[(ani>=anibins[i])&(ani<anibins[i+1])&(xx>=xbins[j])&(xx<xbins[j+1])]
            count[i,j]=len(pnts)
            yplot[i,j]=np.nanmedian(pnts)
    return xplot,yplot,aniplot,count

def skill(d_0,d_old,d_new):
    mad_n=np.nanmedian(np.abs(d_0-d_new))
    mad_o=np.median(np.abs(d_0-d_old))
    ss=1-np.nanmedian(np.abs(d_0-d_new))/np.median(np.abs(d_0-d_old))
    return mad_n,mad_o,ss

def getbins(A,n):
    B=np.sort(A)
    bins=[]
    for i in np.linspace(0,len(A)-1,n):
        i=int(i)
        bins.append(B[i])
    return bins

def add_species(d_,sp,phi,zL,ani,phi_new,phi_old,m,fpsite,xb,stb=False):
    d_[sp]['phi']=phi[m]
    d_[sp]['zL']=zL[m]
    d_[sp]['ani']=ani[m]

    x,y,_,c=binplot1d(zL,phi,ani,stb)

    d_[sp]['p_phi']=y
    d_[sp]['p_zL']=x
    d_[sp]['p_cnt']=c

    mlo=np.abs(zL)<.1
    mhi=np.abs(zL)>.1

    xbi=getbins(xb,11)

    d_[sp]['xbins']=xbi

    d_[sp]['MAD_SC23']=[]
    d_[sp]['MAD_OLD']=[]
    d_[sp]['SS']=[]
    d_[sp]['MAD_SC23_s']={}
    #d_[sp]['MAD_SC23lo_s']={}
    #d_[sp]['MAD_SC23hi_s']={}
    d_[sp]['MAD_OLD_s']={}
    #d_[sp]['MAD_OLDlo_s']={}
    #d_[sp]['MAD_OLDhi_s']={}
    d_[sp]['SS_s']={}
    #d_[sp]['SSlo_s']={}
    #d_[sp]['SShi_s']={}


    for site in sites:
        d_[sp]['MAD_SC23_s'][site]=[]
        d_[sp]['MAD_OLD_s'][site]=[]
        d_[sp]['SS_s'][site]=[]
    d_[sp]['xbins_s']={}
    for i in range(10):
        mxb=(xb<xbi[i+1])&(xb>xbi[i])
        a,b,c=skill(phi[mxb],phi_old[mxb],phi_new[mxb])
        d_[sp]['MAD_SC23'].append(a)
        d_[sp]['MAD_OLD'].append(b)
        d_[sp]['SS'].append(c)

        for site in sites:
            ms=((site.encode('UTF-8'))==fpsite)
            xbi2=getbins(xb[ms],11)
            d_[sp]['xbins_s'][site]=xbi2
            ms2=ms&(xb<xbi2[i+1])&(xb>xbi2[i])
            a,b,c=skill(phi[ms2],phi_old[ms2],phi_new[ms2])
            d_[sp]['MAD_SC23_s'][site].append(a)
            d_[sp]['MAD_OLD_s'][site].append(b)
            d_[sp]['SS_s'][site].append(c)

    return d_

def add_ani(d_,anix,aniy,zL,lc,fpsites):
    mlo=np.abs(zL)<.1
    mhi=np.abs(zL)>.1
    sites2=['ALL']
    sites2.extend(sites)
    for site in sites2:
        mhis=mhi.copy()
        mlos=mlo.copy()
        if site=='ALL':
            ms=np.ones((len(fpsite),),dtype=bool)
        else:
            mlos=mlos&((site.encode('UTF-8'))==fpsite)
            mhis=mhis&((site.encode('UTF-8'))==fpsite)
            ms=((site.encode('UTF-8'))==fpsite)
        d_[site]={}
        for zb in ['xb','yb']:
            d_[site][zb]={}
            if zb=='xb':
                zb_=anix
            elif zb=='yb':
                zb_=aniy
            for stab in ['full','hi','lo']:
                d_[site][zb][stab]={}
                if stab=='full':
                    m_=ms
                elif stab=='hi':
                    m_=mhis
                elif stab=='lo':
                    m_=mlos
                d_[site][zb][stab]['lc']=lc[m_][0]
                d_[site][zb][stab]['median']=np.nanmedian(zb_[m_])
                d_[site][zb][stab]['pct90']=np.nanpercentile(zb_[m_],90)
                d_[site][zb][stab]['pct10']=np.nanpercentile(zb_[m_],10)
                d_[site][zb][stab]['std']=np.nanstd(zb_[m_])

    return d_


####################


# unstable  -> species -> [p_phi,p_zL,phi,zL,ani,SS,SSlo,SShi,SS_s,SSlo_s,SShi_s]
# stable    -> species -> [...]
# aniso_all -> sites   -> [yb,xb] -> [lc,median,pct90,pct10,std]
# aniso_stb -> sites   -> [yb,xb] -> [full,hi,lo] -> [lc,median,pct90,pct10,std]
# aniso_ust -> sites   -> [yb,xb] -> [full,hi,lo] -> [lc,median,pct90,pct10,std]

d_unst={}
d_stbl={}
d_ani_all={}
d_ani_stb={}
d_ani_ust={}

###############################################
#### D_UNST #####
################################################

#### U ####
print('Processing U unstable')
fp=h5py.File(procdir+'NEON_TW_U_UVWT.h5','r')

d_unst['U']={}

phi=np.sqrt(fp['UU'][:])/fp['USTAR'][:]
zL=(fp['zzd'][:])/fp['L_MOST'][:]
ani=fp['ANI_YB'][:]
xb=fp['ANI_XB'][:]
fpsite=fp['SITE'][:]

m=np.zeros((len(ani),))
m[0:100000]=1
np.random.shuffle(m)
m=m.astype(bool)

a=.784-2.582*np.log10(ani)
phi_stp=a*(1-3*zL)**(1/3)
phi_old=2.55*(1-3*zL)**(1/3)

d_unst=add_species(d_unst,'U',phi,zL,ani,phi_stp,phi_old,m,fpsite,xb)

#### V ####
print('Done with U unstable')
print('Processing V unstable',flush=True)
d_unst['V']={}
phi=np.sqrt(fp['VV'][:])/fp['USTAR'][:]
a=.725-2.702*np.log10(ani)
phi_stp=a*(1-3*zL)**(1/3)
phi_old=2.05*(1-3*zL)**(1/3)

d_unst=add_species(d_unst,'V',phi,zL,ani,phi_stp,phi_old,m,fpsite,xb)

#### W ####
print('Done with V unstable')
print('Processing W unstable',flush=True)
d_unst['W']={}
phi=np.sqrt(fp['WW'][:])/fp['USTAR'][:]
a=1.119-0.019*ani-.065*ani**2+0.028*ani**3
phi_stp=a*(1-3*zL)**(1/3)
phi_old=1.35*(1-3*zL)**(1/3)

d_unst=add_species(d_unst,'W',phi,zL,ani,phi_stp,phi_old,m,fpsite,xb)

###############################################
#### D_STBL #####
###############################################

#### U ####
print('Done with CO2 unstable')
print('Processing U stable',flush=True)
fp=h5py.File(procdir+'NEON_TW_S_UVWT.h5','r')

d_stbl['U']={}

phi=np.sqrt(fp['UU'][:])/fp['USTAR'][:]
zL=(fp['zzd'][:])/fp['L_MOST'][:]
ani=fp['ANI_YB'][:]
xb=fp['ANI_XB'][:]
fpsite=fp['SITE'][:]

m=np.zeros((len(ani),))
m[0:100000]=1
np.random.shuffle(m)
m=m.astype(bool)

a_=np.array([2.332,-2.047,2.672])
c_=np.array([.255,-1.76,5.6,-6.8,2.65])
a=0
c=0
for i in range(3):
    a=a+a_[i]*ani**i
for i in range(5):
    c=c+c_[i]*ani**i
phi_stp=a*(1+3*zL)**(c)
phi_old=2.06*np.ones((len(phi_stp),))

d_stbl=add_species(d_stbl,'U',phi,zL,ani,phi_stp,phi_old,m,fpsite,xb,stb=True)

#### V ####
print('Done with U stable')
print('Processing V stable',flush=True)
d_stbl['V']={}
phi=np.sqrt(fp['VV'][:])/fp['USTAR'][:]

a_=np.array([2.385,-2.781,3.771])
c_=np.array([.654,-6.282,21.975,-31.634,16.251])
a=0
c=0
for i in range(3):
    a=a+a_[i]*ani**i
for i in range(5):
    c=c+c_[i]*ani**i
phi_stp=a*(1+3*zL)**(c)

phi_old=2.06*np.ones((len(phi_stp),))

d_stbl=add_species(d_stbl,'V',phi,zL,ani,phi_stp,phi_old,m,fpsite,xb,stb=True)

#### W ####
print('Done with V stable')
print('Processing W stable',flush=True)
d_stbl['W']={}
phi=np.sqrt(fp['WW'][:])/fp['USTAR'][:]

a_=np.array([.953,.188,2.243])
c_=np.array([.208,-1.935,6.183,-7.485,3.077])
a=0
c=0
for i in range(3):
    a=a+a_[i]*ani**i
for i in range(5):
    c=c+c_[i]*ani**i
phi_stp=a*(1+3*zL)**(c)

phi_old=1.6*np.ones((len(phi_stp),))

d_stbl=add_species(d_stbl,'W',phi,zL,ani,phi_stp,phi_old,m,fpsite,xb,stb=True)

print('Pickling',flush=True)

####################################
######## PICKLE ####################
####################################
pickle.dump(d_unst,open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_unst_xbbins_v2.p','wb'))
pickle.dump(d_stbl,open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_stbl_xbbins_v2.p','wb'))
#pickle.dump(d_ani_ust,open('/home/tsw35/soteria/neon_advanced/data/d_ani_ust_v2.p','wb'))
#pickle.dump(d_ani_stb,open('/home/tsw35/soteria/neon_advanced/data/d_ani_stb_v2.p','wb'))

