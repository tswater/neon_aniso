import numpy as np
import h5py
import pickle
import os

sites=[]
for file in os.listdir('/home/tswater/Documents/Elements_Temp/NEON/neon_1m'):
    if 'L1' in file:
        sites.append(file[0:4])
sites.sort()

vmx_a=.9
vmn_a=.1
procdir='/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data/'

anibins=np.linspace(vmn_a,vmx_a,8)
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

def add_species(d_,sp,phi,zL,ani,m,fpsite,stb=False):
    d_[sp]['phi']=phi[m]
    d_[sp]['zL']=zL[m]
    d_[sp]['ani']=ani[m]

    x,y,_,c=binplot1d(zL,phi,ani,stb)

    d_[sp]['p_phi']=y
    d_[sp]['p_zL']=x
    d_[sp]['p_cnt']=c

    d_[sp]['p_phi_s']={}
    d_[sp]['p_zL_s']={}
    d_[sp]['p_cnt_s']={}

    for site in sites:
        ms=((site.encode('UTF-8'))==fpsite)
        x,y,_,c=binplot1d(zL[ms],phi[ms],ani[ms],stb)
        d_[sp]['p_phi_s'][site]=y
        d_[sp]['p_zL_s'][site]=x
        d_[sp]['p_cnt_s'][site]=c

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
zL=(fp['tow_height'][:]-fp['zd'][:])/fp['L_MOST'][:]
ani=fp['ANI_XB'][:]
fpsite=fp['SITE'][:]

m=np.zeros((len(ani),))
m[0:100000]=1
np.random.shuffle(m)
m=m.astype(bool)

d_unst=add_species(d_unst,'U',phi,zL,ani,m,fpsite)

#### V ####
print('Done with U unstable')
print('Processing V unstable',flush=True)
d_unst['V']={}
phi=np.sqrt(fp['VV'][:])/fp['USTAR'][:]

d_unst=add_species(d_unst,'V',phi,zL,ani,m,fpsite)

#### W ####
print('Done with V unstable')
print('Processing W unstable',flush=True)
d_unst['W']={}
phi=np.sqrt(fp['WW'][:])/fp['USTAR'][:]

d_unst=add_species(d_unst,'W',phi,zL,ani,m,fpsite)

#### T ####
print('Done with W unstable')
print('Processing T unstable',flush=True)
d_unst['T']={}
phi=np.abs(fp['T_SONIC_SIGMA'][:]/(fp['WTHETA'][:]/fp['USTAR'][:]))

d_unst=add_species(d_unst,'T',phi,zL,ani,m,fpsite)

##### D_ANI_UST #####
#def add_ani(d_,anix,aniy,zL,lc,fpsites):
print('Done with T unstable')
print('Processing Ani unstable',flush=True)

d_ani_ust=add_ani(d_ani_ust,fp['ANI_XB'][:],fp['ANI_YB'][:],zL,fp['nlcd_dom'][:],fpsite)
fp.close()

print('Done with Ani unstable')
print('Processing H2O unstable',flush=True)

#### H2O ####
fp=h5py.File(procdir+'NEON_TW_U_H2O.h5','r')

d_unst['H2O']={}
molh2o=18.02*10**(-3)
moldry=28.97*10**(-3)
kgdry_m3=fp['RHO'][:]/(fp['H2O'][:]/1000+1)
rr = fp['H2O_SIGMA'][:]*molh2o/moldry
lhv=2500827 - 2360*(fp['T_SONIC'][:]-273)
phi=np.abs(rr/(fp['LE'][:]/lhv/fp['USTAR'][:]))*kgdry_m3/10**3

zL=(fp['tow_height'][:]-fp['zd'][:])/fp['L_MOST'][:]
ani=fp['ANI_XB'][:]
fpsite=fp['SITE'][:]

m=np.zeros((len(ani),))
m[0:100000]=1
np.random.shuffle(m)
m=m.astype(bool)

d_unst=add_species(d_unst,'H2O',phi,zL,ani,m,fpsite)
fp.close()
print('Done with H2O unstable')
print('Processing CO2 unstable',flush=True)


#### CO2 ####
fp=h5py.File(procdir+'NEON_TW_U_CO2.h5','r')

d_unst['CO2']={}
molh2o=18.02*10**(-3)
moldry=28.97*10**(-3)
co2 = fp['CO2_SIGMA'][:]

kgdry_m3=fp['RHO'][:]/(fp['H2O'][:]/1000+1)
moldry_m3=kgdry_m3/moldry
phi=np.abs(co2/(fp['CO2FX'][:]/fp['USTAR'][:]))*moldry_m3
zL=(fp['tow_height'][:]-fp['zd'][:])/fp['L_MOST'][:]
ani=fp['ANI_XB'][:]
fpsite=fp['SITE'][:]

m=np.zeros((len(ani),))
m[0:100000]=1
np.random.shuffle(m)
m=m.astype(bool)

d_unst=add_species(d_unst,'CO2',phi,zL,ani,m,fpsite)
fp.close()

###############################################
#### D_STBL #####
###############################################

#### U ####
print('Done with CO2 unstable')
print('Processing U stable',flush=True)
fp=h5py.File(procdir+'NEON_TW_S_UVWT.h5','r')

d_stbl['U']={}

phi=np.sqrt(fp['UU'][:])/fp['USTAR'][:]
zL=(fp['tow_height'][:]-fp['zd'][:])/fp['L_MOST'][:]
ani=fp['ANI_XB'][:]
fpsite=fp['SITE'][:]

m=np.zeros((len(ani),))
m[0:100000]=1
np.random.shuffle(m)
m=m.astype(bool)

d_stbl=add_species(d_stbl,'U',phi,zL,ani,m,fpsite,stb=True)

#### V ####
print('Done with U stable')
print('Processing V stable',flush=True)
d_stbl['V']={}
phi=np.sqrt(fp['VV'][:])/fp['USTAR'][:]

d_stbl=add_species(d_stbl,'V',phi,zL,ani,m,fpsite,stb=True)

#### W ####
print('Done with V stable')
print('Processing W stable',flush=True)
d_stbl['W']={}
phi=np.sqrt(fp['WW'][:])/fp['USTAR'][:]

d_stbl=add_species(d_stbl,'W',phi,zL,ani,m,fpsite,stb=True)

#### T ####
print('Done with W stable')
print('Processing T stable',flush=True)
d_stbl['T']={}
phi=np.abs(fp['T_SONIC_SIGMA'][:]/(fp['WTHETA'][:]/fp['USTAR'][:]))

d_stbl=add_species(d_stbl,'T',phi,zL,ani,m,fpsite,stb=True)

#### D_ANI_UST #####
#def add_ani(d_,anix,aniy,zL,lc,fpsites):
print('Done with W stable')
print('Processing Ani stable',flush=True)
d_ani_stb=add_ani(d_ani_ust,fp['ANI_XB'][:],fp['ANI_YB'][:],zL,fp['nlcd_dom'][:],fpsite)
fp.close()

#### H2O ####
print('Done with Ani stable')
print('Processing H2O stable',flush=True)
fp=h5py.File(procdir+'NEON_TW_S_H2O.h5','r')

d_stbl['H2O']={}
molh2o=18.02*10**(-3)
moldry=28.97*10**(-3)
kgdry_m3=fp['RHO'][:]/(fp['H2O'][:]/1000+1)

rr = fp['H2O_SIGMA'][:]*molh2o/moldry
lhv=2500827 - 2360*(fp['T_SONIC'][:]-273)

phi=np.abs(rr/(fp['LE'][:]/lhv/fp['USTAR'][:]))*kgdry_m3/10**3
zL=(fp['tow_height'][:]-fp['zd'][:])/fp['L_MOST'][:]
ani=fp['ANI_XB'][:]
fpsite=fp['SITE'][:]

m=np.zeros((len(ani),))
m[0:100000]=1
np.random.shuffle(m)
m=m.astype(bool)

d_stbl=add_species(d_stbl,'H2O',phi,zL,ani,m,fpsite,stb=True)

#### CO2 ####
print('Done with H2O stable')
print('Processing CO2 stable',flush=True)
fp=h5py.File(procdir+'NEON_TW_S_CO2.h5','r')

d_stbl['CO2']={}
molh2o=18.02*10**(-3)
moldry=28.97*10**(-3)
co2 = fp['CO2_SIGMA'][:]

co2=co2/10**6

kgdry_m3=fp['RHO'][:]/(fp['H2O'][:]/1000+1)
moldry_m3=kgdry_m3/moldry
phi=np.abs(co2/(fp['CO2FX'][:]/fp['USTAR'][:]))*moldry_m3

zL=(fp['tow_height'][:]-fp['zd'][:])/fp['L_MOST'][:]
ani=fp['ANI_XB'][:]
fpsite=fp['SITE'][:]
m=np.zeros((len(ani),))
m[0:100000]=1
np.random.shuffle(m)
m=m.astype(bool)

d_stbl=add_species(d_stbl,'CO2',phi,zL,ani,m,fpsite,stb=True)

print('Done with CO2 stable')
print('Pickling',flush=True)

####################################
######## PICKLE ####################
####################################
pickle.dump(d_unst,open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_unst_xb.p','wb'))
pickle.dump(d_stbl,open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_stbl_xb.p','wb'))
#pickle.dump(d_ani_ust,open('/home/tsw35/soteria/neon_advanced/data/d_ani_ust_v2.p','wb'))
#pickle.dump(d_ani_stb,open('/home/tsw35/soteria/neon_advanced/data/d_ani_stb_v2.p','wb'))

