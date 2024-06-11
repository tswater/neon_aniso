import numpy as np
import h5py
import pickle
import os

sites=[]
for file in os.listdir('/home/tsw35/tyche/neon_1m/'):
    if 'L1' in file:
        sites.append(file[0:4])
sites.sort()

vmx_a=.7
vmn_a=.1

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
    return 1-np.nanmedian(np.abs(d_0-d_new))/np.median(np.abs(d_0-d_old))

def add_species(d_,sp,phi,zL,ani,phi_new,phi_old,m,fpsite,stb=False):
    d_[sp]['phi']=phi[m]
    d_[sp]['zL']=zL[m]
    d_[sp]['ani']=ani[m]
    
    x,y,_,c=binplot1d(zL,phi,ani,stb)

    d_[sp]['p_phi']=y
    d_[sp]['p_zL']=x
    d_[sp]['p_cnt']=c

    mlo=np.abs(zL)<.1
    mhi=np.abs(zL)>.1
    
    d_[sp]['SS']=skill(phi,phi_old,phi_new)
    d_[sp]['SSlo']=skill(phi[mlo],phi_old[mlo],phi_new[mlo])
    d_[sp]['SShi']=skill(phi[mhi],phi_old[mhi],phi_new[mhi])
    d_[sp]['SS_s']={}
    d_[sp]['SSlo_s']={}
    d_[sp]['SShi_s']={}

    for site in sites:
        mhis=mhi.copy()
        mlos=mlo.copy()
        mlos=mlos&((site.encode('UTF-8'))==fpsite)
        mhis=mhis&((site.encode('UTF-8'))==fpsite)
        ms=((site.encode('UTF-8'))==fpsite)
        d_[sp]['SS_s'][site]=skill(phi[ms],phi_old[ms],phi_new[ms])
        d_[sp]['SSlo_s'][site]=skill(phi[mlos],phi_old[mlos],phi_new[mlos])
        d_[sp]['SShi_s'][site]=skill(phi[mhis],phi_old[mhis],phi_new[mhis])
    
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
                d_[site][zb][stab]['pct75']=np.nanpercentile(zb_[m_],75)
                d_[site][zb][stab]['pct25']=np.nanpercentile(zb_[m_],25)
                d_[site][zb][stab]['mean']=np.nanmean(zb_[m_])
                d_[site][zb][stab]['min']=np.nanmin(zb_[m_])
                d_[site][zb][stab]['max']=np.nanmax(zb_[m_])
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
fp=h5py.File('/home/tsw35/soteria/neon_advanced/qaqc_data/NEON_TW_U_UVWT.h5','r')

d_unst['U']={}

phi=np.sqrt(fp['UU'][:])/fp['USTAR'][:]
zL=(fp['tow_height'][:]-fp['zd'][:])/fp['L_MOST'][:]
ani=fp['ANI_YB'][:]
fpsite=fp['SITE'][:]

##### D_ANI_UST #####
#def add_ani(d_,anix,aniy,zL,lc,fpsites):
print('Processing Ani unstable',flush=True)

d_ani_ust=add_ani(d_ani_ust,fp['ANI_XB'][:],fp['ANI_YB'][:],zL,fp['nlcd_dom'][:],fpsite)
fp.close()

print('Done with Ani unstable')
###############################################
#### D_STBL #####
###############################################

#### U ####
print('Done with CO2 unstable')
print('Processing U stable',flush=True)
fp=h5py.File('/home/tsw35/soteria/neon_advanced/qaqc_data/NEON_TW_S_UVWT.h5','r')

d_stbl['U']={}

phi=np.sqrt(fp['UU'][:])/fp['USTAR'][:]
zL=(fp['tow_height'][:]-fp['zd'][:])/fp['L_MOST'][:]
ani=fp['ANI_YB'][:]
fpsite=fp['SITE'][:]

#### D_ANI_UST #####
#def add_ani(d_,anix,aniy,zL,lc,fpsites):
print('Processing Ani stable',flush=True)
d_ani_stb=add_ani(d_ani_ust,fp['ANI_XB'][:],fp['ANI_YB'][:],zL,fp['nlcd_dom'][:],fpsite)
fp.close()

print('Pickling',flush=True)

####################################
######## PICKLE ####################
####################################
pickle.dump(d_ani_ust,open('/home/tsw35/soteria/neon_advanced/data/d_ani_ust_v2.p','wb'))
pickle.dump(d_ani_stb,open('/home/tsw35/soteria/neon_advanced/data/d_ani_stb_v2.p','wb'))

