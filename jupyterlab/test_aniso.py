# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 300
sns.set_theme()

# %%
a=np.array([[1,2,3],[4,5,6],[7,8,9]])
b=np.linalg.eig(a)[0]
print(b)
b.sort()
b=b[::-1]
print(b)

# %%
c=np.where(a==19)

# %%
c[0][0]

# %%
fp.close()

# %%
dirt='/home/tsw35/soteria/data/eddy_v2/old/ABBY_L2.h5'
fp2=h5py.File(dirt,'r')

# %%
# Load File
fp=h5py.File('/home/tsw35/soteria/neon_advanced/data/QAQCv4_neon_pntEB1_frez_H_H2_H2_T__US_UU_UV_UW_VV_VW_WS_WW_ZL.h5','r')# Sites
focus_sites=[b'WREF',b'NOGP',b'BART',b'SOAP','MAP',b'ORNL',b'SRER',b'KONZ',b'DSNY']
#focus_sites=[b'ABBY',b'SCBI',b'HARV',b'JERC','MAP',b'ORNL',b'TREE',b'OSBS',b'UNDE']
fpsites=fp['site'][:]
lats=fp['lat'][:]
lons=fp['lon'][:]
sites = []
nlat=[]
nlon=[]
fnlat=[]
fnlon=[]
for i in range(len(fpsites)):
    x=fpsites[i]
    if x not in sites:
        nlat.append(lats[i])
        nlon.append(lons[i])
        sites.append(x)
        if x in focus_sites:
            fnlat.append(lats[i])
            fnlon.append(lons[i])

# %%
fp2.close()

# %%
uu=fp['UU'][:]
uu[uu==-9999]=float('nan')
plt.plot(uu)
plt.show()


# %%
def aniso(uu,vv,ww,uv,uw,vw):
    n=len(uu)
    m=uu>-9999
    k=uu[m]+vv[m]+ww[m]
    aniso=np.ones((n,3,3))*-1
    aniso[m,0,0]=uu[m]/k-1/3
    aniso[m,1,1]=vv[m]/k-1/3
    aniso[m,2,2]=ww[m]/k-1/3
    aniso[m,0,1]=uv[m]/k
    aniso[m,1,0]=uv[m]/k
    aniso[m,2,0]=uw[m]/k
    aniso[m,0,2]=uw[m]/k
    aniso[m,1,2]=vw[m]/k
    aniso[m,2,1]=vw[m]/k
    return aniso


# %%
btij=aniso(fp['UU'][:],fp['VV'][:],fp['WW'][:],fp['UV'][:],fp['UW'][:],fp['VW'][:])
y_b=np.zeros((len(uu),))
for i in range(len(uu)):
    lam3=np.min(np.linalg.eig(btij[i,:,:])[0])
    y_b[i]=np.sqrt(3)/2*(3*lam3+1)

# %%

# %%

# %%
uu=fp['WW'][:]
m=uu>-9999
np.sum(m)

# %%
plt.hist(fp['ZL'][:])
plt.show()

# %%
zl=fp['ZL'][:]

# %%
zl=fp['ZL'][:]
m=m&(zl<0.00001)&(zl>-10**(3))

# %%
plt.hist(y_b[m],bins=np.linspace(0,.9,19))
plt.show()

# %%
normU=np.sqrt(fp['UU'][m])/(fp['USTAR'][m]+.00001)

# %%
binlist=[.1,.25,.4,.55,.7,.85]
colors = plt.cm.terrain(np.linspace(0,1,len(binlist)))
for i in range(len(binlist)):
    ir=binlist[i]
    m2=(y_b[m]<ir)&(y_b[m]>(ir-.1))
    un=normU[m2]
    zlp=-zl[m][m2]
    plt.semilogx(zlp,un,'.',alpha=.2,ms=2,color=colors[i])
    zlbins=np.logspace(-4,1.2,10)
    zlout=[]
    unout=[]
    for j in range(1,10):
        zlout.append((zlbins[j]+zlbins[j-1])/2)
        m3=(zlp>zlbins[j-1])&(zlp<zlbins[j])&(un<10)
        if np.sum(m3)<=10:
            unout.append(float('nan'))
        else:
            unout.append(np.mean(un[m3]))
    plt.semilogx(zlout,unout,'-',alpha=.75,color=colors[i])
                         
plt.ylim(0,10)
plt.xlim(10**(-3),10**(1))
plt.show()

# %%

# %%
fp=h5py.File('/home/tsw35/soteria/neon_advanced/data/QAQCv4_neon_pntEB1_frez_H_H2_H2_T__US_UU_UV_UW_VV_VW_WS_WW_ZL.h5','r')
binlist=[.05,.15,.25,.35,.45,.55,.65,.75]
import matplotlib.patheffects as pe
plt.figure(figsize=(5,4),dpi=100)

m=(zl<0.000001)
m=m&(fp['LE'][:]>0.1)
ybnorm=np.array(y_b)[m]
ybnorm=ybnorm/(np.sqrt(3)/2)
colors=plt.cm.turbo(1-ybnorm)
#un=np.sqrt(fp['UU'][m])/(fp['USTAR'][m]+.00001)

un=np.sqrt(fp['H2O_SIGMA'][m])*(fp['USTAR'][m])/fp['LE'][m]
normU=np.sqrt(fp['H2O_SIGMA'][m])*(fp['USTAR'][m])/fp['LE'][m]

plt.scatter(-zl[m],un,alpha=.1,s=.1,color=colors,label=None)




colors = plt.cm.turbo(np.linspace(1,0,len(binlist)))

for i in range(1,len(binlist)):
    ir=binlist[i]
    m2=(y_b[m]<ir)&(y_b[m]>(ir-.1))
    un=normU[m2]
    zlp=-zl[m][m2]
    #plt.semilogx(zlp,un,'.',alpha=.2,ms=2,color=colors[i])
    zlbins=np.logspace(-4,1.5,15)
    zlout=[]
    unout=[]
    for j in range(1,15):
        zlout.append((zlbins[j]+zlbins[j-1])/2)
        m3=(zlp>zlbins[j-1])&(zlp<zlbins[j])&(un<10)
        if np.sum(m3)<=10:
            unout.append(float('nan'))
        else:
            unout.append(np.median(un[m3]))
    #plt.loglog(zlout,unout,'-',alpha=.75,color=colors[i])
    plt.loglog(zlout,unout,'-',alpha=1,color=colors[i],path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()],label=str(round(ir-.05,1)))
plt.legend(loc='upper left',title=r'$y_b$')
#plt.ylim(.0009,.01)
plt.xlim(10**(-3),10**(1.2))
plt.xticks([10,1,.1,.01,.001],[r'$-10^1$',r'$-10^0$',r'$-10^{-1}$',r'$-10^{-2}$',r'$-10^{-3}$'])
plt.gca().invert_xaxis()
plt.xlabel(r'$\zeta$',fontsize=14)
plt.ylabel(r'$\Phi_q$',fontsize=14)
plt.show()

# %%
fp.close()

# %%
plt.semilogx(a,np.linspace(1,2,15)**5,'.')
plt.show()

# %%
zlbins

# %%
fp=h5py.File('/home/tsw35/soteria/neon_advanced/data/QAQCv4_neon_pntEB1_frez_H_H2_H2_T__US_UU_UV_UW_VV_VW_WS_WW_ZL.h5','r')

# %%
uu=fp['UU'][:]
btij=aniso(fp['UU'][:],fp['VV'][:],fp['WW'][:],fp['UV'][:],fp['UW'][:],fp['VW'][:])
y_b=np.zeros((len(uu),))
for i in range(len(uu)):
    lam3=np.min(np.linalg.eig(btij[i,:,:])[0])
    y_b[i]=np.sqrt(3)/2*(3*lam3+1)

# %%
btij.shape

# %%
zl=fp['ZL'][:]
m=(zl<0.000001)

# %%
normU=np.sqrt(fp['UU'][m])/(fp['USTAR'][m]+.00001)
#normU=np.sqrt(fp['T_SONIC_SIGMA'][m])*(fp['USTAR'][m])/fp['H'][m]

# %%
iva_colors_HEX = ["#410d00","#831901","#983e00","#b56601","#ab8437",
              "#b29f74","#7f816b","#587571","#596c72","#454f51"]
 
#Transform the HEX colors to RGB.
from PIL import ImageColor
 
iva_colors_RGB = np.zeros((np.size(iva_colors_HEX),3),dtype='int')
 
for i in range(0,np.size(iva_colors_HEX)):
    iva_colors_RGB[i,:] = ImageColor.getcolor(iva_colors_HEX[i], "RGB")

iva_colors_RGB = iva_colors_RGB[:,:]/(256)
 
#Transform the array of colors to a list of values.
colors = iva_colors_RGB.tolist()
#----------------------------------------------------
 
#The next few lines create a new colormap using IVA's colors:
from matplotlib.colors import LinearSegmentedColormap,ListedColormap
 
inbetween_color_amount = 10
 
# the 10 is from the original 10 colors, the 4 is for R, G, B, A
newcolvals = np.zeros(shape=(10 * (inbetween_color_amount) - (inbetween_color_amount - 1), 3))
 
# add first one already
newcolvals[0] = colors[0]
 
for i, (rgba1, rgba2) in enumerate(zip(colors[:-1], np.roll(colors, -1, axis=0)[:-1])):
    for j, (p1, p2) in enumerate(zip(rgba1, rgba2)):
        flow = np.linspace(p1, p2, (inbetween_color_amount + 1))
        # discard first 1 since we already have it from previous iteration
        flow = flow[1:]
        newcolvals[ i * (inbetween_color_amount) + 1 : (i + 1) * (inbetween_color_amount) + 1, j] = flow

newcolvals
 
cmap = ListedColormap(newcolvals, name='from_list', N=None)

# %%
np.sqrt(3)/2

# %%
binlist=[.05,.15,.25,.35,.45,.55,.65,.75]
import matplotlib.patheffects as pe
plt.figure(figsize=(5*.8,4*.8),dpi=400)

ybnorm=np.array(y_b)[zl<0.000001]
m=(zl<0.000001)
ybnorm=ybnorm/(.7)
colors=cmap(ybnorm)
un=np.sqrt(fp['UU'][m])/(fp['USTAR'][m]+.00001)
plt.scatter(-zl[m],un,alpha=.15,s=.1,color=colors,label=None)




colors = plt.cm.turbo(np.linspace(1,0,len(binlist)))
#colors =["#410d00","#831901","#983e00","#b56601","#ab8437",
#             "#b29f74","#7f816b","#587571","#596c72"]

colors = ["#410d00","#831901","#b56601","#ab8437",
             "#b29f74","#7f816b","#587571","#596c72"]

for i in range(1,len(binlist)):
    ir=binlist[i]
    m2=(y_b[m]<ir)&(y_b[m]>(ir-.1))
    un=normU[m2]
    zlp=-zl[m][m2]
    #plt.semilogx(zlp,un,'.',alpha=.2,ms=2,color=colors[i])
    zlbins=np.logspace(-4,1.5,15)
    zlout=[]
    unout=[]
    for j in range(1,15):
        zlout.append((zlbins[j]+zlbins[j-1])/2)
        m3=(zlp>zlbins[j-1])&(zlp<zlbins[j])&(un<10)
        if np.sum(m3)<=5:
            unout.append(float('nan'))
        else:
            unout.append(np.mean(un[m3]))
    #plt.loglog(zlout,unout,'-',alpha=.75,color=colors[i])
    plt.semilogx(zlout,unout,'-',alpha=1,color=colors[i],path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()],label=str(round(ir-.05,1)))
plt.legend(loc='upper right',title=r'$y_b$',ncol=2,handlelength=1)
plt.ylim(0,10)
plt.xlim(10**(-3),10**(1.2))
plt.xticks([10,1,.1,.01,.001],[r'$-10^1$',r'$-10^0$',r'$-10^{-1}$',r'$-10^{-2}$',r'$-10^{-3}$'])
plt.gca().invert_xaxis()
plt.xlabel(r'$\zeta$',fontsize=14)
plt.ylabel(r'$\Phi_u$',fontsize=14)
plt.savefig('aniso_5.png')

# %%
round()

# %%
fpsites=fp['site'][:]
sites=np.unique(fpsites)

# %%
# ------------------- #
# NLCD PREP NO FIGURE #
# ------------------- #
nlcd_dom = fp['nlcd_dom'][:]
sites =[]
fpsites = fp['site'][:]
npoints =0
apoints =0
nsites=0
asites=0
nlcds = {}
site_nlcds={}
nlcd_sums={}
site_nlcd_dom={}
i=0
for point in fpsites:
    if point not in sites:
        sites.append(point)
        site_nlcds[point]=[]
        for k in fp.keys():
            if 'nlcd' in str(k):
                if k=='nlcd_dom':
                    continue
                site_nlcds[point].append(fp[k][i])
                if str(k) not in nlcds.keys():
                    nlcds[str(k)]=[]
                    nlcd_sums[str(k)]=0
                nlcds[str(k)].append(fp[k][i])
                nlcd_sums[str(k)]=nlcd_sums[str(k)]+fp[k][i]
    if 'x-' in str(point):
        npoints=npoints+1
    else:
        apoints=apoints+1
    i = i+1
for site in sites:
    site_nlcd_dom[site]=nlcd_dom[fpsites==site][0]
    if 'x-' in str(site):
        nsites=nsites+1
    else:
        asites=asites+1
for k in nlcd_sums.keys():
    print(k+": "+str(nlcd_sums[k]))
class_names={11:'Water',12:'Snow',21:'Open Dev',22:'Low Dev',23:'Med Dev',24:'High Dev',31:'Barren',\
             41:'Decid.',42:'Everg.',43:'Mix Forest',51:'Dwarf Scrub',52:'Shrub',\
             71:'Grass',72:'AK:Sedge',73:'AK:Lichen',74:'AK:Moss',81:'Pasture',82:'Crops',90:'Wetland',95:'Herb Wet',0:'NaN'}
class_colors={11:'royalblue',12:'whitesmoke',21:'pink',22:'lightcoral',23:'red',24:'maroon',31:'silver',\
              41:'limegreen',42:'darkgreen',43:'yellowgreen',51:'darkgoldenrod',52:'tan',\
              71:'wheat',72:'khaki',73:'darkkhaki',74:'darkseagreen',81:'gold',82:'sandybrown',90:'skyblue',95:'cadetblue',0:'white'}
nlcd_labels=[]
nlcd_colors=[]
nlcd_tots=np.zeros((len(nlcds['nlcd21'],)),dtype='float')
start = 0

# %%
site_nlcd_dom[b'BONA']=43
site_nlcd_dom[b'TOOL']=51
site_nlcd_dom[b'BARR']=72
site_nlcd_dom[b'HEAL']=52
site_nlcd_dom[b'DEJU']=42
site_nlcd_dom[b'LAJA']=81
site_nlcd_dom[b'GUAN']=42

# %%
zl=fp['ZL'][:]
ybs={}
ybs0={}
for site in sites:
    m = (fpsites[:]==site)
    ybs0[site]=[]
    btij=aniso(fp['UU'][m],fp['VV'][m],fp['WW'][m],fp['UV'][m],fp['UW'][m],fp['VW'][m])
    for i in range(np.sum(m)):
        lam3=np.min(np.linalg.eig(btij[i,:,:])[0])
        ybs0[site].append(np.sqrt(3)/2*(3*lam3+1))
    ybs[site]=np.mean(ybs0[site])
    

# %%

# %%
# ------------------------------------ #
# Figure 9 (Original) Slopes Bar Graph #
# ------------------------------------ #
from matplotlib.patches import Patch

dom_color=[]
ybs1=[]
nme=[]

plt.figure(figsize=(13,3.5))
ordered_sitelist=[]
flip_ybs={}
for k in ybs.keys():
    flip_ybs[ybs[k]]=k
ybs_list=list(flip_ybs.keys())
ybs_list.sort()
for k in ybs_list:
    ordered_sitelist.append(flip_ybs[k])
#plt.plot([-2,40],[1,1],'--',c='black',alpha=.5)
print(np.unique(site_nlcd_dom))
for site in ordered_sitelist:
    dom_color.append(class_colors[site_nlcd_dom[site]])
    ybs1.append(ybs[site])
    nme.append(str(site)[2:-1])

bar1 = plt.bar(nme,ybs1,color=dom_color)
#plt.ylim(.93,1.6)
plt.xticks(rotation=45)
#plt.xlim(-1,39)
i=0
#for rect in bar1:
#    height = rect.get_height()
#    plt.text(rect.get_x() + rect.get_width()/2.0, height,str(int(round(errs[i]*100))), ha='center', va='bottom',color='r')
#    i=i+1

#print(site_nlcd_dom.values())
#plt.title('Best Fit Slope by Site with Landcover Type with Error')
#plt.ylabel('Best Fit Slope')
legend_elements=[]
keep_nlcds=[41,42,43,52,71,81,82,90]
for x in keep_nlcds:
    legend_elements.append(Patch(facecolor=class_colors[x], edgecolor='k',label=class_names[x]))
#plt.legend(title='Dominant Landcover',handles=legend_elements)
plt.ylim(.15,.35)
plt.xlim(-1,45)
plt.xlabel('NEON Tower',fontsize=14)
plt.ylabel('Mean Anisotropy $y_b$',fontsize=14)
plt.show()

# %%
import matplotlib as mpl

# %%
fig=plt.figure(figsize=(15,10))
i=0
for site in ordered_sitelist:
    if i==11:
        ax=plt.subplot(4,3,i+1)
        cmap = mpl.cm.turbo_r
        norm = mpl.colors.Normalize(vmin=0, vmax=np.sqrt(3)/2)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=ax, orientation='horizontal', label='Some Units')
        i=0
        fig=plt.figure(figsize=(15,10))
    plt.subplot(4,3,i+1)
    m = (fpsites[:]==site)
    zl=fp['ZL'][:]
    ybnorm=np.array(ybs0[site])[zl[m]<0.000001]
    m=m&(zl<0.000001)
    ybnorm=ybnorm/(np.sqrt(3)/2)
    colors=plt.cm.turbo(1-ybnorm)
    un=np.sqrt(fp['UU'][m])/(fp['USTAR'][m]+.00001)
    plt.scatter(-zl[m],un,alpha=.5,s=1,color=colors)
    plt.xscale('log')
    plt.ylim(0,10)
    plt.xlim(10**(-3),10**(1.5))
    plt.title(str(site)[2:-1]+': '+str(fp['tow_height'][m][0]))
    plt.scatter([1],[3],color='k')
    i=i+1
plt.show()

# %%
fig=plt.figure(figsize=(15,10))
i=0
for site in ordered_sitelist:
    if i==12:
        i=0
        fig=plt.figure(figsize=(15,10))
    plt.subplot(4,3,i+1)
    m = (fpsites[:]==site)
    zl=fp['ZL'][:]
    ybnorm=np.array(ybs0[site])[zl[m]<0.000001]
    m=m&(zl<0.000001)
    plt.hist(ybnorm,bins=np.linspace(0,.9))
    plt.title(str(site)[2:-1]+': '+str(fp['tow_height'][m][0]))
    i=i+1
plt.show()

# %%
m = (fpsites[:]==site)
zl=fp['ZL'][:]
ybnorm=np.array(ybs0[site])[zl[m]<0.000001]
m=m&(zl<0.000001)
ybnorm=ybnorm/(np.sqrt(3)/2)
colors=plt.cm.turbo(1-ybnorm)
un=np.sqrt(fp['UU'][m])/(fp['USTAR'][m]+.00001)
plt.scatter(-zl[m],un,alpha=.5,s=1,color=colors)
plt.xscale('log')
plt.ylim(0,10)
plt.xlim(10**(-3),10**(1.5))
plt.title(str(site)[2:-1]+': '+str(fp['tow_height'][m][0]))
plt.scatter([1],[3],color='k')

# %%
fig=plt.figure(figsize=(2.5*.8,10*.8),dpi=300)
grid=ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(5, 1),
                 axes_pad=0.05,
                 cbar_mode="each",
                 cbar_location='right',
                 cbar_pad=.02,
                 cbar_size="5%")
i=0
#cmap = mpl.cm.turbo_r
norm = mpl.colors.Normalize(vmin=0, vmax=.7)
for site in [b'NOGP',b'ONAQ',b'BONA',b'MLBS',b'UKFS']:
    m = (fpsites[:]==site)
    zl=fp['ZL'][:]
    ybnorm=np.array(ybs0[site])[zl[m]<0.000001]
    m=m&(zl<0.000001)
    ybnorm=ybnorm/(.7)
    colors=cmap(ybnorm)
    un=np.sqrt(fp['UU'][m])/(fp['USTAR'][m]+.00001)
    grid[i].scatter(-zl[m],un,alpha=1,s=0.02,color=colors)
    grid[i].set_xscale('log')
    grid[i].set_ylim(0,10)
    grid[i].set_xlim(10**(-3),10**(1.5))
    grid[i].set_yticks([0,2.5,5,7.5,10],['',2.5,'',7.5,''])
    grid[i].set_xticks([10,.1,.001],[r'$-10^1$',r'$-10^{-1}$',r'$-10^{-3}$'])
    if i ==2:
        grid[i].set_ylabel(r'$\Phi_u$',fontsize=14)
    elif i ==4:
        grid[i].set_xlabel(r'$\zeta$',fontsize=14)
    #grid[i].set_title(str(site)[2:-1])
    grid[i].invert_xaxis()
    grid[i].annotate(str(site)[2:-1],[5*10**(-2),6])
    grid.cbar_axes[i].colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),)
    
    i=i+1



# %%
norm = mpl.colors.Normalize(vmin=0, vmax=.7)
grid.cbar_axes[i].colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),)

# %%
lst_std=[]
for site in sites:
    m = (fpsites[:]==site)
    lst=fp['tow_height'][m]
    lst_std.append(np.nanmean(lst[lst>=0]))
plt.scatter(list(ybs.values()),lst_std)
plt.show()

# %%

# %%
plt.hist(fp['UU'][:],bins=np.linspace(-1,10))
plt.show()

# %%
plt.hist(fp['VW'][:],bins=np.linspace(-1,10))
plt.show()

# %%
fp.keys()

# %%
##############################
###### GET BLH FOR SITES #####
##############################
import os
fig=plt.figure(figsize=(15,10))
blh={}
for site in os.listdir('/home/tsw35/soteria/data/eddy_v2/old'):
    f1=h5py.File('/home/tsw35/soteria/data/eddy_v2/old/'+site,'r')
    blh[site[0:4]]=f1['BLH'][:]
    f1.close()

# %%
i=0
fig=plt.figure(figsize=(15,10))
bhms=[]
for site in ordered_sitelist:
    if i==12:
        i=0
        fig=plt.figure(figsize=(15,10))
    plt.subplot(4,3,i+1)
    bh=blh[str(site)[2:-1]]
    bh2=[]
    for j in range(int(len(bh)/48)-1):
        bh2.append(np.nanpercentile(bh[j*48:(j+1)*48],97))
    plt.hist(bh2,bins=np.linspace(250,5000))
    bhms.append(np.mean(bh2))
    plt.title(str(site)[2:-1]+': '+str(np.mean(bh2))[0:6])
    plt.xticks([1000,2000,3000,4000,5000],[])
    i=i+1

# %%
plt.show()

# %%
plt.figure(figsize=(15,5))
plt.bar(ordered_sitelist,bhms)
plt.xticks(rotation=45)
plt.show()

# %%
fp04=h5py.File('/home/tsw35/soteria/data/NEON/dp04/TEAK/NEON.D17.TEAK.DP4.00200.001.nsae.2020-05.basic.h5','r')
fp0p=h5py.File('/home/tsw35/soteria/data/NEON/raw_data/TEAK/NEON.D17.TEAK.IP0.00200.001.ecte.2020-05-01.l0p.h5','r')

# %%
fp0p['TEAK/dp0p/data/'].keys()

# %%

# %%

# %%
fp04['TEAK/dp01/data/co2Stor/'].keys()

# %%
fp04['TEAK/dp01/data/co2Stor/000_050_30m/rtioMoleDryCo2'].dtype

# %%
import cartopy

# %%
ax= plt.subplot(1,1,1,projection=cartopy.crs.PlateCarree())
ax.background_img('ne_shaded','low')
ax.set_xlim(np.min(nlon)-3,np.max(nlon)+1)
ax.set_ylim(np.min(nlat)-1,np.max(nlat)+1)
ax.add_feature(cartopy.feature.COASTLINE)
#ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.BORDERS)
#ax.add_feature(cartopy.feature.LAKES)
ax.set_aspect('auto')
plt.scatter(nlon,nlat,c='g',s=50,alpha=.5)
#plt.scatter(fnlon,fnlat,marker='*',c='y',s=300)
ax.axis('off')
ax.outline_patch.set_edgecolor('white')
plt.show()
print('hi')


# %%
igra_lons=[]
igra_lats=[]
igra_name=[]
fp=open('data/igra.txt','r')
for line in fp.readlines():
    year=float(line[77:81])
    if year==2023:
        pass
    else:
        continue
    if 'USM' not in line[0:11]:
        continue
    ln=line[11:31]
    ln=ln.strip()
    igra_lats.append(float(ln.split(' ')[0]))
    igra_lons.append(float(ln.split(' ')[-1]))
    igra_name.append(line[0:11])

# %%
for i in range(len(igra_lons)):
    print(igra_name[i]+': '+str(igra_lons[i])+', '+str(igra_lats[i]))

# %%
plt.figure(figsize=(13,8),dpi=300)
plt.scatter(nlon,nlat,c='g',s=50,alpha=.5)
plt.scatter(-155.31724,19.55314,c='g',s=50,alpha=.5)
plt.scatter(igra_lons,igra_lats,c='k',s=5)
plt.show()

# %%
import rasterio

# %%
fp=rasterio.open('/home/tsw35/soteria/data/NEON/chm/_tifs/UKFS_chm.tif')

# %%
data=fp.read(1)

# %%
fp.xy(0,0)

# %%
xx,yy=fp.index(nlon[5],nlat[5])

# %%
xx,yy=fp.index(nlon[5],nlat[5])
data[data<0]=float('nan')
data2=data[xx-2000:xx+2000,yy-2000:yy+2000]

# %%

# %%
data2.shape

# %%
plt.imshow(data2,cmap='gist_earth')
plt.colorbar()
plt.scatter([2000],[2000],color='white')
for i, txt in enumerate(['Tower']):
    plt.annotate(txt, (2100, 2050))  
plt.show()

# %%
#####
# TESTING NLCD
#####
from matplotlib import colors

# %%
class_names={11:'Water',12:'Snow',21:'Open Dev',22:'Low Dev',23:'Med Dev',24:'High Dev',31:'Barren',\
             41:'Decid.',42:'Everg.',43:'Mix Forest',51:'Dwarf Scrub',52:'Shrub',\
             71:'Grass',72:'AK:Sedge',73:'AK:Lichen',74:'AK:Moss',81:'Pasture',82:'Crops',90:'Wetland',95:'Herb Wet'}
class_colors={11:'royalblue',12:'whitesmoke',21:'pink',22:'lightcoral',23:'red',24:'maroon',31:'silver',\
              41:'limegreen',42:'darkgreen',43:'yellowgreen',51:'darkgoldenrod',52:'tan',\
              71:'wheat',72:'khaki',73:'darkkhaki',74:'darkseagreen',81:'gold',82:'sandybrown',90:'skyblue',95:'cadetblue'}
bounds=[]
clist=[]
names=[]
for i in class_colors.keys():
    bounds.append(i)
    clist.append(class_colors[i])
    names.append(class_names[i])
bounds.append(100)
cmap = colors.ListedColormap(clist)
norm = colors.BoundaryNorm(bounds, cmap.N)

# %%
fig,ax=plt.subplots(figsize=(5,5))
ax.imshow(data,cmap=cmap,norm=norm,interpolation='none')

# %%
nlcd='/stor/soteria/hydro/private/nc153/data/NLCD/NLCD_2016_Land_Cover_Science_product_L48_20190424.img'
fpr=rasterio.open(nlcd,'r')

# %%
fpr.index(0,0)

# %%
gdal_cmd="gdaltransform -s_srs '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs' -t_srs '+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs=True' -output_xy < data/in_.txt > data/out_.txt"

# %%
from subprocess import run

# %%
run(gdal_cmd,shell=True)

# %%
data_nlcd=fpr.read(1)

# %%
towi,towj=fpr.index(69275.689795717,1779736.14652182)

# %%

# %%
cmap = colors.ListedColormap(clist)
norm = colors.BoundaryNorm(bounds, cmap.N)
fig,ax=plt.subplots(figsize=(5,5))
vw=75
hw=58
ax.imshow(data_nlcd[towi-vw:towi+vw,towj-hw:towj+hw],cmap=cmap,norm=norm,interpolation='none',extent=(-2,2,-2,2))
plt.scatter([0],[0],color='white')
plt.xticks(np.linspace(-2,2,5),[-2.0,-1.0,0,1.0,2.0])
plt.yticks(np.linspace(-2,2,5),[-2.0,-1.0,0,1.0,2.0])
plt.show()

# %%
from mpl_toolkits.axes_grid1 import ImageGrid


# %%
maxh=399


fig=plt.figure(figsize=(2.5*5,9))
subfigs = fig.subfigures(1, 5, hspace=0, wspace=0,frameon=False)
#### NOGP ####
grid=ImageGrid(subfigs[0], 111,  # similar to subplot(111)
                 nrows_ncols=(3, 1),
                 axes_pad=0.1,
                 cbar_mode="each",
                 cbar_location='right',
                 cbar_pad=.02,
                 cbar_size="10%")
towi,towj=fpr.index(-376721.09067976,2651944.55587255)
vw=75
hw=58
im=grid[0].imshow(data_nlcd[towi-vw:towi+vw,towj-hw:towj+hw],cmap=cmap,norm=norm,interpolation='none',extent=(-2,2,-2,2))
grid.cbar_axes[0].colorbar(im,ticks=[])
grid[0].scatter([0],[0],color='white')
grid[0].set_xticks(np.linspace(-2,2,5),[])
grid[0].set_yticks(np.linspace(-2,2,5),[])
grid[0].set_title('NOGP',fontsize=18)
grid[0].set_ylabel('Land Cover')



fp=rasterio.open('/home/tsw35/soteria/data/NEON/chm/_tifs/NOGP_chm.tif')
data=fp.read(1)
xx,yy=fp.index(nlon[7],nlat[7])
data[data<0]=float('nan')
data2=data[xx-2000:xx+2000,yy-2000:yy+2000]
im=grid[1].imshow(data2,cmap='YlGn',extent=(-2,2,-2,2),vmax=35)
grid[1].scatter([0],[0],color='white')
grid[1].set_xticks(np.linspace(-2,2,5),[])
grid[1].set_yticks(np.linspace(-2,2,5),[])
grid[1].set_ylabel('Canopy Height (m)')


fp=rasterio.open('/home/tsw35/soteria/data/NEON/dtm/_tifs/NOGP_dtm.tif')
data=fp.read(1)
xx,yy=fp.index(nlon[7],nlat[7])
data[data<0]=float('nan')
data2=data[xx-2000:xx+2000,yy-2000:yy+2000]
data2=data2-np.min(data2)
im2=grid[2].imshow(data2,cmap='gist_earth',extent=(-2,2,-2,2),vmax=maxh)
grid.cbar_axes[1].colorbar(im,ticks=[])
grid.cbar_axes[2].colorbar(im2,ticks=[])
grid[2].scatter([0],[0],color='white')
grid[2].set_xticks(np.linspace(-2,2,5),[])
grid[2].set_yticks(np.linspace(-2,2,5),[])
grid[2].set_ylabel('Elevation (m)')



#### ONAQ ####
grid=ImageGrid(subfigs[1], 111,  # similar to subplot(111)
                 nrows_ncols=(3, 1),
                 axes_pad=0.1,
                 cbar_mode="each",
                 cbar_location='right',
                 cbar_pad=.02,
                 cbar_size="10%")
towi,towj=fpr.index(-1381875.87445141,2026792.91133654)
vw=75
hw=58
im=grid[0].imshow(data_nlcd[towi-vw:towi+vw,towj-hw:towj+hw],cmap=cmap,norm=norm,interpolation='none',extent=(-2,2,-2,2))
grid.cbar_axes[0].colorbar(im,ticks=[])
grid[0].scatter([0],[0],color='white')
grid[0].set_xticks(np.linspace(-2,2,5),[])
grid[0].set_yticks(np.linspace(-2,2,5),[])
grid[0].set_title('ONAQ',fontsize=18)


fp=rasterio.open('/home/tsw35/soteria/data/NEON/chm/_tifs/ONAQ_chm.tif')
data=fp.read(1)
xx,yy=fp.index(nlon[15],nlat[15])
data[data<0]=float('nan')
data2=data[xx-2000:xx+2000,yy-2000:yy+2000]
im=grid[1].imshow(data2,cmap='YlGn',extent=(-2,2,-2,2),vmax=35)
grid[1].scatter([0],[0],color='white')
grid[1].set_xticks(np.linspace(-2,2,5),[])
grid[1].set_yticks(np.linspace(-2,2,5),[])


fp=rasterio.open('/home/tsw35/soteria/data/NEON/dtm/_tifs/ONAQ_dtm.tif')
data=fp.read(1)
xx,yy=fp.index(nlon[15],nlat[15])
data[data<0]=float('nan')
data2=data[xx-2000:xx+2000,yy-2000:yy+2000]
data2=data2-np.min(data2)
im2=grid[2].imshow(data2,cmap='gist_earth',extent=(-2,2,-2,2),vmax=maxh)
grid.cbar_axes[1].colorbar(im,ticks=[])
grid.cbar_axes[2].colorbar(im2,ticks=[])
grid[2].scatter([0],[0],color='white')
grid[2].set_xticks(np.linspace(-2,2,5),[])
grid[2].set_yticks(np.linspace(-2,2,5),[])





#### BONA ####
grid=ImageGrid(subfigs[2], 111,  # similar to subplot(111)
                 nrows_ncols=(3, 1),
                 axes_pad=0.1,
                 cbar_mode="each",
                 cbar_location='right',
                 cbar_pad=.02,
                 cbar_size="10%")
towi=int(67844/2-11355)
towj=int(124236/2+22450-12)
vw=75
hw=58
im=grid[0].imshow(dataAK[towi-vw:towi+vw,towj-hw:towj+hw],cmap=cmap,norm=norm,interpolation='none',extent=(-2,2,-2,2))
grid.cbar_axes[0].colorbar(im,ticks=[])
grid[0].scatter([0],[0],color='white')
grid[0].set_xticks(np.linspace(-2,2,5),[])
grid[0].set_yticks(np.linspace(-2,2,5),[])
grid[0].set_title('BONA',fontsize=18)


fp=rasterio.open('/home/tsw35/soteria/data/NEON/chm/_tifs/BONA_chm.tif')
data=fp.read(1)
xx,yy=fp.index(nlon[10],nlat[10])
data[data<0]=float('nan')
data2=data[xx-2000:xx+2000,yy-2000:yy+2000]
im=grid[1].imshow(data2,cmap='YlGn',extent=(-2,2,-2,2),vmax=35)
grid.cbar_axes[1].colorbar(im,label='Hello')
grid[1].scatter([0],[0],color='white')
grid[1].set_xticks(np.linspace(-2,2,5),[])
grid[1].set_yticks(np.linspace(-2,2,5),[])


fp=rasterio.open('/home/tsw35/soteria/data/NEON/dtm/_tifs/BONA_dtm.tif')
data=fp.read(1)
xx,yy=fp.index(nlon[10],nlat[10])
data[data<0]=float('nan')
data2=data[xx-2000:xx+2000,yy-2000:yy+2000]
data2=data2-np.nanmin(data2)
im2=grid[2].imshow(data2,cmap='gist_earth',extent=(-2,2,-2,2),vmax=maxh)
grid.cbar_axes[1].colorbar(im,ticks=[])
grid.cbar_axes[2].colorbar(im2,ticks=[])
grid[2].scatter([0],[0],color='white')
grid[2].set_xticks(np.linspace(-2,2,5),[])
grid[2].set_yticks(np.linspace(-2,2,5),[])





#### MLBS ####
grid=ImageGrid(subfigs[3], 111,  # similar to subplot(111)
                 nrows_ncols=(3, 1),
                 axes_pad=0.1,
                 cbar_mode="each",
                 cbar_location='right',
                 cbar_pad=.02,
                 cbar_size="10%")
towi,towj=fpr.index(1351405.12357651,1703423.11999617)
vw=79
hw=46
im=grid[0].imshow(data_nlcd[towi-vw:towi+vw,towj-hw:towj+hw],cmap=cmap,norm=norm,interpolation='none',extent=(-2,2,-2,2))
grid.cbar_axes[0].colorbar(im,ticks=[])
grid[0].scatter([0],[0],color='white')
grid[0].set_xticks(np.linspace(-2,2,5),[])
grid[0].set_yticks(np.linspace(-2,2,5),[])
grid[0].set_title('MLBS',fontsize=18)


fp=rasterio.open('/home/tsw35/soteria/data/NEON/chm/_tifs/MLBS_chm.tif')
data=fp.read(1)
xx,yy=fp.index(nlon[28],nlat[28])
data[data<0]=float('nan')
data2=data[xx-2000:xx+2000,yy-2000:yy+2000]
im=grid[1].imshow(data2,cmap='YlGn',extent=(-2,2,-2,2),vmax=35)
grid[1].scatter([0],[0],color='white')
grid[1].set_xticks(np.linspace(-2,2,5),[])
grid[1].set_yticks(np.linspace(-2,2,5),[])


fp=rasterio.open('/home/tsw35/soteria/data/NEON/dtm/_tifs/MLBS_dtm.tif')
data=fp.read(1)
xx,yy=fp.index(nlon[28],nlat[28])
data[data<0]=float('nan')
data2=data[xx-2000:xx+2000,yy-2000:yy+2000]
data2=data2-np.min(data2)
im2=grid[2].imshow(data2,cmap='gist_earth',extent=(-2,2,-2,2),vmax=maxh)
grid.cbar_axes[1].colorbar(im,ticks=[])
grid.cbar_axes[2].colorbar(im2,ticks=[])
grid[2].scatter([0],[0],color='white')
grid[2].set_xticks(np.linspace(-2,2,5),[])
grid[2].set_yticks(np.linspace(-2,2,5),[])






#### UKFS ####
grid=ImageGrid(subfigs[4], 111,  # similar to subplot(111)
                 nrows_ncols=(3, 1),
                 axes_pad=0.1,
                 cbar_mode="each",
                 cbar_location='right',
                 cbar_pad=.02,
                 cbar_size="10%")
towi,towj=fpr.index(69275.689795717,1779736.14652182)
vw=75
hw=58
im=grid[0].imshow(data_nlcd[towi-vw:towi+vw,towj-hw:towj+hw],cmap=cmap,norm=norm,interpolation='none',extent=(-2,2,-2,2))
grid.cbar_axes[0].colorbar(im,ticks=[])
grid[0].scatter([0],[0],color='white')
grid[0].set_xticks(np.linspace(-2,2,5),[])
grid[0].set_yticks(np.linspace(-2,2,5),[])
grid[0].set_title('UKFS',fontsize=18)


fp=rasterio.open('/home/tsw35/soteria/data/NEON/chm/_tifs/UKFS_chm.tif')
data=fp.read(1)
xx,yy=fp.index(nlon[5],nlat[5])
data[data<0]=float('nan')
data2=data[xx-2000:xx+2000,yy-2000:yy+2000]
im=grid[1].imshow(data2,cmap='YlGn',extent=(-2,2,-2,2),vmax=35)
grid[1].scatter([0],[0],color='white')
grid[1].set_xticks(np.linspace(-2,2,5),[])
grid[1].set_yticks(np.linspace(-2,2,5),[])


fp=rasterio.open('/home/tsw35/soteria/data/NEON/dtm/_tifs/UKFS_dtm.tif')
data=fp.read(1)
xx,yy=fp.index(nlon[5],nlat[5])
data[data<0]=float('nan')
data2=data[xx-2000:xx+2000,yy-2000:yy+2000]
data2=data2-np.min(data2)
im2=grid[2].imshow(data2,cmap='gist_earth',extent=(-2,2,-2,2),vmax=maxh)
grid.cbar_axes[1].colorbar(im,label='Canopy Height (m)')
grid.cbar_axes[2].colorbar(im2,label='Relative Elev. (m)')
grid[2].scatter([0],[0],color='white')
grid[2].set_xticks(np.linspace(-2,2,5),[])
grid[2].set_yticks(np.linspace(-2,2,5),[])

plt.show()

# %%
grid.cbar_axes[0].cla()

# %%
grid.cbar_axes[0].colorbar(im,shrink=0)

# %%
sites

# %%
np.where(np.array(sites)==b'ONAQ')

# %%
print([nlat[28],nlon[28]])

# %%

# %%
fpAK=rasterio.open('/home/tsw35/xSot_shared/alaska_nlcd/NLCD_2016_Land_Cover_AK_20200724.img','r')

# %%
dataAK=fpAK.read(1)

# %%
dataAK.shape

# %%
towi=int(67844/2-11355)
towj=int(124236/2+22450-12)
vw=120
hw=72
plt.imshow(dataAK[towi-vw:towi+vw,towj-hw:towj+hw],cmap=cmap,norm=norm,interpolation='none',extent=(-2,2,-2,2))
plt.scatter([0],[0],color='white')
plt.xticks(np.linspace(-2,2,5),[])
plt.yticks(np.linspace(-2,2,5),[])

# %%
fig=plt.figure(figsize=(20,1.5))
ax = fig.add_subplot(1, 1, 1)
line=[0,1]
for i in clist:
    plt.bar(line,line,color=i)
ax.set_facecolor('white')
plt.legend(names,ncol=7,handletextpad=.1,fontsize=10,framealpha=0)
plt.grid(False)
plt.ylim(0,10)

# %%
fig=plt.figure(figsize=(3.5,3.5))
ax = fig.add_subplot(1, 1, 1)
line=[0,1]
for i in clist:
    plt.bar(line,line,color=i)
ax.set_facecolor('white')
plt.legend(names,ncol=3,handletextpad=.1,fontsize=10,framealpha=1)
plt.grid(False)
plt.ylim(0,10)

# %%
fp.keys()

# %%
fp4=h5py.File('/home/tsw35/soteria/data/NEON/dp04/UKFS/NEON.D06.UKFS.DP4.00200.001.nsae.2021-09.basic.h5','r')


# %%
fp4['UKFS/dp01/data/soni/000_060_30m/veloYaxsErth']['mean']

# %%
fp4['UKFS/dp04/data/foot'].keys()

# %%
fp4['UKFS/dp04/data/foot/stat'].attrs.keys()

# %%
fp4['UKFS/dp04/data/foot/stat'][:]

# %%
height=np.array(fp4['UKFS'].attrs['DistZaxsLvlMeasTow'][:],dtype='float')

# %%
nn=len(fp4['UKFS/dp01/data/tempAirLvl/000_040_30m/temp']['mean'][:])

# %%
Tp=np.zeros((nn,len(height)))
for i in range(len(height)):
    if i==(len(height)-1):
        Tp[:,i]=fp4['UKFS/dp01/data/tempAirTop/000_0'+str(i+1)+'0_30m/temp']['mean'][:]
    else:
        Tp[:,i]=fp4['UKFS/dp01/data/tempAirLvl/000_0'+str(i+1)+'0_30m/temp']['mean'][:]

# %%
plt.figure(figsize=(9,4))
plt.plot(Tp[:,0])
plt.plot(Tp[:,1])
plt.plot(Tp[:,2])
plt.plot(Tp[:,3])
plt.plot(Tp[:,4])
plt.plot(Tp[:,5])

# %%
plt.plot(Tp[0,:],height)

# %%
from scipy import optimize

# %%
plt.hist(z0s[qf==0])

# %%
plt.figure(figsize=(8,3))
plt.plot(z0s[0:120])

# %%
z0=np.nanmedian(z0s[z0s<2.4])
z0


# %%
def interspline(z_,b,c,d):
    a=-d*np.log(z0)
    uz=a+b*z_+c*z_**2+d*np.log(z_)
    return uz


# %%
def interspline2(z_,b,c):
    a=-c*np.log(z0)
    uz=a+b*z_+c*np.log(z_)
    return uz


# %%
def interspline3(z_,a,b,c):
    uz=a+b*z_+c*np.log(z_)
    return uz


# %%

# %%
bcd,out=optimize.curve_fit(interspline,height,Tp[0,:])

# %%

# %%
Tp[0,:]

# %%
dtdz=np.zeros((nn,))
for i in range(nn):
    try:
        bcd,out=optimize.curve_fit(interspline3,height,Tp[i,:])
    except:
        continue
    zs=np.linspace(height[-1]-.5,height[-1]+.5,11)
    
    Us=interspline3(zs,bcd[0],bcd[1],bcd[2])
    dudz[i]=np.gradient(Us,zs)[5]

# %%
bcd

# %%

# %%
plt.plot(Us,zs)
plt.plot(Tp[50,:],height,'k-o')

# %%
np.gradient(Tp[0,:],height)

# %%
np.log(1)

# %%
plt.plot(dudz[0:48])

# %%
zs.extend(5)

# %%
dudz=fp2['DUDZ'][:]
dtdz=fp2['DTDZ'][:]

# %%
plt.plot(dtdz[dtdz>-9999])

# %%
plt.plot(dudz[dudz>-9999])

# %%
fp2.close()

# %%
fp2

# %%
testu=fp2['vertical_wind/WIND_4.81'][:]

# %%
plt.plot(testu[testu>-9999])

# %%
fp2.close()
fp4.close()
fp.close()

# %%
fptest=h5py.File('/home/tsw35/soteria/data/eddy_v2/old/YELL_L2.h5','r')

# %%
dtdz=fptest['DTDZ'][:]

# %%
plt.hist(dtdz[dtdz>-8888])

# %%
#fp=h5py.File('/home/tsw35/soteria/neon_advanced/data/QAQCv4_2_neon_pntEB0.25_frez_DT_DU_H2_T__US_UU_UV_UW_VV_VW_WW_ZL.h5','r')# Sites
fp=h5py.File('/home/tsw35/soteria/neon_advanced/data/QAQCv4_2_neon_dayEB1_frez_DT_DU_H2_T__US_UU_UV_UW_VV_VW_WW_ZL.h5','r')# Sites
focus_sites=[b'WREF',b'NOGP',b'BART',b'SOAP','MAP',b'ORNL',b'SRER',b'KONZ',b'DSNY']
#focus_sites=[b'ABBY',b'SCBI',b'HARV',b'JERC','MAP',b'ORNL',b'TREE',b'OSBS',b'UNDE']
fpsites=fp['site'][:]
lats=fp['lat'][:]
lons=fp['lon'][:]
sites = []
nlat=[]
nlon=[]
fnlat=[]
fnlon=[]
for i in range(len(fpsites)):
    x=fpsites[i]
    if x not in sites:
        nlat.append(lats[i])
        nlon.append(lons[i])
        sites.append(x)
        if x in focus_sites:
            fnlat.append(lats[i])
            fnlon.append(lons[i])

# %%
phim=.4*fp['tow_height'][:]/fp['USTAR'][:]*fp['DUDZ'][:]
rho=fp['RHO'][:]
H=fp['H'][:]/1005/rho
TSTAR=-H/fp['USTAR'][:]
phih=.4*fp['tow_height'][:]/TSTAR*fp['DTDZ'][:]

# %%
np.mean(rho)

# %%
uu=fp['UU'][:]
btij=aniso(fp['UU'][:],fp['VV'][:],fp['WW'][:],fp['UV'][:],fp['UW'][:],fp['VW'][:])
y_b=np.zeros((len(uu),))
for i in range(len(uu)):
    lam3=np.min(np.linalg.eig(btij[i,:,:])[0])
    y_b[i]=np.sqrt(3)/2*(3*lam3+1)

# %%
zl=fp['ZL'][:]
m=(zl<0.000001)

# %%
binlist=[.05,.15,.25,.35,.45,.55,.65,.75]
import matplotlib.patheffects as pe
plt.figure(figsize=(5*.6,3.5*.6),dpi=400)

m=(zl<0.000001)&(phih>=0)
ybnorm=np.array(y_b)[m]
ybnorm=ybnorm/(.7)
colors=cmap(ybnorm)
un=phih[m]
plt.scatter(-zl[m],un,alpha=.15,s=.1,color=colors,label=None)
plt.xscale('log')
plt.yscale('log')



colors = plt.cm.turbo(np.linspace(1,0,len(binlist)))
#colors =["#410d00","#831901","#983e00","#b56601","#ab8437",
#             "#b29f74","#7f816b","#587571","#596c72"]

colors = ["#410d00","#831901","#b56601","#ab8437",
             "#b29f74","#7f816b","#587571","#596c72"]
for i in range(1,len(binlist)):
    ir=binlist[i]
    m2=(y_b[m]<ir)&(y_b[m]>(ir-.1))
    un=phih[m][m2]
    zlp=-zl[m][m2]
    #plt.semilogx(zlp,un,'.',alpha=.2,ms=2,color=colors[i])
    zlbins=np.logspace(-4,1.5,15)
    zlout=[]
    unout=[]
    for j in range(1,15):
        zlout.append((zlbins[j]+zlbins[j-1])/2)
        m3=(zlp>zlbins[j-1])&(zlp<zlbins[j])
        if np.sum(m3)<=5:
            unout.append(float('nan'))
        else:
            unout.append(np.median(un[m3]))
    print(binlist[i])
    print(unout)
    print()
    #plt.loglog(zlout,unout,'-',alpha=.75,color=colors[i])
    plt.plot(zlout,unout,'-',alpha=1,color=colors[i],path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()],label=str(round(ir-.05,1)))
    plt.xscale('log')
    plt.yscale('log')
    
#plt.legend(loc='upper right',title=r'$y_b$',ncol=2,handlelength=1)
plt.ylim(10**(-2),5*10**1)
plt.xlim(3*10**(-4),10**(1.2))
#plt.xticks([10,1,.1,.01,.001],[r'$-10^1$',r'$-10^0$',r'$-10^{-1}$',r'$-10^{-2}$',r'$-10^{-3}$'])
plt.gca().invert_xaxis()
plt.xlabel(r'-$\zeta$',fontsize=14)
plt.ylabel(r'$\Phi_h$',fontsize=14)


# %%
binlist=[.05,.15,.25,.35,.45,.55,.65,.75]
import matplotlib.patheffects as pe
plt.figure(figsize=(5*.6,3.5*.6),dpi=400)

m=(zl<0.000001)&(phim>=0)
ybnorm=np.array(y_b)[m]
ybnorm=ybnorm/(.7)
colors=cmap(ybnorm)
un=phim[m]
plt.scatter(-zl[m],un,alpha=.15,s=.1,color=colors,label=None)
plt.xscale('log')
plt.yscale('log')



colors = plt.cm.turbo(np.linspace(1,0,len(binlist)))
#colors =["#410d00","#831901","#983e00","#b56601","#ab8437",
#             "#b29f74","#7f816b","#587571","#596c72"]

colors = ["#410d00","#831901","#b56601","#ab8437",
             "#b29f74","#7f816b","#587571","#596c72"]
for i in range(1,len(binlist)):
    ir=binlist[i]
    m2=(y_b[m]<ir)&(y_b[m]>(ir-.1))
    un=phim[m][m2]
    zlp=-zl[m][m2]
    #plt.semilogx(zlp,un,'.',alpha=.2,ms=2,color=colors[i])
    zlbins=np.logspace(-4,1.5,15)
    zlout=[]
    unout=[]
    for j in range(1,15):
        zlout.append((zlbins[j]+zlbins[j-1])/2)
        m3=(zlp>zlbins[j-1])&(zlp<zlbins[j])
        if np.sum(m3)<=5:
            unout.append(float('nan'))
        else:
            unout.append(np.median(un[m3]))
    print(binlist[i])
    print(unout)
    print()
    #plt.loglog(zlout,unout,'-',alpha=.75,color=colors[i])
    plt.plot(zlout,unout,'-',alpha=1,color=colors[i],path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()],label=str(round(ir-.05,1)))
    plt.xscale('log')
    plt.yscale('log')
    
#plt.legend(loc='upper right',title=r'$y_b$',ncol=2,handlelength=1)
plt.ylim(10**(-1),5*10**1)
plt.xlim(3*10**(-4),10**(1.2))
#plt.xticks([10,1,.1,.01,.001],[r'$-10^1$',r'$-10^0$',r'$-10^{-1}$',r'$-10^{-2}$',r'$-10^{-3}$'])
plt.gca().invert_xaxis()
plt.xlabel(r'-$\zeta$',fontsize=14)
plt.ylabel(r'$\Phi_m$',fontsize=14)

# %%
binlist=[.05,.15,.25,.35,.45,.55,.65,.75]
import matplotlib.patheffects as pe
plt.figure(figsize=(5*.6,3.5*.6),dpi=400)

m=(zl<0.000001)&(phih>=0)
ybnorm=np.array(y_b)[m]
ybnorm=ybnorm/(.7)
colors=cmap(ybnorm)
un=phih[m]
plt.scatter(-zl[m],un,alpha=.15,s=.1,color=colors,label=None)
plt.xscale('log')
plt.yscale('log')

zlfake=-np.logspace(-4,1.5,50)
MOSTphih=.96*(1-11.6*zlfake)**(-1/2)
plt.plot(-zlfake,MOSTphih,'k')
    
#plt.legend(loc='upper right',title=r'$y_b$',ncol=2,handlelength=1)
plt.ylim(10**(-2),10**3)
plt.xlim(3*10**(-4),10**(1.2))
#plt.xticks([10,1,.1,.01,.001],[r'$-10^1$',r'$-10^0$',r'$-10^{-1}$',r'$-10^{-2}$',r'$-10^{-3}$'])
plt.gca().invert_xaxis()
plt.xlabel(r'-$\zeta$',fontsize=14)
plt.ylabel(r'$\Phi_h$',fontsize=14)

# %%
binlist=[.05,.15,.25,.35,.45,.55,.65,.75]
import matplotlib.patheffects as pe
plt.figure(figsize=(5*.6,3.5*.6),dpi=400)

m=(zl<0.000001)&(phim>=0)
ybnorm=np.array(y_b)[m]
ybnorm=ybnorm/(.7)
colors=cmap(ybnorm)
un=phim[m]
plt.scatter(-zl[m],un,alpha=.15,s=.1,color=colors,label=None)
plt.xscale('log')
plt.yscale('log')

zlfake=-np.logspace(-4,1.5,50)
MOSTphim=(1-19*zlfake)**(-1/4)
plt.plot(-zlfake,MOSTphim,'k')

#plt.legend(loc='upper right',title=r'$y_b$',ncol=2,handlelength=1)
plt.ylim(10**(-1),5*10**1)
plt.xlim(3*10**(-4),10**(1.2))
#plt.xticks([10,1,.1,.01,.001],[r'$-10^1$',r'$-10^0$',r'$-10^{-1}$',r'$-10^{-2}$',r'$-10^{-3}$'])
plt.gca().invert_xaxis()
plt.xlabel(r'-$\zeta$',fontsize=14)
plt.ylabel(r'$\Phi_m$',fontsize=14)

# %%
plt.hist(phih,bins=np.linspace(-.1,.1,50))

# %%
dudz=fp['DUDZ'][:]
dtdz=fp['DTDZ'][:]
plt.hist(dudz[dudz!=0],bins=np.linspace(-.2,.2))

# %%
plt.hist(phih,bins=np.linspace(-100,50))

# %%
dudz=fp['DUDZ'][:]
plt.hist(dudz[np.abs(dudz)>0])

# %%
for site in sites:
    print(str(site)+': '+str(np.mean(fp['DUDZ'][fpsites==site])))

# %%
for site in sites:
    print(str(site)+': '+str(np.sum(fp['DTDZ'][fpsites==site]==0)/np.sum(fpsites==site)*100)[0:6]+'   '+str(np.sum(fpsites==site)))

# %%
fpa=h5py.File('/home/tsw35/soteria/data/eddy_v2/lst/ABBY_L2.h5','r')

# %%
dudz=fpa['DUDZ'][:]
dtdz=fpa['DTDZ'][:]

# %%
print(len(dudz))
print(len(dudz[dudz>-9999]))
print(len(dudz[(dudz>-9999)&(np.abs(dudz)>0)]))

# %%
fpa.close()

# %%
m7=(dudz>-9999)&(np.abs(dudz)>0)
dudz[m7]

# %%
fpa['vertical_wind'].keys()

# %%
wind=np.zeros((5,96336))

# %%
wind[0,:]=fpa['vertical_wind/WIND_0.59'][:]
wind[1,:]=fpa['vertical_wind/WIND_4.81'][:]
wind[2,:]=fpa['vertical_wind/WIND_9.78'][:]
wind[3,:]=fpa['vertical_wind/WIND_13.07'][:]
wind[4,:]=fpa['WS'][:]

# %%
m8=(wind[0,:]>-9999)&(wind[1,:]>-9999)&(wind[2,:]>-9999)&(wind[3,:]>-9999)&(wind[4,:]>-9999)

# %%

# %%
for i in range(5):
    plt.plot(wind[i,m8][10000:10100])
plt.ylim(0,5)
plt.legend(['0.5','5','10','13','18.5'])

# %%
fpa.attrs['tow_height']

# %%
prof1=np.mean(wind[:,m8][:,10000:10100],axis=1)

# %%
hts=[.59,4.81,9.78,13.07,18.59]

# %%
plt.scatter(prof1,hts)
plt.plot(outs,hts[-3:])

# %%
z0=.05
zd=float(fpa.attrs['zd'])


# %%

# %%
def intT(z_,a,b,c):
    return a+b*(z_-zd)+c*np.log(z_-zd)
def intU(z_,b,c):
    a=-c*np.log(z0)
    return a+b*(z_-zd)+c*np.log(z_-zd)


# %%
hts=[.59,4.81,9.78,13.07,18.59]
hts=np.array(hts)

# %%
bcd,out=optimize.curve_fit(intU,hts[1:],prof1[1:])
zs=np.linspace(zd+.1,hts[-1]+.5,50)#np.linspace(hts[-1]-.5,hts[-1]+.5,11)
outs=intU(zs,bcd[0],bcd[1])
DUDZ=np.gradient(outs,zs)


# %%
plt.scatter(prof1,hts)
plt.plot(outs,zs)

# %%

# %%
hts=[.59,4.81,9.78,13.07,18.59]
bcd,out=optimize.curve_fit(intT,hts[1:],prof1[1:])
zs=np.linspace(zd+.1,hts[-1]+.5,50) #np.linspace(hts[-1]-.5,hts[-1]+.5,11)
outs=intT(zs,bcd[0],bcd[1],bcd[2])
DUDZ=np.gradient(outs,zs)

# %%
plt.figure(figsize=(3,4))
plt.scatter(prof1,hts,c='k')
plt.plot(outs,zs,'b--')
plt.xlabel('Velocity (m/s)')
plt.ylabel('Height (m)')


# %%
zs[-3]

# %%
DUDZ[-13]

# %%
swin=wind[:,m8][:,10025:10125]
DUDZ=np.zeros((48,50))
outsU=np.zeros((48,50))
for j in range(48):
    hts=[.59,4.81,9.78,13.07,18.59]
    bcd,out=optimize.curve_fit(intT,hts[1:],swin[1:,j])
    zs=np.linspace(zd+.1,hts[-1]+.5,50) #np.linspace(hts[-1]-.5,hts[-1]+.5,11)
    outs=intT(zs,bcd[0],bcd[1],bcd[2])
    outsU[j,:]=outs
    DUDZ[j,:]=np.gradient(outs,zs)

# %%
plt.plot(DUDZ[:,-3])
plt.plot((swin[-1,0:48]-swin[-2,0:48])/10)

# %%
dtdz2=dtdz[dtdz>-999]
plt.plot(dtdz2[10000:10200])

# %%
np.argmax((np.array(hts)-zd)>0)

# %%
fpa.close()

# %%
fp.close()

# %%
300*(101325/101000)**(2/7)

# %%
fp4=h5py.File('/home/tsw35/soteria/data/NEON/dp04/UKFS/NEON.D06.UKFS.DP4.00200.001.nsae.2021-09.basic.h5','r')

# %%
fp4.close()

# %%
fp.close()

# %%
phim=.4*fp['tow_height'][:]/fp['USTAR'][:]*fp['DUDZ'][:]
rho=fp['RHO'][:]
H=fp['H'][:]/1005/rho
TSTAR=-H/fp['USTAR'][:]
phih=.4*fp['tow_height'][:]/TSTAR*fp['DTDZ'][:]

# %%

# %%
m=(zl<0)

MOSTphih=.96*(1-11.6*zl[m])**(-1/2)
MOSTphim=(1-19*zl[m])**(-1/4)

a=.48+1.8*y_b[m]
#a=.48+1.5*y_b[m]
STPphih=a*((3-2.5*zl[m])/(1-10*zl[m]+50*zl[m]**2))**(1/3)

a=(.24-.38*y_b[m])
a[y_b[m]>.6]=.012
b=.061
c=.45-.53*y_b[m]
n = -0.12 + 6.4*y_b[m]
STPphim=(a+b*(-zl[m])**(n))/(a+(-zl[m])**n)+c*(-zl[m])**(1/3)

# %%
from sklearn import metrics
#sklearn.metrics.r2_score

# %%
plt.figure(figsize=(4,4))
plt.hexbin(np.log(STPphih),np.log(phih[m]),cmap='terrain',mincnt=1,vmax=25,extent=(-2,1,-2,1))
plt.plot([-2,1],[-2,1],'k--')
#plt.xticks([0,.25,.5,.75,1,1.25,1.5])
#plt.yticks([0,.25,.5,.75,1,1.25,1.5])
plt.xticks([-2,-1,0,1])
plt.yticks([-2,-1,0,1])
plt.xlabel(r'Stiperski $\Phi_h$',fontsize=18)
plt.ylabel(r'Observed $\Phi_h$',fontsize=18)
print(metrics.r2_score(phih[m],STPphih))

# %%
plt.figure(figsize=(4,4))
plt.hexbin(np.log(MOSTphih),np.log(phih[m]),vmax=25,cmap='terrain',mincnt=1,extent=(-2,1,-2,1))
plt.plot([-2,1],[-2,1],'k--')
#plt.xticks([0,.25,.5,.75,1,1.25,1.5])
#plt.yticks([0,.25,.5,.75,1,1.25,1.5])
plt.xticks([-2,-1,0,1])
plt.yticks([-2,-1,0,1])

plt.xlabel(r'MOST $\Phi_h$',fontsize=18)
plt.ylabel(r'Observed $\Phi_h$',fontsize=18)
print(metrics.r2_score(phih[m],MOSTphih))

# %%
plt.figure(figsize=(4,4))
plt.hexbin(STPphim,phim[m],cmap='terrain',mincnt=1,vmax=25,extent=(.01,1.5,.01,1.5))
plt.plot([0,1.5],[0,1.5],'k--')
plt.xticks([0,.25,.5,.75,1,1.25,1.5])
plt.yticks([0,.25,.5,.75,1,1.25,1.5])
plt.xlabel(r'Stiperski $\Phi_m$',fontsize=18)
plt.ylabel(r'Observed $\Phi_m$',fontsize=18)
print(metrics.r2_score(phim[m],STPphim))

# %%
plt.figure(figsize=(4,4))
plt.hexbin(MOSTphim,phim[m],vmax=25,cmap='terrain',mincnt=1,extent=(.01,1.5,.01,1.5))
plt.plot([0,1.5],[0,1.5],'k--')
plt.xticks([0,.25,.5,.75,1,1.25,1.5])
plt.yticks([0,.25,.5,.75,1,1.25,1.5])
plt.xlabel(r'MOST $\Phi_h$',fontsize=18)
plt.ylabel(r'Observed $\Phi_h$',fontsize=18)
print(metrics.r2_score(phim[m],MOSTphim))


# %%
def stpphih(zl_,yb):
    a=.48+1.8*yb
    return a*((3-2.5*zl_)/(1-10*zl_+50*zl_**2))**(1/3)


# %%
def stpphim(zl_,yb):
    a=(.24-.38*yb)
    if yb>.6:
        a=.012
    b=.061
    c=.45-.53*yb
    n = -0.12 + 6.4*yb
    return (a+b*(-zl_)**(n))/(a+(-zl_)**n)+c*(-zl_)**(1/3)



# %%
binlist=[.05,.15,.25,.35,.45,.55,.65,.75]
import matplotlib.patheffects as pe
plt.figure(figsize=(5*.6,3.5*.6),dpi=400)

m=(zl<0.000001)&(phih>=0)
ybnorm=np.array(y_b)[m]
ybnorm=ybnorm/(.7)
colors=cmap(ybnorm)
un=phih[m]
plt.scatter(-zl[m],un,alpha=.15,s=.1,color=colors,label=None)
plt.xscale('log')
plt.yscale('log')


colors = ["#410d00","#831901","#b56601","#ab8437",
             "#b29f74","#7f816b","#587571","#596c72"]
binlist=[.05,.15,.25,.35,.45,.55,.65,.75]

zlfake=-np.logspace(-4,1.5,50)
for i in range(1,8):
    STPphih=stpphih(zlfake,(binlist[i]+binlist[i-1])/2)
    plt.plot(-zlfake,STPphih,color=colors[i],path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])

MOSTphih=.96*(1-11.6*zlfake)**(-1/2)
plt.plot(-zlfake,MOSTphih,'k')
#plt.legend(loc='upper right',title=r'$y_b$',ncol=2,handlelength=1)
plt.ylim(5*10**(-2),10**1)
plt.xlim(3*10**(-4),10**(1.2))
#plt.legend([r'$y_b=.1$',r'$y_b=.2$',r'$y_b=.3$',r'$y_b=.4$',r'$y_b=.5$',r'$y_b=.6$',r'$y_b=.7$','MOST'])
#plt.xticks([10,1,.1,.01,.001],[r'$-10^1$',r'$-10^0$',r'$-10^{-1}$',r'$-10^{-2}$',r'$-10^{-3}$'])
plt.gca().invert_xaxis()
plt.xlabel(r'-$\zeta$',fontsize=14)
plt.ylabel(r'$\Phi_h$',fontsize=14)

# %%

# %%
binlist=[.05,.15,.25,.35,.45,.55,.65,.75]
import matplotlib.patheffects as pe
plt.figure(figsize=(5*.6,3.5*.6),dpi=400)

m=(zl<0.000001)&(phim>=0)
ybnorm=np.array(y_b)[m]
ybnorm=ybnorm/(.7)
colors=cmap(ybnorm)
un=phim[m]
plt.scatter(-zl[m],un,alpha=.15,s=.1,color=colors,label=None)
plt.xscale('log')
plt.yscale('log')

zlfake=-np.logspace(-4,1.5,50)

colors = ["#410d00","#831901","#b56601","#ab8437",
             "#b29f74","#7f816b","#587571","#596c72"]
binlist=[.05,.15,.25,.35,.45,.55,.65,.75]

zlfake=-np.logspace(-4,1.5,50)
for i in range(1,8):
    STPphim=stpphim(zlfake,(binlist[i]+binlist[i-1])/2)
    plt.plot(-zlfake,STPphim,color=colors[i],path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])

MOSTphim=(1-19*zlfake)**(-1/4)
plt.plot(-zlfake,MOSTphim,'k')

#plt.legend(loc='upper right',title=r'$y_b$',ncol=2,handlelength=1)
plt.ylim(10**(-1),5*10**1)
plt.xlim(3*10**(-4),10**(1.2))
#plt.xticks([10,1,.1,.01,.001],[r'$-10^1$',r'$-10^0$',r'$-10^{-1}$',r'$-10^{-2}$',r'$-10^{-3}$'])
plt.gca().invert_xaxis()
plt.xlabel(r'-$\zeta$',fontsize=14)
plt.ylabel(r'$\Phi_m$',fontsize=14)

# %%
yb=.3
a=(.24-.38*yb)
b=.061
c=.45-.53*yb
n = -0.12 + 6.4*yb
#(a+b*(-zl_)**(n))/(a+(-zl_)**n)-c*(-zl_)**(1/3)
plt.plot(-zlfake,-c*(-zlfake)**(1/3))
plt.plot(-zlfake,(a+b*(-zlfake)**(n))/(a+(-zlfake)**n))
plt.plot(-zlfake,(a+b*(-zlfake)**(n))/(a+(-zlfake)**n)+c*(-zlfake)**(1/3))

plt.xscale('log')

# %%
plt.scatter(-zl[m],STPphim,s=.1)
plt.xscale('log')
plt.yscale('log')
plt.gca().invert_xaxis()


# %%
SS_h=1-np.median(np.abs(np.log(phih[m])-np.log(STPphih)))/np.median(np.abs(np.log(phih[m])-np.log(MOSTphih)))
SS_m=1-np.nanmedian(np.abs(np.log(phim[m])-np.log(STPphim)))/np.nanmedian(np.abs(np.log(phim[m])-np.log(MOSTphim)))



# %%
SS_h

# %%
SS_m

# %%
m2=zl[m]>-.01
SS_h=1-np.median(np.abs(np.log(phih[m][m2])-np.log(STPphih[m2])))/np.median(np.abs(np.log(phih[m][m2])-np.log(MOSTphih[m2])))
SS_m=1-np.nanmedian(np.abs(np.log(phim[m][m2])-np.log(STPphim[m2])))/np.nanmedian(np.abs(np.log(phim[m][m2])-np.log(MOSTphim[m2])))

# %%
print(SS_h)
print(SS_m)

# %%
zl=fp['ZL'][:]
SSh={}
SSm={}

for site in sites:
    m = (fpsites[:]==site)&(zl<0)
    MOSTphih=.96*(1-11.6*zl[m])**(-1/2)
    MOSTphim=(1-19*zl[m])**(-1/4)

    a=.48+1.8*y_b[m]
    #a=.48+1.5*y_b[m]
    STPphih=a*((3-2.5*zl[m])/(1-10*zl[m]+50*zl[m]**2))**(1/3)

    a=(.24-.38*y_b[m])
    a[y_b[m]>.6]=.012
    b=.061
    c=.45-.53*y_b[m]
    n = -0.12 + 6.4*y_b[m]
    STPphim=(a+b*(-zl[m])**(n))/(a+(-zl[m])**n)+c*(-zl[m])**(1/3)

    SSh[site]=1-np.median(np.abs(np.log(phih[m])-np.log(STPphih)))/np.median(np.abs(np.log(phih[m])-np.log(MOSTphih)))
    SSm[site]=1-np.nanmedian(np.abs(np.log(phim[m])-np.log(STPphim)))/np.nanmedian(np.abs(np.log(phim[m])-np.log(MOSTphim)))


# %%

# ------------------------------------ #
# Figure 9 (Original) Slopes Bar Graph #
# ------------------------------------ #
from matplotlib.patches import Patch

dom_color=[]
ybs1=[]
nme=[]

plt.figure(figsize=(13,3.5))
ordered_sitelist=[]
flip_ybs={}
for k in SSh.keys():
    flip_ybs[SSh[k]]=k
ybs_list=list(flip_ybs.keys())
ybs_list.sort()
for k in ybs_list:
    ordered_sitelist.append(flip_ybs[k])
#plt.plot([-2,40],[1,1],'--',c='black',alpha=.5)
print(np.unique(site_nlcd_dom))
for site in ordered_sitelist:
    dom_color.append(class_colors[site_nlcd_dom[site]])
    ybs1.append(SSh[site])
    nme.append(str(site)[2:-1])

bar1 = plt.bar(nme,ybs1,color=dom_color)
#plt.ylim(.93,1.6)
plt.xticks(rotation=45)
#plt.xlim(-1,39)
i=0
#for rect in bar1:
#    height = rect.get_height()
#    plt.text(rect.get_x() + rect.get_width()/2.0, height,str(int(round(errs[i]*100))), ha='center', va='bottom',color='r')
#    i=i+1

#print(site_nlcd_dom.values())
#plt.title('Best Fit Slope by Site with Landcover Type with Error')
#plt.ylabel('Best Fit Slope')
legend_elements=[]
keep_nlcds=[41,42,43,52,71,81,82,90]
for x in keep_nlcds:
    legend_elements.append(Patch(facecolor=class_colors[x], edgecolor='k',label=class_names[x]))
plt.legend(title='Dominant Landcover',handles=legend_elements)
#plt.ylim(.15,.35)
plt.xlim(-1,40)
plt.xlabel('NEON Tower',fontsize=14)
plt.ylabel('Skill Score',fontsize=14)
plt.title(r'Skill Score for $\phi_h$',fontsize=20)
plt.show()

# %%

# ------------------------------------ #
# Figure 9 (Original) Slopes Bar Graph #
# ------------------------------------ #
from matplotlib.patches import Patch

dom_color=[]
ybs1=[]
nme=[]

plt.figure(figsize=(13,3.5))
ordered_sitelist=[]
flip_ybs={}
for k in SSm.keys():
    flip_ybs[SSm[k]]=k
ybs_list=list(flip_ybs.keys())
ybs_list.sort()
for k in ybs_list:
    ordered_sitelist.append(flip_ybs[k])
#plt.plot([-2,40],[1,1],'--',c='black',alpha=.5)
print(np.unique(site_nlcd_dom))
for site in ordered_sitelist:
    dom_color.append(class_colors[site_nlcd_dom[site]])
    ybs1.append(SSm[site])
    nme.append(str(site)[2:-1])

bar1 = plt.bar(nme,ybs1,color=dom_color)
#plt.ylim(.93,1.6)
plt.xticks(rotation=45)
#plt.xlim(-1,39)
i=0
#for rect in bar1:
#    height = rect.get_height()
#    plt.text(rect.get_x() + rect.get_width()/2.0, height,str(int(round(errs[i]*100))), ha='center', va='bottom',color='r')
#    i=i+1

#print(site_nlcd_dom.values())
#plt.title('Best Fit Slope by Site with Landcover Type with Error')
#plt.ylabel('Best Fit Slope')
legend_elements=[]
keep_nlcds=[41,42,43,52,71,81,82,90]
for x in keep_nlcds:
    legend_elements.append(Patch(facecolor=class_colors[x], edgecolor='k',label=class_names[x]))
#plt.legend(title='Dominant Landcover',handles=legend_elements)
#plt.ylim(.15,.35)
plt.xlim(-1,40)
plt.xlabel('NEON Tower',fontsize=14)
plt.ylabel('Skill Score',fontsize=14)
plt.title(r'Skill Score for $\phi_m$',fontsize=20)
plt.show()

# %%
site=b'JORN'

# %%
m = (fpsites[:]==site)&(zl<0)
MOSTphih=.96*(1-11.6*zl[m])**(-1/2)
MOSTphim=(1-19*zl[m])**(-1/4)

a=.48+1.8*y_b[m]
#a=.48+1.5*y_b[m]
STPphih=a*((3-2.5*zl[m])/(1-10*zl[m]+50*zl[m]**2))**(1/3)

a=(.24-.38*y_b[m])
a[y_b[m]>.6]=.012
b=.061
c=.45-.53*y_b[m]
n = -0.12 + 6.4*y_b[m]
STPphim=(a+b*(-zl[m])**(n))/(a+(-zl[m])**n)+c*(-zl[m])**(1/3)

# %%
plt.scatter(-zl[m],phih[m],s=.2)
plt.xscale('log')
plt.yscale('log')
zlfake=-np.logspace(-4,1.5,50)
for i in range(1,8):
    STPphih=stpphih(zlfake,(binlist[i]+binlist[i-1])/2)
    plt.plot(-zlfake,STPphih,color=colors[i],path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()])

MOSTphih=.96*(1-11.6*zlfake)**(-1/2)
plt.plot(-zlfake,MOSTphih,'k')
plt.gca().invert_xaxis()

# %%
plt.figure(figsize=(3,3),dpi=400)
norm = mpl.colors.Normalize(vmin=0, vmax=.7)
plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),label=r'$y_b$')

# %%
fp=h5py.File('/home/tsw35/soteria/neon_advanced/qaqc_data/NEON_TW_U_UVWT.h5','r')

# %%
fp.keys()

# %%
import matplotlib.patheffects as pe
import matplotlib as mpl
import matplotlib


# %%
vmx_a=.7
vmn_a=.1
indir='/home/tsw35/soteria/neon_advanced/qaqc_data/'

# %%
iva_colors_HEX = ["#410d00","#831901","#983e00","#b56601","#ab8437",
              "#b29f74","#7f816b","#587571","#596c72","#454f51"]
#Transform the HEX colors to RGB.
from PIL import ImageColor
iva_colors_RGB = np.zeros((np.size(iva_colors_HEX),3),dtype='int')
for i in range(0,np.size(iva_colors_HEX)):
    iva_colors_RGB[i,:] = ImageColor.getcolor(iva_colors_HEX[i], "RGB")
iva_colors_RGB = iva_colors_RGB[:,:]/(256)
colors = iva_colors_RGB.tolist()
#----------------------------------------------------
from matplotlib.colors import LinearSegmentedColormap,ListedColormap
inbetween_color_amount = 10
newcolvals = np.zeros(shape=(10 * (inbetween_color_amount) - (inbetween_color_amount - 1), 3))
newcolvals[0] = colors[0]
for i, (rgba1, rgba2) in enumerate(zip(colors[:-1], np.roll(colors, -1, axis=0)[:-1])):
    for j, (p1, p2) in enumerate(zip(rgba1, rgba2)):
        flow = np.linspace(p1, p2, (inbetween_color_amount + 1))
        # discard first 1 since we already have it from previous iteration
        flow = flow[1:]
        newcolvals[ i * (inbetween_color_amount) + 1 : (i + 1) * (inbetween_color_amount) + 1, j] = flow
newcolvals
cmap_ani = ListedColormap(newcolvals, name='from_list', N=None)


# %%
def cani_norm(x):
    try:
        x_=x.copy()
        x_[x_<vmn_a]=vmn_a
        x_[x_>vmx_a]=vmx_a
    except:
        x_=x.copy()
        if x_>vmx_a:
            x_=vmx_a
        elif x_<vmn_a:
            x_=vmn_a
    
    x_=(x_-vmn_a)/(vmx_a-vmn_a)
    return cmap_ani(x_)


# %%
zL=(fp['tow_height'][:]-fp['zd'][:])/fp['L_MOST'][:]
ani=fp['ANI_YB'][:]

# %%

# %%
m=np.zeros((len(zL),))
m[0:10000]=1
np.random.shuffle(m)
m=m.astype(bool)

# %%
plt.scatter(-zL[m],pvw[m],s=1,color=cani_norm(ani[m]),alpha=.5)
plt.xlim(.001,10)
plt.ylim(-.5,2)
plt.xscale('log')


# %%
def binplot1d(xx,yy,ani,xbins,anibins=np.array([-.05,.05,.15,.25,.35,.45,.55,.65,.75,.85]),mincnt=10):
    xplot=(xbins[0:-1]+xbins[1:])/2
    yplot=np.zeros((len(anibins)-1,len(xplot)))
    aniplot=(anibins[0:-1]+anibins[1:])/2
    for i in range(len(anibins)-1):
        for j in range(len(xbins)-1):
            pnts=yy[(ani>=anibins[i])&(ani<anibins[i+1])&(xx>=xbins[j])&(xx<xbins[j+1])]
            if len(pnts)<mincnt:
                yplot[i,j]=float('nan')
            else:
                yplot[i,j]=np.nanmedian(pnts)
    return xplot,yplot,aniplot


# %%
sigu=np.sqrt(fp['UU'][:])
sigv=np.sqrt(fp['VV'][:])
sigw=np.sqrt(fp['WW'][:])
pvw=fp['VW'][:]
puv=fp['UV'][:]
puw=fp['UW'][:]
ani=fp['ANI_YB'][:]
ustar=fp['USTAR'][:]
lmost=fp['L_MOST'][:]
zzd=fp['tow_height'][:]-fp['zd'][:]

# %%
xplt,yplt,aplt=binplot1d(zzd/lmost,np.abs(puv),ani,-(np.logspace(-4,2,21)[-1:0:-1]),np.linspace(vmn_a,vmx_a,11),mincnt=50)
xx=-(zzd/lmost)
yy=np.abs(puv)
cc=cani_norm(ani)
m=np.zeros((len(cc),))
m[0:100000]=1
np.random.shuffle(m)
m=m.astype(bool)
plt.scatter(xx[m],yy[m],color=cc[m],s=1,alpha=.1)

for i in range(yplt.shape[0]):
    plt.semilogx(-xplt,yplt[i,:],color=cani_norm(aplt[i]),linewidth=2,path_effects=[pe.Stroke(linewidth=3, foreground='w'), pe.Normal()])

ax=plt.gca()
ax.tick_params(which="both", bottom=True)
locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )) 
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xticks([10**-3,10**-2,10**-1,1,10],[r'$-10^{-3}$','',r'$-10^{-1}$','',r'$-10^{1}$'])
ax.set_xlim(10**(-3.5),10**(1.1))
plt.ylim(-.02,.25)
plt.gca().invert_xaxis()

# %%
plt.scatter(np.abs(puw[m])/sigu[m]/sigw[m],np.abs(pvw[m])/sigv[m]/sigw[m],color=cc[m],s=.1,alpha=.5)
plt.xlim(-.05,.5)
plt.ylim(-.05,.5)

# %%
plt.hexbin(np.abs(puw[m]),np.abs(pvw[m]),mincnt=1,cmap='terrain',gridsize=50,extent=(0,.15,0,.15))
plt.colorbar()


# %%
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


# %%
rho,phi=cart2pol(np.abs(puw[m])/sigu[m]/sigw[m],np.abs(pvw[m])/sigv[m]/sigw[m])
plt.hexbin(rho,phi,mincnt=10,cmap='terrain',gridsize=20,extent=(0,.6,0,1.6))
plt.colorbar()

# %%
plt.hist(fp['USTAR'][m],bins=np.linspace(0,2))

# %%
fp=h5py.File('/home/tsw35/soteria/neon_advanced/qaqc_data/NEON_TW_U_UVWT.h5','r')
fp.keys()


# %%
def aniso(uu,vv,ww,uv,uw,vw):
    uu=np.array(uu)
    uv=np.array(uv)
    uw=np.array(uw)
    vv=np.array(vv)
    ww=np.array(ww)
    vw=np.array(vw)
    n=len(uu)
    m=uu>-9999
    k=uu[m]+vv[m]+ww[m]
    aniso=np.ones((n,3,3))*-1
    aniso[m,0,0]=uu[m]/k-1/3
    aniso[m,1,1]=vv[m]/k-1/3
    aniso[m,2,2]=ww[m]/k-1/3
    aniso[m,0,1]=uv[m]/k
    aniso[m,1,0]=uv[m]/k
    aniso[m,2,0]=uw[m]/k
    aniso[m,0,2]=uw[m]/k
    aniso[m,1,2]=vw[m]/k
    aniso[m,2,1]=vw[m]/k
    return aniso


# %%
btij=aniso(fp['UU'][:],fp['VV'][:],fp['WW'][:],fp['UV'][:],fp['UW'][:],fp['VW'][:])
y_b=np.zeros((len(fp['UU'][:]),))
for i in range(len(fp['UU'][:])):
    lam3=np.min(np.linalg.eig(btij[i,:,:])[0])
    y_b[i]=np.sqrt(3)/2*(3*lam3+1)

# %%
len(btij)

# %%
btij.shape

# %%
np.mean(btij,axis=0)

# %%
from random import randrange

# %%
N=1000
btij2=np.zeros((N,3,3))
for t in range(N):
    btij2[t,:,:]=btij[randrange(1310454),:,:]

# %%
y_b=np.zeros((N,))
for i in range(N):
    lam3=np.min(np.linalg.eig(btij2[i,:,:])[0])
    y_b[i]=np.sqrt(3)/2*(3*lam3+1)
    

# %%
plt.hist(btij2[:,0,0])

# %%
bij=btij2[0]

# %%
uu=[.984]
vv=[1.159]
ww=[0.301]
uv=[-.128] #-.75,.75
uw=[.3]
vw=[.1]
bij=aniso(uu,vv,ww,uv,uw,vw)
lam3=np.min(np.linalg.eig(bij)[0])
yb=np.sqrt(3)/2*(3*lam3+1)
print(yb)

# %%

# %%
rmin=-.3
rmax=.3
y_b=np.zeros((50,))
uu=[2]  #.1,5
vv=[4] #.1,6
ww=[0.301] #.02,1.5
uv=[.0075] #-.75,.75
uw=[-.05]  #-.3,.3
vw=[-.05]  #-.1,1
for i in range(50):
    uw=[np.linspace(rmin,rmax,50)[i]]
    bij=aniso(uu,vv,ww,uv,uw,vw)
    lam3=np.min(np.linalg.eig(bij)[0])
    y_b[i]=np.sqrt(3)/2*(3*lam3+1)

# %%
plt.scatter(np.linspace(rmin,rmax,50),y_b)

# %%
var='VW'
plt.hist(fp[var][:],bins=np.linspace(np.percentile(fp[var][:],2),np.percentile(fp[var][:],98)))

# %%
np.sum(uw_vw_r>0)/np.sum(uw_vw_r<0)

# %%
plt.scatter(np.abs(fp['UW'][0:10000]),np.abs(fp['VW'][0:10000]),alpha=.1,s=1)
plt.xlim(-.01,.1)
plt.ylim(-.05,.5)

# %%
from scipy import stats

# %%
stats.spearmanr(np.abs(fp['UV']),np.abs(fp['UW']))

# %%
