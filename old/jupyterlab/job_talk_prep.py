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
import numpy as np
import pickle
import h5py
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib as mpl
from matplotlib.lines import Line2D
from scipy import optimize
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import matplotlib as pl
import scipy as sci
from IPython.display import HTML
from sklearn.metrics import mean_squared_error
from matplotlib.gridspec import GridSpec
import cmasher as cmr
import matplotlib.ticker as tck
from scipy import stats
import matplotlib
from datetime import datetime,timedelta
import datetime as dtime
mpl.rcParams['figure.dpi'] = 300
sns.set_theme()
plt.rcParams.update({'figure.max_open_warning': 0})

# %%
fp['TIME']

# %%
# marc quick
fp=h5py.File('/home/tswater/Documents/Elements_Temp/NEON/neon_1m/WREF_L1.h5','r')
print(list(fp.keys()))

# %%
datetime.fromtimestamp(fp['TIME'][1440*2191+8*60],dtime.UTC)-timedelta(hours=8)

# %%
datetime.fromtimestamp(fp['TIME'][1440*2191+8*60+364*60*24],dtime.UTC)+timedelta(hours=8)

# %%
t0=1440*2191+8*60

# %%
t=fp['T_SONIC'][:]
np.nanmean(t[t>-9999])

# %%
u=fp['U'][t0:]
v=fp['V'][t0:]
w=fp['W'][t0:]
uu=fp['U_SIGMA'][t0:]
vv=fp['V_SIGMA'][t0:]
ww=fp['W_SIGMA'][t0:]
t=fp['T_SONIC'][t0:]
tt=fp['T_SONIC_SIGMA'][t0:]

# %%
to_out=[u,v,w,uu,vv,ww,t,tt]
out_names=['mean_u','mean_v','mean_w','var_u','var_v','var_w','mean_T','var_T']
for i in range(8):
    data=to_out[i]
    data[np.isnan(data)]=-9999
    np.savetxt('/home/tswater/Downloads/'+out_names[i]+'.csv',data,delimiter=',')

# %%
tlist=[]
for t in fp['TIME'][t0:]:
    dt=datetime.fromtimestamp(t,dtime.UTC)-timedelta(hours=8)
    tlist.append(str(dt)[0:19])

# %%
tarray=np.array(tlist)
fpout=open('/home/tswater/Downloads/local_time.csv','w')
for t in tlist:
    fpout.write(t+'\n')
fpout.close()

# %%
np.savetxt('/home/tswater/Downloads/local_time.csv',tarray,delimiter=',')

# %%
fpu=h5py.File('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data_v2/NEON_TW_U_UVWT.h5','r')
fps=h5py.File('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L2_qaqc_data_v2/NEON_TW_S_UVWT.h5','r')
d_u=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_unst_v4.p','rb'))
d_s=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_stbl_v4.p','rb'))

# %%
vmx_a=.7
vmn_a=.1
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
anibins=np.linspace(vmn_a,vmx_a,11)
anilvl=(anibins[0:-1]+anibins[1:])/2
anic=cani_norm(anilvl)

zL_u=-np.logspace(-4,2,40)
zL=zL_u.copy()
zL=zL.reshape(1,40).repeat(10,0)

ani=anilvl.copy()
ani=ani.reshape(10,1).repeat(40,1)

# U Unstable
a=.784-2.582*np.log10(ani)
u_u_stp=a*(1-3*zL)**(1/3)
u_u_old=2.55*(1-3*zL)**(1/3)
w_u_old=1.15*(1-3*zL)**(1/3)

# %%

# %%
sz=.75
fig,axs=plt.subplots(1,2,figsize=(5*sz,3*sz),gridspec_kw={'width_ratios': [1,.06]},dpi=400)
fig.patch.set_alpha(0)
plt.style.use("dark_background")
axs[0].set_facecolor('grey')
ss=.5
alph=.3
minpct=1e-04

ylabels=[r'$\Phi_{u}$',r'$\Phi_{v}$',r'$\frac{\sigma_{w}}{u*}$',r'$\Phi_{\theta}$',r'$\Phi_{q}$',r'$\Phi_{C}$']

j=0
v='W'
cc=cani_norm(d_u[v]['ani'][:])
cc_s=cani_norm(d_s[v]['ani'][:])
phi=d_u[v]['phi']
phi_s=d_s[v]['phi']

ymin=min(np.nanpercentile(phi,.1),np.nanpercentile(phi_s,.1))
ymax=max(np.nanpercentile(phi,99),np.nanpercentile(phi_s,99))

##### UNSTABLE #####
# SCATTER UNSTABLE
axs[0].scatter(-d_u[v]['zL'][:][0::5],phi[0::5],color=cc[0::5],s=ss,alpha=alph)
#axs[0].hexbin(-d_u[v]['zL'][:],phi,gridsize=200,mincnt=3,cmap='terrain',extent=(-3.5,1.1,1,4.1),xscale='log')

# LINES UNSTABLE
xplt=d_u[v]['p_zL']
yplt=d_u[v]['p_phi']
cnt=d_u[v]['p_cnt']
tot=np.sum(cnt)
yplt[cnt/tot<minpct]=float('nan')

#for i in range(yplt.shape[0]):
#    axs[0].semilogx(-xplt,yplt[i,:],color=anic[i],linewidth=2,path_effects=[pe.Stroke(linewidth=3.5, foreground='w'), pe.Normal()])

yplt=w_u_old[0]
axs[0].semilogx(-zL_u,yplt,'--',color='k',linewidth=2,path_effects=[pe.Stroke(linewidth=4, foreground='w'), pe.Normal()],zorder=5)

# LABELING
axs[0].tick_params(which="both", bottom=True)
locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )) 
axs[0].xaxis.set_minor_locator(locmin)
axs[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axs[0].set_xticks([10**-3,10**-2,10**-1,1,10],[r'$-10^{-3}$','',r'$-10^{-1}$','',r'$-10^{1}$'])
axs[0].set_xlim(10**(-3.5),10**(1.1))
axs[0].set_xlabel(r'$z/L$',fontsize=18)

#axs[j,0].xaxis.set_minor_locator(tck.AutoMinorLocator())
axs[0].set_ylabel(ylabels[j+2],fontsize=18)
axs[0].set_ylim(.8,4.1)
axs[0].invert_xaxis()


#### COLORBAR ####
axs[1].imshow(anic.reshape(10,1,4),origin='lower',interpolation=None)
axs[1].set_xticks([],[])
inta=[anilvl[0]-(anilvl[1]-anilvl[0])]
inta.extend(anilvl)
inta.extend([anilvl[-1]+(anilvl[1]-anilvl[0])])
xtc=np.interp([.2,.3,.4,.5,.6,.7],inta,np.linspace(-1,10,12))
axs[1].set_yticks(xtc,[.2,.3,.4,.5,.6,.7])
axs[1].grid(False)
#plt.yticks([.2,.3,.4,.5,.6,.7]),[.2,.3,.4,.5,.6,.7])
axs[1].yaxis.tick_right()

plt.subplots_adjust(hspace=.08,wspace=.02)
plt.savefig('../../plot_output/jbtlk_most145_.png', bbox_inches = "tight",transparent=True)

# %%
sz=2
fig,axs=plt.subplots(1,2,figsize=(5*sz,3*sz),gridspec_kw={'width_ratios': [1,.06]},dpi=400)
ss=.5
alph=.3
minpct=1e-04

ylabels=[r'$\Phi_{u}$',r'$\Phi_{v}$',r'$\Phi_{w}$',r'$\Phi_{\theta}$',r'$\Phi_{q}$',r'$\Phi_{C}$']

j=0
v='U'
cc=cani_norm(d_u[v]['ani'][:])
cc_s=cani_norm(d_s[v]['ani'][:])
phi=d_u[v]['phi']
phi_s=d_s[v]['phi']

ymin=min(np.nanpercentile(phi,.1),np.nanpercentile(phi_s,.1))
ymax=max(np.nanpercentile(phi,99),np.nanpercentile(phi_s,99))

##### UNSTABLE #####
# SCATTER UNSTABLE
ani=d_u[v]['ani'][:]
a=.784-2.582*np.log10(ani)
axs[0].scatter(-d_u[v]['zL'][:][0::3],(phi/a*2.55)[0::3],color=cc[0::3],s=ss,alpha=alph)
#axs[0].hexbin(-d_u[v]['zL'][:],(phi/a*2.55),gridsize=200,mincnt=3,cmap='terrain',extent=(-3.5,1.1,1.5,7.2),xscale='log')

# LINES UNSTABLE
xplt=d_u[v]['p_zL']
yplt=d_u[v]['p_phi']
cnt=d_u[v]['p_cnt']
tot=np.sum(cnt)
yplt[cnt/tot<minpct]=float('nan')

a=.784-2.582*np.log10(anilvl)

for i in range(yplt.shape[0]):
    axs[0].semilogx(-xplt,yplt[i,:]/a[i]*2.55,color=anic[i],linewidth=2,path_effects=[pe.Stroke(linewidth=3.5, foreground='w'), pe.Normal()])

yplt=u_u_old[0]
#axs[0].semilogx(-zL_u,yplt,'--',color='k',linewidth=1,path_effects=[pe.Stroke(linewidth=2, foreground='w'), pe.Normal()],zorder=5)

# LABELING
axs[0].tick_params(which="both", bottom=True)
locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 )) 
axs[0].xaxis.set_minor_locator(locmin)
axs[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axs[0].set_xticks([10**-3,10**-2,10**-1,1,10],[r'$-10^{-3}$','',r'$-10^{-1}$','',r'$-10^{1}$'])
axs[0].set_xlim(10**(-3.5),10**(1.1))
axs[0].set_xlabel(r'$\zeta$')

#axs[j,0].xaxis.set_minor_locator(tck.AutoMinorLocator())
axs[0].set_ylabel(ylabels[j])
axs[0].set_ylim(1.5,7.1)
axs[0].invert_xaxis()


#### COLORBAR ####
axs[1].imshow(anic.reshape(10,1,4),origin='lower',interpolation=None)
axs[1].set_xticks([],[])
inta=[anilvl[0]-(anilvl[1]-anilvl[0])]
inta.extend(anilvl)
inta.extend([anilvl[-1]+(anilvl[1]-anilvl[0])])
xtc=np.interp([.2,.3,.4,.5,.6,.7],inta,np.linspace(-1,10,12))
axs[1].set_yticks(xtc,[.2,.3,.4,.5,.6,.7])
axs[1].grid(False)
#plt.yticks([.2,.3,.4,.5,.6,.7]),[.2,.3,.4,.5,.6,.7])
axs[1].yaxis.tick_right()

plt.subplots_adjust(hspace=.08,wspace=.02)
plt.savefig('../../plot_output/jbtlk_most3.png', bbox_inches = "tight",transparent=True)


# %%

# %%

# %%
def getbins(A,n):
     B=np.sort(A)
     bins=[]
     for i in np.linspace(0,len(A)-1,n):
         i=int(i)
         bins.append(B[i])
     return bins
def getbins2D(A,B,n):
    bina=getbins(A,n)
    binb=np.zeros((n-1,n))
    for i in range(n-1):
        m=(A>bina[i])&(A<bina[i+1])
        binb[i,:]=getbins(B[m],n)
    return bina,binb
def old_phi(zLL,a,b):
    return a*(1-3*zLL)**(b)


# %%
Nbine=31
vari=['U']
xbtrue=np.ones((3,2,Nbine-1,Nbine-1))*float('nan')
ybtrue=np.ones((3,2,Nbine-1,Nbine-1))*float('nan')
error=np.ones((3,2,Nbine-1,Nbine-1))*float('nan')

for i in range(len(vari)):
    v=vari[i]
    print(v)
    for j in range(2):
        print(j)
        if j==0:
            fp=fpu
        elif j==1:
            continue
        ybine,xbine=getbins2D(fp['ANI_YB'][:],fp['ANI_XB'][:],Nbine)
        phi_=np.sqrt(fp[v+v][:])/fp['USTAR'][:]
        zL_=(fp['zzd'][:])/fp['L_MOST'][:]
        xb_=fp['ANI_XB'][:]
        yb_=fp['ANI_YB'][:]
        match v:
            case 'U': oldu=old_phi(zL_,2.55,1/3); olds=old_phi(zL_,2.06,0);ylim=[1.5,6]
            case 'V': oldu=old_phi(zL_,2.05,1/3); olds=old_phi(zL_,2.06,0);ylim=[1.25,6]
            case 'W': oldu=old_phi(zL_,1.35,1/3); olds=old_phi(zL_,1.6,0);ylim=[.8,3]
        if j==0:
            old=oldu
        elif j==1:
            old=olds
        for ii in range(len(ybine)-1):
            for jj in range(len(xbine[0])-1):
                #print(str(ii)+','+str(jj)+'   ',end='',flush=True)
                m=(xb_<xbine[ii,jj+1])&(xb_>xbine[ii,jj])&(yb_<ybine[ii+1])&(yb_>ybine[ii])
                xbtrue[i,j,ii,jj]=np.nanmean(xb_[m])
                ybtrue[i,j,ii,jj]=np.nanmean(yb_[m])
                error[i,j,ii,jj]=np.nanmedian(np.abs(phi_[m]-old[m]))

# %%
sz=.8
fig,axs=plt.subplots(1,1,figsize=(4*sz,3.5*sz))
vari=['U']
stbs=[-1]
for i in range(len(vari)):
    for j in range(1):
        vs=vari[i]+str(j)
        v=vari[i]
        #match vs:
        #    case :vmin=;vmax=
        ax=axs
        xc = np.array([0, 1, 0.5])
        yc = np.array([0, 0, np.sqrt(3)*0.5])
        ccc=['k','r','b']
        for k in np.arange(3):
            ip1 = (k+1)%3
            ax.plot([xc[k], xc[ip1]], [yc[k], yc[ip1]], 'k', linewidth=2)
            ax.fill_between([xc[k], xc[ip1]], [yc[k], yc[ip1]],y2=[0,0],color=(0.9176470588235294, 0.9176470588235294, 0.9490196078431372, 1.0),zorder=0)
            
        xbtick=[.2,.4,.6,.8]
        ybtick=[0,np.sqrt(3)/8,np.sqrt(3)/4,3*np.sqrt(3)/8,np.sqrt(3)/2]
        for ii in range(5):
            if ii in[0,2,4]: nm=.05
            else: nm=0
            ax.plot([ybtick[ii]/(np.sqrt(3)/2)/2-nm,1-ybtick[ii]/(np.sqrt(3)/2)/2],[ybtick[ii],ybtick[ii]],'--k',linewidth=.5)
        for ii in range(4):
            if ii<2:
                ulim=xbtick[ii]
            else:
                ulim=1-xbtick[ii]
            ax.plot([xbtick[ii],xbtick[ii]],[-0.025,np.sqrt(3)/2*(ulim/.5)],'--k',linewidth=.5)

        c1ps = np.array([2/3, 1/3, 0])
        x1ps = np.dot(xc, c1ps.transpose())
        y1ps = np.dot(yc, c1ps.transpose())
        c2ps = np.array([0, 0, 1])
        x2ps = np.dot(xc, c2ps.transpose())
        y2ps = np.dot(yc, c2ps.transpose())
        #ax.plot([x1ps, x2ps], [y1ps, y2ps], '-k', linewidth=2)
        
        #for k in np.arange(3):
        #    ax.text(lbpx[k], lbpy[k], labels[k], ha='center', fontsize=12)
        label_side1 = 'Prolate'
        label_side2 = 'Oblate'
        label_side3 = 'Two-component'
        #axs.text((xc[1]+xc[2])/2, (yc[0]+yc[2])/2+0.08*lc, label_side1, ha='center', va='center', rotation=-65)
        #axs.text((xc[0]+xc[2])/2, (yc[1]+yc[2])/2+0.08*lc, label_side2, ha='center', va='center', rotation=65)
        #axs.text((xc[0]+xc[1])/2, (yc[0]+yc[1])/2-0.04*lc, label_side3, ha='center', va='center')
        print(vs)
        print(np.nanmin(error[i,j,0:,:]))
        print(np.nanmax(error[i,j,0:,:]))
        data=error[i,j,:,:]
        vmax=max(np.abs(np.nanpercentile(data,5)),np.abs(np.nanpercentile(data,95)))
        
        im=ax.pcolormesh(xbtrue[i,j,0:,:],ybtrue[i,j,0:,:],data,cmap='terrain',shading='gouraud',vmin=.25,vmax=2)
        cb=fig.colorbar(im, cax=ax.inset_axes([0.95, 0.05, 0.05, .92]),label='$MAD$ $\Phi_'+v+'$')

        #im=ax.pcolormesh(xbtrue[i,j,0:,:],ybtrue[i,j,0:,:],data,cmap='Spectral_r',shading='gouraud',vmin=-vmax,vmax=vmax)
        #cb=fig.colorbar(im, cax=ax.inset_axes([0.95, 0.05, 0.05, .92]),label='$Bias$ $\Phi_'+v+'$')
        
        cb.ax.zorder=100
        ax.patch.set_alpha(0)
        ax.grid(False)
        if j==0:
            ax.set_yticks([0,np.sqrt(3)/8,np.sqrt(3)/4,3*np.sqrt(3)/8,np.sqrt(3)/2],['0',r'$0.21$',r'$0.43$',r'$0.65$',r'$0.86$'])
            ax.set_ylabel('$y_b$')
            ax.set_xticks([0,.2,.4,.6,.8,1],['',.2,.4,.6,.8,''])
            ax.set_xlabel('$x_b$')
        else:
            ax.set_xticks([],[])
        ax.spines[['left', 'bottom']].set_visible(False)

        ax.scatter(b,a,color='white',s=55,zorder=10)
        ax.scatter(b,a,color='black',s=40,zorder=10)
        
    plt.subplots_adjust(wspace=.4)
    #fig.suptitle(' Unstable ($\zeta<0$)  ')
plt.savefig('../../plot_output/jbtlk_most4.png', bbox_inches = "tight",transparent=True)

# %%
a=.37*.82
b=.402*1

# %%
zl=fpu['zzd'][:]/fpu['L_MOST'][:]
m=zl>-.1

# %%
np.nanmedian(fpu['ANI_YB'][m]/fpu['ANID_YB'][m])

# %%
np.nanmedian(fpu['ANI_XB'][m]/fpu['ANID_XB'][m])


# %%

# %%
def sort_together(X,Y):
    # X is an N length array to sort based on. Y is an M x N array of things that will sort
    X=X.copy()
    dic={}
    for i in range(len(X)):
        dic[X[i]]=[]
    for i in range(len(Y)):
        for j in range(len(X)):
            dic[X[j]].append(Y[i][j])
    X=np.array(X)
    X.sort()
    Yout=[]
    for i in range(len(Y)):
        Yout.append([])
    for i in range(len(Y)):
        for j in range(len(X)):
            Yout[i].append(dic[X[j]][i])
    return X,Yout

class_names={41:'Deciduous',42:'Evergreen',43:'Mix Forest',51:'Dwarf Scrub',52:'Shrub',\
             71:'Grassland',72:'AK:Sedge',81:'Pasture',82:'Crops',90:'Wetland'}
class_colors={11:'royalblue',12:'whitesmoke',21:'pink',22:'lightcoral',23:'red',24:'maroon',31:'silver',\
              41:'limegreen',42:'darkgreen',43:'yellowgreen',51:'darkgoldenrod',52:'tan',\
              71:'wheat',72:'khaki',73:'darkkhaki',74:'darkseagreen',81:'gold',82:'sandybrown',90:'skyblue',95:'cadetblue',0:'white'}

# %%
var='U'
mad_most=[]
other=[[],[],[],[]]
fpsite=fpu['SITE'][:]
for i in range(47):
    site=np.unique(fpsite)[i]
    other[0].append(str(site)[2:-1])
    ss=str(site)[2:-1]
    ffu=h5py.File('/home/tswater/tyche/data/neon/foot_stats/'+ss+'_U.h5','r')
    std=np.nanmedian(ffu['std_dsm'][:])
    mean=np.nanmedian(ffu['mean_chm'][:])
    if (std>20):
        other[1].append(1)
    elif (std>10):
        other[1].append(.5)
    else:
        other[1].append(0)
    other[2].append(float(stats.mode(ffu['nlcd_dom'])[0]))
    other[3].append(d_u[var]['MAD_SC23_s'][ss])
    mad_most.append(d_u[var]['MAD_OLD_s'][ss])
    #other[3].append(-d_uabs[var]['MD_SC23_s'][ss])
    #mad_most.append(-d_uabs[var]['MD_OLD_s'][ss])

# %%
frst=[b'CLBJ',b'UNDE',b'MLBS',b'STEI',b'TREE',b'HARV',b'DEJU',b'JERC',b'SERC',b'SCBI',b'LENO',b'PUUM',b'ORNL',b'DELA']
grss=[b'OAES',b'DCFS',b'WOOD',b'KONZ',b'NOGP',b'CLBJ',b'CPER']
arid=[b'SRER',b'NIWO',b'JORN',b'MOAB',b'ONAQ',]
cmplx=[b'TALL',b'UKFS',b'RMNP',b'OSBS',b'YELL',b'GUAN',b'BART',b'BONA',b'ABBY',b'GRSM',b'WREF',b'SOAP',b'SJER',b'TEAK']
crops=[b'DSNY',b'LAJA',b'KONA',b'STER',b'BLAN']
alask=[b'TOOL',b'HEAL',b'BARR']

# %%
a=[]
b=[]
for i in range(47):
    site=np.unique(fpsite)[i]
    ss=str(site)[2:-1]
    a.append(d_u[var]['MAD_SC23_s'][ss])
    b.append(d_u[var]['MAD_OLD_s'][ss])

# %%
np.mean(b)

# %%
np.mean(a)

# %%
a=np.zeros((6,))
b=np.zeros((6,))
for i in range(47):
    site=np.unique(fpsite)[i]
    ss=str(site)[2:-1]
    if site in frst:
        idx=4
    elif site in grss:
        idx=0
    elif site in arid:
        idx=3
    elif site in cmplx:
        idx=5
    elif site in crops:
        idx=1
    elif site in alask:
        idx=2
    a[idx]=a[idx]+d_u[var]['MAD_SC23_s'][ss]
    b[idx]=b[idx]+d_u[var]['MAD_OLD_s'][ss]
cnt=np.array([len(grss),len(crops),len(alask),len(arid),len(frst),len(cmplx)])
a=a/cnt
b=b/cnt

# %%
plt.style.use("default")
sns.set_theme()
fig=plt.figure(figsize=(5,3),dpi=400)
#plt.style.use("dark_background")
#fig.patch.set_facecolor('grey')
yerr=[np.zeros((6,)),b-a]
yerr=np.array(yerr)
yerr[0,:][yerr[1,:]<0]=yerr[1,:][yerr[1,:]<0]*(-1)
yerr[1,:][yerr[1,:]<0]=0
lnd=['Grassland','Crops','Tundra','Arid','Forest\n(simple)','Forest\n(complex)']
colors=['darkgreen','yellowgreen','tan','cadetblue','darkgoldenrod','wheat']
colors=['wheat','gold','mediumaquamarine','tan','yellowgreen','darkgreen']
plt.bar(lnd,a,color=colors,edgecolor='black',yerr=yerr,capsize=8)
plt.text(5.75,.9,'$MAD_{MOST}$',fontsize=10)
plt.text(5.75,.44,'$MAD_{SC23}$',fontsize=10)
plt.plot([5,5.7],[.921,.921],'k--',linewidth=.5)
plt.plot([5,5.7],[.455,.455],'k--',linewidth=.5)
plt.xlim(-.6,5.7)
plt.ylim(0.2,1)
plt.xticks(rotation=45,fontsize=12)
plt.ylabel(r'$MAD$ (error)',fontsize=12)
plt.savefig('../../plot_output/a1_graphical_TOC.png', bbox_inches = "tight",transparent=False)

# %%
np.mean(a)

# %%

# %%
d_utw=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_unst_tw_v5.p','rb'))
d_utw2=pickle.load(open('/home/tswater/Documents/Elements_Temp/NEON/neon_processed/L3_plotting_data/d_unst_tw_v3.p','rb'))

# %%

# %%

# %%

# %%
sz=.75
j=0

fig,axs=plt.subplots(1,1,figsize=(6*sz,2.7*sz),dpi=500)
#names=[r'$|\zeta|>.1$','all',r'$|\zeta|<.1$',r'$|\zeta|<.1$','all',r'$|\zeta|>.1$']
names=['$\Phi_u$','$\Phi_v$','$\Phi_w$','$\Phi_T$','$\Phi_{H2O}$','$\Phi_{CO2}$']

ss=[]
ss.append(list(d_utw['U']['SS_s'].values()))
ss.append(list(d_utw['V']['SS_s'].values()))
ss.append(list(d_utw['W']['SS_s'].values()))
ss.append(list(d_utw2['T']['SS_s'].values()))
ss.append(list(d_utw2['H2O']['SS_s'].values()))
ss.append(list(d_utw2['CO2']['SS_s'].values()))
    
ss=np.array(ss)
axs.plot([0,7],[0,0],color='w',linewidth=3)
pos=[1,2,3,4,5,6]
pos=np.array([.75,1.5,2.25,3,3.75,4.5])+.2
bplot=axs.boxplot(ss.T,labels=names,positions=pos,patch_artist=True)
plt.xlim(0,5.5)

for patch, color in zip(bplot['boxes'], ['mediumpurple']*6):
    patch.set_facecolor(color)
    patch.set_alpha(.35)

axs.set_ylabel(r'$SS$ (% improved)')
axs.set_ylim(-.4,.8)
axs.set_xlim(.5,5.15)
plt.xticks(fontsize=14)
    
plt.subplots_adjust(hspace=.08,wspace=.02)
plt.savefig('../../plot_output/jbtlk_most6.png', bbox_inches = "tight",transparent=True)


# %%

# %%
def databig(data):
    dout=np.zeros((1039,1559))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            dout[i*20:(i+1)*20,j*20:(j+1)*20]=data[i,j]
    return dout


# %%
data=(fp2d['MsKElo_hmg'][:]-fp2d['MsKElo_het'][:])/fp2d['MsKElo_het'][:]*100

# %%
data=databig(data)

# %%
data[msk]=float('nan')

# %%
plt.figure(dpi=400)
plt.imshow(data,origin='lower',cmap='PiYG',vmin=-25,vmax=25)
plt.axis(False)
plt.colorbar(shrink=.73,label='$\Delta\ MsKE\ (\%)$')
plt.savefig('../../plot_output/jbtlk_mske.png', bbox_inches = "tight",transparent=True)

# %%

# %%
data_het=fp2d['LH_60'][:]/(fp2d['LH_60'][:]+fp2d['HFX_60'][:])
data_het=databig(data_het)
data_hmg=fp2d['LH_hmg'][:]/(fp2d['LH_hmg'][:]+fp2d['HFX_hmg'][:])
data=(data_hmg-data_het)/data_het*100
data[msk]=float('nan')
plt.figure(dpi=400)
plt.imshow(data,origin='lower',cmap='BrBG',vmin=-30,vmax=30)
plt.axis(False)
plt.colorbar(shrink=.73,label='$\Delta\ EF\ (\%)$')
plt.savefig('../../plot_output/jbtlk_ef.png', bbox_inches = "tight",transparent=True)

# %%
data=(fp2d['RAINNC_hmg'][:]-fp2d['RAINNC_het'][:])/3
data[msk]=float('nan')
plt.figure(dpi=400)
plt.imshow(data,origin='lower',cmap='coolwarm',vmin=-200,vmax=200)
plt.axis(False)
plt.colorbar(shrink=.73,label='$\Delta\ Precipitation\ (mm)$')
plt.savefig('../../plot_output/jbtlk_rain.png', bbox_inches = "tight",transparent=True)

# %%

# %%

# %%

# %%
import netCDF4 as nc

# %%
frkdir='/home/tswater/Documents/Elements_Temp/WRF/'
fpscl=nc.Dataset(frkdir+'agg_files/agg_scaling.nc','r')
fpagg=nc.Dataset(frkdir+'agg_files/agg_full.nc','r')
fp2d =nc.Dataset(frkdir+'agg_files/agg_2d.nc','r')
fp=nc.Dataset(frkdir+'WRF_basic/20210605_conv.nc','r')
fp=nc.Dataset(frkdir+'scaling_compressed/het/wrfout_d01_2023-07-07_19\uf02200\uf02200','r')
fpst=nc.Dataset(frkdir+'static_data.nc','r')
msk=fpst['LU_INDEX'][0,:,:]==17

# %%
os.listdir('/home/tswater/Documents/Elements_Temp/WRF/scaling_compressed/het/')


# %%
#list(fp.variables)

# %%
def avg(data,dx):
    


# %%
T=fp['T'][0,0,:,:]+300
T[msk]=float('nan')
plt.imshow(T[375:550,0:250],origin='lower',cmap='terrain',interpolation=None)
plt.colorbar()
plt.scatter([155],[432-375])

# %%
os.listdir('/home/tswater/Documents/tyche/data/random/')

# %%
fp=h5py.File('/home/tswater/Documents/tyche/data/random/ECOSTRESS_L2_LSTE_33252_002_20240518T010946_0601_01.h5','r')
fpg=h5py.File('/home/tswater/Documents/tyche/data/random/ECOSTRESS_L1B_GEO_33252_002_20240518T010946_0601_01.h5','r')

# %%
fpg['Geolocation'].keys()

# %%
plt.imshow(fpg['Geolocation']['longitude'][:],cmap='terrain',interpolation=None)
plt.colorbar()

# %%
plt.imshow((fp['SDS']['LST'][:]*0.02),cmap='terrain',interpolation=None,vmin=280,vmax=315)
plt.colorbar()

# %%
fp['StandardMetadata'].keys()

# %%
fp['StandardMetadata']['EastBoundingCoordinate'][:]

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # No Anim. Testing

# %%
fpmsw=nc.Dataset('/run/media/tswater/Elements/WRF/mswep_gaea/2023161.21.nc','r')

# %%
data=fpmsw['Band1'][:]
data[msk]=float('nan')
plt.imshow(data,origin='lower',cmap='terrain',vmin=0,vmax=25)
plt.colorbar()

# %%

# %%
1559/1039*400

# %%

# %%
fp['P_4D_het'][:].shape

# %%
pres=databig(fp['P_4D_het'][17,0,:])
t0=fp['T0_het'][17,:]
th=t0/(pres/1000)**(2/7)
tout=th*(pres/100000)**(2/7)

# %%
fp['TTspatial_4D_het'][16,0,20:24,6:10]
ll=[]
for i in range(49):
    ll.append(np.nanmean(fp['TTspatial_4D_het'][i,0,20:24,6:10]))

# %%
ll[37]

# %%
plt.plot(ll[30:44])

# %%
fpsc=nc.Dataset('/run/media/tswater/Elements/WRF/scaling_compressed/het/wrfout_d01_2023-07-08_180000','r')

# %%
data=fpsc['U'][0,0,:,:-1]
#data=fpsc['RAINNC'][0,:,:]
data2=data
data2[msk]=float('nan')
plt.imshow(data2[0:300,1150:1400],origin='lower',cmap='terrain')
plt.axis(False)
plt.colorbar()
plt.scatter([1300-1150],[170])

# %%
dT=np.ones((49,150,20))*float('nan')
dU=np.ones((49,150,20))*float('nan')
i=0
for f in os.listdir('/run/media/tswater/Elements/WRF/scaling_compressed/het'):
    fp=nc.Dataset('/run/media/tswater/Elements/WRF/scaling_compressed/het/'+f,'r')
    for j in range(6):
        dT[i,:,j]=fp['HFX'][0,170,1225:1375]
    dU[i,:,5:]=fp['U'][0,0:15,170,1225:1375].T
    i=i+1

# %%
t=8
plt.imshow(dT[t,:,:].T,origin='lower',cmap='coolwarm')
plt.imshow(dU[t,:,:].T,origin='lower',cmap='PiYG',vmin=-10,vmax=10,extent=(0,75,0,30))
plt.axis(False)

# %%
#155:432
t_cmc=[]
t_cal=[]
pres=databig(fp['P_4D_het'][37,0,:])
t0=fp['T0_het'][37,:]
th=t0/(pres/1000)**(2/7)
tout=th*(pres/100000)**(2/7)-273.15
t_cal.append(tout[593,63])
t_cmc.append(tout[432,155])
for i in range(2,51):
    data=tout.data[0:659,0:301]
    d2=hmgz(data,i)
    t_cmc.append(d2[432,155])
    t_cal.append(d2[593,63])

# %%
432/20

# %%
fp['Z_het'][:].shape
plt.imshow(fp['Z_het'][15,13,:,:],origin='lower')
plt.colorbar()

# %%
plt.scatter(np.linspace(3,3*50,50),t_cmc)
plt.plot([0,150],[t_cmc[0],t_cmc[0]])
plt.plot([0,150],[np.nanmean(t_cmc),np.nanmean(t_cmc)])

# %%
plt.scatter(np.linspace(3,3*50,50),t_cal,color=None)

# %% [markdown]
# # Animations 

# %%
fig = plt.figure(figsize=(6,4),dpi=200)
ax1=fig.add_subplot(111)

def animate(t):
    plt.imshow(dT[t,:,:].T,origin='lower',cmap='coolwarm')
    plt.imshow(dU[t,:,:].T,origin='lower',cmap='PiYG',vmin=-10,vmax=10,extent=(0,75,0,30))
    plt.axis(False)
    return fig


ani=FuncAnimation(fig,animate,frames=49,interval=200,repeat=True)
#FFwriter = animation.FFMpegWriter(fps=10)
#ani.save('ani_1.mp4',writer=FFwriter)
HTML(ani.to_jshtml())
#HTML(ani.to_html5_video())

# %%
sz=.75
fig=plt.figure(figsize=(9*sz,5*sz),dpi=500)
sbf = fig.subfigures(1, 2, hspace=0,wspace=0,width_ratios= [1.25,1],frameon=False)
ax=sbf[0].subplots(1,1)
sbf[1].subplots(2,1)


# %% [markdown]
# # Draft Final Push

# %% [markdown]
# ## HMG Loss Gif

# %%
def hmgz(data_,dx):
    out=np.zeros(data_.shape)
    n=np.floor(np.array(data_.shape)/dx)
    for i in range(int(n[0])+1):
        for j in range(int(n[1])+1):
            idx=int(i*dx)
            jdx=int(j*dx)
            out[idx:idx+dx,jdx:jdx+dx]=np.nanmean(data_[idx:idx+dx,jdx:jdx+dx])
    return out


# %%
# data prep
#155:432
fp=nc.Dataset('/run/media/tswater/Elements/WRF/WRF_basic/20230721_conv.nc','r')
t_cmc=[]
t_cal=[]
pres=databig(fp['P_4D_het'][37,0,:])
t0=fp['T0_het'][37,:]
th=t0/(pres/1000)**(2/7)
tout=th*(pres/100000)**(2/7)-273.15
t_cal.append(tout[593,63])
t_cmc.append(tout[432,155])
data=np.zeros((50,1001,1001))
d1=np.zeros(tout.shape)
d1[:]=tout.data[:]
d1[msk]=float('nan')
data[0,:]=d1[:1001,:1001]
for i in range(2,51):
    d1=tout.data[:]
    d2=hmgz(d1,i)
    d2[msk]=float('nan')
    data[i-1,:]=d2[0:1001,0:1001]
    t_cmc.append(d2[432,155])
    t_cal.append(d2[593,63])

# %%
plt.style.use("default")
sns.set_theme()
sz=.75
fig=plt.figure(figsize=(9*sz,5*sz),dpi=250)
fig.patch.set_alpha(0)
fig.set_tight_layout(True)
sbf = fig.subfigures(1, 2, hspace=0,wspace=0,width_ratios= [1.25,1],frameon=False)
ax=sbf[0].subplots(1,1)
ax2,ax1=sbf[1].subplots(2,1)
text_color='black'
ccmc=[]
ccal=[]
cmap=pl.colormaps['coolwarm']
for i in range(50):
    hm=t_cmc[i]
    ha=t_cal[i]
    ccmc.append(cmap((hm-13)/(34)))
    ccal.append(cmap((ha-13)/(34)))
    
def animate(i):
    print(i,end=',',flush=True)
    lcmc=t_cmc[0:i+1]
    lcal=t_cal[0:i+1]
    ax1.cla()
    ax2.cla()
    ax.cla()
    ax1.plot([-10,160],[lcmc[0],lcmc[0]],'w--',linewidth=2)
    ax1.scatter(np.linspace(3,3*50,50)[0:i+1],lcmc,s=25,color=ccmc[0:i+1],edgecolors='black',linewidths=.5)
    ax1.set_xlim(0,155)
    ax1.set_title('Temperature: Claremont, CA',color=text_color)
    #ax1.set_ylabel(r'T ($\degree C$)')
    ax1.set_yticks([27,29,31,35,37,39],[27,'',31,35,'',39])
    ax1.set_ylim(25,39)
    ax1.yaxis.set_label_position("right")
    ax1.yaxis.tick_right()
    ax1.set_xlabel('Resolution ($km$)',color=text_color)
    
    ax2.scatter(np.linspace(3,3*50,50)[0:i+1],lcal,s=25,color=ccal[0:i+1],edgecolors='black',linewidths=.5)
    ax2.plot([-10,160],[lcal[0],lcal[0]],'w--',linewidth=2)
    ax2.set_yticks([18,20,22,24,28,30,32,34,36],['',20,'',24,28,'',32,'',36])
    #ax2.set_ylabel(r'T ($\degree C$)')
    ax2.set_xticks([0,50,100,150],[])
    ax2.set_xlim(0,155)
    ax2.set_ylim(17,38)
    ax2.set_title('Temperature: Berkeley, CA',color=text_color)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    
    im=ax.imshow(data[i,0:1000,0:1000],origin='lower',interpolation=None,cmap='coolwarm',vmin=13,vmax=47)
    ax.set_title('Temperature',color=text_color)
    ax.axis(False)
    sbf[0].colorbar(im,cax=ax.inset_axes([0.95, 0, 0.05, 1]),label='T ($\degree C$)')
    return fig

#ani=FuncAnimation(fig,animate,frames=49,interval=500,repeat=False)
#FFwriter = pl.animation.FFMpegWriter(fps=2)
#Pwriter= pl.animation.PillowWriter(fps=2)
#ani.save('../../plot_output/jbtlk_p2_3ani.gif',writer=Pwriter)
#HTML(ani.to_jshtml())
#HTML(ani.to_html5_video())
animate(49)
#plt.savefig('../../plot_output/jbtlk_p2_3v7.png', bbox_inches = "tight")

# %%

# %%

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## HFX Average

# %%
data1=fp['HFX_het'][37,:,:]
data2=hmgz(data1,20)
data1[msk]=float('nan')
data2[msk]=float('nan')
plt.figure(dpi=400)
plt.imshow(data1,cmap='turbo',origin='lower',vmin=-25,vmax=600)
plt.colorbar(label=r'Heat Flux ($W\ m^{-2}$)',shrink=.73)
plt.axis(False)
plt.savefig('../../plot_output/jbtlk_p2_6a.png', bbox_inches = "tight")


# %%
plt.figure(dpi=400)
plt.imshow(data2,cmap='turbo',origin='lower',vmin=-25,vmax=600)
plt.colorbar(label=r'Heat Flux ($W\ m^{-2}$)',shrink=.73)
plt.axis(False)
plt.savefig('../../plot_output/jbtlk_p2_6b.png', bbox_inches = "tight")

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## RAINNC Average

# %%

# %%
data1=fpmsw['Band1'][:]
data2=hmgz(data1,20)
data1[msk]=float('nan')
data2[msk]=float('nan')
plt.figure(dpi=400)
plt.imshow(data1,cmap='gist_ncar',origin='lower',vmin=0,vmax=20)
plt.colorbar(label=r'Precipitation ($mm$)',shrink=.73)
plt.axis(False)
plt.savefig('../../plot_output/jbtlk_p2_19a.png', bbox_inches = "tight")

# %%
plt.figure()
plt.imshow(data2,cmap='gist_ncar',origin='lower',vmin=0,vmax=20)
plt.colorbar(label=r'Precipitation ($mm$)',shrink=.73)
plt.axis(False)
plt.savefig('../../plot_output/jbtlk_p2_19b.png', bbox_inches = "tight")

# %%
## 

# %%

# %% [markdown]
# ## MSWEP Gif

# %%
#fp=nc.Dataset('/run/media/tswater/Elements/WRF/WRF_basic/20230721_conv.nc','r')
d0=fpmsw['Band1'][:]
data=np.zeros((50,d0.shape[0],d0.shape[1]))
d1=np.zeros(d0.shape)
d1[:]=d0[:]
d1[msk]=float('nan')
data[0,:]=d1[:]
for i in range(2,51):
    d1=d0[:]
    d2=hmgz(d1,i)
    d2[msk]=float('nan')
    data[i-1,:]=d2[:]

# %%
sz=.75
fig=plt.figure(figsize=(9.13*sz,5*sz),dpi=250)
plt.style.use("dark_background")
#fig.patch.set_facecolor('grey')
fig.patch.set_alpha(0)
fig.set_tight_layout(True)
ax=fig.subplots(1,1)
cax=ax.inset_axes([0.95, 0, 0.05, 1])

def animate(i):
    print(i,end=',',flush=True)
    ax.clear()
    cax=ax.inset_axes([0.95, 0, 0.05, 1])
    im=ax.imshow(data[i,:],origin='lower',interpolation=None,cmap='gist_ncar',vmin=0,vmax=20)
    #ax.set_title('MSWEP Precipitation',color=text_color)
    ax.axis(False)
    cb=sbf[0].colorbar(im,cax=cax,label='Precipitation ($mm$)')
    return fig
    
ani=FuncAnimation(fig,animate,frames=49,interval=500,repeat=True)
FFwriter = pl.animation.FFMpegWriter(fps=2)
Pwriter= pl.animation.PillowWriter(fps=2)
ani.save('../../plot_output/jbtlk_p2_18ani.gif',writer=Pwriter,savefig_kwargs={"transparent": True})
#HTML(ani.to_jshtml())
#HTML(ani.to_html5_video())
#plt.savefig('../../plot_output/jbtlk_p2_3v2.png', bbox_inches = "tight")

# %% [markdown]
# ## LES Gif Circulation

# %%
import netCDF4 as nc
leshet='/home/tswater/tyche/LES_temp/fr2_20170717_00/'
leshmg='/home/tswater/tyche/LES_temp/fr2_20170717_01/'
flist=os.listdir(leshet)
flist.sort()
dH=20
dataUt=np.ones((len(flist),110+dH,400))*float('nan')
dataWt=np.ones((len(flist),110+dH,400))*float('nan')
dataHt=np.ones((len(flist),110+dH,400))*float('nan')

dataUg=np.ones((len(flist),110+dH,400))*float('nan')
dataWg=np.ones((len(flist),110+dH,400))*float('nan')
dataHg=np.ones((len(flist),110+dH,400))*float('nan')

for i in range(len(flist)):
    fpt=nc.Dataset(leshet+flist[i],'r')
    fpg=nc.Dataset(leshmg+flist[i],'r')

    dataUt[i,dH:,:]=fpt['AVV_U'][0,0:110,250,0:400]
    dataWt[i,dH:,:]=fpt['AVV_W'][0,0:110,250,0:400]

    dataUg[i,dH:,:]=fpg['AVV_U'][0,0:110,250,0:400]
    dataWg[i,dH:,:]=fpg['AVV_W'][0,0:110,250,0:400]

    for j in range(dH+1):
        dataHt[i,j,:]=fpt['AVS_SH'][0,250,0:400]
        dataHg[i,j,:]=fpg['AVS_SH'][0,250,0:400]


# %%
def animate(t,quiver=True,fig=fig):
    if t<6:
        i=t
    elif t in [6,7,8,9]:
        i=6
    elif t in [10,11,12,13]:
        i=7
    elif t in [14,15,16,17]:
        i=8
    elif t in [18,19,20,21]:
        i=9
    elif t in [21,22,23,24]:
        i=10
    elif t in [25,26,27,28]:
        i=11
    else:
        i=t+12-29
    
    print(t,end=',',flush=True)
    ax1.clear()
    ax2.clear()

    cax1=ax1.inset_axes([0.98, 0, 0.02, 1])
    cax2=ax2.inset_axes([0.98, 0, 0.02, 1])
    
    vmin=np.nanpercentile(dataHt[i,:,:],4)
    vmax=np.nanpercentile(dataHt[i,:,:],96)
    ax1.imshow(dataHt[i,:,:],origin='lower',cmap='coolwarm',vmin=vmin,vmax=vmax)
    im=ax1.imshow(dataUt[i,:,:],origin='lower',cmap='PuOr',vmin=-5,vmax=5)

    fig.colorbar(im,cax=cax1,label='Velocity ($m\ s^{-1}$)')
    
    U=hmgz(dataUt[i,:,:],10)[15+dH:-29:32,15:-29:71]
    V=hmgz(dataWt[i,:,:],10)[15+dH:-29:32,15:-29:71]
    if quiver:
        a=ax1.quiver(X,Y,U,V,color='white',pivot='tail',edgecolors='black',linewidth=.5)
    ax1.axis(False)
    ax1.set_title('Heterogeneous',color='white')

    ax2.imshow(dataHg[i,:,:],origin='lower',cmap='coolwarm',vmin=vmin,vmax=vmax)
    im=ax2.imshow(dataUg[i,:,:],origin='lower',cmap='PuOr',vmin=-5,vmax=5)

    fig.colorbar(im,cax=cax2,label='Velocity ($m\ s^{-1}$)')
    
    U=hmgz(dataUg[i,:,:],10)[15+dH:-29:32,15:-29:71]
    V=hmgz(dataWg[i,:,:],10)[15+dH:-29:32,15:-29:71]
    if quiver:
        ax2.quiver(X,Y,U,V,color='white',pivot='tail',edgecolors='black',linewidth=.5,scale=a.scale)
    ax2.axis(False)
    ax2.set_title('Homogeneous',color='white')

    return fig


# %%
fig=plt.figure(figsize=(9*sz,5*sz),dpi=250)
fig.patch.set_alpha(0)
fig.set_tight_layout(True)
ax1,ax2=fig.subplots(2,1)
X=np.meshgrid(np.linspace(15+dH,400+dH-30,6))
Y=np.meshgrid(np.linspace(15+dH,110+dH-30,3))
X,Y=np.meshgrid(X,Y)
plt.style.use("dark_background")
    
ani=FuncAnimation(fig,animate,frames=33,interval=500,repeat=True)
FFwriter = pl.animation.FFMpegWriter(fps=2)
Pwriter= pl.animation.PillowWriter(fps=2)
ani.save('../../plot_output/jbtlk_p3_5ani.gif',writer=Pwriter,savefig_kwargs={"transparent": True})

# %%
fig=plt.figure(figsize=(9*sz,5*sz),dpi=250)
fig.patch.set_alpha(0)
fig.set_tight_layout(True)
ax1,ax2=fig.subplots(2,1)
X=np.meshgrid(np.linspace(15+dH,400+dH-30,6))
Y=np.meshgrid(np.linspace(15+dH,110+dH-30,3))
X,Y=np.meshgrid(X,Y)
plt.style.use("dark_background")

animate(9,False,fig)
plt.savefig('../../plot_output/jbtlk_p3_les.png', bbox_inches = "tight",transparent=True)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Turb Time Series

# %%
fp=h5py.File('/run/media/tswater/Elements/NEON/neon_raw/UKFS/NEON.D06.UKFS.IP0.00200.001.ecte.2023-07-18.l0p.h5','r')

# %%
#plt.figure(dpi=300,figsize=(7,4))
sz=.75
plt.style.use("default")
sns.set_theme()
fig.patch.set_alpha(0)
#ax.set_facecolor('dimgrey')
fig, ax = plt.subplots(nrows=1, ncols=1,dpi=300,figsize=(7*sz,4*sz))
u=fp['UKFS/dp0p/data/soni/000_060/veloXaxs'][0:5*60*20]
ax.plot([-60*2,60*8],[0,0],linewidth=3,color='white')
ax.plot(np.linspace(0,60*5,5*60*20),u,linewidth=.5)
ax.set_ylim(-5,5)
ax.set_xlim(-.1*60,5.1*60)
ax.text(255,2.2,r'$\mu_{u}= $'+'\n'+str(np.nanmean(u))[0:5])
ax.set_xlabel('Seconds')
ax.set_ylabel('Velocity ($m\ s^{-1}$)')
plt.savefig('../../plot_output/jbtlk_p1_3a_.png', bbox_inches = "tight")

# %%
#plt.figure(dpi=300,figsize=(7,4))
sz=.75
fig, ax = plt.subplots(nrows=1, ncols=1,dpi=300,figsize=(7*sz,4*sz))
fig.patch.set_alpha(0)
#ax.set_facecolor('dimgrey')
u=fp['UKFS/dp0p/data/soni/000_060/veloXaxs'][0:5*60*20]
up=u-np.nanmean(u)
ax.plot([-60*2,60*8],[0,0],linewidth=3,color='white')
ax.plot(np.linspace(0,60*5,5*60*20),up,linewidth=.5)
ax.set_ylim(-5,5)
ax.set_xlim(-.1*60,5.1*60)
ax.text(255,2.2,r'$\sigma^{2}_{u}= $'+'\n'+str(np.nanvar(up))[0:5])
ax.set_xlabel('Seconds')
ax.set_ylabel('Velocity ($m\ s^{-1}$)')
plt.savefig('../../plot_output/jbtlk_p1_3b_.png', bbox_inches = "tight")

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Performance MOST/SC23

# %%
ani=fpu['ANI_YB'][:]
zL=fpu['zzd'][:]/fpu['L_MOST'][:]

a=.784-2.582*np.log10(ani)
u_u_stp=a*(1-3*zL)**(1/3)
u_u_old=2.55*(1-3*zL)**(1/3)

uu=fpu['UU'][:]
m_uu=(u_u_old*fpu['USTAR'][:])**2
s_uu=(u_u_stp*fpu['USTAR'][:])**2

#.70,.41

# %%
sz=.75
plt.style.use("dark_background")
#plt.style.use("default")
#sns.set_theme()
fig, ax = plt.subplots(nrows=1, ncols=1,dpi=300,figsize=(4.5*sz,4*sz))
fig.patch.set_alpha(0)
ax.set_facecolor('dimgrey')
#ax.scatter(uu[0::50],s_uu[0::50],s=.1,alpha=.3)
a=ax.hexbin(uu,s_uu,cmap='gist_ncar',gridsize=500,mincnt=5,extent=(0,5,0,5),vmax=150)
ax.plot([-1,10],[-1,10],'w--')
ax.set_xlim(-.5,5)
ax.set_ylim(-.5,5)
ax.text(0.1,4.1,r'$|Bias|=$'+'\n0.41')
ax.set_xticks([0,2,4])
ax.set_yticks([0,2,4])
ax.set_xlabel('Observed $\sigma_{u}^{2}$  ($m^2\ s^{-2}$)',fontsize=14)
ax.set_ylabel('Model $\sigma_{u}^{2}$  ($m^2\ s^{-2}$)',fontsize=14)
plt.savefig('../../plot_output/jbtlk_p1_11b.png', bbox_inches = "tight")

# %%

# %%
#plt.style.use("dark_background")
fig, ax = plt.subplots(nrows=1, ncols=1,dpi=300,figsize=(4.5*sz,4*sz))
fig.patch.set_alpha(0)
ax.set_facecolor('dimgrey')
#ax.scatter(uu[0::50],m_uu[0::50],s=.1,alpha=.3
ax.hexbin(uu,m_uu,cmap='gist_ncar',gridsize=500,mincnt=5,extent=(0,5,0,5),vmax=150)
ax.plot([-1,10],[-1,10],'w--')
ax.set_xlim(-.5,5)
ax.set_ylim(-.5,5)
ax.text(0.1,4.1,r'$|Bias|=$'+'\n0.70')
ax.set_xticks([0,2,4])
ax.set_yticks([0,2,4])
ax.set_xlabel('Observed $\sigma_{u}^{2}$  ($m^2\ s^{-2}$)',fontsize=14)
ax.set_ylabel('Model $\sigma_{u}^{2}$  ($m^2\ s^{-2}$)',fontsize=14)
plt.savefig('../../plot_output/jbtlk_p1_11a.png', bbox_inches = "tight")

# %%

# %%
