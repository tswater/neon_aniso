# Core tools to be imported and used elsewhere
import numpy as np
from scipy.interpolate import interp1d
import datetime
import matplotlib.pyplot as plt
import matplotlib
############## CONSTANTS ################
SITES =['ABBY', 'BARR', 'BART', 'BLAN', 'BONA', 'CLBJ', 'CPER', 'DCFS', 'DEJU',  'DELA', 'DSNY', 'GRSM', 'GUAN', 'HARV', 'HEAL', 'JERC', 'JORN', 'KONA', 'KONZ',  'LAJA', 'LENO', 'MLBS', 'MOAB', 'NIWO', 'NOGP', 'OAES', 'ONAQ', 'ORNL', 'OSBS',  'PUUM', 'RMNP', 'SCBI', 'SERC', 'SJER', 'SOAP', 'SRER', 'STEI', 'STER', 'TALL',  'TEAK', 'TOOL', 'TREE', 'UKFS', 'UNDE', 'WOOD', 'WREF', 'YELL']




#############################################################
#############################################################
###################### INTERNAL FUNCTIONS ###################
def _confirm_user(msg):
    while True:
        user_input = input(f"{msg} (Y/N): ").strip().lower()
        if user_input in ('y', 'yes'):
            return True
        elif user_input in ('n', 'no'):
            return False
        else:
            print("Invalid input. Please enter 'Y' or 'N'.")



################### FUNCTION ########################
#####################################################
def static2full(fmsk,data,debug=False):
    ''' Take site information (L1 attrs) and turn into timeseries '''
    sitelist=list(fmsk.keys())
    sitelist.sort()
    out=[]
    for i in range(len(sitelist)):
        N=np.sum(fmsk[sitelist[i]])
        if debug:
            print(sitelist[i])
            print(N)
            print(data[i],flush=True)
        out.extend([data[i]]*N)
    return out


######################### HMG LIST OF LISTS ###########################
def homogenize_list(a,fill=float('nan')):
    b = np.ones([len(a),len(max(a,key = lambda x: len(x)))])*fill
    for i,j in enumerate(a):
        b[i][0:len(j)] = j
    return b

########################## OUT TO H5 ##################################
# Take a dictionary that mimics h5 filestructure and output to said h5
def out_to_h5(_fp,_ov,overwrite,desc={},attrs={}):
    for k in _ov.keys():
        if type(_ov[k]) is dict:
            ov2=_ov[k]
            if k not in _fp.keys():
                _f=_fp.create_group(k)
            else:
                _f=_fp[k]
            out_to_h5(_f,ov2,overwrite)

        else:
            _f=_fp
            if '/' in k:
                split=k.split('/')
                n=len(split)
                for i in range(n-1):
                    if split[i] not in _f.keys():
                        _f=_f.create_group(k)
                    else:
                        _f=_f[split[i]]
                kout=split[-1]
            else:
                kout=k
            try:
                _f.create_dataset(kout,data=np.array(_ov[k][:]))
            except Exception as e:
                if overwrite:
                    _f[kout][:]=np.array(_ov[k][:])
                else:
                    print('Skipping output of '+str(kout))
            _f[kout].attrs['missing_value']=-9999
            if k in desc.keys():
                _f[kout].attrs['description']=desc[k]
            if k in attrs.keys():
                for kk in attrs[k].keys():
                    _f[kout].attrs[kk]=attrs[k][kk]
    _fp.attrs['last_updated_utc']=str(datetime.datetime.utcnow())
    return None

##############################################################################
############################# GAPFILL ########################################
# Fills a timseries with a fill value to complete it
def ngapfill(t_out,t_in,d_in,fill=float('nan')):
    tout=np.array(t_out)
    tin=np.array(t_in)
    din=np.array(d_in)
    dout=np.ones((len(tout),))*fill
    if len(tin)==0:
        return dout
    msk=np.zeros((len(tout),)).astype(bool)

    # determine i0 and if
    t0=tin[0]
    tf=tin[-1]
    i0=np.where(tout==t0)[0][0]
    ie=np.where(tout==tf)[0][0]

    j=0
    for i in range(i0,ie+1):
        if i>=len(tout):
            break
        t=tout[i]
        if tin[j]==t:
            j=j+1
            msk[i]=True
        else:
            continue
    dout[msk]=din[:]
    return dout


#####################################################
############################# NSCALE ###################################
# Wrapper function; interpolates (ninterp) or upscales (nupscale) as appropriate
def nscale(t_out,t_in,d_in,scl=None,maxdelta=60,nearest=True,extrap=True,nanth=.2,debug=False):
    # ensure inputs are arrays and not lists
    tout=np.array(t_out)
    tin=np.array(t_in)
    din=np.array(d_in)

    try:
        # ensure tin and din same shape
        if not (len(tin)==len(din)):
            raise ValueError('Length of input time '+str(len(tin))+' and data '+\
                str(len(din))+' is not the same!')

        # use nearest if data is covering an averaging period
        # use linear (nearest=false) if data is instantaneous
        tdelta=tin[1:]-tin[0:-1]
        toutdelta=tout[1:]-tout[0:-1]

        if len(tdelta)<2:
            raise ValueError('Input time is too short: '+str(tin))
        if np.all(np.isnan(tdelta)):
            raise ValueError('Change in time evaluates to all nan: '+str(tin))
    except ValueError as e:
        print(e)
        msg='There is an error in the timeseries for interpolation\n'+\
            'Would you like to return all NaN?'
        if _confirm_user(msg):
            return np.ones(tout.shape)*float('nan')
        else:
            print('OKAY; will try to provide debug info: ')
            try:
                print('Input time size: '+str(len(tin)))
                print('Input data size: '+str(len(din)))
                print('First timestep: '+str(tin[0]))
                print('Second timestep: '+str(tin[1]))
                print('Last Timestep: '+str(tin[-1]))
                print(len(tdelta))
                print(tdelta.shape)
                print(tdelta[0])
                print(tdelta[1])

            except Exception as e2:
                print('Failed to provide some debug info due to: ')
                print(e2)
                print()
                print('Halting execution')
                pass
            raise(e)
    if np.nanmin(toutdelta)>np.nanmin(tdelta):
        return nupscale(tout,tin,din,scl,maxdelta,nearest,nanth,debug)
    else:
        return ninterp(tout,tin,din,maxdelta,nearest,extrap,debug)

############################# NEON INTERP ###############################
# Interpolate data from one timeseries to another
def ninterp(tout,tin,din,maxdelta=60,nearest=True,extrap=True,debug=False):
    ''' Interpolate data, with a special mind for gaps
        tout     : timeseries to interpolate to
        tin      : time of data
        din      : data; all non-usable data should be 'NaN'
        maxdelta : maximum gap to interpolate with; larger will be filled
                   with NaN
        nearest  : if true, will use nearest neighbor interpolation
        extrap   : if true, will extrapolate at the edges an ammount equal
                   to min(tin[1:]-tin[0:-1])/2
    '''

    # check if the output time resolution is bigger than input
    tdelta=tin[1:]-tin[0:-1]
    toutdelta=tout[1:]-tout[0:-1]
    inscl=np.nanmin(tdelta)/60
    if np.nanmin(toutdelta)>np.nanmin(tdelta):
        print('NINTERP WARNING! Output resolution '+str(np.nanmin(toutdelta))+\
                ' coarser than input '+str(np.nanmin(tdelta))+\
                '; averaging with nupscale suggested')
        try:
            if debug:
                dbg='::::::::::DEBUG::::::::::::\n'
                f=len(np.where(tdelta<np.nanmin(toutdelta)))
                dbg=dbg+'Frequency of input finer than output: '+str(f)+'/'+str(len(tdelta))+'\n'
                if (f>0):
                    idxs=np.where(tdelta<np.nanmin(toutdelta))[0]
                    i=0
                    for idx in idxs:
                        dbg=dbg+'    At index '+str(idx)+': ['+str(tin[idx])+\
                                ','+str(tin[idx+1])+']\n'
                        i=i+1
                        if i>50:
                            dbg=dbg+'    ... stopping printout\n'
                            break
                print(dbg+'::::::::::DEBUG::::::::::::',flush=True)
        except Exception as e:
            print('Ninterp Debug exception')
            print(e)
    # INDEXING/GAPS EXPLAINED:
    # lets for input indicies 0,1,2 we have times 5,15,25 minutes and for
    # index 3 we have 145 minutes. splt_idi, index of the gap in input tin
    # space, would be 2+1=3. In input space, we interpolate/evaluate tin[0:3]
    # which excludes the gap, and then separately interpolate/evaluate tin[3:...]
    # which also excludes the gap. Lets say output space is 1 minute resolution.
    # We will evaluate/interpolate from 5 to 25 minutes, then from 145 to ...
    # minutes. splt_tmini[0] corresponds to 145 minutes, and split_tminf[0]
    # is 25 minutes. splt_idof and split_idoi get these points in time as
    # output indicies.

    # split into smaller timeseries based on gaps
    splt_idi=np.array(np.where(tdelta>maxdelta*60)[0])+1 # gap index in input
    splt_tmini=tin[splt_idi]
    splt_tminf=tin[splt_idi-1]

    if extrap:
        ext=inscl*60/2
    else:
        ext=0

    # convert split to indicies in tout space
    splt_idof=np.interp(splt_tminf+ext,tout,np.linspace(0,len(tout)-1,len(tout)))
    splt_idoi=np.interp(splt_tmini-ext,tout,np.linspace(0,len(tout)-1,len(tout)))
    t0=int(np.round(np.interp(tin[0]-ext,tout,np.linspace(0,len(tout)-1,len(tout)))))
    t0i=0

    out=np.ones((len(tout),))*float('nan')

    # loop and interate through to interpolate
    for i in range(len(splt_idi)):
        tf=int(np.round(splt_idof[i]))+1
        tfi=splt_idi[i]
        if nearest:
            interp=interp1d(tin[t0i:tfi],din[t0i:tfi],kind='nearest',\
                            bounds_error=False,fill_value=(din[t0i],din[tfi-1]))
            out[t0:tf]=interp(tout[t0:tf])
        else:
            out[t0:tf]=np.interp(tout[t0:tf],tin[t0i:tfi],din[t0i:tfi])
        t0=int(np.round(splt_idoi[i]))+1
        t0i=tfi

    # do final interpolation
    tf=int(np.round(np.interp(tin[-1]+ext,tout,np.linspace(0,len(tout)-1,len(tout)))))+1
    tfi=None
    if nearest:
        interp=interp1d(tin[t0i:tfi],din[t0i:tfi],kind='nearest',\
                        bounds_error=False,fill_value=(din[t0i],din[-1]))
        out[t0:tf]=interp(tout[t0:tf])
    else:
        out[t0:tf]=np.interp(tout[t0:tf],tin[t0i:tfi],din[t0i:tfi])
    return out


############################ NEON UPSCALE ################################
# turn a higher resolution time series into a lower resolution time series
# by averaging
def nupscale(tout,tin,din,outscl=None,maxdelta=60,nearest=True,nanth=.2,debug=False):
    dlt=int(np.nanmin(tout[1:]-tout[0:-1])/60)
    if outscl in [None]:
        outscl=dlt

    if debug:
        dbg='::::::::::DEBUG::::::::::::\n'
        dbg=dbg+'outscale: '+str(outscl)+'\n'+\
                'dlt     : '+str(dlt)+'\n'+\
                'len(out): '+str(len(tout))+'\n'+\
                '::::::::::DEBUG::::::::::::'
        print(dbg)

    # if nearest, will use a constant value over the entire averaging period
    # if input is continous
    if np.min(tout[1:]-tout[0:-1])==np.max(tout[1:]-tout[0:-1]):
        inscl=int(np.nanmin(tin[1:]-tin[0:-1])/60)

        # interpolate to 1 minute, then average up
        tmid=np.linspace(tout[0]-outscl*30,tout[-1]+outscl*30,dlt*(len(tout)-1)+outscl+1)
        dmid=ninterp(tmid,tin,din,maxdelta=max(maxdelta,outscl),nearest=nearest,debug=debug)

        if debug:
            dbg='::::::::::DEBUG::::::::::::\n'
            dbg=dbg+'tmid[0]   : '+str(tmid[0])+'\n'+\
                'tmid[-1]  : '+str(tmid[-1])+'\n'+\
                'delta(tmd): '+str(tmid[1]-tmid[0])+'\n'+\
                'len(tmd)  : '+str(len(tmid))+'\n'+\
                '::::::::::DEBUG::::::::::::'
            print(dbg)

        out=np.zeros((len(tout),))
        nancnt=np.zeros((len(tout),))
        dmid[dmid==-9999]=float('nan')

        # get a count of nans in each output averaging period
        for i in range(outscl):
            nancnt=nancnt+np.isnan(dmid[i::dlt])[0:len(tout)]

        # average data, ignoring nans as long as nancnt is less than [nanth]%
        for i in range(outscl):
            data=dmid[i::dlt][0:len(tout)]/(outscl-nancnt+.000000000001)
            data[np.isnan(data)]=0
            out=out+data
        out[nancnt>nanth*(outscl)]=float('nan')
    else:
        raise RuntimeError('Output timeseries is either non-continuous,'+\
                'or uses variable delta.')

    return out

############################ GETBINS ####################################
# Get equal size bins
def getbins(A,B=None,n=31):
    if not (B is None):
        bina=getbins(A,n=n)
        binb=np.zeros((n-1,n))
        for i in range(n-1):
            m=(A>bina[i])&(A<bina[i+1])
            binb[i,:]=getbins(B[m],n=n)
        return bina,binb
    else:
        C=np.sort(A)
        bins=[]
        for i in np.linspace(0,len(A)-1,n):
            i=int(i)
            bins.append(C[i])
        return bins


############################## SORT TOGETHER #############################
# Sort 2 or more lists (arrays) based on the content of one array
def sort_together(X,Y):
    if len(np.array(Y).shape)==1:
        Y=Y[:,None]
        Y=Y.T
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

############################################################################
########################### COLORING #######################################
def get_cmap_ani():
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
            newcolvals[i*(inbetween_color_amount)+1:(i+1)*\
                    (inbetween_color_amount)+1,j] = flow
    return ListedColormap(newcolvals, name='from_list', N=None)

def cani_norm(x,vmn_a=.1,vmx_a=.7):
    cmapa=get_cmap_ani()
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
    return cmapa(x_)


#############################################################################
################################# GET PHI ###################################
''' VARS
    'UU'
    'VV'
    'WW'
    'TT'
    'QQ'
    'CC'
'''
def get_phi(fp,var,m=None):
    if m in [None]:
        m=np.ones((len(fp['TIME'][:]),)).astype(bool)
    if ((len(var)==2) and (var[0]==var[1])) or (var=='THETATHETA'):
        if var in ['UU','VV','WW']:
            varstar=fp['USTAR'][:][m]
        elif var in ['THETATHETA']:
            varstar=1/fp['USTAR'][:][m]*(fp['WTHETA'][:][m])
        else:
            varstar=1/fp['USTAR'][:][m]*(fp['W'+var[0]][:][m])
        phi=np.sqrt(fp[var][m])/np.abs(varstar)
    return phi

def get_phio(var,stab,fp=None,zL=None):
    if stab:
        vv=var+'s'
    else:
        vv=var+'u'
    if zL is None:
        try:
            zL=fp['zL'][:]
        except KeyError:
            zL=fp['zzd'][:]/fp['L_MOST'][:]

    match vv:
        case 'UUu':
            phio=2.55*(1-3*zL)**(1/3)
        case 'UUs':
            phio=2.06*np.ones(zL.shape)
        case 'VVu':
            phio=2.05*(1-3*zL)**(1/3)
        case 'VVs':
            phio=2.06*np.ones(zL.shape)
        case 'WWu':
            phio=1.35*(1-3*zL)**(1/3)
        case 'WWs':
            phio=1.6*np.ones(zL.shape)
        case 'THETATHETAu':
            phio=.99*(.067-zL)**(-1/3)
            phio[zL>-0.05]=.015*(-zL[zL>-0.05])**(-1)+1.76
        case 'THETATHETAs':
            phio=0.00087*(zL)**(-1.4)+2.03
        case 'QQu':
            phio=np.sqrt(30)*(1-25*zL)**(-1/3)
        case 'QQs':
            phio=2.74*np.ones(zL.shape)
        case 'CCu':
            phio=np.sqrt(30)*(1-25*zL)**(-1/3)
        case 'CCs':
            phio=2.74*np.ones(zL.shape)
    return phio


#############################################################################
################################ TIME STUFF #################################
def get_hours(time,utcoff):
    utcoff=np.array(utcoff)
    time=np.array(time)
    time=time+utcoff*60*60
    second_of_day=time%86400
    return second_of_day/60/60

##############################################################################
################################ LUMLEY PREP #################################
def _iqr(data):
    return np.nanpercentile(data,75)-np.nanpercentile(data,25)

def datagrid(x,y,z,nbin=31,fxn=np.nanmedian,gridtype='equalx',mincnt=None):
    ''' get bin values by x and y
        gridtype: ['linear','equalx','equaly']
                   'equalx' x spacing equal, y spacing equal for each x bin
                   'equaly' y spacing equal, x spacing equal for each y bin
    '''
    if mincnt is None:
        mincnt=max(len(x)/nbin**2*.05,10)
    if gridtype == 'equalx':
        binx,biny=getbins(x,y,nbin)
    elif gridtype in ['linear','lin']:
        binx=np.linspace(np.nanmin(x),np.nanmax(x),nbin)
        biny=np.linspace(np.nanmin(y),np.nanmax(y),nbin)
        biny=[biny]*(nbin-1)
        biny=np.array(biny)
    out = []
    if len(z.shape)==1:
        idxs=1
    elif len(z.shape)==2:
        idxs=z.shape[0]
    for v in range(idxs):
        oo=np.zeros((nbin-1,nbin-1))
        xx=np.zeros((nbin-1,nbin-1))
        yy=np.zeros((nbin-1,nbin-1))
        for i in range(nbin-1):
            m=(x>binx[i])&(x<binx[i+1])
            for j in range(nbin-1):
                m2=m&(y>biny[i,j])&(y<biny[i,j+1])
                if idxs==1:
                    z_=z
                else:
                    z_=z[v]
                xx[i,j]=np.nanmedian(x[m2])
                yy[i,j]=np.nanmedian(y[m2])
                if xx[i,j]!=xx[i,j]:
                    xx[i,j]=(binx[i]+binx[i+1])/2
                if yy[i,j]!=yy[i,j]:
                    yy[i,j]=(biny[i,j]+biny[i,j+1])/2
                oo[i,j]=fxn(z_[m2])
                cnt=np.sum(m2)
                if cnt<mincnt:
                    oo[i,j]=float('nan')
        if idxs==1:
            return xx,yy,oo
        out.append(oo)
        print('ISSUE')
        break
    return xx,yy,out






##############################################################################
################################# PLOTTING ###################################
def get_errorlines(x,y,c,xbins,cbins,minpct=0.00001):
    Nc=len(cbins)-1
    Nx=len(xbins)-1
    n=len(x)

    # check if xbins is increasing; if not, flip it
    if xbins[1]<xbins[0]:
        xbins=np.flip(xbins)

    ctrue=np.ones((Nc,))*float('nan')
    xtrue=np.ones((Nc,Nx))*float('nan')
    ytrue=np.ones((Nc,Nx))*float('nan')
    y25=np.ones((Nc,Nx))*float('nan')
    y75=np.ones((Nc,Nx))*float('nan')

    for i in range(Nc):
        mc=(c>cbins[i])&(c<=cbins[i+1])
        ctrue[i]=np.nanmedian(c[mc])
        for j in range(Nx):
            m=mc&(x>xbins[j])&(x<=xbins[j+1])
            if np.nansum(m)/n<minpct:
                xtrue[i,j]=float('nan')
                ytrue[i,j]=float('nan')
                y25[i,j]=float('nan')
                y75[i,j]=float('nan')
            else:
                ym=y[m]
                xtrue[i,j]=np.nanmedian(x[m])
                ytrue[i,j]=np.nanpercentile(ym,50)
                y25[i,j]=np.nanpercentile(ym,25)
                y75[i,j]=np.nanpercentile(ym,75)

    return ytrue,xtrue,ctrue,y25,y75

##############################################################################
########################### SITEBAR #########################################
def sitebar(fig,ax,sites,data1,data2,color,hatch=None,issorted=False,\
        ymin=None,ymax=None,vmin='pct',vmax='pct',colorvals=None,\
        cbar=None,legend=None,fntsm=7,fntlg=10,lw=1,xticklbls=True):
    ''' Make a sitebar graph
        data2: bar colored data
        data1: error spline data
        color: can be N length colors, or N length integers/strings etc if
               used with colorvals
        colorvals: translates "color" to actual colors
        legend: can be a dictionary of kwargs for legend function or None for
                no legend. kwargs can include patches etc. for custom legend
        can also supply "sorted data" to cover the first 5 arguments

    '''
    if not (cbar is None):
        raise NotImplementedError('Colorbar is not yet implemented')

    # booleans for how color is working
    if colorvals is None:
        direct_color=True
    elif type(colorvals) in [dict]:
        direct_color=False
        cmap=False
    elif type(colorvals) in [str]:
        direct_color=False
        cmap=True

    # hatches
    if hatch is None:
        hatch=np.zeros((len(sites),))

    # setup sorting
    to_sort=[sites,data2]
    if len(np.array(color).shape)>1:
        for i in range(np.array(color).shape[1]):
            to_sort.append(color[:,i])
    else:
        to_sort.append(color)
    to_sort.append(hatch)
    nn=len(to_sort)
    if not issorted:
        X,Y=sort_together(data1,to_sort)
    else:
        X=data1
        Y=to_sort

    # reconstruct color
    if direct_color:
        n=nn-3
        color=np.zeros((len(Y[0]),n))
        for i in range(2,2+n):
            color[:,i]=Y[i]
    elif cmap:

        c0=Y[2]
        if vmin=='pct':
            vmin=np.nanpercentile(color,2.5)
        if vmax=='pct':
            vmax=np.nanpercentile(color,97.5)
        cnorm=(c0-vmin)/(vmax-vmin)
        cmp=matplotlib.cm.get_cmap(colorvals)
        color=cmp(cnorm)

    else:
        c0=Y[2]
        color=[]
        for c in c0:
            if c in colorvals.keys():
                color.append(colorvals[c])
            elif 'default' in colorvals.keys():
                color.append(colorvals['default'])
            else:
                color.append(float('nan'))
        color=np.array(color)
        color=matplotlib.colors.to_rgba_array(color)

    # construct sorted data
    sorted_data=(Y[0],X,Y[1],color,Y[nn-1],True)

    # convert hatch
    hatch=[]
    for i in range(len(X)):
        hc=Y[nn-1][i]
        if hc==1:
            hatch.append('OOOO')
        elif hc==0.5:
            hatch.append('....')
        else:
            hatch.append('')

    # final data prep
    yerr=[np.zeros((len(X),)),np.array(X[:])-np.array(Y[1])]
    yerr=np.array(yerr)
    yerr[0,:][yerr[1,:]<0]=yerr[1,:][yerr[1,:]<0]*(-1)
    yerr[1,:][yerr[1,:]<0]=0

    # plot!
    a=ax.bar(Y[0],Y[1],color=color,hatch=hatch,edgecolor='black',yerr=yerr,\
            capsize=3,error_kw={'lw':lw,'capthick':lw},linewidth=lw)
    if xticklbls:
        ax.set_xticks(np.linspace(0,len(X)-1,len(X)),Y[0],rotation=45,fontsize=fntsm)
    else:
        ax.set_xticks(np.linspace(0,len(X)-1,len(X)),[])
    ax.tick_params(labelsize=fntsm)
    ax.set_xlim(-.5,len(X))


    # ylim determination
    if ymax is None:
        actual_max=max(np.nanmax(Y[1]),np.nanmax(X))
        pct90=max(np.nanpercentile(Y[1],90),np.nanpercentile(X,90))
        if (actual_max>pct90*2):
            cutoff=True
            ymax=pct90*1.03
        else:
            cutoff=False
            ymax=actual_max*1.03
    if ymin is None:
        actual_min=min(np.nanmin(Y[1]),np.nanmin(X))
        if actual_min<0:
            ymin=actual_min*1.03
        else:
            ymin=actual_min-(ymax-actual_min)*.1
            if ymin<0:
                ymin=0

    # handle cutoffs
    if cutoff:
        for i in range(len(X)):
            if (X[i]>ymax) or (Y[1][i]>ymax):
                mxmx=max(X[i],Y[1][i])
                ax.text(i-.25,ymax-(ymax-ymin)*1/8,str(mxmx)[0:4],rotation=90,fontsize=fntsm)

    # legend construction
    if legend is None:
        pass
    elif type(legend) == dict:
        ax.legend(fontsize=fntsm,title_fontsize=fntlg,**legend)

    # sorted data must be (sites,data1,data2,color,issorted)
    return fig,ax,sorted_data


#############################################################################
########################## LUMLEY TRIANGLE PLOTTING  ########################
def lumley(fig,ax,xb,yb,data,cmap='Spectral_r',leftedge=False,\
        bottomedge=False,vmin='abspct',vmax='abspct',shading='gouraud',cbarlabel='',fntsm=8,fntlg=12):
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
    ax.plot([x1ps, x2ps], [y1ps, y2ps], '-k', linewidth=2)

    #for k in np.arange(3):
    #    ax.text(lbpx[k], lbpy[k], labels[k], ha='center', fontsize=12)
    label_side1 = 'Prolate'
    label_side2 = 'Oblate'
    label_side3 = 'Two-component'
    #axs.text((xc[1]+xc[2])/2, (yc[0]+yc[2])/2+0.08*lc, label_side1, ha='center', va='center', rotation=-65)
    #axs.text((xc[0]+xc[2])/2, (yc[1]+yc[2])/2+0.08*lc, label_side2, ha='center', va='center', rotation=65)
    #axs.text((xc[0]+xc[1])/2, (yc[0]+yc[1])/2-0.04*lc, label_side3, ha='center', va='center')
    if vmax=='abspct':
        vmax=max(np.abs(np.nanpercentile(data,5)),np.abs(np.nanpercentile(data,95)))
    elif vmax=='pct':
        vmax=np.nanpercentile(data,95)
    if vmin=='abspct':
        vmin=-vmax
    elif vmin=='pct':
        vmin=np.nanpercentile(data,5)
    im=ax.pcolormesh(xb,yb,data,cmap='Spectral_r',shading=shading,vmin=vmin,vmax=vmax)
    cb=fig.colorbar(im, cax=ax.inset_axes([0.95, 0.05, 0.05, .92]),label=cbarlabel)
    cb.set_label(label=cbarlabel,fontsize=fntlg)
    cb.ax.zorder=100
    ax.patch.set_alpha(0)
    cb.ax.tick_params(labelsize=fntsm)
    ax.grid(False)
    if leftedge:
        ax.text(-.07,-.02,'0',horizontalalignment='right',fontsize=fntsm)
        ax.text(.18,np.sqrt(3)/4-.02,r'$\sqrt{3}/4$',horizontalalignment='right',fontsize=fntsm)
        ax.text(.43,np.sqrt(3)/2-.02,r'$\sqrt{3}/2$',horizontalalignment='right',fontsize=fntsm)
        ax.set_ylabel('$y_b$',fontsize=fntlg)
    ax.set_yticks([],[])
    if bottomedge:
        ax.set_xticks([0,.2,.4,.6,.8,1],['',.2,.4,.6,.8,''],fontsize=fntsm)
        ax.set_xlabel('$x_b$',fontsize=fntlg)
    else:
        ax.set_xticks([],[])
    ax.spines[['left', 'bottom']].set_visible(False)
    return fig,ax

def plt_scaling(zeta,phi,c,ymin,ymax,varlist,\
        figaxs=None,p25=None,p75=None,sc_zeta=None,sc_phi=None,\
        stab=[False,True],zeta_range=[10**-3.5,10**1.3],plot_old=True,\
        colorbar='yb',width=6,skip=False,xyscale='semilog',\
        linearg={'marker':'o','markersize':2,'linewidth':.2},\
        scatarg={},wthscl=1,fntsm=10,fntlg=14,ticrot=0):
    '''
    zeta,phi   : (2,N,M,K) array; 2 columns, N variables, M lines, K points
    p25,p75    : (^) error bar bottom and top
    sc_zeta    : (2,N,L) scatterplot points if using
    sc_phi     : (2,N,L) scatterplot points if using
    c          : (2,N,M) array
    ymin       : (N) array of minimum y values for each variable
    ymax       : (N) array of maximum y values for each variable
    varlist    : (N) array of variables
    figaxs     : list [fig,axs] to supply a figure/axes group ahead
    stab       : stability of each column
    zeta_range : range of zeta values (xlims)
    colobar    : 'empty','yb','none' for an emtpy axis, yb colorbar, or no cb
    width      : width in inches of plot
    skip       : if true, will skip the first line (yb very low)
    xyscale    : string or (N) array of strings; semilog or loglog
    plot_old   : if true, will plot phi_old for the variable
    linearg    : arguments to be passed to the basic lines
    scatarg    : arguments to be passed to the scatter
    '''

    N=zeta.shape[1]
    M=zeta.shape[2]
    K=zeta.shape[3]
    ylabels={'UU':r'$\Phi_u$','VV':r'$\Phi_v$','WW':r'$\Phi_w$',\
             'THETATHETA':r'$\Phi_{\theta}$','QQ':r'$\Phi_q$','CC':r'$\Phi_c$'}

    if colorbar in ['empty','yb']:
        L=3
        wdrt=[1,1,.1]
    else:
        L=2
        wdrt=[1,1]

    if type(xyscale)==str:
        xyscale=[xyscale]*N
    print(xyscale)
    if figaxs is None:
        fig,axs=plt.subplots(N,L,figsize=(width,wthscl*width*1.125/3*N),gridspec_kw={'width_ratios':wdrt})
    else:
        fig=figaxs[0]
        axs=figaxs[1]

    for j in range(N):
        var=varlist[j]
        for s in range(2):
            yplt=phi[s,j,:,:]
            anic=c[s,j,:]
            if stab[s]:
                sts=1
            else:
                sts=-1

            # set axis
            ax=axs[j,s]

            # plot scatter
            if not ((sc_zeta is None)|(sc_phi is None)):
                ax.scatter(sts*sc_zeta[s,j,:],sc_phi[s,j,:],**scatarg)

            for i in range(M):
                if skip and (i==0):
                    continue
                xplt=zeta[s,j,i,:]
                # plot phi
                if xyscale[j]=='semilog':
                    ax.semilogx(sts*xplt,yplt[i,:],color=cani_norm(anic[i]),\
                            zorder=4,**linearg)
                elif xyscale[j]=='loglog':
                    ax.loglog(sts*xplt,yplt[i,:],color=cani_norm(anic[i]),\
                            zorder=4,**linearg)

                # plot fill
                if (p25 is None) or (p75 is None):
                    pass
                else:
                    ax.fill_between(sts*xplt,p25[s,j,i,:],p75[s,j,i,:],\
                            color=cani_norm(anic[i]),alpha=.15,zorder=3)

            # plot old
            if plot_old:
                zLi=sts*np.logspace(np.log10(zeta_range[0]),np.log10(zeta_range[1]))
                phio=get_phio(var,stab[s],zL=zLi)
                if xyscale[j]=='semilog':
                    ax.semilogx(sts*zLi,phio[:],'--',color='k',linewidth=1.5,zorder=5)
                elif xyscale[j]=='loglog':
                    ax.loglog(sts*zLi,phio[:],'--',color='k',linewidth=1.5,zorder=5)

            # labeling
            if j==(N-1):
                ax.tick_params(which="both", bottom=True)
                locmin = matplotlib.ticker.LogLocator(base=10.0,\
                        subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 ))
                ax.xaxis.set_minor_locator(locmin)
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                if stab[s]:
                    lbl=[r' $10^{-3}$','',r' $10^{-1}$','',r' $10^{1}$']
                    xlbl=r'$\zeta$'
                else:
                    lbl=[r'$-10^{-3}$','',r'$-10^{-1}$','',r'$-10^{1}$']
                    xlbl=r'$-\zeta$'
                ax.set_xticks([10**-3,10**-2,10**-1,1,10],lbl,rotation=ticrot)
                ax.set_xlim(zeta_range[0],zeta_range[1])
                ax.set_xlabel(xlbl,fontsize=fntlg)
            else:
                locmin = matplotlib.ticker.LogLocator(base=10.0,\
                        subs=(0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10 ))
                ax.xaxis.set_minor_locator(locmin)
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                ax.set_xticks([10**-3,10**-2,10**-1,1,10],[])
                ax.set_xlim(zeta_range[0],zeta_range[1])
            ax.set_ylim(ymin[j],ymax[j])

            if s==0:
                ax.set_ylabel(ylabels[var],fontsize=fntlg)
            else:
                ax.tick_params(labelleft=False)
            if not stab[s]:
                ax.invert_xaxis()

            ax.tick_params(axis='both',which='major',labelsize=fntsm)

        # colorbar stuff
        if (colorbar is None) or (colorbar=='none'):
            continue
        elif colorbar=='empty':
            ax=axs[j,2]
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.grid(False)
        elif colorbar=='yb':
            anicc=cani_norm(np.array([.05,.15,.25,.35,.45,.55,.65,.75]))
            ax=axs[j,2]
            ax.imshow(anicc.reshape(8,1,4),origin='lower',interpolation=None)
            ax.set_xticks([],[])
            inta=[-.05]
            inta.extend([.05,.15,.25,.35,.45,.55,.65,.75])
            inta.extend([.85])
            xtc=np.interp([.1,.2,.3,.4,.5,.6,.7],inta,np.linspace(-1,8,10))
            ax.set_yticks(xtc,[.1,.2,.3,.4,.5,.6,.7])
            ax.grid(False)
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(r'$y_b$',fontsize=fntsm)
            ax.tick_params(axis='both',which='major',labelsize=10)

    fig.subplots_adjust(hspace=.08,wspace=.02)
    return fig, axs



