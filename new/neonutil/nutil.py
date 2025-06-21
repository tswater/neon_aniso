# Core tools to be imported and used elsewhere
import numpy as np
from scipy.interpolate import interp1d

############## CONSTANTS ################
SITES =['ABBY', 'BARR', 'BART', 'BLAN', 'BONA', 'CLBJ', 'CPER', 'DCFS', 'DEJU',  'DELA', 'DSNY', 'GRSM', 'GUAN', 'HARV', 'HEAL', 'JERC', 'JORN', 'KONA', 'KONZ',  'LAJA', 'LENO', 'MLBS', 'MOAB', 'NIWO', 'NOGP', 'OAES', 'ONAQ', 'ORNL', 'OSBS',  'PUUM', 'RMNP', 'SCBI', 'SERC', 'SJER', 'SOAP', 'SRER', 'STEI', 'STER', 'TALL',  'TEAK', 'TOOL', 'TREE', 'UKFS', 'UNDE', 'WOOD', 'WREF', 'YELL']



################### FUNCTION ########################

#####################################################
def static2full():
    ''' Take site information (L1 attrs) and turn into timeseries '''

#####################################################

############################# NSCALE ###################################
# Wrapper function; interpolates (ninterp) or upscales (nupscale) as appropriate
def nscale(tout,tin,din,maxdelta=60,nearest=False:
    tdelta=tin[1:]-tin[0:-1]
    toutdelta=tout[1:]-tout[0:-1]
    if np.nanmin(toutdelta)>np.nanmin(tdelta):
        return nupscale(tout,tin,din,maxdelta)
    else:
        return ninterp(tout,tin,din,maxdelta,nearest)

############################# NEON INTERP ###############################
# Interpolate data from one timeseries to another
def ninterp(tout,tin,din,maxdelta=60,nearest=False):
    ''' Interpolate data, with a special mind for gaps
        tout     : timeseries to interpolate to
        tin      : time of data
        din      : data; all non-usable data should be 'NaN'
        maxdelta : maximum gap to interpolate with; larger will be filled
                   with NaN
        nearest  : if true, will use nearest neighbor interpolation
    '''

    # check if the output time resolution is bigger than input
    tdelta=tin[1:]-tin[0:-1]
    toutdelta=tout[1:]-tout[0:-1]
    if np.nanmin(toutdelta)>np.nanmin(tdelta):
        print('WARNING! Output resolution coarser than input; averaging with nupscale suggested')

    # split into smaller timeseries based on gaps
    splt_idi=np.where(tdelta>maxdelta*60)[0]
    splt_tmin=tin(splt_idi)
    # convert split to indicies in tout space
    splt_ido=np.interp(splt_tmin,tout,np.linspace(0,len(tout)-1,lent(tout)))
    t0=int(tin[0],tout,np.linspace(0,len(tout)-1,lent(tout)))
    t0i=0

    out=np.ones((len(tout),))*float('nan')

    # loop and interate through to interpolate
    for i in range(len(splt_idi)):
        tf=splt_ido[i]
        tfi=splt_idi[i]
        if nearest:
            interp=interp1d(tin[t0i:tfi],din[t0i:tfi])
            out[t0:tf]=interp(tout[t0:tf])
        else:
            out[t0:tf]=np.interp(tout[t0:tf],tin[t0i:tfi],din[t0i:tfi],\
                             left=float('nan'),right=float('nan'))
        t0=tf
        t0i=tfi

    # do final interpolation
    tf=-1
    tfi=-1
    if nearest:
        interp=interp1d(tin[t0i:tfi],din[t0i:tfi])
        out[t0:tf]=interp(tout[t0:tf])
    else:
        out[t0:tf]=np.interp(tout[t0:tf],tin[t0i:tfi],din[t0i:tfi],\
                         left=float('nan'),right=float('nan'))
    return out


############################ NEON UPSCALE ################################
# turn a higher resolution time series into a lower resolution time series
# by averaging
# FIXME
def nupscale(tout,tin,din,maxdelta=60):
    # if input is continous
    if np.min(tout[1:]-tout[0:-1])==np.max(tout[1:]-tout[0:-1]):
        outscl=int(np.min(tout[1:]-tout[0:-1])/60)

        # interpolate to 1 minute, then average up
        tmid=np.linspace(tout[0],tout[-1],outscl*len(tout))
        dmid=ninterp(tmid,tin,din,maxdelta=max(maxdelta,outscl))

        out=np.zeros((len(tout),))
        for i in range(outscl):
            out=out+dmid[i::outscl]/outscl
    else:
        raise RuntimeError('Output timeseries is either non-continuous, or uses variable delta')

    return out

############################## SORT TOGETHER #############################
# Sort 2 or more lists (arrays) based on the content of one array
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




