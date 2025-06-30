# Core tools to be imported and used elsewhere
import numpy as np
from scipy.interpolate import interp1d

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
def static2full():
    ''' Take site information (L1 attrs) and turn into timeseries '''

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

############################## SORT TOGETHER #############################
# Sort 2 or more lists (arrays) based on the content of one array
def sort_together(X,Y):
    Y=np.array(Y)
    if len(Y.shape)==1:
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
    return X,np.array(Yout)




