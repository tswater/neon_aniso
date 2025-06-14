# Core tools to be imported and used elsewhere
import numpy as np

############## CONSTANTS ################
SITES =['ABBY', 'BARR', 'BART', 'BLAN', 'BONA', 'CLBJ', 'CPER', 'DCFS', 'DEJU',  'DELA', 'DSNY', 'GRSM', 'GUAN', 'HARV', 'HEAL', 'JERC', 'JORN', 'KONA', 'KONZ',  'LAJA', 'LENO', 'MLBS', 'MOAB', 'NIWO', 'NOGP', 'OAES', 'ONAQ', 'ORNL', 'OSBS',  'PUUM', 'RMNP', 'SCBI', 'SERC', 'SJER', 'SOAP', 'SRER', 'STEI', 'STER', 'TALL',  'TEAK', 'TOOL', 'TREE', 'UKFS', 'UNDE', 'WOOD', 'WREF', 'YELL']



################### FUNCTION ########################

#####################################################
def static2full():
    ''' Take site information (L1 attrs) and turn into timeseries '''

#####################################################

############################# NEON INTERP ###############################
# Interpolate data from one timeseries to another
def ninterp(tout,tin,din,maxdelta=60):
    ''' Interpolate data, with a special mind for gaps
        tout     : timeseries to interpolate to
        tin      : time of data
        din      : data; all non-usable data should be 'NaN'
        maxdelta : maximum gap to interpolate with; larger will be filled
                   with NaN
    '''

    # check if the output time resolution is bigger than input
    tdelta=tin[1:]-tin[0:-1]
    toutdelta=tout[1:]-tout[0:-1]
    if np.nanmin(toutdelta)>np.nanmin(tdelta):
        print('WARNING! Output resolution bigger than input; averaging with nupscale suggested')

    # split into smaller timeseries based on gaps
    splits=np.where(tdelta>maxdelta*60)
    # FIXME

############################ NEON UPSCALE ################################
# turn a higher resolution time series into a lower resolution time series
# by averaging
# FIXME
def nupscale(tout,tin,din):
    return

