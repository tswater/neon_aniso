# Driver function for creating and adding to L2 files
# This does NOT include curvefitting information
# Will also create files for adding L2 sub-case and L2 curvefit info

#### Organization:
# 0:OPTIONS: options for running this file note covered by case
# 1:CASE: first you build the 'case' which is a collection of options for the
#         quality assurance etc.
# 2:CASEGEN: generate the case and L2 file
# 3:ADD DATA: this section adds any non-L1 data (data that needs to be
#             compiled here and cannot be pulled from an L1 file)

from neonutil.l2tools import casegen,datagen, add_from_l1

# ---------------------------------- #
#           Driver Options           #
# ---------------------------------- #
include=['UsUs','VsVs','WW','ANID_YBs','ANID_XBs','ANI_YBs','ANI_XBs','USTAR','H','LE','THETA','WTHETA'] # list of variables to include in L2 data (leave this or exclude empty)
exclude=[] # list of variables to exclude from L2 data (will pull in all L1 data
           # except for the specified variables
static=None  # list of static variables to include; if empty will include
           # all L1 attrs
zeta=['zL','zzd','zd','z0']    # list of forms of zeta to include; options are
           # ['zL','L_MOST','z','zd','zzd']
conv_nan=True
overwrite=False
debug=False

# ----------------------------------- #
#                CASE                 #
# ----------------------------------- #
case={}
case['name']      = 'Testing' # descriptive name for the case
case['scale']     = 30 # scale for the case
case['l1dir']     = '/home/tswater/tyche/data/neon/L1/neon_30m/' # l1 directory for input
case['fpath']     = '/home/tswater/tyche/data/neon/L2/L2_ybd/ybd_stable30.h5' # filepath
case['stab']      = True # None, True, False for (All, stable, unstable)
case['basecase']  = None # unclear here...
case['core_vars'] = ['ANID_XBs','ANID_YBs','THETA','WTHETA'] # list of variables that must have all points valid
case['core_q']    = ['qUVW','qH','qUSTAR'] # list of quality flags that must be good
case['limvars']   ={'ST_UsUs_5':[-1,50],'ST_VsVs_5':[-1,50],'USTAR/Us':[.05,100]} # dictionary of VAR:[MIN,MAX] pairs; limits must be met
case['months']    = [] # list of months to include; empty means all
case['years']     = [] # list of years to include; empty means all
case['sites']     = [] # list of sites to include; empty means all
case['precip']    = True # if True, will exclude timesteps with precipitation
case['wind_sys']  = 'streamwise' # if wind is in [streamwise, earth] coordinate system
case['counter']   = False # if True, will exclude countergradient fluxes.
                          # can also be a dictionary of variables

# ----------------------------------- #
#              RUN CASEGEN            #
# ----------------------------------- #
casegen(case,debug)

# ----------------------------------- #
#                ADD DATA             #
# ----------------------------------- #
datagen(case['fpath'],case['l1dir'],include,exclude,\
        static,'C',zeta,conv_nan,overwrite,debug=debug)

ivars = ['f_dsm','f_dtm','f_chm','f_std_dsm','f_std_chm','f_std_dtm',\
            'farea','f_std_lai','f_lai',\
            'f_treecover','f_pct_water','f_nlcd_dom','f_pct_forest']

add_from_l1(case['fpath'],'main',ivars,l1dir=case['l1dir'])

