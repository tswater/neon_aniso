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


# ---------------------------------- #
#           Driver Options           #
# ---------------------------------- #





# ----------------------------------- #
#                CASE                 #
# ----------------------------------- #
case={}
case['name']      = '' # descriptive name for the case
case['scale']     = 0 # scale for the case
case['l1dir']     = '' # l1 directory for input
case['fpath']     = '' # filepath
case['stab']      = # None, True, False for (All, stable, unstable)
case['basecase']  = None # unclear here...
case['core_vars'] = [] # list of variables that must have all points valid
case['core_q']    = [] # list of quality flags that must be goo
case['limvars']   = {} # dictionary of VAR:[MIN,MAX] pairs; limits must be met
case['months']    = [] # list of months to include; empty means all
case['years']     = [] # list of years to include; empty means all
case['sites']     = [] # list of sites to include; empty means all
case['precip']    = False # if True, will exclude timesteps with precipitation
case['wind_sys']  = '' # if wind is in [streamwise, earth] coordinate system

case['exclude']   = [] # list of variables to exclude from L1
case['include']   = [] # list of variables to include from L1
case['zeta']      = [] # list of zeta related variables to include;
                       # ['ZL','L_MOST','zzd','z']
case['static']    = [] # list of static variables to pass over; default all
case['conv_nan']  = True # if True, will convert -9999 to NaN on variables
case['counter']   = False # if True, will exclude countergradient fluxes.
                          # can also be a dictionary of variables



# ----------------------------------- #
#              RUN CASEGEN            #
# ----------------------------------- #



# ----------------------------------- #
#                ADD DATA             #
# ----------------------------------- #




