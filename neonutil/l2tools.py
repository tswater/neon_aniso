# tools for L2 processing, mostly used by L2 driver script


# enable the choice of streamwise or earth coordinates and change
#   variable names. (or both)
# should probably make a "case" class

#####################################################################
#####################################################################
######################## CASE DETAILS ###############################


#####################################################################
#####################################################################
##################### INTERNAL FUNCTIONS ############################
# Functions internal to L2 tools

#### OUTPUT to L2 H5
def _out_to_h5():
    return

#### CONVERT VARLIST
# iterate through varlist to "remove" groups (i.e. profiles)
def _convert_varlist():
    return

#### GET USER CONFIRMATION
def _confirm_user(msg):
    while True:
        user_input = input(f"{msg} (Y/N): ").strip().lower()
        if user_input in ('y', 'yes'):
            return True
        elif user_input in ('n', 'no'):
            return False
        else:
            print("Invalid input. Please enter 'Y' or 'N'.")





#####################################################################
#####################################################################
###################### CONSTRUCTION FUNCTIONS #######################
# Functions for building and manipulating L2 file and mask
def maskgen(fp,mask,cvar=None,flags=None,precip=None,stb=None,limvars=None,\
            counter=None,months=None,years=None):
    ''' Generate a Mask '''
    if flags not in [None,[]]:
        for flag in flags:
            mask=mask&(~np.isnan(fp[flag][:]))
            mask=mask&(fp[flag][:]==0)
    if cvar not in [None,[]]:
        for var in cvars:
            n0=np.sum(mask)/len(mask)*100
            mask=mask&(~np.isnan(fp[var][:]))
            mask=mask&(fp[var][:]!=-9999)
    if limvars not in [None,{},[]]:
        for var in limvars:
            mn=limvars[var][0]
            mx=limvars[var][0]
            if mn not in [float('nan'),None]:
                mask=mask&(fp[var][:]>=mn)
            if mx not in [float('nan'),None]:
                mask=mask&(fp[var][:]<=mx)
    if stb not in [None]:
        if stb:
            mask=mask&(fp['L_MOST'][:]>0)
        if not stb:
            mask=mask&(fp['L_MOST'][:]<0)
    if precip not in [None,False]:
        mask=mask&(fp['P']<=0)
    if counter not in [None,False]:
        raise NotImplementedError('Masking countergradient fluxes TBI')
    yrbool= (year not in [None,[]])
    mnbool= (month not in [None,[]])
    if yrbool | mnbool:
        time = fp['TIME'][:]
        yrmsk=[]
        mnmsk=[]
        d0=datetime.datetime(1970,1,1,0,0)
        for t in time:
            dt = d0+datetime.timedelta(seconds=t)
            if yrbool:
                yrmsk.append(dt.year in year)
            else:
                yrmsk.append(True)
            if mnbool:
                mnmsk.append(dt.month in month)
            else:
                mnmsk.append(True)
        mask=mask&np.array(yrmsk)
        mask=mask&np.array(mnmsk)


    return mask


##############################################################
def time_maskgen():
    ''' Generate a mask based on times '''

#############################################################
def build_L2_file():
    ''' Generate an empty L2 file '''

#############################################################
def casegen(case):
    ''' Essentially a driver for constructing a case, including
        calling maskgen, add (L1) data, static information. See case
        details near the top for options for the case
    '''


#####################################################################
#####################################################################
###################### DATA FUNCTIONS ###############################
# Functions for adding new data


##############################################################
def add_foot():
    ''' Add footprint statistics/information. SLOW! '''

#############################################################
def pull_var():
    ''' Tries to pull an L2 variable from existing L2 file '''

##############################################################
def add_grad():
    ''' Add gradients from profiles to L2 '''

###############################################################
def add_from_l1():
    ''' Add a new L1 variable to an existing L2 file'''

##############################################################
def remove_var():
    ''' Remove variable from L2 file'''

