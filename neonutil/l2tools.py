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
def maskgen():
    ''' Generate a Mask '''

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

