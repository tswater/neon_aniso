# tools for L2 processing, mostly used by L2 driver script


# enable the choice of streamwise or earth coordinates and change
#   variable names. (or both)
# should probably make a "case" class

#############################################################
def add_from_l1():
    ''' Add a new L1 variable to an existing L2 file'''

##############################################################
def remove_var():
    ''' Remove variable from L2 file'''

##############################################################
def core_q():
    ''' Conduct core quality assurance and return updated mask'''

##############################################################
def make_mask():
    ''' Make mask for a given case '''

##############################################################
def add_foot():
    ''' Add footprint statistics/information. SLOW! '''

##############################################################
def add_grad():
    ''' Add gradients from profiles to L2 '''

##############################################################
def check_old_L2():
    ''' Pull a variable from old L2 file to minimize computation'''
    # essentially, if something like a footprint exists in another L2
    # file, pull it in so nothing needs to be recalculated. Return
    # a mask of what does need to be calculated. Should be called
    # by add_foot and add_grad as an option
