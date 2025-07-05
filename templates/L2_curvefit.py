# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CURVE FITTING TEMPLATE

# %% [markdown]
# There are two main types of curve fitting available:
# - sequential ('2_stage')
# - full
# Full curvefitting is usually better but two stage has its merits
# #### Two stage: 
# Fits parameters a,b,c,d,e in a base function (presumed from the literature) N number of times, for N different levels of anisotropy.
# The resulting best fit values for a,b,c,d and e are then fit with a polynomial curve, resulting in a(yb), b(yb) ... parameters
# #### Full
# Each parameter in a traditional base function (from the literature) is an equation a(yb)=a0+a1*yb+a2*yb**2 ... 
# The user defines a list of parameters to fit (i.e. [a0,a2,b0,b1,b2]) and sets any constants (c=1,d=2,e=1).
# The parameters are then fit across the whole data at once.

# %% [markdown]
# ## How To Run a Two Stage Case
# First set the input values as follows:

# %%
base='cbase' # this is the base function; see neonutil.curvefit.py for details/options
             # You will often want 'cbase', which is cbase(...)=a*(b+c*zL)**(d)+e

const={'a':None,'b':1,'c':-3,'d':1/3,'e':0} # for the parameters in 'base' set their
                                            # values. "None" means you will fit to this parameter

p0=[2.5] # list of initial guesses for parameter values; one for each "None" in const

bnds = ([.1],[10]) # bounds for parameters. Form ([MIN1,MIN2,...],[MAX1,MAX2,...])

loss = 'cauchy' # loss function. For options see scipy.optimize.least_squares documentation for options
                # cauchy is good default for many outliers; arctan as well. linear sometimes if 
                # there are few outliers

deg=[1] # list of polynomial degrees for each parameter. "1" means there will be 2 params (i.e. [a0,a1])

typ='lin' # function applied to yb before fit. Can be 'lin' or 'log'. Log will make params
          # a function of np.log10(yb) instead of (yb)

name='test' # name for your fit

var = 'UU' # variable to fit; in this case sigma U

stab = False # if true, is stable conditions, if false, unstable conditions.

# %%
