#!/usr/bin/env python
"""
user_{...}.py is the main user-facing interface to trochograph, a test-particle
tracing program.  This user file serves as both
* configuration: user specifies input parameters, fields, particles
* main program: calls run_trochograph(...) to start the evolution loop.

Usage:

    NUMBA_NUM_THREADS={xx} python user_{...}.py

"""

from __future__ import division, print_function

import numba
import numpy as np
#import matplotlib.pyplot as plt
from os import path
import sys

from trochograph import Fields, Particles, run_trochograph, tprint

# global prtl arrays: struct w/ p.{x,y,z,u,v,w,ind,proc}
# global fld arrays: flds.{ex,ey,ez,bx,by,bz}
# global scalar constants: par.{c,qm}

# user must provide parameters:  c, interval, lapst, last, pltstart, qm
# and also fields and prtls, of course

# global user parameter
# because I am too tired to engineer a one-time use feature.
#BTHETA = 25  # angle between B and shock normal in degrees
BTHETA = int(sys.argv[1])
tprint("initializing BTHETA=",BTHETA)

def user_input():
    """Define a bunch of input parameters"""
    par = {}

    # <time>
    par['last']     = 4000
    par['interval'] = 20

    # <boundaries>
    #par['periodicx'] = False # True=periodic, False="outflow" (escaping particles get NaN-ed)
    par['boundary_xl'] = "reflect"
    par['boundary_xr'] = "outflow"
    par['periodicy'] = True  # True=periodic, False="outflow" (escaping particles get NaN-ed)
    par['periodicz'] = True  # True=periodic, False="outflow" (escaping particles get NaN-ed)

    # NOTE important requirement: istep=1 is ASSUMED for tracing

    # get from param file
    par['c']        = 0.45
    par['lapst']    = 0  # lap we (re-)start from
    par['pltstart'] = 0  # match output numbering to TRISTAN

    #####################
    # setup charge and q/m in tristan units
    c       = par['c']
    c_omp   = 10    # electron skin depth in cells
    gamma0  = 0.05  # flow drift gamma. If < 1, interpreted as v/c
    me      = 1     # electron mass
    mi      = 49    # ion mass
    ppc0    = 64    # number of particles per cell
    sigma   = 4#0.0948  # magnetization, to set Binit

    if gamma0 < 1:
        gamma0 = (1./(1.-gamma0**2))**0.5

    # note that gamma0 is used in the definition of omega_p to get charge.
    qe  = -((c/c_omp)**2*gamma0)/((ppc0*.5)*(1+me/mi))  # no sqrt because qe/me = 1
    qi  = -qe
    me  = me*abs(qi)  # modifies user input
    mi  = mi*abs(qi)
    qme = qe/me
    qmi = qi/mi

    if gamma0 >= 2:
        Binit = (gamma0*ppc0*.5*c**2*(mi+me)*sigma)**0.5  # relativistic
        tprint("USING RELATIVISTIC BINIT INITIALIZATION")
    else:
        Binit = ((gamma0-1)*ppc0*.5*c**2*(mi+me)*sigma)**0.5  # nonrelativistic
        tprint("USING NON-RELATIVISTIC BINIT INITIALIZATION")
    tprint("Binit=",Binit)

    #####################
    # store the things we actually need in rest of code
    par['qm'] = qme  # choose to just use electrons...
    par['Binit'] = Binit  # not actually used by mover, but hitchike onto dict to get Binit value into user_flds, user_prtl

    return par


def user_flds(par):
    """
    Return flds.{ex,ey,ez,bx,by,bz} in which to trace prtl
    particle domain is:
      x in [0, fld.shape[0]-1)
      y in [0, fld.shape[1]-1)
      z in [0, fld.shape[2]-1)
    """
    # lecs travel <=9 cells over interval=20,
    # so mz0=10 lets user reconstruct x,z trajectory from outputs
    shape = (200, 10, 10)

    # numpy default should be C ordering already, but force just to be safe
    # row- vs column-order seems to affect numba+interp performance at ~10% level
    # (~3e-4 vs 3.5e-4, for 1000 prtl on shape=(100+1,10+1,1+1))
    ones = np.ones(shape, dtype=np.float64, order='C')
    zeros = np.zeros(shape, dtype=np.float64, order='C')

    flds = Fields()
    flds.ex = zeros  # in downstream rest frame, zero ExB speed
    flds.ey = zeros
    flds.ez = zeros
    flds.bx = par['Binit']*np.cos(BTHETA*np.pi/180) * ones
    flds.by = par['Binit']*np.sin(BTHETA*np.pi/180) * ones
    flds.bz = zeros

    return flds


def user_prtl(flds):
    """
    Return p.{x,y,z,u,v,w} to initialize prtl
    flds = fields /with/ ghost cells attached, in case needed for prtl init
        user is responsible for initializing prtls in correct region,
        accounting for presence or absence of ghost cells.
    """
    # in cell coordinates
    p = Particles()

    # careful, arrays must be dtype=np.float64 (or some kind of float)
    # if ints, results will be bad

    nprtl = 10000
    dimf = flds.ex.shape

    p.x = np.random.uniform( 1, 2, size=nprtl)
    p.y = np.random.uniform(  0, dimf[1]-1, size=nprtl)
    p.z = np.random.uniform(  0, dimf[2]-1, size=nprtl)
    p.u = np.random.normal(0, scale=0.0001/3**0.5, size=nprtl)
    p.v = np.random.normal(0, scale=0.0001/3**0.5, size=nprtl)
    p.w = np.random.normal(0, scale=0.0001/3**0.5, size=nprtl)
    #p.u = np.zeros(nprtl)
    #p.v = np.zeros(nprtl)
    #p.w = np.zeros(nprtl)
    # attach a net negative vprll, which induces prtl to reflect at left wall
    p.u = p.u + -0.1*np.cos(BTHETA*np.pi/180)
    p.v = p.v + -0.1*np.sin(BTHETA*np.pi/180)

    # because of my incorrect, non-rel init, enforce v < c...
    vmag = (p.u**2+p.v**2+p.w**2)**0.5
    assert np.all(vmag < 1)

    # attach gamma factor, to convert from 3- to 4-velocity
    gamma=1./(1-p.u**2-p.v**2-p.w**2)**0.5
    p.u = gamma*p.u
    p.v = gamma*p.v
    p.w = gamma*p.w

    p.proc = np.zeros((nprtl,))
    p.ind = np.arange(nprtl)

    return p


if __name__ == '__main__':
    print("Trochograph started from:", path.basename(__file__))
    run_trochograph(user_input, user_flds, user_prtl)
