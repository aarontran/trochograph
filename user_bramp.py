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

from trochograph import Fields, Particles, run_trochograph, tprint

# global prtl arrays: struct w/ p.{x,y,z,u,v,w,ind,proc}
# global fld arrays: flds.{ex,ey,ez,bx,by,bz}
# global scalar constants: par.{c,qm}

# user must provide parameters:  c, interval, lapst, last, pltstart, qm
# and also fields and prtls, of course

def user_input():
    """Define a bunch of input parameters"""
    par = {}

    # <time>
    par['last']     = 20000
    par['interval'] = 20

    # <boundaries>
    par['periodicx'] = False # True=periodic, False="outflow" (escaping particles get NaN-ed)
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
    par['qm'] = qme
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
    shape = (200, 1, 10)  # for interval=20, lecs travel at most 9 cells in one step, so 10 cells allows to reconstruct x,z traj
    btheta = 65*np.pi/180  # angle between B and shock normal in radians
    rRH = 2  # density jump
    # ramp at x=100+/-30 cell
    xx = np.arange(shape[0])
    ramp_profile = (1 + (rRH-1)*0.5*(np.tanh((100-xx)/30)+1))  # 1D array

    # numpy default should be C ordering already, but force just to be safe
    # row- vs column-order seems to affect numba+interp performance at ~10% level  (~3e-4 vs 3.5e-4, for 1000 prtl on shape=(100+1,10+1,1+1))
    ones = np.ones(shape, dtype=np.float64, order='C')
    zeros = np.zeros(shape, dtype=np.float64, order='C')

    flds = Fields()
    flds.ex = zeros
    flds.ey = zeros
    flds.ez = +0.01 * par['Binit']*np.sin(btheta) * ones  # constant, so force "shock" frame
    flds.bx = par['Binit']*np.cos(btheta) * ones
    flds.by = par['Binit']*np.sin(btheta) * ramp_profile[:,np.newaxis,np.newaxis] * ones
    flds.bz = zeros

    #plt.plot(xx,np.mean(flds.by,axis=(-2,-1)))
    #plt.show()

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

    nprtl = 1000
    dimf = flds.ex.shape

    p.x = np.random.uniform(190, 191, size=nprtl)
    p.y = np.random.uniform(  0, dimf[1]-1, size=nprtl)
    p.z = np.random.uniform(  0, dimf[2]-1, size=nprtl)
    p.u = -0.01 + np.random.normal(0, scale=0.05/3**0.5, size=nprtl)
    p.v =         np.random.normal(0, scale=0.05/3**0.5, size=nprtl)
    p.w =         np.random.normal(0, scale=0.05/3**0.5, size=nprtl)

    p.proc = np.zeros((nprtl,))
    p.ind = np.arange(nprtl)

    return p


if __name__ == '__main__':
    print("Trochograph started from:", path.basename(__file__))
    run_trochograph(user_input, user_flds, user_prtl)
