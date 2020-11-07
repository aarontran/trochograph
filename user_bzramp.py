#!/usr/bin/env python
"""
trochograph is a test particle tracing code designed to work nicely with
TRISTAN-MP flds and prtl outputs.  Charged particles in uniform fields follow
trochoid trajectories, hence "trochograph", a wheel-curve drawing program.
--Aaron Tran, 2020 July 16

trochograph_user.py should be altered by the user to fit their needs.
The user specifies particles, fields, and boundary conditions.

Coordinates:
* x \in [0, dimf[0]), y \in [0, dimf[1]), z \in [0, dimf[2])
* No yee mesh for fields

todo idk what to do about offset issues
todo this is trivially parallelizable...
"""

from __future__ import division, print_function

import numba
import numpy as np
#import matplotlib.pyplot as plt

from trochograph import Fields, Particles, run_trochograph, tprint

# global prtl arrays: struct w/ p.{x,y,z,u,v,w,ind,proc} (all dprec)
#   for prtl tracking: p.{ev,evprl,evx,evy,evz}
# global fld arays: flds.{ex,ey,ez,bx,by,bz}
# global scalar constants: par.{c, qm}

# user must provide parameters:  c, interval, lapst, last, pltstart, qm
# and also fields and prtls, of course


def user_input():
    """Define a bunch of input parameters"""
    par = {}

    # <time>
    par['last']     = 20000
    par['interval'] = 20

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
    par['Binit'] = Binit

    return par


def user_flds(par):
    """
    Return flds.{ex,ey,ez,bx,by,bz} in which to trace prtl
    particle domain is:
      x in [0, fld.shape[0]-1)
      y in [0, fld.shape[1]-1)
      z in [0, fld.shape[2]-1)
    """
    dimf = (200, 10, 1)  # final dimf will have +1 to everything to make grid cells volume filling...

    # numpy default should be C ordering already, but force just to be safe
    # row- vs column-order seems to affect numba+interp performance at ~10% level  (~3e-4 vs 3.5e-4, for 1000 prtl on dimf=(100+1,10+1,1+1))
    ones = np.ones(dimf, dtype=np.float64, order='C')
    zeros = np.zeros(dimf, dtype=np.float64, order='C')

    flds = Fields()
    flds.ex = zeros
    flds.ey = -0.01 * par['Binit'] * ones  # constant, so force "shock" frame
    flds.ez = zeros
    flds.bx = zeros
    flds.by = zeros
    #flds.bz = par['Binit'] * ones

    # ramp at x=100+/-30 cell
    rRH = 2  # density jump
    xx = np.arange(dimf[0])
    bprofile = par['Binit'] * (1 + (rRH-1)*0.5*(np.tanh((100-xx)/30)+1))
    flds.bz = bprofile[:,np.newaxis,np.newaxis] * ones

    #plt.plot(xx,np.mean(flds.bz,axis=(-2,-1)))
    #plt.show()

    # tack periodic edges onto cross-shock y- and z-dimension, ONLY at "far" edges
    # WARNING this assumes a certain set of user prtl BCs...
    def add_ghost(fld):
        #fldp = np.concatenate(( fld[:,-1:,:],  fld,  fld[:,0:1,:]), axis=1)
        #fldp = np.concatenate((fldp[:,:,-1:], fldp, fldp[:,:,0:1]), axis=2)
        ##fldp = np.concatenate((fldp[-1:,:,:], fldp, fldp[0:1,:,:]), axis=0)
        fldp = np.concatenate(( fld,  fld[:,0:1,:]), axis=1)
        fldp = np.concatenate((fldp, fldp[:,:,0:1]), axis=2)
        #fldp = np.concatenate((fldp, fldp[0:1,:,:]), axis=0)
        return fldp

    flds.ex = add_ghost(flds.ex)
    flds.ey = add_ghost(flds.ey)
    flds.ez = add_ghost(flds.ez)
    flds.bx = add_ghost(flds.bx)
    flds.by = add_ghost(flds.by)
    flds.bz = add_ghost(flds.bz)

    return flds


def user_prtl(par):
    """Return p.{x,y,z,u,v,w} to initialize prtl"""
    # in cell coordinates
    p = Particles()

    # careful, arrays must be dtype=np.float64 (or some kind of float)
    # if ints, results will be bad

    nprtl = 1000

    p.x = np.random.uniform(190, 191, size=nprtl)
    p.y = np.random.uniform(  0,  10, size=nprtl)
    p.z = np.random.uniform(  0,   1, size=nprtl)
    p.u = -0.01 + np.random.normal(0, scale=0.05/3**0.5, size=nprtl)
    p.v =         np.random.normal(0, scale=0.05/3**0.5, size=nprtl)
    p.w =         np.random.normal(0, scale=0.05/3**0.5, size=nprtl)

    p.proc = np.zeros((nprtl,))
    p.ind = np.arange(nprtl)

    p.wtot = np.zeros_like(p.x)
    p.wprl = np.zeros_like(p.x)
    p.wx   = np.zeros_like(p.x)
    p.wy   = np.zeros_like(p.x)
    p.wz   = np.zeros_like(p.x)

    p.ex   = np.zeros_like(p.x)
    p.ey   = np.zeros_like(p.x)
    p.ez   = np.zeros_like(p.x)
    p.bx   = np.zeros_like(p.x)
    p.by   = np.zeros_like(p.x)
    p.bz   = np.zeros_like(p.x)

    return p


@numba.njit(parallel=True)
def user_prtl_bc(px, py, pz, dimf):
    """Given p, dimf; update p according to desired BCs for dimf"""
    for ip in numba.prange(px.size):
        # x boundary condition - enforce no prtl exit domain
        #assert px[ip] >= 0             # asserts prevent numba parallelism
        #assert px[ip] <= dimf[0] - 1
        if px[ip] < 0 or px[ip] > dimf[0]-1:
            px[ip] = np.nan
        # y periodic boundary condition
        #py[ip] = np.mod(py[ip], dimf[1]-1)  # modulo func is slow
        if py[ip] > (dimf[1]-1):
            py[ip] = py[ip] - (dimf[1]-1)
        # z periodic boundary condition
        #pz[ip] = np.mod(pz[ip], dimf[2]-1)  # modulo func is slow
        if pz[ip] > (dimf[2]-1):
            pz[ip] = pz[ip] - (dimf[2]-1)
    return


if __name__ == '__main__':
    run_trochograph(user_input, user_flds, user_prtl, user_prtl_bc)
