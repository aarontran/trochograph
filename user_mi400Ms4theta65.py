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

import faulthandler
faulthandler.enable()

from mark import TristanRun
from trochograph import Fields, Particles, interp, run_trochograph, tprint

# global prtl arrays: struct w/ p.{x,y,z,u,v,w,ind,proc}
# global fld arrays: flds.{ex,ey,ez,bx,by,bz}
# global scalar constants: par.{c,qm}

# user must provide parameters:  c, interval, lapst, last, pltstart, qm
# and also fields and prtls, of course

RUNDIR = "/rigel/astro/users/at3222/aaron_heating/mi400Ms4betap0.25theta25phi90_1d_comp20_ntimes64_later"
RUN = TristanRun(RUNDIR)
SCENE = RUN[0]

U0 = 1.6863e-2  # inflow velocity
USH = 0.036684958  # shock velocity, Gamma=5/3 assumed
VBOOST = USH - U0  # non-rel boost from lab to shock frame, so vboost > 0 points along +\hat{x}

def user_input():
    """Define a bunch of input parameters"""
    par = {}

    # <time>
    par['last']     = SCENE.lap() + 400000 # rough guess
    par['interval'] = 200

    # <boundaries>
    par['periodicx'] = True  # True=periodic, False="outflow" (escaping particles get NaN-ed)
    par['periodicy'] = False # True=periodic, False="outflow" (escaping particles get NaN-ed)
    par['periodicz'] = True  # True=periodic, False="outflow" (escaping particles get NaN-ed)

    # get from param file
    par['c']        = SCENE.param['c']  # speed of light
    par['lapst']    = SCENE.lap()  # lap we (re-)start from
    par['pltstart'] = SCENE.param['pltstart']  # match output numbering to TRISTAN
    par['qm']       = -1*SCENE.param['qi']/SCENE.param['me']  # charge to mass ratio, signed

    # istep > 1 is not supported right now, results will be all wrong
    assert SCENE.param['istep'] == 1

    return par


def user_flds(par):
    """
    Return flds.{ex,ey,ez,bx,by,bz} in which to trace prtl
    particle domain is:
      x in [0, fld.shape[0]-1)
      y in [0, fld.shape[1]-1)
      z in [0, fld.shape[2]-1)
    """
    tflds = SCENE.flds('b','e', ysel=slice(None,10000))

    # tile along BOTH z,x to help w/ prtl tracing in post-processing
    # length is 1/2 interval, so c-speed prtl travels < half box
    # be careful abt memory usage
    ones = np.ones( (10, tflds['e'][0].shape[1], 100), dtype=np.float32, order='C')  # single prec

    flds = Fields()
    #flds.ex = np.tile( tflds['e'][0], (100,1,100) )
    #flds.ey = np.tile( tflds['e'][1], (100,1,100) )
    #flds.ez = np.tile( tflds['e'][2], (100,1,100) )
    #flds.bx = np.tile( tflds['b'][0], (100,1,100) )
    #flds.by = np.tile( tflds['b'][1], (100,1,100) )
    #flds.bz = np.tile( tflds['b'][2], (100,1,100) )
    flds.ex = tflds['e'][0] * ones
    flds.ey = tflds['e'][1] * ones
    flds.ez = tflds['e'][2] * ones
    flds.bx = tflds['b'][0] * ones
    flds.by = tflds['b'][1] * ones
    flds.bz = tflds['b'][2] * ones

    print("done tiling e,b")

    # 2020 july 29 - apply boost into shock frame...
    vboost_fld = np.zeros_like(tflds['e'])
    vboost_fld[1,:,:] = VBOOST  # axis=1 for loadlbalance yshock
    eboost = np.cross(vboost_fld, tflds['b'], axis=0)
    #eboost = np.tile( eboost, (100,1,100) )  # have to tile here too?
    flds.ex = flds.ex + eboost[0] * ones
    flds.ey = flds.ey + eboost[1] * ones
    flds.ez = flds.ez + eboost[2] * ones

    return flds


def user_prtl(flds):
    """
    Return p.{x,y,z,u,v,w} to initialize prtl
    flds = fields /with/ ghost cells attached, in case needed for prtl init
        user is responsible for initializing prtls in correct region,
        accounting for presence or absence of ghost cells.
    """

    tprtl = SCENE.prtl('xe','ye','ze','ue','ve','we','proce','inde')

    # First selection: pick prtl slab far ahead of whistler precursor

    sel = np.logical_and(tprtl['ye'] >= 9000, tprtl['ye'] < 9999)#10000)  # for non-periodic BCs, have to trim off edge...

    p = Particles()
    p.x = tprtl['xe'][sel]
    p.y = tprtl['ye'][sel]
    p.z = tprtl['ze'][sel]
    p.u = tprtl['ue'][sel]
    p.v = tprtl['ve'][sel]
    p.w = tprtl['we'][sel]
    p.proc = tprtl['proce'][sel]
    p.ind  = tprtl['inde'][sel]

    # Second selection: pick prtl with vprll < -0.1
    # To do so, break prtl momenta into perp/prll
    # AND, also downsample prtl by 5x to get ~10,000 prtl

    bx = interp(flds.bx,p.x,p.y,p.z)
    by = interp(flds.bx,p.x,p.y,p.z)
    bz = interp(flds.bx,p.x,p.y,p.z)
    b = np.array([bx,by,bz])
    bmag = np.sum(b**2,axis=0)**0.5

    gamma = (1 + p.u**2 + p.v**2 + p.w**2)**0.5
    v3 = np.array([p.u, p.v, p.w]) / gamma  # three-velocity, normed by c (so, actually beta)
    # v_\parallel = \vec{v}\cdot\vec{B} / |\vec{B}| = \vec{v}\cdot\hat{b}
    # \vec{v}_\perp = \vec{v} - (\vec{v}\cdot\hat{b}) \hat{b}
    v3_prll = np.sum(v3*b,axis=0) / bmag  # a signed quantity
    v3_perp = v3 - v3_prll * b / bmag

    sel2 = v3_prll < 0.0

    p.x = p.x[sel2][::5]
    p.y = p.y[sel2][::5]
    p.z = p.z[sel2][::5]
    p.u = p.u[sel2][::5]
    p.v = p.v[sel2][::5]
    p.w = p.w[sel2][::5]
    p.proc = p.proc[sel2][::5]
    p.ind  = p.ind[sel2][::5]

    # Non-rel boost into shock frame... (for yshock simulation)
    p.u = p.u - VBOOST

    return p


if __name__ == '__main__':
    print("Trochograph started from:", path.basename(__file__))
    run_trochograph(user_input, user_flds, user_prtl)
