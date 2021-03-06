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

from mark import TristanRun
from trochograph import Fields, Particles, run_trochograph, tprint

# global prtl arrays: struct w/ p.{x,y,z,u,v,w,ind,proc}
# global fld arrays: flds.{ex,ey,ez,bx,by,bz}
# global scalar constants: par.{c,qm}

# user must provide parameters:  c, interval, lapst, last, pltstart, qm
# and also fields and prtls, of course

RUNDIR = "/rigel/astro/users/at3222/aaron_heating/mi625Ms7betap0.25theta90_2d_gamma1.67_dgamr0.5_comp20_my1200_ntimes64ppc128_later_wxyz"
RUN = TristanRun(RUNDIR)
SCENE = RUN[0]

# 2020 july 29 - apply boost into shock frame...
U0 = 0.023243835670501263
USH = 0.03632950746848023  # Gamma=5/3 assumed
VBOOST = USH - U0  # boost from lab to shock frame, so vboost points along +\hat{x}

SMOOTH = False  # whether to smooth fields along y-axis or not...

def user_input():
    """Define a bunch of input parameters"""
    par = {}

    # <time>
    #par['last']     = SCENE.lap() + 160000 # need 400x400 laps to get through shock... rightnow 800 laps = 2.8sec, so 160000 laps =560 sec = 9.33 minutes
    par['last']     = SCENE.lap() + 400000 # need 400x400 laps to get through shock... rightnow 800 laps = 2.8sec, so 160000 laps =560 sec = 9.33 minutes
                                           # actually, NEED MUCH MORE cuz shock is not moving forward... so progress is slow
    par['interval'] = 400

    # <boundaries>
    par['periodicx'] = False # True=periodic, False="outflow" (escaping particles get NaN-ed)
    par['periodicy'] = True  # True=periodic, False="outflow" (escaping particles get NaN-ed)
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
    tflds = SCENE.flds('b','e', xsel=slice(None,3500))#slice(3000,4500))

    # I think mark reads as Fortran (column-major) ordered array stored as
    # [z,y,x] contiguous; Python-level axis transpose to [x,y,z] is a view.
    # So force C ordering in memory.
    # row- vs column-order seems to affect numba+interp performance at ~10% level  (~3e-4 vs 3.5e-4, for 1000 prtl on dimf=(100+1,10+1,1+1))
    flds = Fields()
    flds.ex = np.ascontiguousarray(tflds['e'][0])
    flds.ey = np.ascontiguousarray(tflds['e'][1])
    flds.ez = np.ascontiguousarray(tflds['e'][2])
    flds.bx = np.ascontiguousarray(tflds['b'][0])
    flds.by = np.ascontiguousarray(tflds['b'][1])
    flds.bz = np.ascontiguousarray(tflds['b'][2])

    # 2020 july 29 - apply boost into shock frame... (for yshock simulation)
    vboost_fld = np.zeros_like(tflds['e'])
    vboost_fld[0,:,:] = VBOOST
    eboost = np.cross(vboost_fld, tflds['b'], axis=0)
    flds.ex = flds.ex + eboost[0]
    flds.ey = flds.ey + eboost[1]
    flds.ez = flds.ez + eboost[2]

    # 2020 july 31 - collapse the y axis!!!!!
    if SMOOTH:
        flds.ex = np.mean(flds.ex,axis=-2)[:,np.newaxis,:]  # need to restore y-axis after collapsing
        flds.ey = np.mean(flds.ey,axis=-2)[:,np.newaxis,:]
        flds.ez = np.mean(flds.ez,axis=-2)[:,np.newaxis,:]
        flds.bx = np.mean(flds.bx,axis=-2)[:,np.newaxis,:]
        flds.by = np.mean(flds.by,axis=-2)[:,np.newaxis,:]
        flds.bz = np.mean(flds.bz,axis=-2)[:,np.newaxis,:]

    return flds


def user_prtl(flds):
    """
    Return p.{x,y,z,u,v,w} to initialize prtl
    flds = fields /with/ ghost cells attached, in case needed for prtl init
        user is responsible for initializing prtls in correct region,
        accounting for presence or absence of ghost cells.
    """

    tprtl = SCENE.prtl('xe','ye','ze','ue','ve','we','proce','inde')

    sel = np.logical_and(tprtl['xe'] >= 3000, tprtl['xe'] < 3010)  # narrow slice of 0.5 c/omp from hi-res prtl tracking run

    # in cell coordinates
    p = Particles()

    p.x = tprtl['xe'][sel] #- 1000 #3000  # because of finite field selection
    p.y = tprtl['ye'][sel]
    p.z = tprtl['ze'][sel]
    p.u = tprtl['ue'][sel]
    p.v = tprtl['ve'][sel]
    p.w = tprtl['we'][sel]

    # 2020 july 31 - collapse the y axis!!!!!
    #if SMOOTH:
    #    p.y = np.mod(p.y, 1)  # shouldn't be needed, actually
    #                          # the code should enforce prtl bc before entering
    #                          # the evolution loop

    # 2020 july 29 - apply boost into shock frame... (for yshock simulation)
    p.u = p.u - VBOOST

    p.proc = tprtl['proce'][sel]
    p.ind  = tprtl['inde'][sel]

    return p


if __name__ == '__main__':
    print("Trochograph started from:", path.basename(__file__))
    run_trochograph(user_input, user_flds, user_prtl)
