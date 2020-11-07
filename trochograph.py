#!/usr/bin/env python
"""
trochograph is a test particle tracing code designed to work nicely with
TRISTAN-MP flds and prtl outputs.  Charged particles in uniform fields follow
trochoid trajectories, hence "trochograph", a wheel-curve drawing program.
--Aaron Tran, 2020 July 16

trochograph_mainloop.py contains routines for moving particles and stuff
"""

from __future__ import division, print_function

from datetime import datetime
import numpy as np
import os
#from os import path

import h5py
import numba
#from scipy.interpolate import interpn, RegularGridInterpolator
#from interp3d import interp_3d

class Fields(object):
    pass


class Particles(object):
    pass


#def run_trochograph(p,flds,par,user_prtl_bc):
def run_trochograph(user_input, user_flds, user_prtl, user_prtl_bc):
    """Main execution loop of trochograph"""

    # -----------------------------------------
    # initialize

    t0 = datetime.now()
    tprint("Initializing input, flds, prtl")

    par = user_input()
    flds = user_flds(par)
    p = user_prtl(par)

    last        = par['last']
    lapst       = par['lapst']
    c           = par['c']
    qm          = par['qm']

    assert len(p.x.shape) == 1
    tprint("  lapst =",lapst, "last =",last)
    tprint("  flds shape =", flds.ex.shape)
    tprint("  prtl number =", p.x.size)
    tprint("  qm =", par['qm'])

    tprint("Pre-enforce prtl BCs")
    user_prtl_bc(p,flds.ex.shape)

    tprint("Get prtl starting flds for initial output")
    # E-component interpolations:
    p.ex = interp5(flds.ex,p.x,p.y,p.z)
    p.ey = interp5(flds.ey,p.x,p.y,p.z)
    p.ez = interp5(flds.ez,p.x,p.y,p.z)
    # B-component interpolations:
    p.bx = interp5(flds.bx,p.x,p.y,p.z)
    p.by = interp5(flds.by,p.x,p.y,p.z)
    p.bz = interp5(flds.bz,p.x,p.y,p.z)


    tprint("Initial output")
    if not os.path.exists("output"):
        os.mkdir("output")
    #os.makedirs("output", exist_ok=True)  # needs python >=3.2
    output(p,par,lapst)

    # -----------------------------------------
    # main loop

    t1 = datetime.now()
    tlaptot = 0
    tlaprestmov = 0
    tlaprestout = 0
    tlapfirst = 0

    for lap in range(lapst+1, last+1):

        tlap0 = datetime.now()
        tprint("Lap {:10d}".format(lap),end='')

        #mover(p,flds,qm,c)
        mover2(
            flds.bx,flds.by,flds.bz,flds.ex,flds.ey,flds.ez,
            p.x,p.y,p.z,p.u,p.v,p.w,
            p.wtot,p.wprl,p.wx,p.wy,p.wz,
            p.ex,p.ey,p.ez,p.bx,p.by,p.bz,
            qm,c
        )
        user_prtl_bc(p,flds.ex.shape)

        tlap1 = datetime.now()

        fwrote = output(p,par,lap)

        tlap2 = datetime.now()

        # lap stdout and time accounting

        dtlap1_0 = (tlap1-tlap0).total_seconds()
        dtlap2_1 = (tlap2-tlap1).total_seconds()
        dtlap2_0 = (tlap2-tlap0).total_seconds()

        tprint("  move {:.3e} out {:.3e} tot {:.3e}".format(
            dtlap1_0,
            dtlap2_1,
            dtlap2_0,
        ))
        if fwrote:
            tprint("  wrote", fwrote)

        tlaptot += dtlap2_0
        if lap == lapst+1:
            tlapfirst = dtlap2_0
        else:
            tlaprestmov += dtlap1_0
            tlaprestout += dtlap2_1

    # -----------------------------------------
    # finalize

    t2 = datetime.now()

    tprint("Done, total time:", (t2-t0).total_seconds())
    tprint("  init time", (t1-t0).total_seconds())
    tprint("  loop time", (t2-t1).total_seconds())
    tprint("    first lap", tlapfirst)
    tprint("    rest laps", tlaptot - tlapfirst)
    tprint("      rest mover", tlaprestmov)
    tprint("      rest output", tlaprestout)

    tprint("Numba threading")
    tprint("  NUM_THREADS", numba.config.NUMBA_NUM_THREADS)
    tprint("  DEFAULT_NUM_THREADS", numba.config.NUMBA_DEFAULT_NUM_THREADS)
    tprint("  get_num_threads()", numba.get_num_threads())
    tprint("  THREADING_LAYER", numba.config.THREADING_LAYER)
    tprint("  threading_layer()", numba.threading_layer())

    #interp5.parallel_diagnostics(level=4)
    #mover2.parallel_diagnostics(level=4)

    return


def mover(p,flds,qm,c):
    """Boris particle mover"""

    cinv=1./c

    ## E-component interpolations:
    #ex0 = interp(flds.ex,p.x,p.y,p.z) * 0.25*qm
    #ey0 = interp(flds.ey,p.x,p.y,p.z) * 0.25*qm
    #ez0 = interp(flds.ez,p.x,p.y,p.z) * 0.25*qm
    ## B-component interpolations:
    #bx0 = interp(flds.bx,p.x,p.y,p.z) * 0.125*qm*cinv
    #by0 = interp(flds.by,p.x,p.y,p.z) * 0.125*qm*cinv
    #bz0 = interp(flds.bz,p.x,p.y,p.z) * 0.125*qm*cinv

    # E-component interpolations:
    ex0 = interp5(flds.ex,p.x,p.y,p.z) * 0.5*qm
    ey0 = interp5(flds.ey,p.x,p.y,p.z) * 0.5*qm
    ez0 = interp5(flds.ez,p.x,p.y,p.z) * 0.5*qm
    # B-component interpolations:
    bx0 = interp5(flds.bx,p.x,p.y,p.z) * 0.5*qm*cinv
    by0 = interp5(flds.by,p.x,p.y,p.z) * 0.5*qm*cinv
    bz0 = interp5(flds.bz,p.x,p.y,p.z) * 0.5*qm*cinv

    # First half electric acceleration, with relativity's gamma:
    u0 = c*p.u+ex0
    v0 = c*p.v+ey0
    w0 = c*p.w+ez0

    # --- START DIAGNOSTIC E \cdot v = E \cdot (v_{pre-rot} + v_{post-rot})/2 ---
    ev = ex0*u0+ey0*v0+ez0*w0  # first part of Edotv work in mover
    evx = ex0*u0
    evy = ey0*v0
    evz = ez0*w0
    # --- END DIAGNOSTIC E \cdot v = E \cdot (v_{pre-rot} + v_{post-rot})/2 ---

    # First half magnetic rotation, with relativity's gamma:
    g = c/(c**2+u0**2+v0**2+w0**2)**0.5
    bx0 = g*bx0
    by0 = g*by0
    bz0 = g*bz0
    f = 2./(1.+bx0*bx0+by0*by0+bz0*bz0)
    u1 = (u0+v0*bz0-w0*by0)*f
    v1 = (v0+w0*bx0-u0*bz0)*f
    w1 = (w0+u0*by0-v0*bx0)*f
    ## Second half mag. rot'n & el. acc'n:
    #u0 = u0 + v1*bz0-w1*by0+ex0
    #v0 = v0 + w1*bx0-u1*bz0+ey0
    #w0 = w0 + u1*by0-v1*bx0+ez0

    # break {second half mag. rot'n & el. acc'n} into two pieces
    # in order to compute diagnostic E \cdot v
    # without E dot v diagnostic, we would normally do all in one chunk

    # Second half mag. rot'n
    u0=u0+v1*bz0-w1*by0
    v0=v0+w1*bx0-u1*bz0
    w0=w0+u1*by0-v1*bx0

    # --- START DIAGNOSTIC E \cdot v = E \cdot (v_{pre-rot} + v_{post-rot})/2 ---
    # E_prll \cdot v = (E \cdot B) (B \cdot v) / B^2
    # Restore 2x to ex0 and remove Lorentz gamma.
    # v_prll, Lorentz gamma are unchanged by mag rotation,
    # so OK to compute either before or after.
    evprl = 2*(ex0*bx0+ey0*by0+ez0*bz0) * g*(bx0*u0+by0*v0+bz0*w0) / (bx0**2+by0**2+bz0**2)
    p.wprl = p.wprl + evprl*cinv*cinv  # store as dimensionless Lorentz gamma

    # E \cdot v = E \cdot (v_{pre-rot} + v_{post-rot})/2
    ev = g*(ev + ex0*u0+ey0*v0+ez0*w0)  # second part, remove Lorentz gamma
    p.wtot = p.wtot + ev*cinv*cinv  # store as dimensionless Lorentz gamma

    evx = g*(evx + ex0*u0)
    evy = g*(evy + ey0*v0)
    evz = g*(evz + ez0*w0)
    p.wx = p.wx + evx*cinv*cinv
    p.wy = p.wy + evy*cinv*cinv
    p.wz = p.wz + evz*cinv*cinv
    # --- END DIAGNOSTIC E \cdot v = E \cdot (v_{pre-rot} + v_{post-rot})/2 ---

    # Second half el. acc'n:
    u0=u0+ex0
    v0=v0+ey0
    w0=w0+ez0

    # Normalized 4-velocity advance:
    p.u = u0*cinv
    p.v = v0*cinv
    p.w = w0*cinv
    # Position advance:
    g = c/(c**2+u0**2+v0**2+w0**2)**0.5
    p.x = p.x + p.u*g*c
    p.y = p.y + p.v*g*c
    p.z = p.z + p.w*g*c

    return


# up-front compile time is big!!  (~10 seconds),
# but this dramatically cuts mover time from ~{4-6}e-4 to {1-1.5}e-4 sec per particle-lap
@numba.njit()#cache=True,parallel=True)
def mover2(
        bx,by,bz,ex,ey,ez,
        px,py,pz,pu,pv,pw,
        pwtot,pwprl,pwx,pwy,pwz,
        pex,pey,pez,pbx,pby,pbz,
        qm,c
):
    """Boris particle mover"""

    cinv=1./c

    ## E-component interpolations:
    #ex0 = interp(flds.ex,p.x,p.y,p.z) * 0.25*qm  # TRISTAN interp has 2x factor due to Yee mesh edge-centering; see Buneman's comments
    #ey0 = interp(flds.ey,p.x,p.y,p.z) * 0.25*qm
    #ez0 = interp(flds.ez,p.x,p.y,p.z) * 0.25*qm
    ## B-component interpolations:
    #bx0 = interp(flds.bx,p.x,p.y,p.z) * 0.125*qm*cinv  # TRISTAN interp has 4x factor due to Yee mesh face-centering; see Buneman's comments
    #by0 = interp(flds.by,p.x,p.y,p.z) * 0.125*qm*cinv
    #bz0 = interp(flds.bz,p.x,p.y,p.z) * 0.125*qm*cinv

    # E-component interpolations:
    ex0 = interp5(ex,px,py,pz) * 0.5*qm
    ey0 = interp5(ey,px,py,pz) * 0.5*qm
    ez0 = interp5(ez,px,py,pz) * 0.5*qm
    # B-component interpolations:
    bx0 = interp5(bx,px,py,pz) * 0.5*qm*cinv
    by0 = interp5(by,px,py,pz) * 0.5*qm*cinv
    bz0 = interp5(bz,px,py,pz) * 0.5*qm*cinv

    # First half electric acceleration, with relativity's gamma:
    u0 = c*pu+ex0
    v0 = c*pv+ey0
    w0 = c*pw+ez0

    # --- START DIAGNOSTIC E \cdot v = E \cdot (v_{pre-rot} + v_{post-rot})/2 ---
    ev = ex0*u0+ey0*v0+ez0*w0  # first part of Edotv work in mover
    evx = ex0*u0
    evy = ey0*v0
    evz = ez0*w0
    # --- END DIAGNOSTIC E \cdot v = E \cdot (v_{pre-rot} + v_{post-rot})/2 ---

    # First half magnetic rotation, with relativity's gamma:
    g = c/(c**2+u0**2+v0**2+w0**2)**0.5
    bx0 = g*bx0
    by0 = g*by0
    bz0 = g*bz0
    f = 2./(1.+bx0*bx0+by0*by0+bz0*bz0)
    u1 = (u0+v0*bz0-w0*by0)*f
    v1 = (v0+w0*bx0-u0*bz0)*f
    w1 = (w0+u0*by0-v0*bx0)*f
    ## Second half mag. rot'n & el. acc'n:
    #u0 = u0 + v1*bz0-w1*by0+ex0
    #v0 = v0 + w1*bx0-u1*bz0+ey0
    #w0 = w0 + u1*by0-v1*bx0+ez0

    # break {second half mag. rot'n & el. acc'n} into two pieces
    # in order to compute diagnostic E \cdot v
    # without E dot v diagnostic, we would normally do all in one chunk

    # Second half mag. rot'n
    u0=u0+v1*bz0-w1*by0
    v0=v0+w1*bx0-u1*bz0
    w0=w0+u1*by0-v1*bx0

    # --- START DIAGNOSTIC E \cdot v = E \cdot (v_{pre-rot} + v_{post-rot})/2 ---
    # E_prll \cdot v = (E \cdot B) (B \cdot v) / B^2
    # Restore 2x to ex0 and remove Lorentz gamma.
    # v_prll, Lorentz gamma are unchanged by mag rotation,
    # so OK to compute either before or after.
    evprl = 2*(ex0*bx0+ey0*by0+ez0*bz0) * g*(bx0*u0+by0*v0+bz0*w0) / (bx0**2+by0**2+bz0**2)
    pwprl[:] = pwprl + evprl*cinv*cinv  # store as dimensionless Lorentz gamma

    # E \cdot v = E \cdot (v_{pre-rot} + v_{post-rot})/2
    ev = g*(ev + ex0*u0+ey0*v0+ez0*w0)  # second part, remove Lorentz gamma
    pwtot[:] = pwtot + ev*cinv*cinv  # store as dimensionless Lorentz gamma

    evx = g*(evx + ex0*u0)
    evy = g*(evy + ey0*v0)
    evz = g*(evz + ez0*w0)
    pwx[:] = pwx + evx*cinv*cinv
    pwy[:] = pwy + evy*cinv*cinv
    pwz[:] = pwz + evz*cinv*cinv
    # --- END DIAGNOSTIC E \cdot v = E \cdot (v_{pre-rot} + v_{post-rot})/2 ---

    # Second half el. acc'n:
    u0=u0+ex0
    v0=v0+ey0
    w0=w0+ez0

    # Normalized 4-velocity advance:
    pu[:] = u0*cinv
    pv[:] = v0*cinv
    pw[:] = w0*cinv
    # Position advance:
    g = c/(c**2+u0**2+v0**2+w0**2)**0.5
    px[:] = px + pu*g*c
    py[:] = py + pv*g*c
    pz[:] = pz + pw*g*c

    # Save fields used (NOTICE THAT THERE'S AN OFFSET!!!!!... fields used at PREVIOUS LOCATION, or "in-between" locations if you like)
    pex[:] = ex0 / (0.5*qm)
    pey[:] = ey0 / (0.5*qm)
    pez[:] = ez0 / (0.5*qm)
    pbx[:] = bx0 / (0.5*qm*cinv)
    pby[:] = by0 / (0.5*qm*cinv)
    pbz[:] = bz0 / (0.5*qm*cinv)

    return


#def interp(fld,x,y,z):
#    """Linearly interpolate fld to input x,y,z position(s)
#    Inputs:
#        fld = scalar fld, dimensions must match
#        x, y, z = position(s) in TRISTAN code units
#            If using prtl.tot coords, you must subtract off ghost cells.
#    Output:
#        fld value(s) at x,y,z
#    """
#    # x, y, z coords for a regular grid
#    #gridp = (np.arange(0, fld.shape[0]),
#    #         np.arange(0, fld.shape[1]),
#    #         np.arange(0, fld.shape[2]))
#
#    # x, y, z coords for a regular grid with periodic edges attached
#    gridp = (np.arange( 0, fld.shape[0]    ),
#             np.arange(-1, fld.shape[1] + 1),
#             np.arange(-1, fld.shape[2] + 1))
#
#    # tack periodic edges onto cross-shock y- and z-dimension
#    # WARNING TODO this assumes a certain set of user prtl BCs...
#    fldp = np.concatenate((fld[:,-1:,:], fld, fld[:,0:1,:]), axis=1)
#    fldp = np.concatenate((fldp[:,:,-1:], fldp, fldp[:,:,0:1]), axis=2)
#
#    return interpn(gridp,  # shapes are (m1,), (m2,), (m3,)
#                   fldp,  # shape is (m1, m2, m3)
#                   np.array([x, y, z]).T,  # shape is (anything, 3)
#                   method='linear')  # returns shape (anything)


#global interper
#interper = {}
#def interp2(fld,x,y,z,name=None):
#    if name not in interper:
#        # x, y, z coords for a regular grid with periodic edges attached
#        gridp = (np.arange( 0, fld.shape[0]    ),
#                 np.arange(-1, fld.shape[1] + 1),
#                 np.arange(-1, fld.shape[2] + 1))
#        # tack periodic edges onto cross-shock y- and z-dimension
#        # WARNING TODO this assumes a certain set of user prtl BCs...
#        fldp = np.concatenate((fld[:,-1:,:], fld, fld[:,0:1,:]), axis=1)
#        fldp = np.concatenate((fldp[:,:,-1:], fldp, fldp[:,:,0:1]), axis=2)
#        interper[name] = RegularGridInterpolator(
#            gridp,  # shapes are (m1,), (m2,), (m3,)
#            fldp,  # shape is (m1, m2, m3)
#            method='linear',
#        )
#    myfldinterper = interper[name]
#    dat = np.array([x,y,z]).T  # might be inefficient
#    return myfldinterper(dat)


#global interper
#interper = {}
#def interp3(fld,x,y,z,name=None):
#    if name not in interper:
#        # x, y, z coords for a regular grid with periodic edges attached
#        gridp = (np.arange( 0, fld.shape[0]    ),
#                 np.arange(-1, fld.shape[1] + 1),
#                 np.arange(-1, fld.shape[2] + 1))
#        # tack periodic edges onto cross-shock y- and z-dimension
#        # WARNING TODO this assumes a certain set of user prtl BCs...
#        fldp = np.concatenate((fld[:,-1:,:], fld, fld[:,0:1,:]), axis=1)
#        fldp = np.concatenate((fldp[:,:,-1:], fldp, fldp[:,:,0:1]), axis=2)
#        interper[name] = interp_3d.Interp3D(
#            fldp,  # shape is (m1, m2, m3)
#            gridp[0],  # shapes are (m1,), (m2,), (m3,)
#            gridp[1],
#            gridp[2],
#        )
#    myfldinterper = interper[name]
#    #dat = np.array([x,y,z]).T  # might be inefficient
#    #return myfldinterper(dat)
#    return myfldinterper((x,y,z))


#def interp4(fld,x,y,z):
#    """Linearly interpolate fld to input x,y,z position(s)
#    Inputs:
#        fld = scalar fld, dimensions must match
#        x, y, z = position(s) in TRISTAN code units
#            If using prtl.tot coords, you must subtract off ghost cells.
#    Output:
#        fld value(s) at x,y,z
#    """
#    i=x.astype(np.int)
#    dx=x-i
#    j=y.astype(np.int)
#    dy=y-j
#    k=z.astype(np.int)
#    dz=z-k
#
#    ix=1
#    iy=fld.shape[0]
#    iz=fld.shape[0]*fld.shape[1]
#    l=i+iy*(j-1)+iz*(k-1)
#
#    # Field interpolations are tri-linear (linear in x times linear in y
#    # times linear in z). This amounts to the 3-D generalisation of "area
#    # weighting".
#    f=fld.flatten(order='F')  # match buneman's convention
#    return (
#            (1-dx)*(1-dy)*(1-dz)*f[l]
#          +    dx *(1-dy)*(1-dz)*f[l+ix]
#          + (1-dx)*   dy *(1-dz)*f[l+iy]
#          +    dx *   dy *(1-dz)*f[l+ix+iy]
#          + (1-dx)*(1-dy)*   dz *f[l+iz]
#          +    dx *(1-dy)*   dz *f[l+ix+iz]
#          + (1-dx)*   dy *   dz *f[l+iy+iz]
#          +    dx *   dy *   dz *f[l+ix+iy+iz]
#    )


@numba.njit()#cache=True,parallel=True)
def interp5(fld,x,y,z):
    """Linearly interpolate fld to input x,y,z position(s)

    Particle x,y,z has domain:
        x in [0, fld.shape[0]-1)
        y in [0, fld.shape[1]-1)
        z in [0, fld.shape[2]-1)

    Inputs:
        fld = scalar fld, dimensions must match (x,y,z)
        x, y, z = position(s) in TRISTAN code units
            If using prtl.tot coords, you must subtract off ghost cells.
            REQUIRES x,y,z to be 1-D arrays
    Output:
        fld value(s) at x,y,z
    """
    #assert len(x.shape) == 1
    #assert x.shape == y.shape
    #assert x.shape == z.shape

    #f=fld.flatten(order='F')  # match buneman's convention
    #ix=1
    #iy=fld.shape[0]
    #iz=fld.shape[0]*fld.shape[1]

    # numba only supports C ordering...
    # but for a 2-D x-y run with shape[2]==2, interpolation with C-ordering is
    # actually more contiguous in memory.  but idk if contiguity matters,
    # I think compilers can optimize for strided access too...
    f=np.ravel(fld)#,order='C')  # ravel=view, flatten=copy
    ix=fld.shape[1]*fld.shape[2]
    iy=fld.shape[2]
    iz=1

    fout=np.empty_like(x, dtype=np.float64)  # in case user happens to give all ints, casting can result in rather subtle errors

    for ip in range(x.size):
        i=int(x[ip])
        dx=x[ip]-i
        j=int(y[ip])
        dy=y[ip]-j
        k=int(z[ip])
        dz=z[ip]-k
        #l=i+iy*(j-1)+iz*(k-1)  # fortran 1-based index
        #l=i+iy*j+iz*k  # fortran column-major ordering
        l=i*ix+j*iy+k*iz  # row-major ordering

        # Field interpolations are tri-linear (linear in x times linear in y
        # times linear in z). This amounts to the 3-D generalisation of "area
        # weighting".
        fout[ip] = (
              (1.-dx)*(1.-dy)*(1.-dz)*f[l]
            +     dx *(1.-dy)*(1.-dz)*f[l+ix]
            + (1.-dx)*    dy *(1.-dz)*f[l+iy]
            +     dx *    dy *(1.-dz)*f[l+ix+iy]
            + (1.-dx)*(1.-dy)*    dz *f[l+iz]
            +     dx *(1.-dy)*    dz *f[l+ix+iz]
            + (1.-dx)*    dy *    dz *f[l+iy+iz]
            +     dx *    dy *    dz *f[l+ix+iy+iz]
        )

        #print("interp5: x,y,z=",x[ip],y[ip],z[ip])
        #print("interp5: i,j,k=",i,j,k)
        #print("interp5: dx,dy,dz=",dx,dy,dz)
        #print()
        #print("interp5: l         =", l           )
        #print("interp5: l+ix      =", l+ix        )
        #print("interp5: l+iy      =", l+iy        )
        #print("interp5: l+ix+iy   =", l+ix+iy     )
        #print("interp5: l+iz      =", l+iz        )
        #print("interp5: l+ix+iz   =", l+ix+iz     )
        #print("interp5: l+iy+iz   =", l+iy+iz     )
        #print("interp5: l+ix+iy+iz=", l+ix+iy+iz  )
        #print()
        #print("interp5: f[l]         =", f[l]           )
        #print("interp5: f[l+ix]      =", f[l+ix]        )
        #print("interp5: f[l+iy]      =", f[l+iy]        )
        #print("interp5: f[l+ix+iy]   =", f[l+ix+iy]     )
        #print("interp5: f[l+iz]      =", f[l+iz]        )
        #print("interp5: f[l+ix+iz]   =", f[l+ix+iz]     )
        #print("interp5: f[l+iy+iz]   =", f[l+iy+iz]     )
        #print("interp5: f[l+ix+iy+iz]=", f[l+ix+iy+iz]  )
        #print("interp5: f[l]          weight=", (1.-dx)*(1.-dy)*(1.-dz) )
        #print("interp5: f[l+ix]       weight=",     dx *(1.-dy)*(1.-dz) )
        #print("interp5: f[l+iy]       weight=", (1.-dx)*    dy *(1.-dz) )
        #print("interp5: f[l+ix+iy]    weight=",     dx *    dy *(1.-dz) )
        #print("interp5: f[l+iz]       weight=", (1.-dx)*(1.-dy)*    dz  )
        #print("interp5: f[l+ix+iz]    weight=",     dx *(1.-dy)*    dz  )
        #print("interp5: f[l+iy+iz]    weight=", (1.-dx)*    dy *    dz  )
        #print("interp5: f[l+ix+iy+iz] weight=",     dx *    dy *    dz  )
        #print()
        #print("interp5: fout=", fout[ip])

    return fout


def output(p,par,lap):
    """Write outputs"""

    interval    = par['interval']
    last        = par['last']
    pltstart    = par['pltstart']

    if lap % interval != 0:
        return False

    n = int(lap/interval) - int(pltstart/interval)  # starts at 0000
    fname = "output/troch.{:04d}".format(n)

    # savez_compressed is really slow;
    # savez is really big.  Only use for testing/debugging etc
#    np.savez_compressed(
#    np.savez(
#        fname,
#        x=p.x, y=p.y, z=p.z,
#        u=p.u, v=p.v, w=p.w,
#        proc=p.proc, ind=p.ind,
#        # --- START DIAGNOSTIC E \cdot v = E \cdot (v_{pre-rot} + v_{post-rot})/2 ---
#        wtot=p.wtot, wprl=p.wprl, wx=p.wx, wy=p.wy, wz=p.wz,
#        # --- END DIAGNOSTIC E \cdot v = E \cdot (v_{pre-rot} + v_{post-rot})/2 ---
#    )

    # gzip doesn't help much
    with h5py.File(fname, "w") as f:
        f.create_dataset("x", data=p.x)#, compression="gzip")
        f.create_dataset("y", data=p.y)#, compression="gzip")
        f.create_dataset("z", data=p.z)#, compression="gzip")
        f.create_dataset("u", data=p.u)#, compression="gzip")
        f.create_dataset("v", data=p.v)#, compression="gzip")
        f.create_dataset("w", data=p.w)#, compression="gzip")
        f.create_dataset("proc", data=p.proc)#, compression="gzip")
        f.create_dataset("ind", data=p.ind)#, compression="gzip")
        # --- START DIAGNOSTIC E \cdot v = E \cdot (v_{pre-rot} + v_{post-rot})/2 ---
        f.create_dataset("wtot", data=p.wtot)#, compression="gzip")
        f.create_dataset("wprl", data=p.wprl)#, compression="gzip")
        f.create_dataset("wx", data=p.wx)#, compression="gzip")
        f.create_dataset("wy", data=p.wy)#, compression="gzip")
        f.create_dataset("wz", data=p.wz)#, compression="gzip")
        # --- END DIAGNOSTIC E \cdot v = E \cdot (v_{pre-rot} + v_{post-rot})/2 ---
        f.create_dataset("bx", data=p.bx)#, compression="gzip")
        f.create_dataset("by", data=p.by)#, compression="gzip")
        f.create_dataset("bz", data=p.bz)#, compression="gzip")
        f.create_dataset("ex", data=p.ex)#, compression="gzip")
        f.create_dataset("ey", data=p.ey)#, compression="gzip")
        f.create_dataset("ez", data=p.ez)#, compression="gzip")

    return fname


def tprint(*args, **kwargs):
    #tstr = (datetime.now().strftime("%m-%d-%Y %H:%M:%S.%f"))[:-3]
    #print("[troch {}] ".format(tstr), end='')
    return print(*args, **kwargs)