#!/usr/bin/env python
"""
Body of code for the program trochograph:
main evolution loop, mover, field interpolation, outputs.
To start a simulation run, call the method run_trochograph(...).
"""

from __future__ import division, print_function

from datetime import datetime
import h5py
import numba
import numpy as np
import os


class Fields(object):
    pass


class Particles(object):
    pass


def run_trochograph(user_input, user_flds, user_prtl, user_prtl_bc):
    """Main execution loop of trochograph"""

    # -----------------------------------------
    # initialize

    t0 = datetime.now()
    tprint("Initializing input, flds, prtl")

    par = user_input()
    flds = user_flds(par)
    dimf = flds.ex.shape  # dimf always means fld shape WITHOUT ghost cells
    p = user_prtl(dimf)

    last        = par['last']
    lapst       = par['lapst']
    c           = par['c']
    qm          = par['qm']

    for arr in [p.x, p.y, p.z, p.u, p.v, p.w]:
        assert arr.ndim  == 1
        assert arr.dtype.kind == 'f'
    for arr in [flds.ex, flds.ey, flds.ez, flds.bx, flds.by, flds.bz]:
        assert arr.ndim  == 3
        assert arr.dtype.kind == 'f'

    tprint("  lapst =",lapst, "last =",last)
    tprint("  flds shape =", flds.ex.shape)
    tprint("  prtl number =", p.x.size)
    tprint("  qm =", par['qm'])

    # -----------------------------------------
    # Apply ghost cells, enforce array memory layout, enforce particle BCs

    flds.ex = add_ghost(flds.ex, par)
    flds.ey = add_ghost(flds.ey, par)
    flds.ez = add_ghost(flds.ez, par)
    flds.bx = add_ghost(flds.bx, par)
    flds.by = add_ghost(flds.by, par)
    flds.bz = add_ghost(flds.bz, par)

    # row- vs column-order seems to affect performance at ~10% level
    # (~3e-4 vs 3.5e-4, for 1000 prtl on dimf=(100+1,10+1,1+1))
    flds.ex = np.ascontiguousarray(flds.ex)
    flds.ey = np.ascontiguousarray(flds.ey)
    flds.ez = np.ascontiguousarray(flds.ez)
    flds.bx = np.ascontiguousarray(flds.bx)
    flds.by = np.ascontiguousarray(flds.by)
    flds.bz = np.ascontiguousarray(flds.bz)

    tprint("Pre-enforce prtl BCs")

    prtl_bc(
        p.x,p.y,p.z,dimf,
        par['periodicx'],par['periodicy'],par['periodicz']
    )

    user_prtl_bc(p.x,p.y,p.z,dimf)

    # -----------------------------------------
    # Initial output

    tprint("Get prtl tracking flds for initial output")
    # E-component interpolations:
    p.ex = interp(flds.ex,p.x,p.y,p.z)
    p.ey = interp(flds.ey,p.x,p.y,p.z)
    p.ez = interp(flds.ez,p.x,p.y,p.z)
    # B-component interpolations:
    p.bx = interp(flds.bx,p.x,p.y,p.z)
    p.by = interp(flds.by,p.x,p.y,p.z)
    p.bz = interp(flds.bz,p.x,p.y,p.z)
    # E dot v work tracking
    p.wtot = np.zeros_like(p.x)
    p.wprl = np.zeros_like(p.x)
    p.wx   = np.zeros_like(p.x)
    p.wy   = np.zeros_like(p.x)
    p.wz   = np.zeros_like(p.x)

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
    tlaprestbc = 0
    tlaprestout = 0
    tlapfirst = 0

    for lap in range(lapst+1, last+1):

        tlap0 = datetime.now()
        tprint("Lap {:10d}".format(lap),end='')

        mover(
            flds.bx,flds.by,flds.bz,flds.ex,flds.ey,flds.ez,
            p.x,p.y,p.z,p.u,p.v,p.w,
            p.wtot,p.wprl,p.wx,p.wy,p.wz,
            p.ex,p.ey,p.ez,p.bx,p.by,p.bz,
            qm,c
        )

        tlap1 = datetime.now()

        prtl_bc(
            p.x,p.y,p.z,dimf,
            par['periodicx'],par['periodicy'],par['periodicz']
        )

        user_prtl_bc(p.x,p.y,p.z,dimf)

        tlap2 = datetime.now()

        fwrote = output(p,par,lap)

        tlap3 = datetime.now()

        # lap stdout and time accounting

        dtlap1_0 = (tlap1-tlap0).total_seconds()  # deltas
        dtlap2_1 = (tlap2-tlap1).total_seconds()
        dtlap3_2 = (tlap3-tlap2).total_seconds()
        dtlap3_0 = (tlap3-tlap0).total_seconds()  # total

        tprint("  move {:.3e} bc {:.3e} out {:.3e} tot {:.3e}".format(
            dtlap1_0,
            dtlap2_1,
            dtlap3_2,
            dtlap3_0,
        ))
        if fwrote:
            tprint("  wrote", fwrote)

        tlaptot += dtlap3_0
        if lap == lapst+1:
            tlapfirst = dtlap3_0
        else:
            tlaprestmov += dtlap1_0
            tlaprestbc += dtlap2_1
            tlaprestout += dtlap3_2

    # -----------------------------------------
    # finalize

    t2 = datetime.now()

    tprint("Done, total time:", (t2-t0).total_seconds())
    tprint("  init time", (t1-t0).total_seconds())
    tprint("  loop time", (t2-t1).total_seconds())
    tprint("    first lap", tlapfirst)
    tprint("    rest laps", tlaptot - tlapfirst)
    tprint("      rest mover", tlaprestmov)
    tprint("      rest bc", tlaprestbc)
    tprint("      rest output", tlaprestout)

    tprint("Numba threading")
    tprint("  NUM_THREADS", numba.config.NUMBA_NUM_THREADS)
    tprint("  DEFAULT_NUM_THREADS", numba.config.NUMBA_DEFAULT_NUM_THREADS)
    tprint("  get_num_threads()", numba.get_num_threads())
    tprint("  THREADING_LAYER", numba.config.THREADING_LAYER)
    tprint("  threading_layer()", numba.threading_layer())

    #interp.parallel_diagnostics(level=4)
    #mover.parallel_diagnostics(level=4)

    return


@numba.njit(cache=True,fastmath=True,parallel=True)
def mover(
        bx,by,bz,ex,ey,ez,
        px,py,pz,pu,pv,pw,
        pwtot,pwprl,pwx,pwy,pwz,
        pex,pey,pez,pbx,pby,pbz,
        qm,c
):
    """Boris particle mover"""

    cinv=1./c

    # Fortran/TRISTAN indexing convention
    #ix=1
    #iy=fld.shape[0]
    #iz=fld.shape[0]*fld.shape[1]

    # numba only supports C ordering...
    # but for a 2-D x-y run with shape[2]==2, interpolation with C-ordering is
    # actually more contiguous in memory.  but idk if contiguity matters,
    # I think compilers can optimize for strided access too...
    ix=ex.shape[1]*ex.shape[2]
    iy=ex.shape[2]
    iz=1
    exview=np.ravel(ex)#,order='C')  # ravel=view, flatten=copy
    eyview=np.ravel(ey)#,order='C')  # ravel=view, flatten=copy
    ezview=np.ravel(ez)#,order='C')  # ravel=view, flatten=copy
    bxview=np.ravel(bx)#,order='C')  # ravel=view, flatten=copy
    byview=np.ravel(by)#,order='C')  # ravel=view, flatten=copy
    bzview=np.ravel(bz)#,order='C')  # ravel=view, flatten=copy

    for ip in numba.prange(px.size):

        # Linearly interpolate fields to prtl positions
        # particle domain is:
        #   x in [0, fld.shape[0]-1)
        #   y in [0, fld.shape[1]-1)
        #   z in [0, fld.shape[2]-1)
        # if using prtl.tot coordinates

        i=int(px[ip])
        dx=px[ip]-i
        j=int(py[ip])
        dy=py[ip]-j
        k=int(pz[ip])
        dz=pz[ip]-k
        #l=i+iy*(j-1)+iz*(k-1)  # fortran 1-based index
        #l=i+iy*j+iz*k  # fortran column-major ordering
        l=i*ix+j*iy+k*iz  # row-major ordering

        # TRISTAN mover uses (0.25*qm) for E fld, (0.125*qm*cinv) for B fld
        # TRISTAN interp includes 2x, 4x factors respectively
        # due to Yee mesh edge/face-centering for E/B flds, see Buneman's comments
        # whereas trochograph uses vertex-centered flds, hence 0.5*qm, 0.5*qm*cinv

        # Field interpolations are tri-linear (linear in x times linear in y
        # times linear in z). This amounts to the 3-D generalisation of "area
        # weighting".
        ex0 = (
              (1.-dx)*(1.-dy)*(1.-dz)*exview[l]
            +     dx *(1.-dy)*(1.-dz)*exview[l+ix]
            + (1.-dx)*    dy *(1.-dz)*exview[l+iy]
            +     dx *    dy *(1.-dz)*exview[l+ix+iy]
            + (1.-dx)*(1.-dy)*    dz *exview[l+iz]
            +     dx *(1.-dy)*    dz *exview[l+ix+iz]
            + (1.-dx)*    dy *    dz *exview[l+iy+iz]
            +     dx *    dy *    dz *exview[l+ix+iy+iz]
        ) * 0.5*qm

        ey0 = (
              (1.-dx)*(1.-dy)*(1.-dz)*eyview[l]
            +     dx *(1.-dy)*(1.-dz)*eyview[l+ix]
            + (1.-dx)*    dy *(1.-dz)*eyview[l+iy]
            +     dx *    dy *(1.-dz)*eyview[l+ix+iy]
            + (1.-dx)*(1.-dy)*    dz *eyview[l+iz]
            +     dx *(1.-dy)*    dz *eyview[l+ix+iz]
            + (1.-dx)*    dy *    dz *eyview[l+iy+iz]
            +     dx *    dy *    dz *eyview[l+ix+iy+iz]
        ) * 0.5*qm

        ez0 = (
              (1.-dx)*(1.-dy)*(1.-dz)*ezview[l]
            +     dx *(1.-dy)*(1.-dz)*ezview[l+ix]
            + (1.-dx)*    dy *(1.-dz)*ezview[l+iy]
            +     dx *    dy *(1.-dz)*ezview[l+ix+iy]
            + (1.-dx)*(1.-dy)*    dz *ezview[l+iz]
            +     dx *(1.-dy)*    dz *ezview[l+ix+iz]
            + (1.-dx)*    dy *    dz *ezview[l+iy+iz]
            +     dx *    dy *    dz *ezview[l+ix+iy+iz]
        ) * 0.5*qm

        bx0 = (
              (1.-dx)*(1.-dy)*(1.-dz)*bxview[l]
            +     dx *(1.-dy)*(1.-dz)*bxview[l+ix]
            + (1.-dx)*    dy *(1.-dz)*bxview[l+iy]
            +     dx *    dy *(1.-dz)*bxview[l+ix+iy]
            + (1.-dx)*(1.-dy)*    dz *bxview[l+iz]
            +     dx *(1.-dy)*    dz *bxview[l+ix+iz]
            + (1.-dx)*    dy *    dz *bxview[l+iy+iz]
            +     dx *    dy *    dz *bxview[l+ix+iy+iz]
        ) * 0.5*qm*cinv

        by0 = (
              (1.-dx)*(1.-dy)*(1.-dz)*byview[l]
            +     dx *(1.-dy)*(1.-dz)*byview[l+ix]
            + (1.-dx)*    dy *(1.-dz)*byview[l+iy]
            +     dx *    dy *(1.-dz)*byview[l+ix+iy]
            + (1.-dx)*(1.-dy)*    dz *byview[l+iz]
            +     dx *(1.-dy)*    dz *byview[l+ix+iz]
            + (1.-dx)*    dy *    dz *byview[l+iy+iz]
            +     dx *    dy *    dz *byview[l+ix+iy+iz]
        ) * 0.5*qm*cinv

        bz0 = (
              (1.-dx)*(1.-dy)*(1.-dz)*bzview[l]
            +     dx *(1.-dy)*(1.-dz)*bzview[l+ix]
            + (1.-dx)*    dy *(1.-dz)*bzview[l+iy]
            +     dx *    dy *(1.-dz)*bzview[l+ix+iy]
            + (1.-dx)*(1.-dy)*    dz *bzview[l+iz]
            +     dx *(1.-dy)*    dz *bzview[l+ix+iz]
            + (1.-dx)*    dy *    dz *bzview[l+iy+iz]
            +     dx *    dy *    dz *bzview[l+ix+iy+iz]
        ) * 0.5*qm*cinv

        #print("interp: x,y,z=",px[ip],py[ip],pz[ip])
        #print("interp: i,j,k=",i,j,k)
        #print("interp: dx,dy,dz=",dx,dy,dz)
        #print()
        #print("interp: l         =", l           )
        #print("interp: l+ix      =", l+ix        )
        #print("interp: l+iy      =", l+iy        )
        #print("interp: l+ix+iy   =", l+ix+iy     )
        #print("interp: l+iz      =", l+iz        )
        #print("interp: l+ix+iz   =", l+ix+iz     )
        #print("interp: l+iy+iz   =", l+iy+iz     )
        #print("interp: l+ix+iy+iz=", l+ix+iy+iz  )
        #print()
        #print("interp: ex[l]         =", exview[l]           )
        #print("interp: ex[l+ix]      =", exview[l+ix]        )
        #print("interp: ex[l+iy]      =", exview[l+iy]        )
        #print("interp: ex[l+ix+iy]   =", exview[l+ix+iy]     )
        #print("interp: ex[l+iz]      =", exview[l+iz]        )
        #print("interp: ex[l+ix+iz]   =", exview[l+ix+iz]     )
        #print("interp: ex[l+iy+iz]   =", exview[l+iy+iz]     )
        #print("interp: ex[l+ix+iy+iz]=", exview[l+ix+iy+iz]  )
        #print("interp: ex[l]          wt=", (1.-dx)*(1.-dy)*(1.-dz) )
        #print("interp: ex[l+ix]       wt=",     dx *(1.-dy)*(1.-dz) )
        #print("interp: ex[l+iy]       wt=", (1.-dx)*    dy *(1.-dz) )
        #print("interp: ex[l+ix+iy]    wt=",     dx *    dy *(1.-dz) )
        #print("interp: ex[l+iz]       wt=", (1.-dx)*(1.-dy)*    dz  )
        #print("interp: ex[l+ix+iz]    wt=",     dx *(1.-dy)*    dz  )
        #print("interp: ex[l+iy+iz]    wt=", (1.-dx)*    dy *    dz  )
        #print("interp: ex[l+ix+iy+iz] wt=",     dx *    dy *    dz  )
        #print()
        #print("interp: ex0=", ex0)


        # First half electric acceleration, with relativity's gamma:
        u0 = c*pu[ip]+ex0
        v0 = c*pv[ip]+ey0
        w0 = c*pw[ip]+ez0

        # START DIAGNOSTIC E \cdot v = E \cdot (v_{pre-rot} + v_{post-rot})/2
        ev = ex0*u0+ey0*v0+ez0*w0  # first part of Edotv work in mover
        evx = ex0*u0
        evy = ey0*v0
        evz = ez0*w0
        # END DIAGNOSTIC E \cdot v = E \cdot (v_{pre-rot} + v_{post-rot})/2

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

        # START DIAGNOSTIC E \cdot v = E \cdot (v_{pre-rot} + v_{post-rot})/2
        # E_prll \cdot v = (E \cdot B) (B \cdot v) / B^2
        # Restore 2x to ex0 and remove Lorentz gamma.
        # v_prll, Lorentz gamma are unchanged by mag rotation,
        # so OK to compute either before or after.
        evprl = 2*(ex0*bx0+ey0*by0+ez0*bz0) * g*(bx0*u0+by0*v0+bz0*w0) / (bx0**2+by0**2+bz0**2)
        pwprl[ip] = pwprl[ip] + evprl*cinv*cinv  # store as dimensionless Lorentz gamma

        # E \cdot v = E \cdot (v_{pre-rot} + v_{post-rot})/2
        ev = g*(ev + ex0*u0+ey0*v0+ez0*w0)  # second part, remove Lorentz gamma
        pwtot[ip] = pwtot[ip] + ev*cinv*cinv  # store as dimensionless Lorentz gamma

        evx = g*(evx + ex0*u0)
        evy = g*(evy + ey0*v0)
        evz = g*(evz + ez0*w0)
        pwx[ip] = pwx[ip] + evx*cinv*cinv
        pwy[ip] = pwy[ip] + evy*cinv*cinv
        pwz[ip] = pwz[ip] + evz*cinv*cinv
        # END DIAGNOSTIC E \cdot v = E \cdot (v_{pre-rot} + v_{post-rot})/2

        # Second half el. acc'n:
        u0=u0+ex0
        v0=v0+ey0
        w0=w0+ez0

        # Normalized 4-velocity advance:
        pu[ip] = u0*cinv
        pv[ip] = v0*cinv
        pw[ip] = w0*cinv
        # Position advance:
        g = c/(c**2+u0**2+v0**2+w0**2)**0.5
        px[ip] = px[ip] + pu[ip]*g*c
        py[ip] = py[ip] + pv[ip]*g*c
        pz[ip] = pz[ip] + pw[ip]*g*c

        # Save fields used (NOTICE THAT THERE'S AN OFFSET!!!!!... fields used at PREVIOUS LOCATION, or "in-between" locations if you like)
        pex[ip] = ex0 / (0.5*qm)
        pey[ip] = ey0 / (0.5*qm)
        pez[ip] = ez0 / (0.5*qm)
        pbx[ip] = bx0 / (0.5*qm*cinv)
        pby[ip] = by0 / (0.5*qm*cinv)
        pbz[ip] = bz0 / (0.5*qm*cinv)

    return


@numba.njit(cache=True,fastmath=True,parallel=True)
def interp(fld,x,y,z):
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

    #ix=1
    #iy=fld.shape[0]
    #iz=fld.shape[0]*fld.shape[1]
    #f=fld.flatten(order='F')  # match buneman's convention

    # numba only supports C ordering...
    # but for a 2-D x-y run with shape[2]==2, interpolation with C-ordering is
    # actually more contiguous in memory.  but idk if contiguity matters,
    # I think compilers can optimize for strided access too...
    ix=fld.shape[1]*fld.shape[2]
    iy=fld.shape[2]
    iz=1
    f=np.ravel(fld)#,order='C')  # ravel=view, flatten=copy

    fout=np.empty_like(x)

    for ip in numba.prange(x.size):
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

        #print("interp: x,y,z=",x[ip],y[ip],z[ip])
        #print("interp: i,j,k=",i,j,k)
        #print("interp: dx,dy,dz=",dx,dy,dz)
        #print()
        #print("interp: l         =", l           )
        #print("interp: l+ix      =", l+ix        )
        #print("interp: l+iy      =", l+iy        )
        #print("interp: l+ix+iy   =", l+ix+iy     )
        #print("interp: l+iz      =", l+iz        )
        #print("interp: l+ix+iz   =", l+ix+iz     )
        #print("interp: l+iy+iz   =", l+iy+iz     )
        #print("interp: l+ix+iy+iz=", l+ix+iy+iz  )
        #print()
        #print("interp: f[l]         =", f[l]           )
        #print("interp: f[l+ix]      =", f[l+ix]        )
        #print("interp: f[l+iy]      =", f[l+iy]        )
        #print("interp: f[l+ix+iy]   =", f[l+ix+iy]     )
        #print("interp: f[l+iz]      =", f[l+iz]        )
        #print("interp: f[l+ix+iz]   =", f[l+ix+iz]     )
        #print("interp: f[l+iy+iz]   =", f[l+iy+iz]     )
        #print("interp: f[l+ix+iy+iz]=", f[l+ix+iy+iz]  )
        #print("interp: f[l]          weight=", (1.-dx)*(1.-dy)*(1.-dz) )
        #print("interp: f[l+ix]       weight=",     dx *(1.-dy)*(1.-dz) )
        #print("interp: f[l+iy]       weight=", (1.-dx)*    dy *(1.-dz) )
        #print("interp: f[l+ix+iy]    weight=",     dx *    dy *(1.-dz) )
        #print("interp: f[l+iz]       weight=", (1.-dx)*(1.-dy)*    dz  )
        #print("interp: f[l+ix+iz]    weight=",     dx *(1.-dy)*    dz  )
        #print("interp: f[l+iy+iz]    weight=", (1.-dx)*    dy *    dz  )
        #print("interp: f[l+ix+iy+iz] weight=",     dx *    dy *    dz  )
        #print()
        #print("interp: fout=", fout[ip])

    return fout


def add_ghost(fld, par):
    """
    Attach ghost cells to far edges of fld for periodic boundaries
    Returns copy of fld; input array is not modified
    """
    if par['periodicx']:
        #fld = np.concatenate((fld[-1:,:,:], fld, fld[0:1,:,:]), axis=0)
        fld = np.concatenate((fld, fld[0:1,:,:]), axis=0)
    if par['periodicy']:
        #fld = np.concatenate(( fld[:,-1:,:],  fld,  fld[:,0:1,:]), axis=1)
        fld = np.concatenate((fld, fld[:,0:1,:]), axis=1)
    if par['periodicz']:
        #fld = np.concatenate((fld[:,:,-1:], fld, fld[:,:,0:1]), axis=2)
        fld = np.concatenate((fld, fld[:,:,0:1]), axis=2)
    return fld



@numba.njit(cache=True,parallel=True)
def prtl_bc(px, py, pz, dimf, periodicx, periodicy, periodicz):
    """Given p, dimf; update p according to desired BCs for dimf
    dimf = fld shape provided by user, NOT including ghost cells
        for periodic bdry
    """
    for ip in numba.prange(px.size):

        # x boundary condition
        if periodicx:
            #px[ip] = np.mod(px[ip], dimf[0])  # modulo func is slow
            if px[ip] > dimf[0]:
                px[ip] = px[ip] - dimf[0]
        else:
            #assert px[ip] >= 0             # asserts prevent numba parallelism
            #assert px[ip] <= dimf[0] - 1
            if px[ip] < 0 or px[ip] > dimf[0]-1:
                px[ip] = np.nan

        # y boundary condition
        if periodicy:
            #py[ip] = np.mod(py[ip], dimf[1])  # modulo func is slow
            if py[ip] > dimf[1]:
                py[ip] = py[ip] - dimf[1]
        else:
            #assert py[ip] >= 0             # asserts prevent numba parallelism
            #assert py[ip] <= dimf[1] - 1
            if py[ip] < 0 or py[ip] > dimf[1]-1:
                py[ip] = np.nan

        # z boundary condition
        if periodicz:
            #pz[ip] = np.mod(pz[ip], dimf[2])  # modulo func is slow
            if pz[ip] > dimf[2]:
                pz[ip] = pz[ip] - dimf[2]
        else:
            #assert pz[ip] >= 0             # asserts prevent numba parallelism
            #assert pz[ip] <= dimf[2] - 1
            if pz[ip] < 0 or pz[ip] > dimf[2]-1:
                pz[ip] = np.nan
    return


def output(p,par,lap):
    """Write outputs"""

    interval    = par['interval']
    last        = par['last']
    pltstart    = par['pltstart']

    if lap % interval != 0:
        return False

    n = int(lap/interval) - int(pltstart/interval)  # starts at 0000
    fname = "output/troch.{:04d}".format(n)

    with h5py.File(fname, "w") as f:
        f.create_dataset("x", data=p.x)
        f.create_dataset("y", data=p.y)
        f.create_dataset("z", data=p.z)
        f.create_dataset("u", data=p.u)
        f.create_dataset("v", data=p.v)
        f.create_dataset("w", data=p.w)
        f.create_dataset("proc", data=p.proc)
        f.create_dataset("ind", data=p.ind)
        # START DIAGNOSTIC E \cdot v = E \cdot (v_{pre-rot} + v_{post-rot})/2
        f.create_dataset("wtot", data=p.wtot)
        f.create_dataset("wprl", data=p.wprl)
        f.create_dataset("wx", data=p.wx)
        f.create_dataset("wy", data=p.wy)
        f.create_dataset("wz", data=p.wz)
        # END DIAGNOSTIC E \cdot v = E \cdot (v_{pre-rot} + v_{post-rot})/2
        f.create_dataset("bx", data=p.bx)
        f.create_dataset("by", data=p.by)
        f.create_dataset("bz", data=p.bz)
        f.create_dataset("ex", data=p.ex)
        f.create_dataset("ey", data=p.ey)
        f.create_dataset("ez", data=p.ez)

    return fname


def tprint(*args, **kwargs):
    #tstr = (datetime.now().strftime("%m-%d-%Y %H:%M:%S.%f"))[:-3]
    #print("[troch {}] ".format(tstr), end='')
    return print(*args, **kwargs)
