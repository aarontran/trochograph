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
import psutil


class Fields(object):
    pass


class Particles(object):
    pass


def run_trochograph(user_input, user_flds, user_prtl):
    """Main execution loop of trochograph"""

    # -----------------------------------------
    # initialize

    t0 = datetime.now()
    tprint("Initializing input, flds, prtl")

    par = user_input()

    flds = user_flds(par)

    dimf0 = flds.ex.shape  # dimf0 means fld shape WITHOUT ghost cells

    flds.ex = add_ghost(flds.ex, par)
    flds.ey = add_ghost(flds.ey, par)
    flds.ez = add_ghost(flds.ez, par)
    flds.bx = add_ghost(flds.bx, par)
    flds.by = add_ghost(flds.by, par)
    flds.bz = add_ghost(flds.bz, par)

    dimf = flds.ex.shape  # dimf means fld shape WITH ghost cells

    # row- vs column-order seems to affect performance at ~10% level
    # (~3e-4 vs 3.5e-4, for 1000 prtl on dimf=(100+1,10+1,1+1))
    flds.ex = np.ascontiguousarray(flds.ex)
    flds.ey = np.ascontiguousarray(flds.ey)
    flds.ez = np.ascontiguousarray(flds.ez)
    flds.bx = np.ascontiguousarray(flds.bx)
    flds.by = np.ascontiguousarray(flds.by)
    flds.bz = np.ascontiguousarray(flds.bz)

    p = user_prtl(flds)

    # machine epsilon at values of dimf bounds, for use with prtl BCs
    # np.spacing(..., dtype=...) cannot be numba compiled,
    # so must compute prior to evolution loop
    dimfeps = (
        np.spacing(dimf[0]-1, dtype=p.x.dtype),
        np.spacing(dimf[1]-1, dtype=p.y.dtype),
        np.spacing(dimf[2]-1, dtype=p.z.dtype),
    )

    # -----------------------------------------
    # some checks, report to stdout

    for arr in [p.x, p.y, p.z, p.u, p.v, p.w]:
        assert arr.ndim  == 1
        assert arr.dtype.kind == 'f'
    for arr in [flds.ex, flds.ey, flds.ez, flds.bx, flds.by, flds.bz]:
        assert arr.ndim  == 3
        assert arr.dtype.kind == 'f'

    tprint("  flds shape =", dimf, "without ghost=", dimf0)
    tprint("  flds ex,ey,ez,bx,by,bz dtype =",
            flds.ex.dtype, flds.ey.dtype, flds.ez.dtype,
            flds.bx.dtype, flds.by.dtype, flds.bz.dtype,
    )
    tprint("  prtl number =", p.x.size)
    tprint("  prtl x,y,z,u,v,w dtype =",
            p.x.dtype, p.y.dtype, p.z.dtype,
            p.u.dtype, p.v.dtype, p.w.dtype
    )
    tprint("Input parameters:")
    for kk in sorted(par):
        tprint("  {:s} = {}".format(kk, par[kk]))

    tprint("Numba threading")
    tprint("  NUM_THREADS", numba.config.NUMBA_NUM_THREADS)
    tprint("  DEFAULT_NUM_THREADS", numba.config.NUMBA_DEFAULT_NUM_THREADS)
    tprint("  get_num_threads()", numba.get_num_threads())
    tprint("  THREADING_LAYER", numba.config.THREADING_LAYER)
    tprint("  threading_layer()", numba.threading_layer())

    # https://stackoverflow.com/questions/938733/total-memory-used-by-python-process
    mem = psutil.Process(os.getpid()).memory_info().rss
    tprint("Process ID", os.getpid(), "using mem", mem/1e6, "MB")

    # -----------------------------------------
    tprint("Pre-enforce prtl BCs")

    prtl_bc(
        p.x,p.y,p.z,p.u,p.v,p.w,par['c'],dimf,dimfeps,
        #par['periodicx'],par['periodicy'],par['periodicz']
        par['boundary_xl'],par['boundary_xr'],par['periodicy'],par['periodicz']
    )

    # -----------------------------------------
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

    # -----------------------------------------
    tprint("Initial output")
    if not os.path.exists("output"):
        os.mkdir("output")
    #os.makedirs("output", exist_ok=True)  # needs python >=3.2
    output(p,par,par['lapst'])

    # -----------------------------------------
    # main loop

    t1 = datetime.now()
    tlaptot = 0
    tlaprestmov = 0
    tlaprestout = 0
    tlapfirst = 0

    for lap in range(par['lapst']+1, par['last']+1):

        tlap0 = datetime.now()
        tprint("Lap {:10d}".format(lap),end='')

        mover(
            flds.bx,flds.by,flds.bz,flds.ex,flds.ey,flds.ez,
            p.x,p.y,p.z,p.u,p.v,p.w,
            p.wtot,p.wprl,p.wx,p.wy,p.wz,
            p.ex,p.ey,p.ez,p.bx,p.by,p.bz,
            par['qm'],par['c'],
            dimf,dimfeps,
            #par['periodicx'],par['periodicy'],par['periodicz']
            par['boundary_xl'],par['boundary_xr'],par['periodicy'],par['periodicz']
        )

        tlap1 = datetime.now()

        fwrote = output(p,par,lap)

        tlap2 = datetime.now()

        # lap stdout and time accounting

        dtlap1_0 = (tlap1-tlap0).total_seconds()  # deltas
        dtlap2_1 = (tlap2-tlap1).total_seconds()
        dtlap2_0 = (tlap2-tlap0).total_seconds()  # total

        tprint("  move {:.3e} out {:.3e} tot {:.3e}".format(
            dtlap1_0,
            dtlap2_1,
            dtlap2_0,
        ))
        if fwrote:
            tprint("  wrote", fwrote)

        tlaptot += dtlap2_0
        if lap == par['lapst']+1:
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

    #interp.parallel_diagnostics(level=4)
    #mover.parallel_diagnostics(level=4)

    return


#boundscheck=True
@numba.njit(fastmath={'ninf','nsz','arcp','contract','afn','reassoc'},parallel=True)
def mover(
        bx,by,bz,ex,ey,ez,
        px,py,pz,pu,pv,pw,
        pwtot,pwprl,pwx,pwy,pwz,
        pex,pey,pez,pbx,pby,pbz,
        qm,c,
        dimf,dimfeps,#periodicx,periodicy,periodicz
        boundary_xl,boundary_xr,periodicy,periodicz
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

        if np.isnan(px[ip]) or np.isnan(py[ip]) or np.isnan(pz[ip]):
            continue

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

        prtl_bc_one(
            ip,px,py,pz,pu,pv,pw,c,
            dimf,dimfeps,#periodicx,periodicy,periodicz
            boundary_xl,boundary_xr,periodicy,periodicz
        )

        # Save fields used (NOTICE THAT THERE'S AN OFFSET!!!!!... fields used at PREVIOUS LOCATION, or "in-between" locations if you like)
        pex[ip] = ex0 / (0.5*qm)
        pey[ip] = ey0 / (0.5*qm)
        pez[ip] = ez0 / (0.5*qm)
        pbx[ip] = bx0 / (0.5*qm*cinv)
        pby[ip] = by0 / (0.5*qm*cinv)
        pbz[ip] = bz0 / (0.5*qm*cinv)

    return


#boundscheck=True
@numba.njit(fastmath={'ninf','nsz','arcp','contract','afn','reassoc'},parallel=True)
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
    #if par['periodicx']:
    if par['boundary_xl'] == "periodic" and par['boundary_xr'] == "periodic":
        #fld = np.concatenate((fld[-1:,:,:], fld, fld[0:1,:,:]), axis=0)
        fld = np.concatenate((fld, fld[0:1,:,:]), axis=0)
    if par['periodicy']:
        #fld = np.concatenate(( fld[:,-1:,:],  fld,  fld[:,0:1,:]), axis=1)
        fld = np.concatenate((fld, fld[:,0:1,:]), axis=1)
    if par['periodicz']:
        #fld = np.concatenate((fld[:,:,-1:], fld, fld[:,:,0:1]), axis=2)
        fld = np.concatenate((fld, fld[:,:,0:1]), axis=2)
    return fld


@numba.njit(fastmath={'ninf','nsz','arcp','contract','afn','reassoc'})#,parallel=True)
#def prtl_bc_one(px, py, pz, ip, dimf, dimfeps, periodicx, periodicy, periodicz):
def prtl_bc_one(ip, px, py, pz, pu, pv, pw, c, dimf, dimfeps, boundary_xl, boundary_xr, periodicy, periodicz):
    """Return x,y,z after applying BC"""

    if np.isnan(px[ip]) or np.isnan(py[ip]) or np.isnan(pz[ip]):
        return

    # domain [0,mx) x [0,my) x [0,mz) is strict,
    # prtl at x=mx, y=my, z=mz will break prtl interp (index out of bounds)
    mx = dimf[0] - 1  # prtl bdry, dimf with ghost cells
    my = dimf[1] - 1
    mz = dimf[2] - 1

    # x boundary condition
    #if periodicx:
    #    if px[ip] >= mx:
    #        px[ip] = max(px[ip] - mx, 0.0)
    #    elif px[ip] < 0:
    #        px[ip] = min(px[ip] + mx, mx-dimfeps[0])
    #else:
    #    if px[ip] < 0 or px[ip] >= mx:
    #        px[ip] = np.nan
    if px[ip] < 0:
        if boundary_xl == "periodic":
            px[ip] = min(px[ip] + mx, mx-dimfeps[0])
        elif boundary_xl == "outflow":
            px[ip] = np.nan
        elif boundary_xl == "reflect":

            # adapted from tristan-mp_xshock/user_shock_twowalls.F90
            # currently implemented for walloc=0, betawall=0 only

            #this algorithm ignores change in y and z coordinates
            #during the scattering. Including it can result in rare
            #conditions where a particle gets stuck in the ghost zones.
            #This can be improved.

            gamma=(1+pu[ip]**2+pv[ip]**2+pw[ip]**2)**0.5

            # unwind x location of particle
            x0=px[ip]-pu[ip]/gamma*c
            y0=py[ip] #-pv[ip]/gamma*c
            z0=pz[ip] #-pw[ip]/gamma*c
            # unwind wall location
            walloc0 = 0.0  # wallloc - betawall*c  # see also code logic px[ip] < 0...

            # where did they meet?
            tfrac=min(abs((x0-walloc0)/max(abs(pu[ip]/gamma*c),1e-9)),1.)
            xcolis=x0+pu[ip]/gamma*c*tfrac
            ycolis=y0 #+pv[ip]/gamma*c*tfrac
            zcolis=z0 #+pw[ip]/gamma*c*tfrac

            # reset particle momentum, getting a kick from the wall
            #pu[ip] = gammawall**2*gamma*(2*betawall - pu[ip]/gamma*(1+betawall**2))
            #gamma=(1+pu[ip]**2+pv[ip]**2+pw[ip]**2)**0.5
            pu[ip] = -1*pu[ip]

            #move particle from the wall position with the new velocity
            tfrac=min(abs((px[ip]-xcolis)/max(abs(px[ip]-x0),1e-9)),1.)
            px[ip] = xcolis + pu[ip]/gamma*c * tfrac
            py[ip] = ycolis #+ pv[ip]/gamma*c * tfrac
            pz[ip] = zcolis #+ pw[ip]/gamma*c * tfrac

    if px[ip] >= mx:
        if boundary_xr == "periodic":
            px[ip] = max(px[ip] - mx, 0.0)
        elif boundary_xr == "outflow":
            px[ip] = np.nan
        elif boundary_xr == "reflect":
            ## not implemented...
            px[ip] = np.nan

    # y boundary condition
    if periodicy:
        if py[ip] >= my:
            py[ip] = max(py[ip] - my, 0.0)
        elif py[ip] < 0:
            py[ip] = min(py[ip] + my, my-dimfeps[1])
    else:
        if py[ip] < 0 or py[ip] >= my:
            py[ip] = np.nan

    # z boundary condition
    if periodicz:
        if pz[ip] >= mz:
            pz[ip] = max(pz[ip] - mz, 0.0)
        elif pz[ip] < 0:
            pz[ip] = min(pz[ip] + mz, mz-dimfeps[2])
    else:
        if pz[ip] < 0 or pz[ip] >= mz:
            pz[ip] = np.nan
    return


@numba.njit(fastmath={'ninf','nsz','arcp','contract','afn','reassoc'},parallel=True)
#def prtl_bc(px, py, pz, pu, pv, pw, dimf, dimfeps, periodicx, periodicy, periodicz):
def prtl_bc(px, py, pz, pu, pv, pw, c, dimf, dimfeps, boundary_xl, boundary_xr, periodicy, periodicz):
    """Given p, dimf; update p according to desired BCs for dimf
    dimf = fld shape provided by user, NOT including ghost cells
        for periodic bdry
    """
    for ip in numba.prange(px.size):
        prtl_bc_one(
            ip,px,py,pz,pu,pv,pw,c,
            #dimf,dimfeps,periodicx,periodicy,periodicz
            dimf,dimfeps,boundary_xl,boundary_xr,periodicy,periodicz
        )
    return


def output(p,par,lap):
    """Write outputs"""

    interval    = par['interval']
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
