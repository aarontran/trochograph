Version history
===============
* 2020 Nov 06-07 cleanup, speedup
* 2020 Jul 16-31(ish) started


Field and particle coordinates
==============================

Coordinate systems
------------------
`trochograph` field and particle positions are zero-indexed;
field array indices and particle positions share the same coordinates.
A particle with x=0.0, y=1.0, z=2.0 sees E,B fields at indices [0,1,2].
There's no mesh staggering in the mover; fields always live at coordinate
vertices.

In comparison, TRISTAN outputs particle positions that are offset
by (+3,+3,+3) with respect to field indices (if you read the fields into a
0-based index array).  A TRISTAN particle at x=3.0, y=3.0, z=3.0 sees E,B
fields at array indices [0,0,0].


Reading in TRISTAN particles and fields
---------------------------------------
To read TRISTAN output fields into `trochograph`: no index offset is needed.
Use numpy.ascontiguousarray(...) or similar to enforce row-major (C-like)
layout in memory, which seems to be marginally faster in practice.

To read TRISTAN output particles into `trochograph`: subtract 3 from all
coordinates (2 ghost cells, and change from 1- to 0-based indexing).


Boundary conditions
-------------------
Periodic boundaries will attach 1 ghost cell layer at each "far" boundary.
Motivation: consider fields with dimf = (5, 1, 1), periodic in all directions.

        y=1 .____.____.____.____.____.      @ = physical vertex where
            |    |    |    |    |    |          flds are defined
            @____@____@____@____@____.
        x,y=0  x=1  x=2  x=3  x=4  x=5      . = ghost vertex

We use 3-D interpolation, so every particle must lie between 2 vertices in each
of x,y,z directions; we do NOT treat 1D/2D simulations as special cases.
The valid particle domain is thus:

    x in [0, 5) = [0, dimf[0])
    y in [0, 1) = [0, dimf[1])
    z in [0, 1) = [0, dimf[2])

For a fully-periodic domain, the field values live at "lower left" corner of
grid, with respect to the cell volume in which particles reside.

Now consider an outflow boundary in x and periodic boundaries in y,z.

        y=1 .____.____.____.____.           @ = physical vertex where
            |    |    |    |    |               flds are defined
            @____@____@____@____@
        x,y=0  x=1  x=2  x=3  x=4           . = ghost vertex

Fields are not defined past x=4, so the particle domain is:

    x in [0, 4) = [0, dimf[0]-1)
    y in [0, 1) = [0, dimf[1])
    z in [0, 1) = [0, dimf[2])


More detail on TRISTAN coordinates
----------------------------------
To check TRISTAN's internal coordinates in overly explicit detail, consider a
2D x,y simulation with domain decomposition along y (generalizes to 3D x,y,z).
The translation from internal to output coordinates uses:

    fields.F90
        mx0 = {user input mx0} + 5  ! global domain size, internal coords
        my0 = {user input my0} + 5
        mx = mx0                    ! local domain size, internal coords
        my = (my0-5)/sizey + 5

    output.F90
        jstart = 3
        jfinish = my-3
        istart = 3
        ifinish = mx-3
        do j=jstart,jfinish,istep
            ny=(j-jstart)/istep + 1
            do i=istart,ifinish,istep
                nx=(i-istart)/istep + 1
                call interpolfld(real(i),real(j),...)
                temporary1(nx,ny) = {stuff}  ! output buffer

Fields are stored at cell edges/centers; `interpolfld` gets cell vertex values.

Internal coordinates (i,j) translate to output indices (nx,ny) as follows,
assuming istep=1 and my0 divisible by sizey (no "leftover" cells):

    i=istart  = 3                           -> nx = 1
    i=ifinish = {user input mx0} + 2        -> nx = {user input mx0}
    j=jstart  = 3                           -> ny = 1
    j=jfinish = {user input my0}/sizey + 2  -> ny = {user input my0}/sizey

We see that output fields omit ghost cells.  But, particle positions are
written in internal coordinates with 1-based indexing and 2 ghost cells.
So we have an offset of (+2,+2,+2) or (+3,+3,+3) between TRISTAN's output field
and particle positions.

Example: for {user input mx0,my0} = 4,1, the internal (i,j) grid looks like:

        j=5 .____.____.____.____.____.____.____.____.
            |    |    |    |    |    |    |    |    |
        j=4 .____.____.____.____.____.____.____.____.   ## = physical domain
            |    |    | ## | ## | ## | ## |    |    |
        j=3 .____.____@____@____@____@____.____.____.   @ = output field locations
            |    |    |    |    |    |    |    |    |
        j=2 .____.____.____.____.____.____.____.____.   . = ghost vertex
            |    |    |    |    |    |    |    |    |
            .____.____.____.____.____.____.____.____.
        i,j=1  i=2  i=3  i=4  i=5  i=6  i=7  i=8  i=9

Internal coordinates run from i=1 to i=mx=9, j=1 to j=my=5.
Particles live within (i,j)=(3,3) to (4,7).


Making things go fastly
=======================

For comparison/inspiration: https://github.com/mikegrudic/pykdgrav

Profiling: `python -m line_profiler asdf.py.lprof`.

2020 Jul 17-31 (ish)
--------------------
Things tried for interpolation:
1. `scipy.interpolate.interpn(...)`, called every loop
2. `scipy.interpolate.RegularGridInterpolator`, pre-cached function
3. `interp_3d.Interp3D(...)` (from https://github.com/jglaser/interp3d),
   pre-cached function
4. Buneman's interp, array broadcasting
5. Buneman's interp, loop over individual prtl

Things tried for output:
* numpy `savez` is really big
* numpy `savez_compressed`
* hdf5 gzip doesn't help much


2020 Nov 06-07
--------------

### Mover speedup
Mover: do explicit prtl loop with numba prange rather than array broadcast,
basically the same as what TRISTAN does.  Test on "bzramp" problem with 10,000
particles, 200 laps:

    array mover     = 0.40 sec / (10000 prtl x 199 lap) = 2e-7 sec/prtl-lap
    loop mover,     = 0.20 sec / (10000 prtl x 199 lap) = 1e-7 sec/prtl-lap
    loop mover,prll = 0.12 sec / (10000 prtl x 199 lap) = 6e-8 sec/prtl-lap

numba fastmath = ~10% speedup in mover.

Redo "bzramp" with fastmath, TBB loaded, slurm allocated 20 procs, and 100,000
particles to check threading performance:

     1 thread, 2.02  sec                       = 1.0e-7 thread-sec/prtl-lap
     5 thread, 0.737 sec = 3.7e-8 sec/prtl-lap = 1.9e-7 thread-sec/prtl-lap
    10 thread, 0.444 sec = 2.2e-8 sec/prtl-lap = 2.2e-7 thread-sec/prtl-lap
    20 thread, 0.288 sec = 1.4e-8 sec/prtl-lap = 2.8e-7 thread-sec/prtl-lap

Compare to TRISTAN run on same hardware (habanero cluster) with AVX2.
My run `mi400Ms4b0.25theta65_2d_my552_comp20_later` with 8.03e8 prtl and 552
CPU takes 0.1 sec to move.  Cost is thus 0.1/(8.03e8/552) = 7e-8 cpu-sec per
prtl-lap.  So trochograph is just a bit slower than Fortran, but for some
reason doesn't scale ideally on threads.

### Prtl BC speedup
Prtl boundary condition: njit on prtl boundary condition is ~10% speedup for BC
time, but ~1% of total time, so small gain.  Also maybe affected by timing
noise.  Replacing costly modulo function with if/else check helps the most.

    baseline: 0.073 sec
    baseline, no x check: 0.067 sec
    numba njit parallel: 0.050 sec, 0.053 sec
    numba njit parallel, no x check: 0.041 sec
    numba njit parallel, yes x check, NO modulo: 0.012 sec

### Output speedup
HDF5 performance? moving to libver='latest' improves performance by about 25%.

### Threading
Try `module load intel-tbb-oss/intel64/2017_20161128oss`
Result: looks the same... maybe TBB loaded by parallel studio already?
Or other threading options work well enough?  Or I didn't configure right.

### Overall gain
Run on shock flds is ~10x faster than version from July.

* Old run from July 2020

    Lap     759997  move 6.986e-03 out 7.000e-06 tot 6.993e-03
    Lap     759998  move 6.958e-03 out 7.000e-06 tot 6.965e-03
    Lap     759999  move 6.992e-03 out 6.000e-06 tot 6.998e-03
    Lap     760000  move 7.197e-03 out 1.364e-02 tot 2.083e-02
      wrote output/troch.1000
    Done, total time: 2540.935682
      init time 8.722618
      loop time 2532.213064
        first lap 8.786596
        rest laps 2513.037071000067
          rest mover 2493.6883080001135
          rest output 19.3487630000371

* Run 24 threads (forgot how many cores given by slurm)

    Lap     759997  move 5.100e-04 bc 8.400e-05 out 2.000e-06 tot 5.960e-04
    Lap     759998  move 4.770e-04 bc 8.300e-05 out 2.000e-06 tot 5.620e-04
    Lap     759999  move 4.920e-04 bc 8.100e-05 out 2.000e-06 tot 5.750e-04
    Lap     760000  move 5.290e-04 bc 8.100e-05 out 1.295e-02 tot 1.356e-02
      wrote output/troch.1000
    Done, total time: 343.257236
      init time 10.645249
      loop time 332.611987
        first lap 4.826299
        rest laps 299.09611899999004
          rest mover 239.98665600002798
          rest bc 37.09661899999347
          rest output 22.012843999882318


2020 Nov 07-08
--------------
Change from user-specified BC to param-specified, but implement on code side?
Small-ish wallclock cost, but should be much more user-friendly this way.

One thread, user- vs param-specified BC

    user-provided function

        Done, total time: 31.870584
          init time 3.861863
          loop time 28.008721
            first lap 4.651909
            rest laps 23.260312000000166
              rest mover 21.887041000000018
              rest bc 0.5587780000000313
              rest output 0.8144929999999242

    param-specified

        Done, total time: 38.073931
          init time 10.022054
          loop time 28.051877
            first lap 4.536094
            rest laps 23.421835000000073
              rest mover 21.860247999999956
              rest bc 0.7777529999999826
              rest output 0.7838339999999334

Four threads, user- vs param-specified BC

    user-provided function

        Done, total time: 33.345343
          init time 16.365275
          loop time 16.980068
            first lap 4.658985
            rest laps 12.201204000000033
              rest mover 11.07100400000007
              rest bc 0.5605820000000101
              rest output 0.56961799999994

    param-specified

        Done, total time: 33.345343
          init time 16.365275
          loop time 16.980068
            first lap 4.658985
            rest laps 12.201204000000033
              rest mover 11.07100400000007
              rest bc 0.5605820000000101
              rest output 0.56961799999994

Cost aint big. BC does not benefit much from parallelizing, should we turn off?

Can we put ghost cell handling under the hood, so user needn't deal with?
Not completely.  User has to reckon with ghost cells during prtl init, and if
they are applying user-specified BCs.

Strategy:
* `user_flds(...)` initializes flds without ghost cells

* `user_prtl(...)` can put prtl on ghost cells; the user is responsible for
  knowing how the prtl domain changes depending on BCs

* the variable "dimf" always means flds shape without ghost cells; i.e., what
  the user originally put in (and not the flds shape used internally)

* code-applied BCs remain in effect, even if user supplies custom BCs
  + code BCs always get applied before user BCs, since code BCs help to
    strictly define the valid prtl domain

Not-used strategies:
* user is responsible for supplying ghost cells + BCs, code doesn't handle
  + pro: user knows how things work; nothing hidden
  + con: more copy-paste code, more hard-coded numbers floating around

* let `user_prtl(...)` take arguments "xlim,ylim,zlim" representing the valid
  particle domain; user is not told whether xlim,ylim,zlim = dimf or not.
  + pro: right meaning
  + con: extra layer of abstraction = conceptual overhead.
    user may bypass "xlim,ylim,zlim" abstraction layer anyways
    should they choose to specify custom BCs
