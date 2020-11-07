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
