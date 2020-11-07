README

`trochograph` is a very simple
[numba](https://numba.pydata.org/)-accelerated
test particle tracing code designed to work with
[TRISTAN-MP](https://github.com/ntoles/tristan-mp-pitp)
flds and prtl outputs.  It borrows heavily from TRISTAN-MP code
(mover, code units, input parameters, etc).
Charged particles in uniform E, B fields follow trochoid trajectories, hence
"trochograph", a wheel-curve drawing program
(see also: [Spirograph](https://en.wikipedia.org/wiki/Spirograph)).

`trochograph` uses `numba` JIT-compilation and parallel threading to go fastly.
But it only does threading; it's not massively parallel.
On an Intel Broadwell (E5-2650v4) node, the mover takes ~1e-7 to 3e-7 seconds
to advance a particle one timestep (thread-sec/prtl-lap).
TRISTAN-MP's mover takes 7e-8 seconds (cpu-sec/prtl-lap) on the same hardware
with `-O3 -ipo -xCORE-AVX2`.

`trochograph` is not much tested or heavily used.  No warranties, no
guarantees, no liability, nothing!  Use at your own risk.

Dependencies: h5py, numba

Usage: trace particles in a 1-D-like B-field tanh ramp

    python user_bzramp.py

Usage: same, but manually set thread count.  See numba docs for details.

    NUMBA_NUM_THREADS={xx} python user_bzramp.py

Usage: trace particles on TRISTAN output flds.  Does not work out of the box
due to dependency on a homebrewed Python layer for reading TRISTAN outputs, so
you must provide your own file read routine for now.

    python user_shock_flds.py
