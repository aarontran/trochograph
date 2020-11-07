README

`trochograph` is a very simple
[numba](https://numba.pydata.org/)-accelerated
test particle tracing code designed to work with
[TRISTAN-MP](https://github.com/ntoles/tristan-mp-pitp)
flds and prtl outputs.  It borrows heavily from TRISTAN-MP code
(mover, code units, input parameters, etc).
Charged particles in uniform E, B fields follow trochoid trajectories, hence
"trochograph", a wheel-curve drawing program.

`trochograph` is not much tested or heavily used.  No warranties, no
guarantees, no liability, nothing!  Use at your own risk.

Dependencies: h5py, numba

Usage:

    python user_bzramp.py
    python user_shock_flds.py

