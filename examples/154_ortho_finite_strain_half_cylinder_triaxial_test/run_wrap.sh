#!/bin/bash
# usage: run_wrap.sh <log> <args...>
log=$1; shift
env -u CXX -u CPPFLAGS PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 PYTHONFAULTHANDLER=1 nice -n 19 /home/tom/miniforge3/envs/edelweiss_next/bin/python ortho_finite_strain_half_cylinder_triaxial_test.py "$@" > $log 2>&1
echo "EXIT STATUS $?" >> $log
