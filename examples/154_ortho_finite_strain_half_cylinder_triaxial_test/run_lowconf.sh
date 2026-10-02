#!/bin/bash
# RKPM h = 3.33 half cylinder at sigma0 = 0 and 5 MPa, sequential (memory ~13 GB per run)
for s in 0 5; do for b in 45 0 90; do
  ./run_wrap.sh HC3c_s${s}_b${b}.log --h 3.3333 --beta $b --confine $s --umax 10.125 --inc 0.01 --kernels centres --support 1.5 --cwf 1 --flux-abs 0.5 --tag HC3c_s${s}_b${b}
done; done
