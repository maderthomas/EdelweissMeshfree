#!/bin/bash
# calibrated card (A_s 4, D_f 0.90): RKPM h = 3.33 at 30 and 5 MPa, beta 0/30/45/60/90, sequential (13 GB each)
for s in 30 5; do for b in 45 0 90 30 60; do
  ./run_wrap.sh HCc_s${s}_b${b}.log --h 3.3333 --beta $b --confine $s --umax 10.125 --inc 0.01 --kernels centres --support 1.5 --cwf 1 --flux-abs 0.5 --tag HCc_s${s}_b${b}
done; done
