#!/bin/sh
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --job-name="neonsclr"
#SBATCH --output="log.sclr"

#python base_h5.py 1
#mpiexec -n 52 python add_dp04.py 1
#mpiexec -n 52 python add_rad.py 1
#mpiexec -n 52 python add_2dws.py 1
#mpiexec -n 52 python add_ghflx.py 1
time mpiexec -n 1 python scalar_covariances.py
#time mpiexec -n 1 python multiscale.py
#python add_nlcd.py 1
#mpiexec -n 110 python fix_aniso.py 
#mpiexec -n 6 python raw_adjusted.py 
#mpiexec -n 52 python add_aniso.py 0
