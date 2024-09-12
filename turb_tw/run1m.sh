#!/bin/sh
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name="neonmrd"
#SBATCH --output="log.mrd"

#python base_h5.py 1
#mpiexec -n 52 python add_dp04.py 1
#mpiexec -n 52 python add_rad.py 1
#mpiexec -n 52 python add_2dws.py 1
#mpiexec -n 52 python add_ghflx.py 1
#python add_nlcd.py 1
#mpiexec -n 52 python add_precp.py 1
mpiexec -n 6 python raw_adjusted.py 
#mpiexec -n 52 python add_aniso.py 0
