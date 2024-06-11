#!/bin/sh
#SBATCH --nodes=4
#SBATCH --tasks-per-node=1
#SBATCH --job-name="neon30m"
#SBATCH --output="log.30m"

python base_h5.py 0
mpiexec -n 54 python add_dp04.py 0
mpiexec -n 54 python add_rad.py 0
mpiexec -n 54 python add_2dws.py 0
mpiexec -n 54 python add_ghflx.py 0
python add_nlcd.py 0
mpiexec -n 54 python add_precp.py 0
mpiexec -n 54 python add_aniso.py 0

