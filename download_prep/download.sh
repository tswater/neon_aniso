#!/bin/sh
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --job-name="NEONdwnld"
#SBATCH --output="dwnld.log"

time python download_soil.py
printf "\n\n\n #### DONE WITH SOIL #### \n\n\n"
#time python download_rad.py
#printf "\n\n\n #### DONE WITH RAD #### \n\n\n"
#time python download_precip.py
#printf "\n\n\n #### DONE WITH PRECIP #### \n\n\n"
#time python download_w2d.py
#printf "\n\n\n #### DONE WITH WIND #### \n\n\n"

