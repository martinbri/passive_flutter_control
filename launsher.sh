#!/bin/bash                                                                                                                                                                                                        
                                                                                                                                                                                                                   
#SBATCH -J optm_cma

# SBATCH -N 1
# SBATCH --ntasks-per-node=1

#SBATCH -n 1
#SBATCH --ntasks-per-core=4
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

#SBATCH --partition=short

#SBATCH --begin=now

#SBATCH --output=/dev/null   # Redirige la sortie standard vers /dev/null
#SBATCH --error=/dev/null    # Redirige les erreurs standard vers /dev/null




module load python
source activate flutter_env
python passive_control_autodif.py


