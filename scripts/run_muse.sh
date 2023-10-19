#!/bin/bash
#SBATCH -C gpu 
#SBATCH -q regular
#SBATCH -t 06:00:00 
#SBATCH --cpus-per-task 4 
#SBATCH --gpus-per-task 1 
#SBATCH --ntasks-per-node 1
#SBATCH --nodes 1
#SBATCH -A m4031
#=
srun /global/homes/j/jaimerz/.julia/juliaup/julia-1.9.0-rc2+0.x64.linux.gnu/bin/julia MUSE.jl $(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
exit 0
# =#
CMBLensing.stop_MPI_workers()
