#!/bin/bash

#SBATCH --job-name=distributed_migration
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=m.vershinin@g.nsu.ru
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition batch
#SBATCH --cpus-per-task=36
#SBATCH --time=10:00:00
#SBATCH --exclusive
#SBATCH --output=distmigr_job_%j_o.log
#SBATCH --error=distmigr_job_%j_e.log

pwd; hostname; date

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo ${OMP_NUM_THREADS}

WORKING_DIR="/home/m_vershinin/distributed-kirchhoff-migration"

cd ${WORKING_DIR}

for i in `seq 1 2 $SLURM_CPUS_PER_TASK`; do
	echo "NUMBER OF THREADS: $i"
	export OMP_NUM_THREADS=$i

	mpirun -np 1 --host gazpromnoc mpi_migration.py ./demo_data/settings.ini
done

date
