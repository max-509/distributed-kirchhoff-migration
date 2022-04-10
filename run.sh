#!/bin/bash

for i_threads in $(seq 1 36); do
    for i_processes in $(seq 1 2); do
        export OMP_NUM_THREADS=$i_threads
        env time --output="time_${i_threads}_${i_processes}.log" mpirun -np $i_processes mpi_migration.py ./demo_data/settings.ini &> "migr_${i_threads}_$i_processes.log"

    # echo "TIME RES FOR $i_threads THREADS, $i_processes PROCESSES: $ELAPSED_TIME" >> time_res.txt
    done
done

