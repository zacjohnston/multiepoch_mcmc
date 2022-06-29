#!/bin/bash

if [[ $# -ne 3 ]]; then
  echo -e "\nusage: `basename $0` <N_STEPS> <N_WALKERS> <MAX_THREADS>\n"
  exit 0
fi

N_STEPS=${1}
N_WALKERS=${2}
MAX_THREADS=${3}

for i in {0..7}; do
	(( N_THREADS=2**i ))

  sbatch --job-name="${MODEL}" \
    --ntasks="${N_THREADS}" \
    --export=n_steps="${N_STEPS}",n_walkers="${N_WALKERS}" \
    --job-name="${N_WALKERS}_${N_THREADS}" \
    --output="temp/${N_WALKERS}_${N_THREADS}.out" \
    submit_job.sb

	if [ "${N_THREADS}" -eq "${MAX_THREADS}" ]; then
		break
	fi
done

