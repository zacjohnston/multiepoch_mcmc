#!/bin/bash

if [[ $# -ne 4 ]]; then
  echo -e "\nusage: `basename $0` <MIN_WALKERS> <MAX_WALKERS> <MAX_THREADS>\n"
  exit 0
fi

MIN_WALKERS=${1}
MAX_WALKERS=${2}
MAX_THREADS=${3}

MAX_STEPS=160

for i in {0..5}; do
  (( N_WALKERS=MIN_WALKERS*2**i ))
  (( N_STEPS=MAX_STEPS/2**i ))

  for j in {0..7}; do
    (( N_THREADS=2**j ))
    
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

  if [ "${N_WALKERS}" -eq "${MAX_WALKERS}" ]; then
    break
  fi
done
