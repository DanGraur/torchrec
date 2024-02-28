#!/usr/bin/env bash

experiment=${1:-"criteo"}
worker_count=${2:-"32"}
experiment_time=${3:-"120"}

# Prepare parameters
batch_sizes="$( for i in {4..14}; do echo $((2**$i)); done)"

# Prepare target experiment documents
EXPERIMENT_FOLDER="experiments/torchrec_${experiment}_$(date +%F_%H_%M_%S_%3N)"
RESULT_FILE="${EXPERIMENT_FOLDER}/results.csv"
mkdir -p ${EXPERIMENT_FOLDER}
command="${experiment}_runner.py"
params=""
if [ "${experiment}" == "criteo" ]; then
  params="--worker_count=${worker_count} --experiment_time=${experiment_time}"
fi
echo "system,sample_count,batch,throughput" > ${RESULT_FILE}

# Run experiment
for i in ${batch_sizes}; do
  echo "Starting experiment for ${i} batch size..."
  experiment_log="${EXPERIMENT_FOLDER}/batch_${i}.log"
  python3 ${command} --batch_size=${i} ${params} > >(tee ${experiment_log}) 2> /dev/null
  throughput=$( cat ${experiment_log} | tail -n 1 | awk '{print $3}' )
  echo "pytorch,10000000,${i},${throughput}" >> ${RESULT_FILE}
  echo "Finished experiment for ${i} batch size!"
done

echo "Results now available in: ${RESULT_FILE}"
