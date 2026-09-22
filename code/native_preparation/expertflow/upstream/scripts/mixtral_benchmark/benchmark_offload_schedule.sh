#!/bin/bash

# Check if exactly one argument is provided
if [ "$#" -ne 1 ] && [ "$#" -ne 2 ]; then
    echo "Usage: $0 {xsum|wmt16} [top_n]"
    exit 1
fi

if [ "$#" -eq 2 ]; then
    top_n=$2
else
    top_n=1
fi


# Default values for parameters
data_name=${1:-xsum} # or wmt16
is_predict=true
in_order=false
seed=1234
max_new_tokens=16
num_batches=10
schedule_size=0

# Define log file and CSV file with dynamic names
log_file="mixtral_offload_schedule_experiment_log.txt"
csv_file="mixtral_offload_schedule_experiment_data.csv"
raw_log_file="mixtral_offload_schedule_raw_experiment_log.txt"

# Initialize CSV file with headers
echo "Data Name,Switch,Offload Size,Batch Size,Is Predict,In Order,Top N,Schedule Size,Seed,Max New Tokens,Num Batches,Elapsed Time,Forward Computation Time,GPU Mem(GB), Hit Rate" >> $csv_file

# Function to log and append to CSV
log_experiment() {
    local model=$1
    local offload_size=$2
    local batch_size=$3
    local is_predict=$4
    local in_order=$5
    local top_n=$6
    local schedule_size=$7
    local seed=$8
    local max_new_tokens=$9
    local num_batches=${10}
    local elapsed_time=${11}
    local forward_time=${12}
    local max_gpu_mem=${13}
    local final_hit_rate=${14}

    # Log to text file
    echo "----------------------------------------" >> $log_file
    echo "Model: $model" >> $log_file
    echo "Offload Size: $offload_size" >> $log_file
    echo "Batch Size: $batch_size" >> $log_file
    echo "Is Predict: $is_predict" >> $log_file
    echo "In Order: $in_order" >> $log_file
    echo "Top N: $top_n" >> $log_file
    echo "Schedule Size: $schedule_size" >> $log_file
    echo "Seed: $seed" >> $log_file
    echo "Max New Tokens: $max_new_tokens" >> $log_file
    echo "Num Batches: $num_batches" >> $log_file
    echo "Elapsed Time: $elapsed_time" >> $log_file
    echo "Forward Computation Time: $forward_time" >> $log_file
    echo "Max GPU memory usage: $max_gpu_mem" >> $log_file
    echo "Final hit rate: $final_hit_rate" >> $log_file
    echo "Data Name: $data_name" >> $log_file
    echo "----------------------------------------" >> $log_file

    # Append to CSV file
    echo "$data_name,$model,$offload_size,$batch_size,$is_predict,$in_order,$top_n,$schedule_size,$seed,$max_new_tokens,$num_batches,$elapsed_time,$forward_time,$max_gpu_mem,$final_hit_rate" >> $csv_file
}

# Execute commands based on the input argument
for offload_size in 7 5 4
do
    cache_size=$((8 - offload_size))
    max_batch_size=$((cache_size * 2))
    batch_size=2
    schedule_size=$((batch_size * 2))
    while [ $batch_size -le $max_batch_size ]
    do
        cmd="python benchmark/benchmark_offload.py --model_path="mistralai/Mixtral-8x7B-Instruct-v0.1" \
            --offload_size=$offload_size \
            --batch_size=$batch_size \
            --max_new_tokens=$max_new_tokens \
            --num_batches=$num_batches \
            --data_name=$data_name \
            --top_n=$top_n \
            --schedule_size=$schedule_size \
            --seed=$seed \
            --num_batches=$num_batches \
            --is_predict 2>&1 | tee -a $raw_log_file"
        echo $cmd
        # output=$(eval $cmd)
        # elapsed_time=$(echo "$output" | grep "Elapsed time" | awk '{print $3}')
        # forward_time=$(echo "$output" | grep "Forward computation time" | awk '{print $4}')
        # max_gpu_mem=$(echo "$output" | grep "Max GPU memory usage" | awk '{print $5}')
        # final_hit_rate=$(echo "$output" | grep "Final hit rate" | awk '{print $4}')
        # log_experiment "$data_name" "$model" "$offload_size" "$batch_size" "$elapsed_time" "$forward_time" "$max_gpu_mem" "$final_hit_rate"
        # echo $cache_size, $batch_size
        batch_size=$((batch_size * 2))
    done
done   
                