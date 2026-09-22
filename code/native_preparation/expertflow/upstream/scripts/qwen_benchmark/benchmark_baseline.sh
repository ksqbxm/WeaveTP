#!/bin/bash

# Default values for parameters
data_name=${1:-alpaca} # or wmt16
model="qwen"
is_predict=false
in_order=false
top_n=0
seed=1234
max_new_tokens=16
num_batches=20

# Define log file and CSV file with dynamic names
log_file="qwen_baseline_experiment_log.txt"
csv_file="qwen_baseline_experiment_data.csv"
raw_log_file="qwen_baseline_raw_experiment_log.txt"


# Initialize CSV file with headers
echo "Data Name,Model,Offload Size,Batch Size,Elapsed Time,Forward Computation Time,GPU Mem(GB),Hit Rate" >> $csv_file

# Function to log and append to CSV
log_experiment() {
    local model=$1
    local offload_size=$2
    local batch_size=$3
    local elapsed_time=$4
    local forward_time=$5
    local max_gpu_mem=$6
    local final_hit_rate=$7

    # Log to text file
    echo "----------------------------------------" >> $log_file
    echo "Model: $model" >> $log_file
    echo "Offload Size: $offload_size" >> $log_file
    echo "Batch Size: $batch_size" >> $log_file
    echo "Elapsed Time: $elapsed_time" >> $log_file
    echo "Forward Computation Time: $forward_time" >> $log_file
    echo "Max GPU memory usage: $max_gpu_mem" >> $log_file
    echo "Final hit rate: $final_hit_rate" >> $log_file
    echo "data_name: $data_name" >> $log_file
    echo "----------------------------------------" >> $log_file

    # Append to CSV file
    echo "$data_name,$model,$offload_size,$batch_size,$elapsed_time,$forward_time,$max_gpu_mem,$final_hit_rate" >> $csv_file
}

# Execute commands based on the input argument

for offload_size in 36 28 20 # 58 56 52 48
do
    cache_size=$((60 - offload_size))
    max_batch_size=32
    batch_size=4
    while [ $batch_size -le $max_batch_size ]
    do
        cmd="python benchmark/benchmark_offload.py --model_path="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
                                        --offload_size=$offload_size \
                                        --batch_size=$batch_size \
                                        --max_new_tokens=$max_new_tokens \
                                        --num_batches=$num_batches \
                                        --data_name=$data_name \
                                        --is_baseline 2>&1 | tee -a $raw_log_file"
        echo $cmd
        output=$(eval $cmd)
        elapsed_time=$(echo "$output" | grep "Elapsed time" | awk '{print $3}')
        forward_time=$(echo "$output" | grep "Forward computation time" | awk '{print $4}')
        max_gpu_mem=$(echo "$output" | grep "Max GPU memory usage" | awk '{print $5}')
        final_hit_rate=$(echo "$output" | grep "Final hit rate" | awk '{print $4}')
        log_experiment "$model" "$offload_size" "$batch_size" "$elapsed_time" "$forward_time" "$max_gpu_mem" "$final_hit_rate"
        echo $cache_size, $batch_size
        batch_size=$((batch_size * 2))
    done
done