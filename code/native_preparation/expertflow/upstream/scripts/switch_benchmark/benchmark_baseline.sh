#!/bin/bash
MODEL_ROOT=${EXPERTFLOW_MODEL_ROOT:-google}

# Check if exactly one argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 {switch-32|switch-64|switch-128}"
    exit 1
fi

# Read the input argument
switch=$1

# Default values for parameters
data_name="wmt16"
is_predict=false
in_order=false
top_n=0
seed=1234
max_new_tokens=16
num_batches=20

# Define log file and CSV file with dynamic names
log_file="baseline_experiment_log.txt"
csv_file="baseline_experiment_data.csv"
raw_log_file="baseline_raw_experiment_log.txt"


# Initialize CSV file with headers
echo "Data Name,Switch,Offload Size,Batch Size,Elapsed Time,Forward Computation Time,GPU Mem(GB),Hit Rate" >> $csv_file

# Function to log and append to CSV
log_experiment() {
    local switch=$1
    local offload_size=$2
    local batch_size=$3
    local elapsed_time=$4
    local forward_time=$5
    local max_gpu_mem=$6
    local final_hit_rate=$7

    # Log to text file
    echo "----------------------------------------" >> $log_file
    echo "Switch: $switch" >> $log_file
    echo "Offload Size: $offload_size" >> $log_file
    echo "Batch Size: $batch_size" >> $log_file
    echo "Elapsed Time: $elapsed_time" >> $log_file
    echo "Forward Computation Time: $forward_time" >> $log_file
    echo "Max GPU memory usage: $max_gpu_mem" >> $log_file
    echo "Final hit rate: $final_hit_rate" >> $log_file
    echo "data_name: $data_name" >> $log_file
    echo "----------------------------------------" >> $log_file

    # Append to CSV file
    echo "$data_name,$switch,$offload_size,$batch_size,$elapsed_time,$forward_time,$max_gpu_mem,$final_hit_rate" >> $csv_file
}

# Execute commands based on the input argument
case $switch in
    switch-32)
        for offload_size in 28 24 16
        do
            cache_size=$((32 - offload_size))
            # max_batch_size=$((cache_size * 2))
            max_batch_size=32
            batch_size=$max_batch_size
            while [ $batch_size -le $max_batch_size ]
            do
                cmd="python benchmark/benchmark_offload.py --model_path=${MODEL_ROOT}/switch-base-32 \
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
                log_experiment "$switch" "$offload_size" "$batch_size" "$elapsed_time" "$forward_time" "$max_gpu_mem" "$final_hit_rate"
                echo $cache_size, $batch_size
                batch_size=$((batch_size * 2))
            done
        done   
        ;;
    switch-64)
        for offload_size in 60 56 48
        do
            cache_size=$((64 - offload_size))
            # max_batch_size=$((cache_size * 2))
            max_batch_size=32
            batch_size=$max_batch_size
            while [ $batch_size -le $max_batch_size ]
            do
                cmd="python benchmark/benchmark_offload.py --model_path=${MODEL_ROOT}/switch-base-64 \
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
                log_experiment "$switch" "$offload_size" "$batch_size" "$elapsed_time" "$forward_time" "$max_gpu_mem" "$final_hit_rate"
                echo $cache_size, $batch_size
                batch_size=$((batch_size * 2))
            done
        done
        ;;
    switch-128)
        for offload_size in 124 120 112
        do
            cache_size=$((128 - offload_size))
            # max_batch_size=$((cache_size * 2))
            max_batch_size=32
            batch_size=$max_batch_size
            while [ $batch_size -le $max_batch_size ]
            do
                cmd="python benchmark/benchmark_offload.py --model_path=${MODEL_ROOT}/switch-base-128 \
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
                log_experiment "$switch" "$offload_size" "$batch_size" "$elapsed_time" "$forward_time" "$max_gpu_mem" "$final_hit_rate"
                echo $cache_size, $batch_size
                batch_size=$((batch_size * 2))
            done
        done
        ;;
    *)
        echo "Invalid argument: $switch"
        echo "Usage: $0 {switch-32|switch-64|switch-128}"
        exit 1
        ;;
esac