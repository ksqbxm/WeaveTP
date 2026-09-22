#!/bin/bash

model_path="${EXPERTFLOW_MODEL_PATH:-google/switch-base-32}"
offload_size=16
batch_size=16
max_new_tokens=16
num_batches=20
is_profile=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment)
      experiment="$2"
      shift 2
      ;;
    --model_path)
      model_path="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

case "$experiment" in
  baseline)
    output_file="offload_switch_baseline"
    script="benchmark/benchmark_offload.py"
    extra_args="--is_baseline"
    ;;
  overlap)
    output_file="offload_switch_overlap"
    script="benchmark/benchmark_offload.py"
    extra_args="--is_predict"
    ;;
  schedule)
    output_file="offload_switch_schedule"
    script="benchmark/benchmark_schedule.py"
    schedule_size=$((2 * $batch_size))
    extra_args="--is_predict --schedule_size $schedule_size"
    ;;
  *)
    echo "Unknown experiment type: $experiment"
    exit 1
    ;;
esac

command="nsys profile --sample=none --cpuctxsw=none -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop -f true -x true -o $output_file python $script"

command+=" --model_path $model_path --offload_size $offload_size --batch_size $batch_size --max_new_tokens $max_new_tokens --num_batches $num_batches --is_profile"

command+=" $extra_args"

echo $command
# eval $command

# bash scripts/profile.sh --experiment baseline
# bash scripts/profile.sh --experiment overlap
# bash scripts/profile.sh --experiment schedule
