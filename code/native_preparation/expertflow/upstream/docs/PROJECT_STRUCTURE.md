# Project Structure

ExpertFlow is organized around the three components in the DAC 2026 paper:
Routing Path Predictor (RPP), Token Scheduler (TS), and Expert Cache Engine
(ECE). The runtime package is `expertflow`.

## Runtime Package

### Routing Path Predictor

RPP predicts future MoE expert activations before decoding reaches the target
layers.

- `expertflow/predictor.py`: compact T5-style predictor used by training,
  evaluation, and predicted-pattern generation.
- `preprocess/gen_decode_pattern.py`: records prompt ids, decode ids, and
  actual router choices from MoE runs.
- `preprocess/gen_predictor_pattern.py`: runs a trained predictor over routing
  datasets and attaches predicted paths.
- `preprocess/predictor_training.py`: shared RPP training implementation.
- `preprocess/train_pattern_predictor.py`: single CLI entry point for RPP
  training; `--model_family` selects Switch, Mixtral, Qwen, or DeepSeek presets.

### Token Scheduler

TS groups tokens with similar predicted routing paths so each scheduled batch
touches fewer experts.

- `expertflow/scheduler.py`: routing-path grouping and token reordering.
- `expertflow/generate.py`: Switch generation with scheduling.
- `expertflow/generate_mixtral.py`: Mixtral generation with scheduling.
- `expertflow/generate_qwen.py`: Qwen-MoE generation with scheduling.
- `expertflow/generate_deepseek.py`: DeepSeek-MoE generation with scheduling.
- `benchmark/benchmark_schedule.py`: scheduled-offload benchmark entry point.

### Expert Cache Engine

ECE keeps predicted active experts on GPU, offloads idle experts to CPU, and
loads missed experts on demand.

- `expertflow/expert_cache.py`: cache state, prefetching, and miss handling.
  `ExpertCache` is the baseline on-demand cache; `ExpertFlowExpertCache` is the
  predictive cache used by the paper implementation.
- `expertflow/expert_wrapper.py`: CPU/GPU expert wrappers.
- `expertflow/custom_layers.py`: MoE layer adapters that call the cache.
- `expertflow/build_model.py`: Switch offload model builder.
- `expertflow/build_mixtral_model.py`: Mixtral offload model builder.
- `expertflow/build_qwen_model.py`: Qwen-MoE offload model builder.
- `expertflow/build_deepseek_model.py`: DeepSeek-MoE offload model builder.
- `expertflow/benchmark_utils.py`: shared benchmark helpers for loading local
  or Hugging Face datasets and selecting tokenizer references.
- `benchmark/benchmark_offload.py`: predictive expert-cache benchmark entry.

## Supporting Directories

- `benchmark/`: direct Python benchmark entry points.
- `scripts/`: shell wrappers for baseline, no-offload, offload-overlap, and
  scheduled-offload sweeps.
- `docs/`: architecture and training notes.

## Ignored Local Artifacts

The repository intentionally ignores generated datasets, logs, checkpoints,
profiler traces, model snapshots, local virtual environments, and the local
paper PDF. Do not place source changes in these ignored paths:

- `logs/`
- `preprocess/logs/`
- `preprocess/eval_results/`
- `preprocess/wandb/`
- `decode_data/`
- `data/`
- `outputs/`
- `checkpoints/`
- `test/`
- `*.nsys-rep`
- `*.pt`, `*.pth`, `*.bin`, `*.safetensors`
- `DAC2026_expertflow.pdf`
