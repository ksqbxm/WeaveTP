# ExpertFlow

Implementation for **ExpertFlow: Efficient Mixture-of-Experts Inference via
Predictive Expert Caching and Token Scheduling**.

ExpertFlow reduces GPU memory pressure for Mixture-of-Experts (MoE) inference by
combining:

- **Routing Path Predictor (RPP)**: predicts future expert activations.
- **Expert Cache Engine (ECE)**: prefetches predicted active experts to GPU and
  handles misses on demand.
- **Token Scheduler (TS)**: batches tokens with similar predicted routing paths.

Runtime package: `expertflow`.

## Paper

**ExpertFlow: Efficient Mixture-of-Experts Inference via Predictive Expert
Caching and Token Scheduling**
DAC 2026

Paper link: <https://doi.org/10.1145/3770743.3804292>

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The benchmark scripts require CUDA and enough CPU memory to hold offloaded
experts.

## Model Checkpoints

Most commands accept `--model_path` or `--state_path`. The value can be a
Hugging Face model id, a local model directory, or a glob that resolves to a
local checkpoint snapshot.

Example model references:

```bash
--model_path google/switch-base-32 --model_name google/switch-base-32
--model_path /path/to/switch-base-32 --model_name google/switch-base-32
```

The shell wrappers under `scripts/` are paper-sweep templates. Before running
them, edit or extend the generated benchmark commands so they pass the explicit
model, tokenizer, dataset, and predictor artifact arguments described below.

For Switch sweep scripts, set `EXPERTFLOW_MODEL_ROOT` when checkpoints are
stored locally, or edit the generated `--model_path` values directly. Without
`EXPERTFLOW_MODEL_ROOT`, the scripts use Hugging Face model roots such as
`google`.

Direct benchmark entry points require explicit artifact references:

- `--model_path`: local checkpoint directory, glob, or Hugging Face model id.
- `--model_name`: model config/name fallback used by the offload builder.
- `--tokenizer_path`: tokenizer directory or Hugging Face id.
- `--dataset_path`: local `datasets.save_to_disk()` directory or Hugging Face
  dataset id.
- `--dataset_split`: dataset split, `train` by default.
- `--predictor_path`: local RPP checkpoint directory or Hugging Face id for
  predictive offload and scheduled benchmarks.

### Pretrained Predictor Weights

Pretrained RPP weights are hosted in the
[`expertflow-dac`](https://huggingface.co/expertflow-dac/models) Hugging Face
namespace. Pass the matching repository id directly to `--predictor_path`;
the checkpoint is downloaded and cached automatically.

| Target MoE model | Dataset | `--predictor_path` |
| --- | --- | --- |
| DeepSeek-MoE-16B | AIME2024 | `expertflow-dac/t5-small_dff2048_dmodel32_token-pattern-predictor_deepseek_moe_16b_aime2024` |
| Qwen1.5-MoE-A2.7B | Alpaca | `expertflow-dac/t5-small_dff2048_dmodel32_token-pattern-predictor_qwen1.5MoEA2.7B_alpaca` |
| Mixtral-8x7B-Instruct-v0.1 | XSum | `expertflow-dac/t5-small_dff2048_dmodel32_token-pattern-predictor_mixtral8x7bInstructv0.1_xsum` |
| Mixtral-8x7B-Instruct-v0.1 | WMT16 | `expertflow-dac/t5-small_dff2048_dmodel32_token-pattern-predictor_mixtral8x7bInstructv0.1_wmt16` |
| Switch-32 | XSum | `expertflow-dac/t5-small_dff2048_dmodel32_token-pattern-predictor_switch32_xsum` |
| Switch-32 | WMT16 | `expertflow-dac/t5-small_dff2048_dmodel32_token-pattern-predictor_switch32_wmt16` |
| Switch-64 | XSum | `expertflow-dac/t5-small_dff2048_dmodel32_token-pattern-predictor_switch64_xsum` |
| Switch-64 | WMT16 | `expertflow-dac/t5-small_dff2048_dmodel32_token-pattern-predictor_switch64_wmt16` |
| Switch-128 | XSum | `expertflow-dac/t5-small_dff2048_dmodel32_token-pattern-predictor_switch128_xsum` |
| Switch-128 | WMT16 | `expertflow-dac/t5-small_dff2048_dmodel32_token-pattern-predictor_switch128_wmt16` |

Example with local model weights and explicit dataset/predictor refs:

```bash
python benchmark/benchmark_offload.py \
  --model_path /path/to/switch-base-32 \
  --model_name google/switch-base-32 \
  --tokenizer_path /path/to/switch-base-32 \
  --dataset_path /path/to/predicted_pattern_dataset \
  --predictor_path /path/to/rpp_checkpoint \
  --is_predict
```

## Quick Start

Run ExpertFlow offloading with predicted paths:

```bash
python benchmark/benchmark_offload.py \
  --model_path /path/to/switch-base-32 \
  --model_name google/switch-base-32 \
  --tokenizer_path /path/to/switch-base-32 \
  --dataset_path /path/to/predicted_pattern_dataset \
  --predictor_path expertflow-dac/t5-small_dff2048_dmodel32_token-pattern-predictor_switch32_wmt16 \
  --data_name wmt16 \
  --offload_size 16 \
  --batch_size 16 \
  --max_new_tokens 16 \
  --num_batches 20 \
  --is_predict
```

Run scheduled offloading:

```bash
python benchmark/benchmark_schedule.py \
  --model_path /path/to/switch-base-32 \
  --model_name google/switch-base-32 \
  --tokenizer_path /path/to/switch-base-32 \
  --dataset_path /path/to/predicted_pattern_dataset \
  --predictor_path expertflow-dac/t5-small_dff2048_dmodel32_token-pattern-predictor_switch32_wmt16 \
  --data_name wmt16 \
  --offload_size 16 \
  --batch_size 16 \
  --schedule_size 32 \
  --max_new_tokens 16 \
  --num_batches 20 \
  --is_predict
```

`benchmark_schedule.py` currently supports Switch models only.

Run the baseline on-demand cache:

```bash
python benchmark/benchmark_offload.py \
  --model_path /path/to/switch-base-32 \
  --model_name google/switch-base-32 \
  --tokenizer_path /path/to/switch-base-32 \
  --dataset_path /path/to/predicted_pattern_dataset \
  --predictor_path expertflow-dac/t5-small_dff2048_dmodel32_token-pattern-predictor_switch32_wmt16 \
  --data_name wmt16 \
  --offload_size 16 \
  --batch_size 16 \
  --max_new_tokens 16 \
  --num_batches 20 \
  --is_baseline
```

## Routing-Path Predictor

Generate routing-pattern data:

```bash
EXPERTFLOW_MODEL_NAME=Qwen/Qwen1.5-MoE-A2.7B-Chat \
EXPERTFLOW_STATE_PATH=Qwen/Qwen1.5-MoE-A2.7B-Chat \
bash preprocess/gen_decode_pattern.sh
```

Generated datasets are written to `decode_data/` by default. Override this with
`EXPERTFLOW_DECODE_DATA_DIR` or `--output_dir`.

Train RPP models:

```bash
cd preprocess
python3 train_pattern_predictor.py --model_family switch --data_path /path/to/pattern_dataset --output_dir ./logs/
python3 train_pattern_predictor.py --model_family mixtral --data_path /path/to/pattern_dataset --output_dir ./logs/
python3 train_pattern_predictor.py --model_family qwen --data_path /path/to/pattern_dataset --output_dir ./logs/
python3 train_pattern_predictor.py --model_family deepseek --data_path /path/to/pattern_dataset --output_dir ./logs/
```

Attach predicted paths to a routing-pattern dataset:

```bash
cd preprocess
python3 gen_predictor_pattern.py \
  --model_family qwen \
  --predictor_path ./logs/<run_name>/model_state_dict \
  --data_path /path/to/pattern_dataset \
  --output_path /path/to/predicted_pattern_dataset
```

See [docs/TRAINING.md](docs/TRAINING.md) for the full RPP data and training
workflow.

## Benchmark Scripts

Paper-style sweep wrappers are under `scripts/`:

```bash
bash scripts/switch_benchmark/benchmark_baseline.sh switch-32
bash scripts/switch_benchmark/benchmark_offload_overlap.sh switch-32
bash scripts/switch_benchmark/benchmark_offload_schedule.sh switch-32
```

Model-family wrappers are available for Switch, Mixtral, Qwen-MoE, and
DeepSeek-MoE where supported by the current benchmark path. The wrappers call
the direct benchmark entry points, so set or edit the required model, dataset,
tokenizer, and predictor artifact arguments before running paper-style sweeps.
Scheduled-offload sweeps are currently Switch-only.

For Nsight Systems profiling:

```bash
EXPERTFLOW_MODEL_PATH=google/switch-base-32 \
bash scripts/profile.sh --experiment overlap
```

`scripts/profile.sh` prints the `nsys profile` command by default. Uncomment
`eval $command` in that script to execute profiling directly.

## Repository Layout

- `expertflow/`: runtime implementation for prediction, caching, scheduling,
  generation, and offload model construction.
- `expertflow/models/`: local model definitions for Switch, Mixtral, Qwen-MoE,
  and DeepSeek-MoE.
- `preprocess/`: routing-pattern dataset generation and RPP training.
- `benchmark/`: direct benchmark entry points.
- `scripts/`: sweep and profiling wrappers.
- `docs/`: implementation notes and paper-to-code mapping.

See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for a component-level
map of the code.

## Citation

```bibtex
@inproceedings{he2026expertflow,
  title = {ExpertFlow: Efficient Mixture-of-Experts Inference via Predictive Expert Caching and Token Scheduling},
  author = {He, Xin and Zhang, Shunkang and Tang, Kaijie and Shi, Shaohuai and Wang, Yuxin and Zeng, Zihao and Tang, Zhenheng and Chu, Xiaowen and Yin, Haiyan and Tsang, Ivor W. and Ong, Yew Soon},
  booktitle = {Proceedings of the 63rd ACM/IEEE Design Automation Conference (DAC)},
  year = {2026},
  doi = {10.1145/3770743.3804292}
}
```
