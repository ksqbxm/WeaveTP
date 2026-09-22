# Routing Predictor Training

This document covers the offline RPP workflow used by ExpertFlow.

## Data Flow

1. Run a target MoE model and collect actual router choices with
   `preprocess/gen_decode_pattern.py`.
2. Train the compact predictor with `preprocess/train_pattern_predictor.py`.
3. Optionally attach predicted routing paths to a dataset with
   `preprocess/gen_predictor_pattern.py`.
4. Use the predicted paths during scheduled/offloaded inference.

Generated datasets and checkpoints are ignored by Git. Keep them under
`decode_data/`, `preprocess/logs/`, or another ignored artifact directory.

## Generate Routing Patterns

```bash
EXPERTFLOW_MODEL_NAME=Qwen/Qwen1.5-MoE-A2.7B-Chat \
EXPERTFLOW_STATE_PATH=Qwen/Qwen1.5-MoE-A2.7B-Chat \
bash preprocess/gen_decode_pattern.sh
```

The generator writes to `decode_data/` unless `EXPERTFLOW_DECODE_DATA_DIR` or
`--output_dir` is set.

## Train Predictors

Use the single training entry point with a paper model-family preset:

```bash
cd preprocess
python3 train_pattern_predictor.py --model_family switch --data_path /path/to/pattern_dataset --output_dir ./logs/
python3 train_pattern_predictor.py --model_family mixtral --data_path /path/to/pattern_dataset --output_dir ./logs/
python3 train_pattern_predictor.py --model_family qwen --data_path /path/to/pattern_dataset --output_dir ./logs/
python3 train_pattern_predictor.py --model_family deepseek --data_path /path/to/pattern_dataset --output_dir ./logs/
```

The presets require `--data_path` and set routing defaults for:

- Switch: 6 sparse layers, 32 experts, top-1 routing.
- Mixtral: 32 sparse layers, 8 experts, top-2 routing.
- Qwen-MoE: 24 sparse layers, 60 experts, top-4 routing.
- DeepSeek-MoE: 27 sparse layers, 64 experts, top-6 routing.

Override any default with normal Hugging Face arguments, for example:

```bash
cd preprocess
python3 train_pattern_predictor.py \
  --model_family qwen \
  --data_path /path/to/local/pattern_dataset \
  --train_max_seq_size 1024 \
  --eval_max_seq_size 1024 \
  --suffix local_seq1024
```

For a model family outside the presets, call the generic entry point and pass a
real MoE config or explicit routing dimensions:

```bash
python3 train_pattern_predictor.py \
  --data_path /path/to/pattern_dataset \
  --moe_model /path/to/moe_model_or_hf_id \
  --model_name_or_path google-t5/t5-small \
  --output_dir ./logs/
```

## Attach Predicted Paths

Use a trained predictor to add `predictor_pattern` to a routing-pattern dataset:

```bash
cd preprocess
python3 gen_predictor_pattern.py \
  --model_family qwen \
  --predictor_path ./logs/<run_name>/model_state_dict \
  --data_path /path/to/pattern_dataset \
  --output_path /path/to/predicted_pattern_dataset
```

The resulting local dataset directory can be passed to benchmark entry points
with `--dataset_path /path/to/predicted_pattern_dataset`; local or Hugging Face
predictor checkpoints can be passed with `--predictor_path`. See the
[pretrained predictor weights](../README.md#pretrained-predictor-weights) for
the published model- and dataset-specific checkpoints.

## Outputs

Each run writes:

- `all_args.json` or `all_args_eval.json`
- `eval_results.json`
- `model_state_dict/`

The output directory is `--output_dir/<run_name>/`, where the run name includes
the predictor base model, dataset, split, learning rates, batch size, seed, and
predictor width.
