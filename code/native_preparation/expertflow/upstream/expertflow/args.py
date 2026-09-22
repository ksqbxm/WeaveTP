import argparse

def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="MoEOffload Arguments")

    # Add arguments
    parser.add_argument('--data_name', type=str, help='Dataset label used in logs', default='xsum')
    parser.add_argument('--model_path', type=str, help='Model state path or Hugging Face id', required=True)
    parser.add_argument('--model_name', type=str, help='Model config/name fallback; required by offload and schedule benchmarks.')
    parser.add_argument('--tokenizer_path', type=str, help='Tokenizer path or Hugging Face id. Defaults to model_path when tokenizer files exist, then model_name.')
    parser.add_argument('--dataset_path', type=str, help='Dataset path or Hugging Face id', required=True)
    parser.add_argument('--dataset_split', type=str, help='Dataset split to run', default='train')
    parser.add_argument('--predictor_path', type=str, help='Predictor checkpoint path or Hugging Face id; required by offload and schedule benchmarks.')
    parser.add_argument('--batch_size', type=int, help="Batch size for inference", default=8)
    parser.add_argument('--offload_size', type=int, help="Number of offload experts in each layer", default=8)
    parser.add_argument('--schedule_size', type=int, help="The total batch size for scheduling", default=128)
    parser.add_argument('--seed', type=int, help="Random seed for shuffling dataset", default=1234)
    parser.add_argument('--max_new_tokens', type=int, help="Maximum number of new generation tokens", default=8)
    parser.add_argument('--top_n', type=int, help='Select top n output as predict pattern', default=0)
    parser.add_argument('--num_batches', type=int, help='Run num batches data', default=8)
    parser.add_argument('--is_baseline', action='store_true', help='Whether run baseline offload')
    parser.add_argument('--is_profile', action='store_true', help='Whether profile the run')
    parser.add_argument('--is_predict', action='store_true', help='Whether run predictor')
    parser.add_argument('--is_schedule_overlap', action='store_true', help='Whether overlap schedule with inference')
    parser.add_argument('--in_order', action='store_true', help='Whether to schedule batch in order')

    # Parse the arguments
    args = parser.parse_args()

    return args
