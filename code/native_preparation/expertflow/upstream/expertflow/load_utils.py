import torch
import logging

def process_dataset(dataset, tokenizer, batch_size, num_expert, top_n=0, moe_type='switch'):
    len_dataset = len(dataset)
    num_batch = len_dataset // batch_size
    num_encoder_layer = 12 # for switch

    if top_n == 0:
        logging.info("Process real decode pattern")
    else:
        logging.info(f"Process top {top_n} pattern")

    for i in range(num_batch):
        prompts = []
        decode_id = []
        decode_pattern = []
        predict_pattern = []

        # Extract the batch info
        for j in range(batch_size):
            sample = dataset[i*batch_size+j]
            prompts.append(sample['prompt_text'])
            decode_id.append(sample['decode_ids'])
            decode_pattern.append(sample['decode_pattern'])
            predict_pattern.append(sample['predictor_pattern'])
        
        # Padding prompts
        input_data = tokenizer(prompts, return_tensors="pt", padding=True, return_attention_mask=True)

        decode_id = torch.Tensor(decode_id).long()
        decode_length = decode_id.shape[-1]

        # Deal with pattner
        decode_pattern = torch.Tensor(decode_pattern).long()
        if moe_type == 'switch':
            decode_pattern = decode_pattern.permute((0, 2, 1))
            predict_pattern = torch.Tensor(predict_pattern).long() # (bs, seq_len, num_layer, top3_indices)
            
            pattern = None
            # Switch Transformer use MoE in non-adjacent layer
            # Currently, we only have pattern for decoder

            # We use real pattern
            if top_n == 0:
                onehot_decode_pattern = torch.nn.functional.one_hot(
                    decode_pattern, num_classes=num_expert
                )
                batch_decode_pattern_real = onehot_decode_pattern # (batch_size, seq_len, num_moe_layers, num_expert))
                batch_encode_pattern = torch.zeros((batch_size, decode_length, num_encoder_layer, num_expert), dtype=torch.long)
                batch_decode_pattern = torch.zeros((batch_size, decode_length, num_encoder_layer, num_expert), dtype=torch.long)
                indices = list(range(1, num_encoder_layer, 2))
                batch_decode_pattern[:, :, indices, :] = batch_decode_pattern_real
                pattern = torch.cat((batch_encode_pattern, batch_decode_pattern), dim=2) # (batch_size, seq_len, num_layer, num_expert)
            else:
                top_predict_pattern = predict_pattern[..., :top_n]
                onehot_predict_pattern = torch.nn.functional.one_hot(
                    top_predict_pattern, num_classes=num_expert
                )
                batch_predict_pattern = onehot_predict_pattern.sum(-2) # Sum along and top n indices
                batch_encode_pattern = torch.zeros((batch_size, decode_length, num_encoder_layer, num_expert), dtype=torch.long)
                batch_decode_pattern = torch.zeros((batch_size, decode_length, num_encoder_layer, num_expert), dtype=torch.long)
                indices = list(range(1, num_encoder_layer, 2))
                batch_decode_pattern[:, :, indices, :] = batch_predict_pattern
                pattern = torch.cat((batch_encode_pattern, batch_decode_pattern), dim=2) # (batch_size, seq_len, num_layer, num_expert)
        elif moe_type == 'mixtral':
            num_moe_layer = 32 # for mixtral
            decode_pattern = decode_pattern.permute((0, 2, 1, 3)) # (bs, seq_len, num_layer, topk_indices)
            predict_pattern = torch.Tensor(predict_pattern).long() # (bs, seq_len, num_layer, topk_indices)
            
            pattern = None
            if top_n == 0:
                onehot_decode_pattern = torch.nn.functional.one_hot(
                    decode_pattern, num_classes=num_expert
                ) # (bs, seq_len, num_layer, topk_indices, num_expert)
                pattern = onehot_decode_pattern.sum(-2) # (batch_size, seq_len, num_moe_layers, num_expert))
            else:
                assert top_n >= 2, 'mixtral selects top 2 experts per layer, top_n should be >= 2, <=4'
                top_predict_pattern = predict_pattern[..., :top_n]
                onehot_predict_pattern = torch.nn.functional.one_hot(
                    top_predict_pattern, num_classes=num_expert
                ) # (bs, seq_len, num_layer, topn_indices, num_expert)
                pattern = onehot_predict_pattern.sum(-2) # (batch_size, seq_len, num_moe_layers, num_expert))
        elif moe_type == 'qwen':
            num_moe_layer = 24 # for qwen
            decode_pattern = decode_pattern.permute((0, 2, 1, 3)) # (bs, seq_len, num_layer, topk_indices)
            predict_pattern = torch.Tensor(predict_pattern).long() # (bs, seq_len, num_layer, topk_indices)
            
            pattern = None
            if top_n == 0:
                onehot_decode_pattern = torch.nn.functional.one_hot(
                    decode_pattern, num_classes=num_expert
                ) # (bs, seq_len, num_layer, topk_indices, num_expert)
                pattern = onehot_decode_pattern.sum(-2) # (batch_size, seq_len, num_moe_layers, num_expert))
            else:
                assert top_n >= 4, 'qwen selects top 4 experts per layer, top_n should be >= 4, <=8'
                top_predict_pattern = predict_pattern[..., :top_n]
                onehot_predict_pattern = torch.nn.functional.one_hot(
                    top_predict_pattern, num_classes=num_expert
                ) # (bs, seq_len, num_layer, topn_indices, num_expert)
                pattern = onehot_predict_pattern.sum(-2) # (batch_size, seq_len, num_moe_layers, num_expert))
        elif moe_type == 'deepseek':
            num_moe_layer = 27 # for deepseek
            decode_pattern = decode_pattern.permute((0, 2, 1, 3)) # (bs, seq_len, num_layer, topk_indices)
            predict_pattern = torch.Tensor(predict_pattern).long() # (bs, seq_len, num_layer, topk_indices)

            pattern = None
            if top_n == 0:
                onehot_decode_pattern = torch.nn.functional.one_hot(
                    decode_pattern, num_classes=num_expert
                ) # (bs, seq_len, num_layer, topk_indices, num_expert)
                pattern = onehot_decode_pattern.sum(-2) # (batch_size, seq_len, num_moe_layers, num_expert))
            else:
                assert top_n >= 6, 'deepseek selects top 6 experts per layer, top_n should be >= 6'
                top_predict_pattern = predict_pattern[..., :top_n]
                onehot_predict_pattern = torch.nn.functional.one_hot(
                    top_predict_pattern, num_classes=num_expert
                ) # (bs, seq_len, num_layer, topn_indices, num_expert)
                pattern = onehot_predict_pattern.sum(-2) # (batch_size, seq_len, num_moe_layers, num_expert))
        yield input_data, decode_id, pattern

def process_schedule_dataset(dataset, tokenizer, batch_size, num_expert, top_n=0, moe_type='switch'):
    return process_dataset(dataset, tokenizer, batch_size, num_expert, top_n, moe_type)

def load_encoder(dataset, tokenizer, batch_size, batch_idx):
    len_dataset = len(dataset)
    num_batch = len_dataset // batch_size
    num_moe_layer = 6
    num_expert = 32
    num_layer = 24
    num_encoder_layer = 12

    prompts = []
    decode_id = []
    decode_pattern = []
    i = batch_idx

    # Extract the batch info
    for j in range(batch_size):
        sample = dataset[i*batch_size+j]
        prompts.append(sample['prompt_text'])
        decode_id.append(sample['decode_ids'])
        decode_pattern.append(sample['decode_pattern'])
    
    # Padding prompts
    input_data = tokenizer(prompts, return_tensors="pt", padding=True, return_attention_mask=True)

    decode_id = torch.Tensor(decode_id)
    decode_length = decode_id.shape[-1]

    decode_pattern = torch.Tensor(decode_pattern) # (128, 6, 8)

    decode_pattern = decode_pattern.permute((0, 2, 1)) # (128, 8, 6)

    token_pattern = torch.zeros((batch_size, decode_length, num_layer, num_expert), dtype=torch.int)
    for batch_id in range(batch_size):
        for token_id in range(decode_length):
            for j in range(num_moe_layer):
                decode_layer_id = num_encoder_layer + j*2 + 1
                batch_pattern = decode_pattern[batch_id][token_id][j].to(int)
                token_pattern[batch_id][token_id][decode_layer_id][batch_pattern] = 1
    
    decode_pattern = decode_pattern.permute((1, 2, 0))
    pattern = torch.zeros((decode_length, num_layer, num_expert), dtype=torch.int)
    for token_id in range(decode_length):
        for j in range(num_moe_layer):
            decode_layer_id = num_encoder_layer + j*2 + 1
            batch_pattern = decode_pattern[token_id][j].to(int).flatten().unique().tolist()
            pattern[token_id][decode_layer_id][batch_pattern] = 1
    
    return input_data, decode_id, pattern, token_pattern

def truncate_input(input_ids, attention_mask, batch_size):
    num_batch = attention_mask.shape[0] // batch_size

    max_length = input_ids.shape[-1]

    truncate_input_ids = []
    for batch_id in range(num_batch):
        batch_mask = attention_mask[batch_id*batch_size:(batch_id+1)*batch_size]
        batch_input_ids = input_ids[batch_id*batch_size:(batch_id+1)*batch_size]
        
        local_length = batch_mask.sum(-1).max()
        truncate_id = max_length - local_length

        truncate_input_ids.append(batch_input_ids[:, truncate_id:])
    
    return truncate_input_ids
