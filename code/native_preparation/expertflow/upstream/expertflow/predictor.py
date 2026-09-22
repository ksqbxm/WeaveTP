from transformers import AutoConfig, T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union


_CONFIG_KWARGS = {
    "cache_dir",
    "force_download",
    "local_files_only",
    "proxies",
    "revision",
    "subfolder",
    "token",
    "trust_remote_code",
}

PREDICTOR_ROUTING_PRESETS = {
    "switch": {
        "num_moe_layers": 6,
        "num_experts_per_layer": 32,
        "num_experts_per_token": 1,
    },
    "mixtral": {
        "num_moe_layers": 32,
        "num_experts_per_layer": 8,
        "num_experts_per_token": 2,
    },
    "qwen": {
        "num_moe_layers": 24,
        "num_experts_per_layer": 60,
        "num_experts_per_token": 4,
    },
    "deepseek": {
        "num_moe_layers": 27,
        "num_experts_per_layer": 64,
        "num_experts_per_token": 6,
    },
}


def predictor_routing_kwargs(moe_type, num_experts_per_layer=None):
    if moe_type not in PREDICTOR_ROUTING_PRESETS:
        choices = ", ".join(sorted(PREDICTOR_ROUTING_PRESETS))
        raise ValueError(f"Unknown moe_type={moe_type!r}; choose one of: {choices}")

    kwargs = dict(PREDICTOR_ROUTING_PRESETS[moe_type])
    if num_experts_per_layer is not None:
        kwargs["num_experts_per_layer"] = int(num_experts_per_layer)
    return kwargs


def set_predictor_routing_config(
    config,
    num_moe_layers,
    num_experts_per_layer,
    num_experts_per_token,
):
    config.num_moe_layers = int(num_moe_layers)
    config.num_experts_per_layer = int(num_experts_per_layer)
    config.num_experts_per_token = int(num_experts_per_token)
    config.tie_word_embeddings = False
    return config


def load_predictor_model(pretrained_model_name_or_path, **kwargs):
    routing_keys = {
        "num_moe_layers",
        "num_experts_per_layer",
        "num_experts_per_token",
    }
    routing = {key: kwargs.pop(key) for key in routing_keys if key in kwargs}
    missing = routing_keys.difference(routing)
    if missing:
        missing_args = ", ".join(sorted(missing))
        raise ValueError(f"Missing predictor routing arguments: {missing_args}")

    config = kwargs.pop("config", None)
    if config is None:
        config_kwargs = {
            key: kwargs[key]
            for key in _CONFIG_KWARGS
            if key in kwargs
        }
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            **config_kwargs,
        )
    config = set_predictor_routing_config(config, **routing)
    return PredictorModel.from_pretrained(
        pretrained_model_name_or_path,
        config=config,
        **kwargs,
    )


class PredictorModel(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.lm_head = nn.Linear(config.d_model, config.num_moe_layers * config.num_experts_per_layer, bias=False)
        self.num_experts_per_token = config.num_experts_per_token
        self.num_experts_per_layer = config.num_experts_per_layer

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            if isinstance(labels, torch.Tensor) and labels.dim() == 2:
                decoder_input_ids = self._shift_right(labels)
            else:
                raise ValueError("decoder_input_ids must be provided when training on routing labels")

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output) # (bs, seq_len, num_layers*num_experts)
        loss = None
        if labels is not None:
            if isinstance(labels, dict):
                label_idx = labels['idx'][..., :self.num_experts_per_token].to(lm_logits.device).long()
                valid_idx = label_idx.ge(0)
                safe_idx = label_idx.clamp(min=0)
                labels = torch.zeros(
                    *label_idx.shape[:-1],
                    self.num_experts_per_layer,
                    device=lm_logits.device,
                    dtype=lm_logits.dtype,
                )
                labels.scatter_add_(-1, safe_idx, valid_idx.to(labels.dtype))
                labels.clamp_(max=1.0)
                labels = labels.view(*lm_logits.shape[:-1], self.config.num_moe_layers, self.num_experts_per_layer)
            else:
                labels = labels.to(lm_logits.device).float()
                labels = labels.view(
                    *lm_logits.shape[:-1],
                    self.config.num_moe_layers,
                    self.config.num_experts_per_layer,
                )
            lm_logits_by_layer = lm_logits.view(
                *lm_logits.shape[:-1],
                self.config.num_moe_layers,
                self.config.num_experts_per_layer,
            )
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                lm_logits_by_layer.view(-1, self.num_experts_per_layer),
                labels.view(-1, self.num_experts_per_layer),
                reduction='none')
            loss_mask = labels.view(-1, self.num_experts_per_layer).sum(-1) != 0
            if loss_mask.any():
                loss = loss[loss_mask].sum() / loss_mask.sum()
            else:
                loss = lm_logits.sum() * 0.0

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
