from typing import Literal
from transformers.models.t5.modeling_t5 import T5Config, T5ForConditionalGeneration


def init_seq2seq_model(model_size: Literal["base", "large"], tokenizer):
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    sep_token_id = tokenizer.sep_token_id

    if model_size == "base":
        d_model = 768
        d_kv = 64
        d_ff = 3072
        num_layers = 12
        num_decoder_layers = 12
        num_heads = 12
    elif model_size == "large":
        d_model = 1024
        d_kv = 64
        d_ff = 4096
        num_layers = 24
        num_decoder_layers = 24
        num_heads = 16
    
    config = T5Config(
        vocab_size=vocab_size,
        d_model=d_model,
        d_kv=d_kv,
        d_ff=d_ff,
        num_layers=num_layers,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        feed_forward_proj="gated-gelu",
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        sep_token_id=sep_token_id,
        decoder_start_token_id=bos_token_id
    )

    model = T5ForConditionalGeneration(config)
    return model
