import torch
from typing import Text, Dict, List, Any

from torch.utils.data import DataLoader

from libs.data_helpers.bytedataset import ByteDataset
from libs.preprocessing.normalization import NFKCNormalizer
from libs.preprocessing.tokenization import SPieceNFKCTokenizer


def get_collate_fn(
    tokenizer: SPieceNFKCTokenizer,
    normalizer: NFKCNormalizer,
    max_prompt_len: int = None,
    max_completion_len: int = None
):
    def collate_fn(items: List[Dict[Text, Any]]):
        batch_prompt_ids = []
        batch_completion_ids = []
        batch_labels = []
        batch_max_prompt_len = 0
        batch_max_completion_len = 0

        if max_prompt_len is None:
            max_input_len = float("inf")
        else:
            max_input_len = max_prompt_len
        if max_completion_len is None:
            max_output_len = float("inf")
        else:
            max_output_len = max_prompt_len

        for item in items:
            prompt_ids = tokenizer.encode(normalizer(item["prompt"]))
            completion_ids = tokenizer.encode(normalizer(item["completion"]))

            if len(prompt_ids) > max_input_len - 2:
                prompt_ids = prompt_ids[:max_input_len - 2]
            if len(completion_ids) > max_output_len - 1:
                completion_ids = completion_ids[:max_output_len - 1]

            prompt_ids = [tokenizer.cls_token_id] + prompt_ids + [tokenizer.sep_token_id]
            labels = completion_ids + [tokenizer.eos_token_id]
            completion_ids = [tokenizer.bos_token_id] + completion_ids

            if batch_max_prompt_len < len(prompt_ids):
                batch_max_prompt_len = len(prompt_ids)
            if batch_max_completion_len < len(completion_ids):
                batch_max_completion_len = len(completion_ids)

            batch_prompt_ids.append(prompt_ids)
            batch_completion_ids.append(completion_ids)
            batch_labels.append(labels)
        
        # padding
        padded_batch_prompt_ids = []
        padded_batch_completion_ids = []
        padded_batch_labels = []
        batch_prompt_attn_mask = []
        batch_completion_attn_mask = []

        for i in range(len(items)):
            # < prompt
            prompt_ids = batch_prompt_ids[i]
            prompt_pad_len = batch_max_prompt_len - len(prompt_ids)

            padded_prompt_ids = prompt_ids + [tokenizer.pad_token_id] * prompt_pad_len
            prompt_attn_mask = [1] * len(prompt_ids) + [0] * prompt_pad_len

            padded_batch_prompt_ids.append(padded_prompt_ids)
            batch_prompt_attn_mask.append(prompt_attn_mask)
            # prompt />

            # < completion
            completion_ids = batch_completion_ids[i]
            labels = batch_labels[i]
            completion_pad_len = batch_max_completion_len - len(completion_ids)

            padded_completion_ids = completion_ids + [tokenizer.pad_token_id] * completion_pad_len
            completion_attn_mask = [1] * len(completion_ids) + [0] * completion_pad_len
            padded_labels = labels + [-100] * completion_pad_len

            padded_batch_completion_ids.append(padded_completion_ids)
            padded_batch_labels.append(padded_labels)
            batch_completion_attn_mask.append(completion_attn_mask)
            # completion />

        batch = {
            "input_ids": torch.tensor(padded_batch_prompt_ids, dtype=torch.int64),
            "attention_mask": torch.tensor(batch_prompt_attn_mask, dtype=torch.int64),
            "decoder_input_ids": torch.tensor(padded_batch_completion_ids, dtype=torch.int64),
            "decoder_attention_mask": torch.tensor(batch_completion_attn_mask, dtype=torch.int64),
            "labels": torch.tensor(padded_batch_labels, dtype=torch.int64)
        }

        return batch

    return collate_fn


def create_dataloader(
    dataset: ByteDataset,
    tokenizer: SPieceNFKCTokenizer,
    normalizer: NFKCNormalizer,
    batch_size: int = 8,
    shuffle: bool = True,
    drop_last: bool = False
):
    collate_fn = get_collate_fn(tokenizer, normalizer)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn
    )
    return dataloader
