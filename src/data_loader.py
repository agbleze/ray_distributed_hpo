
import ray
from datasets import load_dataset
from utils import tokenize_batch
from functools import partial
from transformers import AutoTokenizer


def data_loader(config):
    dataset_name = config.get("dataset_name")
    batch_size = config.get("batch_size")
    model_name = config.get("model_name")
    device = config.get("device")
    
    dataset = load_dataset(dataset_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    collate_fn = partial(tokenize_batch, tokenizer=tokenizer, device=device)
    
    train_data = ray.data.from_items(dataset["train"].to_list()).rename_columns({"label": "labels"})
    test_data = ray.data.from_items(dataset["test"].to_list()).rename_columns({"label": "labels"})
    train_data_iterable = train_data.iter_torch_batches(batch_size=batch_size, collate_fn=collate_fn)
    test_data_iterable = test_data.iter_torch_batches(batch_size=batch_size, collate_fn=collate_fn)
    max_steps_per_epoch = train_data.count() // batch_size
    return train_data_iterable, test_data_iterable, max_steps_per_epoch
        