import evaluate
import numpy as np
import torch


def tokenize_batch(batch, tokenizer, device):
    text = batch["text"]
    label = batch["labels"]
    text = text.tolist() if isinstance(text, np.ndarray) else text
    token = tokenizer(text, max_length=128, truncation=True, 
                    padding="longest", 
                    return_tensors="pt"
                    )
    token["labels"] = torch.tensor(label, dtype=torch.long)
    token = {k: v.to(device) for k,v in token.items()}
    return token

def compute_metric(eval_pred, metric_name):
    metric = evaluate.load(metric_name)
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return metric.compute(predictions=preds, references=labels)
