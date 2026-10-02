from .data_loader import data_loader
from utils import compute_metric
from functools import partial


def train_model(config):
    from ray.train.huggingface.transformers import prepare_trainer
    from transformers import (AutoModelForSequenceClassification, 
                            TrainingArguments, 
                            Trainer, pipeline,
                            PrinterCallback
                            )
    import ray
    
    model_name = config.get("model_name")
    batch_size = config.get("batch_size")
    metric_name = config.get("metric_name", "accuracy")
    epochs = config.get("epochs")
    
    train_data_iterable, test_data_iterable, max_steps_per_epoch = data_loader(config=config)
    
    metric = partial(compute_metric, metric_name=metric_name)


    print("Getting Model")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                                use_safetensors=True,
                                                                num_labels=77
                                                                )
    max_step = max_steps_per_epoch * epochs
    print("Configuring TrainingArguments and Trainer")
    args = TrainingArguments(num_train_epochs=epochs,
                             per_device_train_batch_size=batch_size,
                             per_device_eval_batch_size=batch_size,
                             learning_rate=config.get("learning_rate"),
                             lr_scheduler_type=config.get("lr_scheduler_type"),
                             optim=config.get("optim"),
                             eval_strategy="epoch",
                             save_strategy="best",
                             metric_for_best_model="accuracy",
                             max_steps=max_step,
                             weight_decay=config.get("weight_decay"),
                             )
    trainer = Trainer(model=model,
                      args=args,
                      train_dataset=train_data_iterable,
                      eval_dataset=test_data_iterable,
                      compute_metrics=metric
                      )
    print("add callbacks")
    trainer.add_callback(PrinterCallback())
    print("prepare trainer")
    trainer = prepare_trainer(trainer)
    
    print(" training...")
    trainer.train()
    print("evaluating ...")
    eval_metrics = trainer.evaluate()
    print(f"Evaluation metrics: {eval_metrics}")
    
    ray.tune.report(metrics=eval_metrics)
    
