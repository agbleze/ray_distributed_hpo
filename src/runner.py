from ray import tune
from ray.train.huggingface.transformers import RayTrainReportCallback, prepare_trainer
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.air.config import RunConfig
from utils import load_config_file
import os

config = {"batch_size": tune.choice([4,8,16]),
          "learning_rate": tune.loguniform(1e-5, 1e-1),
          "lr_scheduler_type": tune.choice(categories=["linear", "cosine", "constant", "constant_with_warmup"]),
          "optim": tune.choice(categories=["adamw_torch", "sgd", "adafactor"]),
          "weight_decay": tune.uniform(lower=0.01, upper=0.1),
          "dataset_name": "legacy-datasets/banking77",
          "model_name": "microsoft/deberta-v3-small",
          "device": "cuda",
          "epochs":1,
          "cpu": 2,
          "gpu": 0.33
          }

optuna_search = OptunaSearch(metric="loss", mode="min")
#%%
def main(config_path):
    config = load_config_file(config_path=config_path)
    infra_config = config.get("infrastructure")
    model_config = config.get("model")
    storage_path = infra.get("storage_path") 
    
    model_name = model_config.get("name")
    os.makedirs(storage_path, exist_ok=True)
    print("Configuring scheduler")
    scheduler = ASHAScheduler(time_attr="training_iteration",
                              max_t=model_config.get("epochs"),
                              grace_period=2,
                              reduction_factor=2
                              )
    print("Configuring Tuner")
    tuner = tune.Tuner(trainable=tune.with_resources(trainable=tune.with_parameters(train_model),
                                           resources={"cpu": infra_config.get("cpu"),
                                                      "gpu": infra_config.get("gpu")
                                                      }
                                           ),
                       tune_config=tune.TuneConfig(metric="eval_loss",
                                                   mode="min",
                                                   scheduler=scheduler,
                                                   num_samples=3,
                                                   search_alg=optuna_search,
                                                   ),
                       run_config=RunConfig(name=f"{config.get('dataset_name')}_tune_demo",
                                            storage_path=storage_path
                                            ),
                       param_space=config,
        
                    )
    print(f"fitting tuner")
    results = tuner.fit()
    print(f"successfully fitted tuner")
    best_result = results.get_best_result(metric="eval_loss", mode="min")
    print(f"Best Validation loss: {best_result.metrics['eval_loss']}")
    print(f"Best validation accuracy: {best_result.metrics['eval_accuracy']}")
    
    return results, best_result


# %%
ray.init(
    ignore_reinit_error=True,
    _temp_dir="/tmp/ray_native",
    # 2. OFFLOAD THE HEAVY WEIGHTS: Large arrays spill to your D: drive
    _system_config={
        "object_spilling_config": '{"type": "filesystem", "params": {"directory_path": ["/mnt/d/ray_spill"]}}'
    },
    
)
res = main(config)



