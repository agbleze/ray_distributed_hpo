from ray import tune
from ray.train.huggingface.transformers import RayTrainReportCallback, prepare_trainer
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.air.config import RunConfig
from transformers import AutoModelForSequenceClassification
import ray
from utils import load_config_file
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="HPO with ray")
    

#%%
def main(config_path):
    config = load_config_file(config_path=config_path)
    infra_config = config.get("infrastructure")
    model_config = config.get("model")
    storage_path = infra_config.get("storage_path") 
    hpo_config = config.get("hpo_bounds")
    temp_dir = infra_config.get("temp_dir")
    directory_path = infra_config.get("directory_path")
    
    model_name = model_config.get("name")
    num_labels = model_config.get("num_labels")
    
    optuna_search = OptunaSearch(metric="loss", mode="min")

    
    os.makedirs(storage_path, exist_ok=True)
    
    ray.init(ignore_reinit_error=True,
             _temp_dir=temp_dir,
             _system_config={"object_spilling_config": '{"type": "filesystem", "params": {"directory_path": ["/mnt/d/ray_spill"]}}'},
            )
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                                use_safetensors=True,
                                                                num_labels=num_labels
                                                                )
    
    print("Configuring scheduler")
    scheduler = ASHAScheduler(time_attr="training_iteration",
                              max_t=model_config.get("epochs"),
                              grace_period=2,
                              reduction_factor=2
                              )
    print("Configuring Tuner")
    tuner = tune.Tuner(trainable=tune.with_resources(trainable=tune.with_parameters(model),
                                           resources={"cpu": infra_config.get("cpu"),
                                                      "gpu": infra_config.get("gpu")
                                                      }
                                           ),
                       tune_config=tune.TuneConfig(metric=hpo_config.get("metric"),
                                                   mode=hpo_config.get("mode"),
                                                   scheduler=scheduler,
                                                   num_samples=hpo_config.get("num_samples"),
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
    best_result = results.get_best_result(metric=hpo_config.get("metric"), mode=hpo_config.get("mode"))
    print(f"Best Validation loss: {best_result.metrics[hpo_config.get("metric")]}")
    print(f"Best validation accuracy: {best_result.metrics['eval_accuracy']}")
    return results, best_result



if __name__ == "__main__":
    res = main(config_path)


