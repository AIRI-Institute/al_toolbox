import logging
import os
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf

from acleto.al4nlp.active_selector import ActiveSelector
from acleto.al4nlp.utils.general import (
    log_config,
    get_time_dict_path,
    json_dump,
    json_load,
)
from acleto.al4nlp.utils.main_decorator import main_decorator

log = logging.getLogger()

OmegaConf.register_new_resolver(
    "to_string", lambda x: x.replace("/", "_").replace("-", "_")
)
OmegaConf.register_new_resolver(
    "get_patience_value", lambda dev_size: 1000 if dev_size == 0 else 5
)


@main_decorator
def run_active_learning(config, work_dir):
    # Imports inside function to set environment variables before imports
    from acleto.al4nlp.utils.data.load_data import load_data
    from acleto.al_benchmark.simulated_active_learning import (
        create_models,
        probably_fit_and_evaluate_and_log_models,
        log_model,
    )

    # Log config so that it is visible from the console
    log_config(log, config)
    log.info("Loading data...")
    cache_dir = config.cache_dir if config.cache_model_and_dataset else None
    train_instances, dev_instances, test_instances, labels_or_id2label = load_data(
        config.data, config.acquisition_model.type, config.framework.name, cache_dir,
    )

    init_n = (
        config.al.init_p_or_n
        if isinstance(config.al.init_p_or_n, int)
        else round(config.al.init_p_or_n * len(train_instances))
    )
    step_n = (
        config.al.step_p_or_n
        if isinstance(config.al.step_p_or_n, int)
        else round(config.al.init_p_or_n * len(train_instances))
    )

    if config.al.strategy_kwargs.queries_path is None:
        scores_dir = config.al.strategy_kwargs.rouges_dir
        strategy = config.al.strategy
        selector = ActiveSelector(scores_dir, len(train_instances), strategy=strategy)

        num_instances_to_query = step_n * (config.al.num_queries + 1) + init_n
        query_ids = selector.query(num_instances_to_query)
        json_dump(query_ids, Path(work_dir) / "ids_data_query.json")

    else:
        query_ids = json_load(config.al.strategy_kwargs.queries_path)

    # Set framework
    framework = config.framework.name
    # Set time dict path
    time_dict_path = get_time_dict_path(config)

    instances_on_each_iteration = init_n + np.arange(
        0, step_n * (config.al.num_queries + 1), step=step_n
    )
    json_dump(query_ids[:init_n], Path(work_dir) / f"ids_data_query_0.json")

    models, model_names, _, _ = create_models(
        framework, config, dev_instances, labels_or_id2label, time_dict_path, work_dir,
    )

    for idx_al_iter, num_instances_i in enumerate(instances_on_each_iteration):
        log.info(
            f"=================AL iteration #{idx_al_iter} started.================="
        )
        train_data = train_instances.select(query_ids[:num_instances_i])
        next_query_data = train_instances.select(
            query_ids[num_instances_i : num_instances_i + step_n]
        )
        probably_fit_and_evaluate_and_log_models(
            models,
            model_names,
            [True, True],
            log,
            work_dir,
            None,
            test_instances,
            train_data,
            framework,
        )
        if config.al.evaluate_query:
            log.info(f"############### Evaluating the query. ###############")
            for model_name, model in zip(model_names, models):
                evaluate_dict = model.evaluate(next_query_data)
                log_model(
                    log,
                    work_dir,
                    evaluate_dict,
                    f"{model_name}_evaluate_query",
                    idx_al_iter,
                    framework,
                )

        json_dump(
            query_ids[num_instances_i : num_instances_i + step_n],
            Path(work_dir) / f"ids_data_query_{idx_al_iter + 1}.json",
        )


@hydra.main(
    config_path=os.environ["HYDRA_CONFIG_PATH"],
    config_name=os.environ["HYDRA_CONFIG_NAME"],
)
def main(config):
    run_active_learning(config)


if __name__ == "__main__":
    main()
