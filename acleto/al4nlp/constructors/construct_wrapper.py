from pathlib import Path
from typing import Union

from omegaconf.omegaconf import DictConfig

from ..model_wrappers.pytorch import (
    PytorchClsWrapper,
    PytorchNerWrapper,
)
from ..model_wrappers.transformers import (
    WrapperCls,
    WrapperNer,
)
from ..models import (
    PYTORCH_INIT_MODELS_DICT,
    FLAIR_MODELS,
)
from ..utils.general import get_num_chekpoints

TASK2WRAPPER = {
    "cls": WrapperCls,
    "ner": WrapperNer,
    "pytorch_cls": PytorchClsWrapper,
    "pytorch_ner": PytorchNerWrapper,
}

TASK_SUBCLASS_KWARGS = {
    "cls": ["id2label", "num_labels"],
    "ner": ["id2label", "num_labels"],
    "pytorch_cls": ["id2label", "num_labels", "embeddings", "word2idx"],
    "pytorch_ner": ["id2label", "num_labels", "embeddings", "word2idx"],
}


def construct_wrapper(
    config: DictConfig,
    model_cfg: DictConfig,
    dev_data,
    framework: str = None,
    labels_or_id2label=None,
    name: str = "acquisition",
    time_dict_path: Path or str = None,
    embeddings=None,
    word2idx: dict = None,
):
    if framework is None:
        framework = config.framework.name
    # in some cases we could have models both from hf and pytorch in one config, e. g. PLASM
    if (
        framework == "transformers"
        or (
            framework == "pytorch"
            and (
                model_cfg.checkpoint not in PYTORCH_INIT_MODELS_DICT.keys()
                and not model_cfg.checkpoint.endswith(".py")
            )
        )
        or (framework == "flair" and model_cfg.checkpoint not in FLAIR_MODELS)
    ):
        return construct_transformers_wrapper(
            config, model_cfg, dev_data, labels_or_id2label, name, time_dict_path,
        )
    elif framework == "pytorch" and (
        model_cfg.checkpoint in PYTORCH_INIT_MODELS_DICT.keys()
        or model_cfg.checkpoint.endswith(".py")
    ):
        return construct_pytorch_wrapper(
            config,
            model_cfg,
            dev_data,
            labels_or_id2label,
            name,
            time_dict_path,
            embeddings=embeddings,
            word2idx=word2idx,
            framework=framework,
        )
    elif framework == "flair" and model_cfg.checkpoint in FLAIR_MODELS:
        from ..model_wrappers.flair import FlairModelWrapper

        TASK2WRAPPER["flair_ner"] = FlairModelWrapper
        TASK_SUBCLASS_KWARGS["flair_ner"] = ["id2label", "num_labels"]
        return construct_pytorch_wrapper(
            config,
            model_cfg,
            dev_data,
            labels_or_id2label,
            name,
            time_dict_path,
            embeddings=embeddings,
            word2idx=word2idx,
            framework=framework,
        )
    else:
        raise NotImplementedError()


def construct_transformers_wrapper(
    config: DictConfig,
    model_cfg: DictConfig,
    dev_data,
    id2label=None,
    name: str = "acquisition",
    time_dict_path: Path or str = None,
    default_data_config: Union[dict, DictConfig, None] = None,
    tokenize_dev_data: int = True,
) -> Union[WrapperCls, WrapperNer]:

    task = model_cfg.type
    num_labels = model_cfg.num_labels if id2label is None else len(id2label)
    num_checkpoints_to_save = get_num_chekpoints(config, name)

    if default_data_config is None:
        default_data_config = getattr(config, "data", None)
    training_cfg = model_cfg.training
    dev_data_kwargs = {
        "dev_data": dev_data,
        "shuffle_dev": training_cfg.shuffle_dev,
        "size": training_cfg.dev_size,
        "tokenize_dev_data": tokenize_dev_data,
    }

    wrapper_class = TASK2WRAPPER[task]
    subclass_kwargs = {
        "id2label": id2label,
        "num_labels": num_labels,
    }
    subclass_kwargs = {
        k: v for k, v in subclass_kwargs.items() if k in TASK_SUBCLASS_KWARGS[task]
    }

    model_wrapper = wrapper_class(
        model_config=model_cfg,
        checkpoint=model_cfg.checkpoint,
        task=task,
        name=name,
        default_data_config=default_data_config,
        seed=config.seed,
        dev_data_kwargs=dev_data_kwargs,
        trainer_kwargs=training_cfg.trainer_args,
        batch_size_kwargs=training_cfg.batch_size_args,
        optimizer_kwargs=training_cfg.optimizer_args,
        scheduler_kwargs=training_cfg.scheduler_args,
        time_dict_path=time_dict_path,
        cache_dir=config.cache_dir,
        cache_model=config.cache_model_and_dataset,
        num_checkpoints_to_save=num_checkpoints_to_save,
        **subclass_kwargs,
    )

    return model_wrapper


def construct_pytorch_wrapper(
    config: DictConfig,
    model_cfg: DictConfig,
    dev_data,
    id2label=None,
    name: str = "acquisition",
    time_dict_path: Path or str = None,
    default_data_config: Union[dict, DictConfig, None] = None,
    tokenize_dev_data: int = True,
    embeddings=None,
    word2idx: dict = None,
    framework: str = "pytorch",
) -> Union[PytorchClsWrapper, PytorchNerWrapper]:

    task = model_cfg.type
    framework_task = f"{framework}_{task}"
    num_labels = model_cfg.num_labels if id2label is None else len(id2label)
    if ("tracin" not in config) or (not config.tracin.use) or (name != "target"):
        num_checkpoints_to_save = 1
    else:
        num_checkpoints_to_save = config.tracin.num_model_checkpoints
    if default_data_config is None:
        default_data_config = getattr(config, "data", None)
    training_cfg = model_cfg.training
    dev_data_kwargs = {
        "dev_data": dev_data,
        "shuffle_dev": training_cfg.shuffle_dev,
        "size": training_cfg.dev_size,
        "tokenize_dev_data": tokenize_dev_data,
    }

    wrapper_class = TASK2WRAPPER[framework_task]
    subclass_kwargs = {
        "id2label": id2label,
        "num_labels": num_labels,
        "embeddings": embeddings,
        "word2idx": word2idx,
    }
    subclass_kwargs = {
        k: v
        for k, v in subclass_kwargs.items()
        if k in TASK_SUBCLASS_KWARGS[framework_task]
    }

    model_wrapper = wrapper_class(
        model_config=model_cfg,
        checkpoint=model_cfg.checkpoint,
        task=task,
        name=name,
        default_data_config=default_data_config,
        seed=config.seed,
        dev_data_kwargs=dev_data_kwargs,
        trainer_kwargs=training_cfg.trainer_args,
        batch_size_kwargs=training_cfg.batch_size_args,
        optimizer_kwargs=training_cfg.optimizer_args,
        scheduler_kwargs=training_cfg.scheduler_args,
        time_dict_path=time_dict_path,
        cache_dir=config.cache_dir,
        cache_model=config.cache_model_and_dataset,
        num_checkpoints_to_save=num_checkpoints_to_save,
        **subclass_kwargs,
    )

    return model_wrapper
