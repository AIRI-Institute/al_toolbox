import logging
import os
import time
from abc import abstractmethod
from pathlib import Path

import mlflow
import transformers
from datasets import load_metric
from omegaconf.omegaconf import DictConfig
from torch import cuda
from transformers import (
    set_seed,
    EarlyStoppingCallback,
    TrainerCallback,
    PrinterCallback
)

from .trainer_for_pseudo_labeled import (
    TrainerForPseudoLabeled,
)
from ...utils.general import (
    create_time_dict,
    json_dump,
    json_load,
)
from ...utils.get_train_constants import get_train_constants

log = logging.getLogger()
transformers.logging.set_verbosity_info()

CUDA_IS_AVAILABLE = (
    len(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) > 0 and cuda.is_available()
)


class TransformersBaseWrapper:  # TODO: add interface
    def __init__(
        self,
        model_config,
        checkpoint,
        task: str,
        model=None,
        tokenizer=None,
        name: str = "acquisition",
        default_data_config=None,
        seed: int = 42,
        dev_data_kwargs=None,
        trainer_kwargs=None,
        batch_size_kwargs=None,
        optimizer_kwargs=None,
        scheduler_kwargs=None,
        time_dict_path: Path or str = None,
        cache_dir: Path or str = None,
        cache_model: bool = False,
        num_checkpoints_to_save: int = 1,
    ):
        self.checkpoint = checkpoint
        self.task = task
        self.name = name

        self.model = model
        self.model_config = model_config
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.set_tokenizer()

        if default_data_config is None:
            default_data_config = {
                "dataset_name": None,
                "text_name": "text",
                "label_name": "label",
            }
        self.data_config = dict(default_data_config)

        self._optimizer_kwargs = optimizer_kwargs
        self._trainer_kwargs = trainer_kwargs
        self._scheduler_kwargs = scheduler_kwargs
        self._batch_size_kwargs = batch_size_kwargs

        self.dev_data = dev_data_kwargs.pop("dev_data")
        self._dev_kwargs = DictConfig(dev_data_kwargs)
        if self.dev_data is not None and self._dev_kwargs.pop("tokenize_dev_data"):
            self.tokenized_dev_data = self.tokenize_data(
                self.dev_data,
                self.tokenizer,
                self.data_config["text_name"],
                self.data_config["label_name"],
                test_mode=False,
                **self.get_additional_tokenization_train_kwargs(),
            )
        else:
            self.tokenized_dev_data = None

        self.seed = seed
        if self._trainer_kwargs["serialization_dir"] is None:
            self._trainer_kwargs["serialization_dir"] = "./output/"
        self._num_checkpoints_to_save = num_checkpoints_to_save
        self.use_own_trainer = (
            model_config.training.get("pseudo_labeled_label_smoothing", False) != False
            or model_config.training.get("labeled_weight", 1.0) != 1.0
        )

        self.cache_dir = (
            Path(cache_dir) if cache_dir is not None else Path(f"workdir/cache_{seed}")
        )
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_model = cache_model
        if time_dict_path is None:
            time_dict_path = self.cache_dir / f"time_dict_{name}.json"
            create_time_dict(time_dict_path, name)
        self.time_dict_path = time_dict_path

    def fit(self, train_data, from_scratch=True, is_tokenized=False, data_config=None):

        # Here we "reverse" train and test to have train_test_split returning the data in the reversed way.
        # The idea is that for validation (i.e. development set) we want to use the instances, which would
        # best approximate the "average" instances. Since the first k instances are usually selected randomly,
        # they would be a good approximation of data in contrast to later instances, which are queried
        # according to a query strategy (usually the most uncertain ones). Hence, we would want to create
        # the development set from the randomly selected instances.
        if self._dev_kwargs["size"] > 0:
            splitted = train_data.train_test_split(
                train_size=self._dev_kwargs[
                    "size"
                ],  # this is done to have the sample in reversed way
                shuffle=self._dev_kwargs["shuffle_dev"],
                seed=self.seed,
            )
            # this is done to have the sample in reversed way
            train_data = splitted["test"]
            dev_data = splitted["train"]

            if not is_tokenized:
                data_config = (
                    data_config if data_config is not None else self.data_config
                )
                dev_data = self.tokenize_data(
                    tokenizer=self.tokenizer,
                    data=dev_data,
                    task=self.task,
                    text_name=data_config["text_name"],
                    label_name=data_config["label_name"],
                    **self.get_additional_tokenization_train_kwargs(),
                )
        else:
            dev_data = self.tokenized_dev_data
            if dev_data is None and self.dev_data is not None:
                data_config = (
                    data_config if data_config is not None else self.data_config
                )
                dev_data = self.tokenize_data(
                    tokenizer=self.tokenizer,
                    data=self.dev_data,
                    task=self.task,
                    text_name=data_config["text_name"],
                    label_name=data_config["label_name"],
                    **self.get_additional_tokenization_train_kwargs(),
                )
                self.tokenized_dev_data = dev_data

        log.info(f"Training dataset size: {len(train_data)}")
        log.info(
            f"Validation dataset size: {len(dev_data) if dev_data is not None else 0}"
        )

        (
            batch_size,
            num_epochs,
            steps_per_epoch,
            scheduler_warmup_steps,
        ) = get_train_constants(
            len(train_data),
            self._trainer_kwargs.num_epochs,
            self._batch_size_kwargs.batch_size,
            self._batch_size_kwargs.adjust_batch_size,
            self._batch_size_kwargs.adjust_num_epochs,
            self._batch_size_kwargs.min_num_gradient_steps,
            self._scheduler_kwargs.warmup_steps_factor,
            self._batch_size_kwargs.min_batch_size,
            self._dev_kwargs["size"],
        )

        if from_scratch or self.model is None:
            cache_dir = self.cache_dir if self.cache_model else None
            kwargs = self.get_init_model_kwargs()
            self.model, self.tokenizer = self.get_model_and_tokenizer(
                self.model_config, seed=self.seed, cache_dir=cache_dir, **kwargs
            )

        data_config = data_config if data_config is not None else self.data_config
        if not is_tokenized:
            train_data = self.tokenize_data(
                data=train_data,
                tokenizer=self.tokenizer,
                task=self.task,
                text_name=data_config["text_name"],
                label_name=data_config["label_name"],
                test_mode=False,
                **self.get_additional_tokenization_train_kwargs(),
            )

        data_collator = self.get_data_collator_class()
        if data_collator is not None:
            data_collator_kwargs = dict(tokenizer=self.tokenizer, padding="longest")
            data_collator = data_collator(**data_collator_kwargs)

        gradient_accumulation_steps = (
            self._trainer_kwargs.accumulation.gradient_accumulation_steps
        )
        fp16 = self._trainer_kwargs.fp16.training and CUDA_IS_AVAILABLE
        fp16_full_eval = self._trainer_kwargs.fp16.evaluation and CUDA_IS_AVAILABLE
        label_smoothing_factor = self._trainer_kwargs.get("label_smoothing_factor", 0.0)
        # Only if validation sample is "dynamic"
        load_best = self._dev_kwargs.size > 0 or self._trainer_kwargs.get(
            "load_best_at_end", False
        )
        if dev_data is None:
            logging_and_evaluation_strategy = "no"
        elif (not load_best) or (self._trainer_kwargs.evaluation_strategy != "no"):
            logging_and_evaluation_strategy = self._trainer_kwargs.evaluation_strategy
        else:
            logging_and_evaluation_strategy = "epoch"
        save_strategy = (
            self._trainer_kwargs.save_strategy if (self._trainer_kwargs.get("save_strategy"))
            else "no"
            if (not load_best and self.name != "successor")
            else logging_and_evaluation_strategy
        )
        save_total_limit = self._num_checkpoints_to_save
        validation_metric = "eval_" + self._trainer_kwargs.validation_metric
        other_kwargs = {}
        if self._trainer_kwargs.accumulation.eval_accumulation_steps is not None:
            other_kwargs[
                "eval_accumulation_steps"
            ] = self._trainer_kwargs.accumulation.eval_accumulation_steps
        if logging_and_evaluation_strategy == "steps":
            other_kwargs["logging_steps"] = other_kwargs[
                "save_steps"
            ] = self._trainer_kwargs.get("logging_steps", 100)

            other_kwargs["max_steps"] = self._trainer_kwargs.get(
                "max_steps", self._batch_size_kwargs.min_num_gradient_steps
            )
        log.info(f"Load best at end: {load_best}")

        training_args_class = self.get_training_args_class()
        additional_training_kwargs = self.get_additional_training_kwargs(
            train_data=train_data
        )
        other_kwargs.update(additional_training_kwargs)

        training_args = training_args_class(
            output_dir=self._trainer_kwargs["serialization_dir"]
            or "output",  # output directory
            # Batch size args
            num_train_epochs=num_epochs,  # total number of training epochs
            per_device_train_batch_size=batch_size,  # batch size per device during training
            per_device_eval_batch_size=self._batch_size_kwargs.eval_batch_size,  # batch size for evaluation
            # Optimizer args
            adafactor=self._scheduler_kwargs.use_adafactor,
            learning_rate=self._optimizer_kwargs.lr,
            weight_decay=self._optimizer_kwargs.weight_decay,  # strength of weight decay
            # Gradient args
            max_grad_norm=self._trainer_kwargs.grad_clipping,
            label_smoothing_factor=label_smoothing_factor,
            # Scheduler args
            warmup_ratio=self._scheduler_kwargs.warmup_steps_factor,
            # fp16 args
            fp16=fp16,
            fp16_full_eval=fp16_full_eval,
            # Accumulation args
            gradient_accumulation_steps=gradient_accumulation_steps,
            # Evaluation args
            metric_for_best_model=validation_metric,
            load_best_model_at_end=load_best,
            evaluation_strategy=logging_and_evaluation_strategy,  # Evaluation is done at the end of each epoch
            logging_strategy=logging_and_evaluation_strategy,
            save_strategy=save_strategy,
            save_total_limit=save_total_limit,  # limit the total amount of checkpoints. Deletes the older checkpoints
            # Disable any integrations
            report_to="none",
            # Other args
            log_level="error",
            disable_tqdm=True,
            seed=self.seed,
            **other_kwargs,
        )
        log.info(training_args)

        callbacks = [TrainingMetricsLogger()]
        if load_best:
            callbacks.append(EarlyStoppingCallback(self._trainer_kwargs.patience))

        compute_metrics = self.get_compute_metrics_fn(
            self.data_config["dataset_name"], self._trainer_kwargs.eval_metrics,
        )

        set_seed(self.seed)
        trainer_class = (
            self.get_trainer_class()
            if not self.use_own_trainer
            else TrainerForPseudoLabeled
        )
        trainer = trainer_class(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_data,
            eval_dataset=dev_data,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
        )
        trainer.pop_callback(PrinterCallback)
        # Prevent parallelization of small models
        if (self.name == "acquisition" or self.name == "successor") and len(
            os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
        ) > 1:
            trainer.args._n_gpu = 1
        self.trainer = trainer

        log.info(f"Starting training...")
        start_time = time.time()

        trainer.train()

        self._calculate_time(start_time, phase="fit")

        mlflow.end_run()
        self.trainer.model.eval()
        self.best_metric = getattr(trainer.state, "best_metric", None)

    def evaluate(self, data, is_tokenized=False, data_config=None, **kwargs):
        predictions = self.get_predictions(
            data, is_tokenized, data_config, calculate_time=False, test_mode=False, **kwargs
        )
        return predictions.metrics

    def tokenize_data(
        self,
        data,
        tokenizer,
        text_name="text",
        label_name="labels",
        test_mode: bool = False,
        **kwargs,
    ):
        tokenize_function = self.get_tokenize_function(
            tokenizer, text_name, label_name, test_mode, **kwargs
        )
        columns_to_remove = [
            x for x in data.features.keys() if x not in ["labels", "weight"]
        ]
        # The last two arguments is a temporary fix due to datasets caching error
        return data.map(
            tokenize_function,
            batched=True,
            remove_columns=columns_to_remove,
            load_from_cache_file=False,
            cache_file_name=f"tmp.data",
        )

    def get_compute_metrics_fn(
        self, dataset_name=None, additional_metric_names=None,
    ):
        metrics_cache_dir = self.cache_dir / "metrics"
        datasets_metric_name = self.get_datasets_metric_name()
        metric = load_metric(datasets_metric_name, cache_dir=metrics_cache_dir)
        # Add metric to additional metrics in case dataset metric is available (e.g. glue-cola)
        additional_metrics = [metric]

        if dataset_name is not None:
            try:
                args = [dataset_name] if isinstance(dataset_name, str) else dataset_name
                metric = load_metric(*args, cache_dir=metrics_cache_dir)
            except:
                # Exclude the metric from additional ones
                additional_metrics = additional_metrics[:-1]

        if additional_metric_names is None:
            additional_metric_names = self.get_additional_datasets_metric_names() or []

        for add_metric_name in additional_metric_names:
            add_metric = load_metric(add_metric_name, cache_dir=metrics_cache_dir)
            additional_metrics.append(add_metric)

        kwargs = self.get_kwargs_for_compute_metrics_fn()
        return self.get_compute_metrics_function(
            metric=metric, additional_metrics=additional_metrics, **kwargs
        )

    def set_tokenizer(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.checkpoint)

    def get_init_model_kwargs(self):
        """
        Returns a dictionary of additional arguments that will be passed to model generation
        """
        return {}

    @staticmethod
    def get_additional_datasets_metric_names():
        return []

    @staticmethod
    def get_additional_tokenization_train_kwargs():
        return {}

    @staticmethod
    def get_additional_tokenization_inference_kwargs():
        return {}

    @abstractmethod
    def get_model_and_tokenizer(
        self, model_cfg, seed: int = 42, cache_dir=None, **kwargs
    ):
        """
        Returns the model and the tokenizer
        """
        raise NotImplementedError("get_model_and_tokenizer is not implemented!")

    @abstractmethod
    def get_predictions(self, *args, **kwargs):
        """
        A function to get the model predictions
        """
        raise NotImplementedError("get_predictions is not implemented!")

    @staticmethod
    @abstractmethod
    def get_tokenize_function(
        tokenizer, text_name, label_name, test_mode: bool = False, **kwargs
    ):
        """
        Returns a function to convert a text to the list of ids (of tokens)
        """

        def tokenize_function(instances):
            raise NotImplementedError("tokenize_function is not implemented!")

        return tokenize_function

    @staticmethod
    @abstractmethod
    def get_metric_name():
        """
        Returns the name of the main metric
        """
        raise NotImplementedError("get_metric_name is not implemented!")

    @staticmethod
    @abstractmethod
    def get_datasets_metric_name():
        """
        Returns the name of the metric from `datasets` `load_metric` function
        """
        raise NotImplementedError("get_datasets_metric_name is not implemented!")

    @abstractmethod
    def get_kwargs_for_compute_metrics_fn(self):
        raise NotImplementedError(
            "get_kwargs_for_compute_metrics_fn is not implemented!"
        )

    @staticmethod
    @abstractmethod
    def get_compute_metrics_function(metric, additional_metrics, **kwargs):
        """
        Returns a function to compute metrics
        """

        def compute_metrics(eval_preds):
            raise NotImplementedError("compute_metrics is not implemented!")

        return compute_metrics

    @staticmethod
    @abstractmethod
    def get_model_class():
        """
        Returns the class of model to use
        """
        raise NotImplementedError("get_model_class is not implemented!")

    @staticmethod
    @abstractmethod
    def get_trainer_class():
        """
        Return a class for trainer
        """
        raise NotImplementedError("get_trainer_class is not implemented!")

    @staticmethod
    @abstractmethod
    def get_training_args_class():
        """
        Return a class for training arguments
        """
        raise NotImplementedError("get_training_args_class is not implemented!")

    @abstractmethod
    def get_additional_training_kwargs(self, **kwargs):
        """
        Return additional training kwargs
        """
        raise NotImplementedError("get_additional_training_kwargs is not implemented!")

    @staticmethod
    @abstractmethod
    def get_data_collator_class():
        """
        Returns a data collator that will be used in `Trainer`
        """
        raise NotImplementedError("get_data_collator_class is not implemented!")

    def enable_dropout(self):
        """
        Function to enable the dropout layers during test-time
        """
        assert self.model is not None, "Model must be set to enable dropout!"
        dropout_found = False
        for m in self.model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                dropout_found = True
                m.train()

        if not dropout_found:
            self.model.train()  # e.g., for BART

    def disable_dropout(self):
        """
        Function to disable the dropout layers during test-time
        """
        assert self.model is not None, "Model must be set to disable dropout!"
        for m in self.model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.eval()

    def _calculate_time(self, start_time, phase="fit"):
        try:
            time_work = time.time() - start_time
            time_dict = json_load(self.time_dict_path)
            time_dict[self.name + f"_{phase}"].append(time_work)
            json_dump(time_dict, self.time_dict_path)
            log.info(f"Done with the model {phase}.")
        except:
            pass


class TrainingMetricsLogger(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Get train loss
        log.info(f"Epoch {metrics['epoch']}\nEvaluation Metrics:")
        log.info(
            f"Train Loss: {state.log_history[-2]['loss']}"
        )  # since [-1] is the dict with evaluation metrics
        [
            log.info(f"{k.replace('_', ' ').title()}: {v}")
            for k, v in metrics.items()
            if k != "epoch"
        ]
