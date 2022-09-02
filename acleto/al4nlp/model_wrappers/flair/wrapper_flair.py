import logging
import time
from pathlib import Path
from typing import Union

import flair
import mlflow
import numpy as np
import torch
import transformers
from datasets import load_metric
from datasets.arrow_dataset import Dataset as ArrowDataset
from flair.data import Sentence, Corpus, Label
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import (
    set_seed,
    TrainerCallback,
)

from ...model_wrappers.flair.flair_trainer import FlairModelTrainer
from ...models.bilstm_crf import create_flair_model_tokenizer
from ...utils.general import DictWithGetattr
from ...utils.general import create_time_dict, json_dump, json_load
from ...utils.get_train_constants import get_train_constants
from ...utils.transformers_dataset import TransformersDataset

log = logging.getLogger()
transformers.logging.set_verbosity_info()

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# os.environ["CUDA_LAUNCH_BLOCKING"]="1"


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


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == "O":
            new_tags.append(tag)
        elif tag.split("-")[0] == "B":
            if i + 1 != len(tags) and tags[i + 1].split("-")[0] == "I":
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace("B-", "S-"))
        elif tag.split("-")[0] == "I":
            if i + 1 < len(tags) and tags[i + 1].split("-")[0] == "I":
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace("I-", "E-"))
        else:
            raise Exception("Invalid IOB format!")
    return new_tags


def make_flair_sentences(X, y=None, idx2tag=None, tag_type=None):
    iobes = False
    sentences = [Sentence(tokens) for tokens in X]
    if y is not None:
        assert tag_type, "Tag type is required if tags (y) are defined"
        for sentence, tags in zip(sentences, y):
            if iobes:
                intermediate_tags = [idx2tag[tag] for tag in tags]
                intermediate_tags = iob_iobes(intermediate_tags)
                for (token, tag) in zip(sentence.tokens, intermediate_tags):
                    token.add_tag_label(tag_type, Label(tag))
            else:
                for (token, tag) in zip(sentence.tokens, tags):
                    token.add_tag_label(tag_type, Label(idx2tag[tag]))
    return sentences


def transform_idx2tag(data, tag_type=None):
    unique_tags = {"O"}
    tag_to_idx = {"O": 0}
    for sentence in data:
        ner_tags = []
        for token in sentence.tokens:
            ner_tags.append(token.get_tag(tag_type).value)
        unique_tags.update(ner_tags)
        if unique_tags != set(tag_to_idx.keys()):
            for key in unique_tags:
                if key not in tag_to_idx:
                    tag_to_idx[key] = len(tag_to_idx)
    idx_to_tag = {i: tag for tag, i in tag_to_idx.items()}
    return idx_to_tag


def transformers2flair_dataset(
    data: TransformersDataset,
    idx2tag: dict,
    text_name: str = "tokens",
    label_name: str = "ner_tags",
):
    """Transform transformers dataset to flair corpus.
    Implemented because we use flair alongside transformers.
    """
    if isinstance(data, ArrowDataset):
        instances = data
    else:
        instances = data.instances
        if idx2tag is None:
            idx2tag = data.id2label
    if instances[0].get(label_name) is not None:
        corpus = make_flair_sentences(
            [instance[text_name] for instance in instances],
            [instance[label_name] for instance in instances],
            idx2tag,
            label_name,
        )
    else:
        corpus = make_flair_sentences([instance[text_name] for instance in instances])
    return corpus


class FlairModelWrapper:  # TODO: add interface
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
        num_labels: int = 9,
        id2label=None,
        *args,
        **kwargs,
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

        self.num_labels = num_labels

        if default_data_config is None:
            default_data_config = {
                "dataset_name": None,
                "text_name": "text",
                "label_name": "label",
            }
        self.data_config = dict(default_data_config)
        self.data_config["idx_to_tag"] = id2label
        # TODO: make as an argument
        self.data_config["display_tag_stats"] = "f1"

        self._optimizer_kwargs = optimizer_kwargs
        self._trainer_kwargs = trainer_kwargs
        self._scheduler_kwargs = scheduler_kwargs
        self._batch_size_kwargs = batch_size_kwargs

        self.dev_data = dev_data_kwargs.pop("dev_data")
        self._dev_kwargs = DictWithGetattr(dev_data_kwargs)
        if self.dev_data is not None:
            # now it's flair Corpus
            self.tokenized_dev_data = transformers2flair_dataset(
                self.dev_data,
                self.data_config["idx_to_tag"],
                self.data_config["text_name"],
                self.data_config["label_name"],
            )

        self.seed = seed

        if self._trainer_kwargs["serialization_dir"] is None:
            self._trainer_kwargs["serialization_dir"] = "./output/"
        self._num_checkpoints_to_save = num_checkpoints_to_save
        if time_dict_path is None:
            time_dict_path = "time_dict.json"
            create_time_dict(time_dict_path, name)
        self.time_dict_path = time_dict_path
        self.cache_dir = (
            Path(cache_dir) if cache_dir is not None else Path(f"workdir/cache_{seed}")
        )
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_model = cache_model

        self.device_name = "cuda"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        flair.device = self.device
        if model is not None:
            self.model.to(flair.device)
        # Path(serialization_dir).mkdir(exist_ok=True)

    def set_tokenizer(self):
        self.tokenizer = None

    def fit(self, train_data, from_scratch=True, is_tokenized=False, data_config=None):

        # TODO: find a better way
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
                dev_data = transformers2flair_dataset(
                    dev_data,
                    self.data_config["idx_to_tag"],
                    self.data_config["text_name"],
                    self.data_config["label_name"],
                )
        else:
            dev_data = self.dev_data
            if dev_data is not None:
                if data_config is None:
                    dev_data = self.tokenized_dev_data
                elif dev_data.tokenized_data is not None:
                    dev_data = dev_data.tokenized_data
                else:
                    dev_data = transformers2flair_dataset(
                        dev_data,
                        self.data_config["idx_to_tag"],
                        self.data_config["text_name"],
                        self.data_config["label_name"],
                    )

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
        )

        data_config = data_config if data_config is not None else self.data_config
        if not is_tokenized:
            train_data = transformers2flair_dataset(
                train_data,
                self.data_config["idx_to_tag"],
                self.data_config["text_name"],
                self.data_config["label_name"],
            )
        if from_scratch:
            self.model, self.tokenizer = create_flair_model_tokenizer(
                self.model_config,
                self.data_config["idx_to_tag"],
                self.data_config["label_name"],
                self.seed,
            )
            self.model.to(self.device)

        corpus = Corpus(train=train_data, dev=dev_data, test=[])
        # Here we use base SGD optimizer, as in prev code
        # set seed
        # torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        set_seed(self.seed)
        trainer = FlairModelTrainer(
            self.model,
            corpus,
            device=self.device_name,
            golden_label_type=self.data_config["label_name"],
        )
        start_time = time.time()
        results = trainer.train(
            base_path=self.model_config.training.trainer_args.serialization_dir,
            learning_rate=self._optimizer_kwargs.lr,
            mini_batch_size=batch_size,
            mini_batch_chunk_size=None,
            max_epochs=num_epochs,
            anneal_factor=self._scheduler_kwargs.factor,
            patience=self._scheduler_kwargs.patience,
            min_learning_rate=self._scheduler_kwargs.min_lr,
            train_with_dev=False,
            monitor_train=False,
            monitor_test=False,
            embeddings_storage_mode="cuda",
            checkpoint=True,
            save_final_model=True,
            anneal_with_restarts=True,
            anneal_with_prestarts=False,
            batch_growth_annealing=False,
            shuffle=True,
            param_selection_mode=False,
            num_workers=0,
            sampler=None,
            use_amp=False,
            amp_opt_level="O1",
            eval_on_train_fraction=0.0,
            eval_on_train_shuffle=False,
        )
        self._calculate_time(start_time, phase="fit")

        mlflow.end_run()
        self.trainer = trainer

    def get_predictions(
        self,
        data,
        is_tokenized=False,
        data_config=None,
        calculate_time=False,
        save_length_array=False,
        only_metrics=False,
    ):

        if data_config is None:
            data_config = self.data_config

        text_name = data_config["text_name"]
        label_name = data_config["label_name"]
        dataset_name = data_config["dataset_name"]
        metrics_cache_dir = (
            Path(self.cache_dir) / "metrics" if self.cache_dir is not None else None
        )

        labels = data[label_name]
        if not is_tokenized:
            data = transformers2flair_dataset(
                data, self.data_config["idx_to_tag"], text_name, label_name,
            )

        start_time = time.time()

        result = self._model_predict_loop(data)
        loss, logits, preds = result["loss"], result["logits"], result["predictions"]
        # pad logits and make np array

        compute_metrics_fn = self._return_compute_metrics_func(
            self.task,
            eval_metrics=None,
            idx_to_tag=self.data_config["idx_to_tag"],
            num_labels=self.num_labels,
            dataset_name=dataset_name,
            metrics_cache_dir=metrics_cache_dir,
            preprocessed_logits=False,
        )

        metrics_dict = compute_metrics_fn([logits, labels])
        metrics_dict["loss"] = loss
        metrics_dict = {k: v for k, v in metrics_dict.items()}

        # for normal saving
        metrics_dict = {key: float(value) for key, value in metrics_dict.items()}
        # add f1_score from model
        f1_flair = self.model.evaluate(
            data,
            self.data_config["label_name"],
            embedding_storage_mode="cuda",
            mini_batch_size=self._batch_size_kwargs["eval_batch_size"],
        )
        metrics_dict["f1_flair"] = float(f1_flair.main_score)

        if only_metrics:
            list_logits = []
        else:
            list_logits = logits.tolist()
        predictions = DictWithGetattr(
            {"predictions": list_logits, "metrics": metrics_dict}
        )

        if calculate_time:
            self._calculate_time(start_time, phase="predict")

        if save_length_array:
            self._tmp_length_array = [len(x) for x in labels]
        return predictions

    def predict_logits(
        self, data, is_tokenized=False, data_config=None, save_length_array=False
    ):
        predictions = self.get_predictions(
            data,
            is_tokenized,
            data_config,
            calculate_time=True,
            save_length_array=save_length_array,
        )
        return predictions.predictions

    def predict_proba(
        self,
        data,
        is_tokenized=False,
        data_config=None,
        to_numpy=True,
        remove_padding=False,
    ):
        logits = self.predict_logits(
            data,
            is_tokenized=is_tokenized,
            data_config=data_config,
            save_length_array=remove_padding,
        )
        probas = softmax(torch.Tensor(logits).to(self.device), dim=-1)

        if self.task == "ner" and remove_padding:
            probas = self._remove_padding(probas, self._tmp_length_array)
        elif to_numpy:
            return probas.cpu().detach().numpy()
        return probas

    def evaluate(self, data, is_tokenized=False, data_config=None, *args, **kwargs):

        predictions = self.get_predictions(
            data, is_tokenized, data_config, calculate_time=False, only_metrics=True,
        )
        return dict(predictions.metrics)

    def tokenize_data(
        self,
        tokenizer,
        data,
        task="cls",
        text_name="text",
        label_name="labels",
        **kwargs,
    ):
        # For flair model we simply transform dataset into flair corpus
        # Attention! This function supported only for compatibility with tracin
        if task == "ner":
            tokenized_data = transformers2flair_dataset(
                data, self.data_config["idx_to_tag"], text_name, label_name,
            )
            return tokenized_data
        else:
            raise NotImplementedError

    @staticmethod
    def _get_tokenize_fn_for_cls(
        tokenizer, text_name="text", label_name="labels", **kwargs
    ):
        if label_name == "labels":

            def tokenize_function(instance):
                encoded = tokenizer(instance[text_name], truncation=True, **kwargs)
                return encoded

        else:

            def tokenize_function(instance):
                encoded = tokenizer(instance[text_name], truncation=True, **kwargs)
                encoded["labels"] = instance[label_name]
                return encoded

        return tokenize_function

    @staticmethod
    def _get_tokenize_fn_for_ner(
        tokenizer, tokens_name="tokens", tags_name="ner_tags", **kwargs
    ):
        def tokenize_function(instance):
            encoded = tokenizer(
                instance[tokens_name],
                is_split_into_words=True,
                return_offsets_mapping=True,
                truncation=True,
                **kwargs,
            )
            ### Use offset mapping to restore the positions of original tokens and adapt ner tags to them
            # Initialize with zeros ('O' tag)
            encoded["labels"] = np.zeros(len(encoded["input_ids"]), dtype="int") - 100
            arr_offset = np.array(encoded["offset_mapping"])
            # Substitute values corresponding to first bpe-tokens with the tags of the tokens
            encoded["labels"][
                (arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)
            ] = instance[tags_name]
            # Remove redundant items
            del encoded["offset_mapping"]
            return encoded

        return tokenize_function

    @staticmethod
    def _return_compute_metrics_func(
        task,
        eval_metrics=None,
        num_labels=2,
        dataset_name=None,
        idx_to_tag=None,
        display_tag_stats=None,
        metrics_cache_dir=None,
        preprocessed_logits=False,
    ):

        load_func = lambda task: load_metric(
            "accuracy" if task == "cls" else "seqeval", cache_dir=metrics_cache_dir
        )
        if dataset_name is not None:
            # since metrics are not implemented for all the datasets
            try:
                metric = (
                    load_metric(dataset_name, cache_dir=metrics_cache_dir)
                    if isinstance(dataset_name, str)
                    else load_metric(*dataset_name, cache_dir=metrics_cache_dir)
                )
            except:
                metric = load_func(task)
        elif dataset_name is not None:
            metric = load_metric(*dataset_name, cache_dir=metrics_cache_dir)
        else:
            metric = load_func(task)

        if eval_metrics is not None:
            additional_metrics = [
                load_metric(add_metric, cache_dir=metrics_cache_dir)
                for add_metric in eval_metrics
            ]
        elif task == "cls":
            additional_metrics = [load_metric("f1", cache_dir=metrics_cache_dir)]
            if metric.name != "accuracy":
                additional_metrics.append(
                    load_metric("accuracy", cache_dir=metrics_cache_dir)
                )
        elif task == "ner":
            additional_metrics = (
                [load_metric("seqeval", cache_dir=metrics_cache_dir)]
                if metric.name != "seqeval"
                else []
            )
        else:
            raise NotImplementedError

        if task == "cls":
            return FlairModelWrapper._return_compute_metrics_func_for_cls(
                metric, additional_metrics, num_labels
            )
        elif task == "ner":
            return FlairModelWrapper._return_compute_metrics_func_for_ner(
                metric,
                idx_to_tag,
                additional_metrics,
                num_labels,
                display_tag_stats,
                preprocessed_logits,
            )

    @staticmethod
    def _return_compute_metrics_func_for_cls(
        metric, additional_metrics: Union[tuple, list] = tuple(), num_labels=2
    ):
        def compute_metrics(eval_preds):
            logits, labels = eval_preds
            preds = logits.argmax(axis=-1)
            metrics_dict = metric.compute(predictions=preds, references=labels)
            for add_metric in additional_metrics:
                if add_metric.name == "f1" and num_labels > 2:
                    for average in ["micro", "macro", "weighted"]:
                        add_metric_dict = add_metric.compute(
                            predictions=preds, references=labels, average=average
                        )
                        metrics_dict.update({f"f1_{average}": add_metric_dict["f1"]})
                else:
                    add_metric_dict = add_metric.compute(
                        predictions=preds, references=labels
                    )
                    metrics_dict.update(add_metric_dict)

            return metrics_dict

        return compute_metrics

    @staticmethod
    def _return_compute_metrics_func_for_ner(
        metric,
        idx_to_tag,
        additional_metrics=tuple(),
        num_labels=2,
        display_tag_stats="f1",
        preprocessed_logits=False,
    ):
        def compute_metrics(eval_preds):
            logits_ids, labels_ids = eval_preds
            if not (preprocessed_logits):
                preds_ids = logits_ids.argmax(axis=-1)
                # Convert ids to tags
                length_array = FlairModelWrapper._get_length_array_from_label_ids(
                    labels_ids
                )
                labels = FlairModelWrapper._transform_idx_to_tag(
                    labels_ids, idx_to_tag, length_array
                )
                preds = FlairModelWrapper._transform_idx_to_tag(
                    preds_ids, idx_to_tag, length_array
                )
            else:
                length_array = FlairModelWrapper._get_length_array_from_label_ids(
                    labels_ids
                )
                labels = FlairModelWrapper._transform_idx_to_tag(
                    labels_ids, idx_to_tag, length_array
                )
                preds = logits_ids
            # Compute metrics
            metrics_dict = metric.compute(predictions=preds, references=labels)

            for add_metric in additional_metrics:
                if add_metric.name == "f1" and num_labels > 2:
                    for average in ["micro", "macro", "weighted"]:
                        add_metric_dict = add_metric.compute(
                            predictions=preds, references=labels, average=average
                        )
                        metrics_dict.update({f"f1_{average}": add_metric_dict["f1"]})
                else:
                    add_metric_dict = add_metric.compute(
                        predictions=preds, references=labels
                    )
                    metrics_dict.update(add_metric_dict)
            # Individual entity scores are presented as dicts now. Need to "straighten" them
            output_dict = FlairModelWrapper._construct_metrics_dict_for_ner(
                metrics_dict, display_tag_stats
            )
            return output_dict

        return compute_metrics

    def _token_probas_to_idx(self, sentences, label_name):
        """Get token probas from token.get_tags_proba_dist and
        return its in logits format. START and STOP tags counts as O tag.
        """
        extended_tag2idx = {
            value: key for key, value in self.data_config["idx_to_tag"].items()
        }
        extended_tag2idx["<START>"] = extended_tag2idx["O"]
        extended_tag2idx["<STOP>"] = extended_tag2idx["O"]
        all_probas = []
        all_preds = []
        for sentence in tqdm(sentences):
            probas = np.zeros((len(sentence), len(self.data_config["idx_to_tag"])))
            preds = [token.get_tag(label_name).value for token in sentence]
            all_preds.append(preds)
            for idx, el in enumerate(sentence):
                token = el.get_tags_proba_dist(label_name)
                for label in token:
                    class_idx = extended_tag2idx[label.value]
                    probas[idx][class_idx] += label.score
            all_probas.append(probas)
        return all_probas, all_preds

    def _model_predict_loop(self, data):
        # do the same, but without losses
        predictions = self.model.predict(
            data,
            mini_batch_size=self._batch_size_kwargs.eval_batch_size,
            all_tag_prob=True,
            label_name=self.data_config["label_name"] + "_pred",
            return_loss=True,
        )
        loss = predictions[0].detach().cpu().item() / predictions[1]
        logits, preds = self._token_probas_to_idx(
            data, label_name=self.data_config["label_name"] + "_pred"
        )
        # pad logits to make np array
        max_len = np.max([len(el) for el in logits])
        padded_logits = np.array(
            [np.pad(logit, ((0, max_len - logit.shape[0]), (0, 0))) for logit in logits]
        )
        return {"loss": loss, "logits": padded_logits, "predictions": preds}

    @staticmethod
    def _transform_idx_to_tag(array_idx, idx_to_tag, length_array):
        array_tag = []
        for boundary, instance_idx in zip(length_array, array_idx):
            instance_tag = []
            # TODO: There was 1:boundary - why?
            for idx in instance_idx[:boundary]:
                instance_tag.append(idx_to_tag[idx])
            array_tag.append(instance_tag)
        return array_tag

    @staticmethod
    def _get_length_array_from_label_ids(label_ids, padding_idx=-100):
        length_array = []
        for instance_ids in label_ids:
            inst_length = (
                np.argmax(instance_ids[1:] == padding_idx) + 1
                if instance_ids[-1] == padding_idx
                else len(instance_ids)
            )
            length_array.append(inst_length)
        return length_array

    @staticmethod
    def _construct_metrics_dict_for_ner(
        metrics_dict, display_tag_stats: str or bool = "f1"
    ):
        output_dict = {}
        for key, val in metrics_dict.items():
            if isinstance(val, dict):
                if display_tag_stats is False or display_tag_stats is None:
                    continue
                elif display_tag_stats == "f1":
                    output_dict[key + "_f1"] = val["f1"]
                else:
                    for key_2, val_2 in val.items():
                        output_dict[key + "_" + key_2] = val_2
            else:
                output_dict[key] = val

        return output_dict

    @staticmethod
    def _remove_padding(predictions, length_array):
        preds = []
        for i, pred in enumerate(predictions):
            boundary = (
                length_array[i] - 1
            )  # Since we do not need the prediction for "[SEP]"
            pred_without_padding = pred[1:boundary].cpu().detach().numpy()
            preds.append(pred_without_padding)

        return np.array(preds)

    def _calculate_time(self, start_time, phase="fit"):
        time_work = time.time() - start_time
        time_dict = json_load(self.time_dict_path)
        time_dict[self.name + f"_{phase}"].append(time_work)
        json_dump(time_dict, self.time_dict_path)
        log.info(f"Done with the model {phase}.")
