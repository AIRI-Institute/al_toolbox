import gc
import logging
import os
import time
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Union, Tuple, List

import torch
from requests.models import HTTPError
from torch.nn.functional import softmax
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrainingArguments, Trainer, set_seed, AutoTokenizer

from .transformers_base_wrapper import (
    TransformersBaseWrapper,
)
from ...models import INIT_MODELS_DICT
from ...utils.general import (
    tensor_to_numpy,
    DictWithGetattr,
)
from ...utils.init_model_and_tokenizer import (
    get_classifier_dropout_kwargs,
    get_tokenizer_kwargs,
)
from ...utils.model_modifications import _get_pre_classifier_dropout_activation

log = logging.getLogger()


class WrapperEncoder(TransformersBaseWrapper):
    def __init__(self, id2label: dict, num_labels: int, **base_kwargs):
        super().__init__(**base_kwargs)
        self.id2label = id2label
        self.num_labels = num_labels
        self.disable_tqdm = os.environ.get("NO_TQDM", "False") in {"True", "1", "t"}

    @staticmethod
    def get_training_args_class():
        return TrainingArguments

    @staticmethod
    def get_trainer_class():
        return Trainer

    def get_additional_training_kwargs(self, **kwargs):
        return {}

    @staticmethod
    @abstractmethod
    def get_model_class():
        """
        Returns the model and the tokenizer
        """
        raise NotImplementedError("get_model_class is not implemented!")

    def get_model_and_tokenizer(
        self, model_cfg, id2label: dict = None, seed: int = 42, cache_dir=None
    ):
        set_seed(seed)  # WTF: ?
        pretrained_model_name = model_cfg.checkpoint
        num_labels = model_cfg.num_labels if id2label is None else len(id2label)
        label2id = {v: k for k, v in id2label.items()} if id2label is not None else None
        classifier_dropout = model_cfg.classifier_dropout

        model_cache_dir = Path(cache_dir) / "model" if cache_dir is not None else None
        tokenizer_cache_dir = (
            Path(cache_dir) / "tokenizer" if cache_dir is not None else None
        )

        if model_cfg.exists_in_repo:
            ### Model part
            model_class = self.get_model_class()
            kwargs = get_classifier_dropout_kwargs(
                pretrained_model_name, classifier_dropout
            )
            if num_labels is not None:
                kwargs["num_labels"] = num_labels
            try:
                model = model_class.from_pretrained(
                    pretrained_model_name,
                    id2label=id2label,
                    label2id=label2id,
                    cache_dir=model_cache_dir,
                    **kwargs,
                )
            # If there are troubles with the connection, try to build from local files
            except HTTPError:
                model = model_class.from_pretrained(
                    pretrained_model_name,
                    id2label=id2label,
                    label2id=label2id,
                    cache_dir=model_cache_dir,
                    local_files_only=True,
                    **kwargs,
                )
            if model_cfg.get("use_spectralnorm", False):
                pre_classifier, *_ = _get_pre_classifier_dropout_activation(model)
                spectral_norm(
                    pre_classifier,
                    n_power_iterations=getattr(model_cfg, "n_power_iterations", 1),
                )
            if "xlnet" in pretrained_model_name:
                model.config.use_mems_eval = False

            ### Tokenizer part
            tokenizer_kwargs = get_tokenizer_kwargs(pretrained_model_name, self.task)
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    pretrained_model_name,
                    cache_dir=tokenizer_cache_dir,
                    **tokenizer_kwargs,
                )
            except HTTPError:
                # If there are troubles with the connection, try to build from local files
                tokenizer = AutoTokenizer.from_pretrained(
                    pretrained_model_name,
                    cache_dir=tokenizer_cache_dir,
                    local_files_only=True,
                    **tokenizer_kwargs,
                )
            if model_cfg.tokenizer_max_length is not None:
                tokenizer.model_max_length = model_cfg.tokenizer_max_length
        else:
            assert (
                model_cfg.checkpoint in INIT_MODELS_DICT
            ), f"Model {model_cfg.checkpoint} is not supported currently! Please add its `init` function to active_learning/models/__init__.py"
            path_to_pretrained = Path(model_cfg.path_to_pretrained)
            init_function = INIT_MODELS_DICT[model_cfg.checkpoint]
            model, tokenizer = init_function(
                self.task,
                path_to_pretrained,
                num_labels,
                model_cfg.tokenizer_max_length,
            )

        return model, tokenizer

    def get_init_model_kwargs(self):
        return {"id2label": self.id2label}

    def get_predictions(
        self,
        data,
        is_tokenized: bool = False,
        data_config=None,
        calculate_time: bool = False,
        use_predict_loop: bool = False,
        calculate_metrics: bool = False,
        test_mode: bool = True,
        **predict_loop_kwargs,
    ):

        if data_config is None:
            data_config = self.data_config

        text_name = data_config["text_name"]
        label_name = data_config["label_name"]
        dataset_name = data_config["dataset_name"]

        if not is_tokenized:
            data = self.tokenize_data(
                tokenizer=self.tokenizer,
                data=data,
                task=self.task,
                text_name=text_name,
                label_name=label_name,
                test_mode=test_mode,
                **self.get_additional_tokenization_inference_kwargs(),
            )

        if self.task == "ner":
            self._idx_tmp_first_bpe = data["idx_first_bpe"]
            data = data.remove_columns("idx_first_bpe")

        start_time = time.time()

        if getattr(self, "trainer", None) is not None and not use_predict_loop:
            predictions = self.trainer.predict(data)
        else:
            result = self._model_predict_loop(data, **predict_loop_kwargs)
            loss, logits, extra_data = (
                result["loss"],
                result["logits"],
                result["extra_data"],
            )

            if calculate_metrics and loss is not None:
                compute_metrics_fn = self.get_compute_metrics_fn(
                    dataset_name=dataset_name,
                )
                labels = data["labels"]
                metrics_dict = compute_metrics_fn([logits, labels])
                metrics_dict["loss"] = loss
                metrics_dict = {k: v for k, v in metrics_dict.items()}
                predictions = DictWithGetattr(
                    {
                        "predictions": logits,
                        "metrics": metrics_dict,
                        "extra_data": extra_data,
                    }
                )
            else:
                predictions = DictWithGetattr(
                    {"predictions": logits, "extra_data": extra_data}
                )

        if calculate_time:
            self._calculate_time(start_time, phase="predict")
        torch.cuda.empty_cache()
        gc.collect()

        return predictions

    def predict_logits(self, data, **kwargs):
        predictions = self.get_predictions(data, **kwargs)
        return predictions.predictions

    def predict_proba(
        self,
        data,
        is_tokenized: bool = False,
        data_config=None,
        to_numpy: bool = True,
        use_predict_loop: bool = False,
        calculate_metrics: bool = False,
        test_mode: bool = True,
        **predict_loop_kwargs,
    ):
        logits = self.predict_logits(
            data,
            is_tokenized=is_tokenized,
            data_config=data_config,
            use_predict_loop=use_predict_loop,
            calculate_metrics=calculate_metrics,
            test_mode=test_mode,
            **predict_loop_kwargs,
        )
        probas = softmax(torch.Tensor(logits).to(self.model.device), dim=-1)

        if self.task == "ner":
            probas = self._remove_padding(probas, self._idx_tmp_first_bpe)
        # If padding is removed, transformation to numpy has already been done
        elif to_numpy:
            return tensor_to_numpy(probas)
        return probas

    def _model_predict_loop(
        self,
        data,
        evaluate=True,
        to_eval_mode: bool = True,
        eval_batch_size: Union[int, None] = None,
        extra_keys: Union[List[str], Tuple[str], None] = None,
    ):
        """
        Args:
            data:
            evaluate: if True, assume labels are given in the data. Otherwise, loss is not aclculated

        Returns: dict with loss and logits
        """
        if eval_batch_size is None:
            eval_batch_size = self._batch_size_kwargs.eval_batch_size
        data_collator = self.get_data_collator_class()(tokenizer=self.tokenizer)
        dataloader = DataLoader(
            data, batch_size=eval_batch_size, collate_fn=data_collator,
        )

        # Get device and the dimension of output
        for x in self.model.parameters():
            pass
        dim = x.shape[-1]
        device = x.device
        if self.task == "cls":
            logits = torch.empty(
                (len(dataloader.dataset), dim), dtype=torch.float, device=device
            )
        elif self.task == "ner":
            logits = torch.empty(
                (len(dataloader.dataset), self.tokenizer.model_max_length, dim),
                dtype=torch.float,
                device=device,
            )
        else:
            raise NotImplementedError

        extra_data = defaultdict(list)
        loss = 0
        start = 0
        if extra_keys is not None and "encoder_last_hidden_state" in extra_keys:
            extra_data["encoder_last_hidden_state"] = torch.zeros(
                len(data),
                list(self.model.parameters())[0].shape[1],
                dtype=torch.float32,
                device="cpu",
            )

        device = self.model.device
        if to_eval_mode:
            self.model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader, disable=self.disable_tqdm):
                end = start + len(batch["input_ids"])
                batch = {k: v.to(device) for k, v in batch.items()}
                predictions = self.model(**batch)
                if evaluate:
                    loss += predictions.loss.item() * len(predictions.logits)
                logits[start:end, : predictions.logits.shape[1]].copy_(
                    predictions.logits
                )
                if extra_keys is not None:
                    for key in extra_keys:
                        key_data = getattr(predictions, key)
                        if isinstance(key_data, torch.Tensor):
                            key_data = key_data.cpu()
                        if key == "encoder_last_hidden_state":
                            key_data = torch.stack(
                                [
                                    hid_st[:num_tokens, :].mean(dim=0)
                                    for hid_st, num_tokens in zip(
                                        key_data,
                                        batch["attention_mask"]
                                        .sum(dim=1)
                                        .cpu()
                                        .numpy(),
                                    )
                                ]
                            )
                            extra_data[key][start:end, :].copy_(
                                key_data, non_blocking=True
                            )
                        else:
                            extra_data[key].append(key_data)
                start = end

        logits = logits.cpu().numpy()
        if evaluate:
            loss /= len(data)
        else:
            loss = None
        if extra_keys is not None:
            for key in extra_keys:
                if key != "encoder_last_hidden_state":
                    extra_data[key] = torch.cat(extra_data[key], dim=0)

        return DictWithGetattr(
            {"loss": loss, "logits": logits, "extra_data": extra_data}
        )
