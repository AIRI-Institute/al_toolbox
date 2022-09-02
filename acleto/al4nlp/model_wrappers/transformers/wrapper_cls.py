import logging
from typing import Union, List, Tuple

from datasets.metric import Metric
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding

from .transformers_base_wrapper import (
    TransformersBaseWrapper,
)
from .wrapper_encoder import WrapperEncoder

log = logging.getLogger()


class WrapperCls(WrapperEncoder, TransformersBaseWrapper):

    """
    Returns a function that converts text to list of ids (of tokens)
    """

    @staticmethod
    def get_tokenize_function(
        tokenizer, text_name, label_name, test_mode: bool = False, **kwargs
    ):
        if test_mode:

            def tokenize_function(instances):
                encoding = tokenizer(instances[text_name], truncation=True)
                return encoding

        else:

            def tokenize_function(instances):
                encoding = tokenizer(instances[text_name], truncation=True)
                encoding["labels"] = instances[label_name]
                return encoding

        return tokenize_function

    @staticmethod
    def get_data_collator_class():
        return DataCollatorWithPadding

    @staticmethod
    def get_datasets_metric_name():
        return "accuracy"

    @staticmethod
    def get_additional_datasets_metric_names():
        return ["f1"]

    @staticmethod
    def get_model_class():
        return AutoModelForSequenceClassification

    @staticmethod
    def get_metric_name():
        return "test_accuracy"

    def get_kwargs_for_compute_metrics_fn(self):
        return {"num_labels": self.num_labels}

    @staticmethod
    def get_compute_metrics_function(
        metric,
        num_labels: int = 2,
        additional_metrics: Union[Tuple[Metric], List[Metric]] = tuple(),
    ):
        def compute_metrics(eval_preds):
            logits, labels, *inputs = eval_preds
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
