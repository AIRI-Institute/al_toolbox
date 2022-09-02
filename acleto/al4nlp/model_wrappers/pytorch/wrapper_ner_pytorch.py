import logging
from typing import Union, Tuple, List, Dict

import numpy as np
from datasets.metric import Metric
from torch import Tensor
from transformers import DataCollatorForTokenClassification

from ..pytorch.wrapper_encoder_pytorch import PytorchEncoderWrapper
from ..transformers.transformers_base_wrapper import (
    TransformersBaseWrapper,
)
from ...models import PYTORCH_INIT_MODELS_DICT

log = logging.getLogger()


class PytorchNerWrapper(PytorchEncoderWrapper, TransformersBaseWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def align_labels_with_tokens(
        labels: List[int], word_ids: List[int], tag_for_subtokens=None
    ):
        """
        Function to adjust the labels according to the tokenized input
        :param labels: original list of tags
        :param word_ids: list with ids of the original words in the tokenized list of tokens;
        can be extracted via `inputs.word_ids()`, where `inputs = tokenizer(...)`
        :param tag_for_subtokens: how to tokenize subtokens (BPEs) of a word. Have three options:
        imagine we have a word `loving` -> [`lov`, `###ing`].
        Option 1 (default) = None: `lov` -> B-TAG, `###ing` -> O
        Option 2 = 'i': `lov` -> B-TAG, `###ing` -> I-TAG
        Option 3 = 'b': `lov` -> B-TAG, `###ing` -> B-TAG
        """
        assert tag_for_subtokens in [
            None,
            "i",
            "b",
        ], "Param :tag_for_subtokens must be one of `None`, `'i'`, `'b'`!"
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id is None:
                # Special token
                new_labels.append(-100)
            elif word_id != current_word:
                # Start of a new word!
                current_word = word_id
                new_labels.append(labels[word_id])
            else:
                # Same word as previous token
                if tag_for_subtokens is not None:
                    label = labels[word_id]
                    # If the label is B-XXX we change it to I-XXX
                    if tag_for_subtokens == "i" and label % 2 == 1:
                        label += 1
                else:
                    label = -100
                new_labels.append(label)

        return new_labels

    """
    Returns a function that converts text to list of ids (of tokens)
    """

    @staticmethod
    def get_tokenize_function(
        tokenizer,
        text_name,
        label_name,
        test_mode: bool = False,
        save_first_bpe_mask: bool = True,
        **kwargs
    ):
        if test_mode:

            def tokenize_function(examples):
                tokens = examples[text_name]
                tokenized_inputs = tokenizer(
                    tokens, truncation=True, is_split_into_words=True
                )
                # Save ids of the first BPEs if necessary
                if save_first_bpe_mask:
                    idx_first_bpe = []
                    for i in range(len(tokens)):
                        word_ids = tokenized_inputs.word_ids(i)
                        idx_first_bpe.append(
                            [
                                i
                                for (i, x) in enumerate(word_ids[1:], 1)
                                if ((x != word_ids[i - 1]) and (x is not None))
                            ]
                        )
                    tokenized_inputs["idx_first_bpe"] = idx_first_bpe
                return tokenized_inputs

        else:

            def tokenize_function(examples):
                tokenized_inputs = tokenizer(
                    examples[text_name], truncation=True, is_split_into_words=True
                )
                if not test_mode:
                    all_labels = examples[label_name]
                    new_labels = []
                    idx_first_bpe = []
                    for i, labels in enumerate(all_labels):
                        word_ids = tokenized_inputs.word_ids(i)
                        new_labels.append(
                            PytorchNerWrapper.align_labels_with_tokens(labels, word_ids)
                        )
                        # Save ids of the first BPEs if necessary
                        if save_first_bpe_mask:
                            idx_first_bpe.append(
                                [
                                    i
                                    for (i, x) in enumerate(word_ids[1:], 1)
                                    if ((x != word_ids[i - 1]) and (x is not None))
                                ]
                            )
                    tokenized_inputs["labels"] = new_labels
                    if save_first_bpe_mask:
                        tokenized_inputs["idx_first_bpe"] = idx_first_bpe
                return tokenized_inputs

        return tokenize_function

    def get_model_class(self):
        try:
            return PYTORCH_INIT_MODELS_DICT[self.model_config.checkpoint]["model_class"]
        except:
            model_class, _ = self.get_model_from_path()
            return model_class

    @staticmethod
    def get_additional_tokenization_train_kwargs():
        return {"save_first_bpe_mask": False}

    @staticmethod
    def get_additional_tokenization_inference_kwargs():
        return {"save_first_bpe_mask": True}

    @staticmethod
    def get_data_collator_class():
        return DataCollatorForTokenClassification

    @staticmethod
    def get_datasets_metric_name():
        return "seqeval"

    @staticmethod
    def get_metric_name():
        return "test_overall_f1"

    def get_kwargs_for_compute_metrics_fn(self):
        return {
            "id2label": self.id2label,
            "display_tag_stats": "f1",
            "padding_idx": -100,
        }

    @staticmethod
    def get_compute_metrics_function(
        metric,
        id2label: Union[Dict[int, str], List],
        additional_metrics: Union[Tuple[Metric], List[Metric]] = tuple(),
        display_tag_stats="f1",
        padding_idx=-100,
    ):
        def compute_metrics(eval_preds):
            logits_ids, labels_ids, *inputs = eval_preds
            preds_ids = logits_ids.argmax(axis=-1)
            # Convert ids to tags
            idx_first_bpe = np.array(labels_ids) != padding_idx
            # Remove padding
            labels_without_padding = PytorchNerWrapper._remove_padding(
                labels_ids, idx_first_bpe
            )
            preds_without_padding = PytorchNerWrapper._remove_padding(
                preds_ids, idx_first_bpe
            )
            # Convert idx to tokens
            labels = PytorchNerWrapper._convert_id2label(
                labels_without_padding, id2label
            )
            preds = PytorchNerWrapper._convert_id2label(preds_without_padding, id2label)
            # Compute metrics
            metrics_dict = metric.compute(predictions=preds, references=labels)
            for add_metric in additional_metrics:
                add_metric_dict = add_metric.compute(
                    predictions=preds, references=labels
                )
                metrics_dict.update(add_metric_dict)
            # Individual entity scores are presented as dicts now. Need to "straighten" them
            output_dict = PytorchNerWrapper._construct_metrics_dict_for_ner(
                metrics_dict, display_tag_stats
            )
            return output_dict

        return compute_metrics

    @staticmethod
    def _convert_id2label(labels_or_preds, id2label, padding_idx=-100):
        return [
            [id2label[l] for l in label if l != padding_idx]
            for label in labels_or_preds
        ]

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
    def _remove_padding(predictions, idx_first_bpe):
        preds = []
        for pred, cond in zip(predictions, idx_first_bpe):
            if isinstance(pred, Tensor):
                pred = pred.cpu().detach().numpy()
            pred_without_padding = pred[cond]
            preds.append(pred_without_padding)

        return np.array(preds)
