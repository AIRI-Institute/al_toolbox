import logging

import numpy as np
import torch
import torch.nn.functional as F

from . import alpaca_calibrator as calibrator

log = logging.getLogger("text_classifier")


class TextClassifier:
    def __init__(
        self,
        auto_model,
        bpe_tokenizer,
        max_len=192,
        pred_loader_args={"num_workers": 1},
        pred_batch_size=100,
        training_args=None,
        trainer=None,
    ):
        super().__init__()

        self._auto_model = auto_model
        self._bpe_tokenizer = bpe_tokenizer
        self._pred_loader_args = pred_loader_args
        self._pred_batch_size = pred_batch_size
        self._training_args = training_args
        self._trainer = trainer
        self._named_parameters = auto_model.named_parameters
        self.temperature = 1.0
        self._max_len = max_len

    @property
    def _bert_model(self):
        return self._auto_model

    @property
    def model(self):
        return self._auto_model

    @property
    def tokenizer(self):
        return self._bpe_tokenizer

    def predict(
        self, eval_dataset, calibrate=False, apply_softmax=True, return_preds=True,
    ):
        self._auto_model.eval()

        self._trainer.args.eval_accumulation_steps = 1
        res = self._trainer.predict(eval_dataset)
        self._trainer.args.eval_accumulation_steps = None
        logits = res[0]
        if isinstance(logits, tuple):
            logits = logits[0]
        if calibrate:
            labels = [example["label"] for example in eval_dataset]
            calibr = calibrator.ModelWithTempScaling(self._auto_model)
            calibr.scaling(
                torch.FloatTensor(logits),
                torch.LongTensor(labels),
                lr=1e-3,  # TODO:
                max_iter=100,  # TODO:
            )
            self.temperature = calibr.temperature.detach().numpy()[0]
            self.temperature = np.clip(self.temperature, 0.1, 10)

        logits = np.true_divide(logits, self.temperature)

        if apply_softmax:
            probs = F.softmax(torch.Tensor(logits), dim=-1).numpy()
        else:
            probs = logits

        if not return_preds:
            return [probs] + list(res)

        # preds = np.argmax(probs, axis=1)
        preds = probs[:, 0]

        return [preds, probs] + list(res)
