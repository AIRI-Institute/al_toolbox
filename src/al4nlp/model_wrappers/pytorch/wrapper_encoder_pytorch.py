import sys
from importlib import import_module

from transformers import (
    set_seed,
)

from ..transformers.wrapper_encoder import WrapperEncoder
from ...models import PYTORCH_INIT_MODELS_DICT


class PytorchEncoderWrapper(WrapperEncoder):
    def __init__(self, embeddings, word2idx, **base_kwargs):
        self.embeddings = embeddings
        self.word2idx = word2idx
        super().__init__(**base_kwargs)

    def get_model_and_tokenizer(
        self,
        model_cfg,
        id2label: dict = None,
        seed: int = 42,
        cache_dir=None,
        embeddings=None,
        word2idx=None,
    ):
        set_seed(seed)
        num_labels = model_cfg.num_labels if id2label is None else len(id2label)
        # build CNN for text classification
        init_function = self.get_init_function()
        model, hf_tokenizer = init_function(model_cfg, embeddings, word2idx, num_labels)
        return model, hf_tokenizer

    def get_init_function(self):
        if self.model_config.checkpoint in PYTORCH_INIT_MODELS_DICT.keys():
            # try to get function from implemented in framework functions
            init_function = PYTORCH_INIT_MODELS_DICT[self.model_config.checkpoint][
                "model"
            ]
        else:
            # use a custom function and model
            _, init_function = self.get_model_from_path()
        return init_function

    def get_init_model_kwargs(self):
        return {
            "id2label": self.id2label,
            "embeddings": self.embeddings,
            "word2idx": self.word2idx,
        }

    def set_tokenizer(self):
        init_function = self.get_init_function()
        try:
            num_labels = self.model_config.num_labels
        except:
            # we doesn't really need this param for tokenizer
            num_labels = 2
        _, hf_tokenizer = init_function(
            self.model_config,
            self.embeddings,
            self.word2idx,
            num_labels,
            only_tokenizer=True,
        )
        self.tokenizer = hf_tokenizer

    def get_model_from_path(self):
        model_path = self.model_config.checkpoint
        if (not model_path.endswith("/")) and ("/" in model_path):
            path = model_path[: model_path.rindex("/")]
        elif "/" in model_path:
            path = model_path[: model_path[:-2].rindex("/")]
        else:
            path = "./"
        model_filename = model_path.split("/")[-1] or model_path.split("/")[-2]
        if model_filename.endswith(".py"):
            model_filename = model_filename[:-3]
        sys.path.append(path)

        models_dict = getattr(import_module(model_filename), "MODELS_DICT", None)
        model_class = getattr(import_module(model_filename), models_dict["model"], None)
        model_constructor = getattr(
            import_module(model_filename), models_dict["constructor"], None
        )
        return model_class, model_constructor
