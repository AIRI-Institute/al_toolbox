import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

# in this file one should define model class and constructor function
# to use these models with AL library, one should add at the top of the file
# dict with names of model class and constructor function
MODELS_DICT = {"model": "TextClassifier", "constructor": "init_text_classifier"}


class DictWithGetattr(dict):
    def __getattr__(self, item):
        return self.get(item, None)


def init_text_classifier(
    model_cfg, embeddings, word2idx, num_labels, only_tokenizer=False
):
    """Should return model and tokenizer instances.
    To add any custom model parameters just add them to model_cfg.
    """
    # here one could use custom function for embeddings
    # and override it, if necessary
    """
    if embeddings is None and model_cfg.embeddings_path is not None:
        embeddings, word2idx = load_embeddings(
            model_cfg.embeddings_path, model_cfg.embeddings_cache_dir
        )
    """
    # create tokenizer
    # in this example we use hf tokenizer, but one could also use any custom tokenizer as well
    tokenizer_model = WordLevel(word2idx, "[UNK]")
    tokenizer = Tokenizer(tokenizer_model)
    tokenizer.normalizer = Lowercase()
    tokenizer.pre_tokenizer = Whitespace()
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]"
    )
    if only_tokenizer:
        return None, hf_tokenizer
    # build model
    model = TextClassifier(
        model_config=model_cfg,
        num_labels=num_labels,
        pretrained_embeddings=embeddings,
        word2idx=word2idx,
    )
    return model, hf_tokenizer


class TextClassifier(torch.nn.Module):
    def __init__(
        self,
        model_config: dict,
        num_labels: int,
        pretrained_embeddings: torch.Tensor = None,
        word2idx: dict = None,
    ):
        # model name shouldn't ends with Model
        super().__init__()
        self.model_config = model_config
        self.num_labels = num_labels
        self.pretrained_embeddings = pretrained_embeddings
        self.word2idx = word2idx
        # special key showing does model return embeddings in forward or not
        self.return_embeddings = False
        self.init_model()

    def init_model(self):
        if self.pretrained_embeddings is not None:
            self.vocab_size, self.embed_dim = self.pretrained_embeddings.shape
            # self.embed_dim - necessary argument for embeddings based strategies
            self.embedding = nn.Embedding.from_pretrained(
                self.pretrained_embeddings,
                padding_idx=self.vocab_size - 1,
                freeze=self.model_config.freeze_embedding,
            )
        else:
            self.embed_dim = self.model_config.embed_dim
            self.embedding = nn.Embedding(
                num_embeddings=self.model_config.vocab_size,
                embedding_dim=self.embed_dim,
                padding_idx=0,
                max_norm=5.0,
            )
        self.pre_classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.model_config.classifier_dropout),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
        )
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.model_config.classifier_dropout),
            nn.Linear(self.embed_dim // 2, self.num_labels),
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, input_ids, labels=None, output_hidden_states=False, **kwargs):
        x_embed = self.embedding(input_ids).float()
        sent_embs = torch.mean(x_embed, dim=1)
        x_pre = self.pre_classifier(sent_embs)
        logits = self.head(x_pre)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.view(-1))
        results = {"loss": loss, "logits": logits}
        if output_hidden_states or self.return_embeddings:
            results["last_hidden_state"] = sent_embs
            results["hidden_states"] = [sent_embs]
        results = DictWithGetattr(results)
        return results

    def get_pre_classifier_dropout_activation(self):
        """Additional function - used for embeddings based AL strategies"""
        return self.pre_classifier[-1], self.head[-2], nn.ReLU()

    def save_pretrained(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load_pretrained(self, load_path):
        self.load_state_dict(torch.load(load_path))
