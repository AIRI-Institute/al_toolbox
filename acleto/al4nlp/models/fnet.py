import json

import torch
from scipy import linalg
from tokenizers import SentencePieceBPETokenizer
from tokenizers.processors import TemplateProcessing
from torch import nn
from transformers import PreTrainedTokenizerFast


def init_fnet(task, path_to_pretrained, num_labels=2, tokenizer_max_length=512):
    with open(path_to_pretrained / "config.json") as f:
        pretrained_model_cfg = json.load(f)

    model_class = (
        FNetForSequenceClassification if task == "cls" else FNetForTokenClassification
    )
    model = model_class(pretrained_model_cfg, num_labels)
    model.load_state_dict(
        torch.load(path_to_pretrained / "fnet.statedict.pt"), strict=False
    )

    orig_tokenizer = SentencePieceBPETokenizer.from_file(
        str(path_to_pretrained / "vocab.json"), str(path_to_pretrained / "merges.txt"),
    )
    orig_tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A [SEP] $B:1 </s>:1",
        special_tokens=[("<s>", 1), ("</s>", 2), ("[MASK]", 6), ("[SEP]", 5)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=orig_tokenizer,
        model_max_length=tokenizer_max_length,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    return model, tokenizer


class FNetEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config["vocab_size"],
            config["embedding_size"],
            padding_idx=config["pad_token_id"],
        )
        self.position_embeddings = nn.Embedding(
            config["max_position_embeddings"], config["embedding_size"]
        )
        self.token_type_embeddings = nn.Embedding(
            config["type_vocab_size"], config["embedding_size"]
        )

        self.layer_norm = nn.LayerNorm(
            config["embedding_size"], eps=config["layer_norm_eps"]
        )
        self.hidden_mapping = nn.Linear(config["embedding_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["dropout_rate"])

        self.register_buffer(
            "position_ids",
            torch.arange(config["max_position_embeddings"]).expand((1, -1)),
            persistent=False,
        )

    def forward(self, input_ids, token_type_ids, attention_mask=None):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        position_ids = self.position_ids[:, :seq_length]

        word_embeddings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = word_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.layer_norm(embeddings)
        embeddings = self.hidden_mapping(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class FourierMMLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dft_mat_seq = torch.tensor(linalg.dft(config["max_position_embeddings"]))
        self.dft_mat_hidden = torch.tensor(linalg.dft(config["hidden_size"]))

    def forward(self, hidden_states):
        hidden_states_complex = hidden_states.type(torch.complex128)
        return torch.einsum(
            "...ij,...jk,...ni->...nk",
            hidden_states_complex,
            self.dft_mat_hidden,
            self.dft_mat_seq,
        ).real.type(torch.float32)


class FourierFFTLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        return torch.fft.fft(torch.fft.fft(hidden_states, dim=-1), dim=-2).real


class FNetLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fft = (
            FourierMMLayer(config)
            if config["fourier"] == "matmul"
            else FourierFFTLayer()
        )
        self.mixing_layer_norm = nn.LayerNorm(
            config["hidden_size"], eps=config["layer_norm_eps"]
        )
        self.feed_forward = nn.Linear(
            config["hidden_size"], config["intermediate_size"]
        )
        self.output_dense = nn.Linear(
            config["intermediate_size"], config["hidden_size"]
        )
        self.output_layer_norm = nn.LayerNorm(
            config["hidden_size"], eps=config["layer_norm_eps"]
        )
        self.dropout = nn.Dropout(config["dropout_rate"])
        self.activation = nn.GELU()

    def forward(self, hidden_states):
        fft_output = self.fft(hidden_states)
        fft_output = self.mixing_layer_norm(fft_output + hidden_states)
        intermediate_output = self.feed_forward(fft_output)
        intermediate_output = self.activation(intermediate_output)
        output = self.output_dense(intermediate_output)
        output = self.dropout(output)
        output = self.output_layer_norm(output + fft_output)
        return output


class FNetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config_dict = config
        self.layer = nn.ModuleList(
            [FNetLayer(config) for _ in range(config["num_hidden_layers"])]
        )

    def forward(self, hidden_states):
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states)

        return hidden_states


class FNetPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class FNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config_dict = config

        self.embeddings = FNetEmbeddings(config)
        self.encoder = FNetEncoder(config)
        self.pooler = FNetPooler(config)

    def forward(self, input_ids, token_type_ids, labels=None, attention_mask=None):
        embedding_output = self.embeddings(input_ids, token_type_ids)
        sequence_output = self.encoder(embedding_output)
        pooled_output = self.pooler(sequence_output)

        return sequence_output, pooled_output


class FNetForPreTraining(nn.Module):
    def __init__(self, config):
        super(FNetForPreTraining, self).__init__()
        self.encoder = FNet(config)

        self.embedding_size = config["embedding_size"]
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_hidden_layers"]

        self.mlm_intermediate = nn.Linear(self.hidden_size, self.embedding_size)
        self.activation = nn.GELU()
        self.mlm_layer_norm = nn.LayerNorm(self.embedding_size)
        self.mlm_output = nn.Linear(self.embedding_size, self.vocab_size)

        self.nsp_output = nn.Linear(self.hidden_size, 2)

        self.loss_fn = nn.CrossEntropyLoss()

    def _mlm(self, x):
        x = self.mlm_intermediate(x)
        x = self.activation(x)
        x = self.mlm_layer_norm(x)
        x = self.mlm_output(x)
        return x

    def forward(
        self,
        input_ids,
        token_type_ids,
        labels=None,
        token_type_labels=None,
        attention_mask=None,
    ):
        sequence_output, pooled_output = self.encoder(input_ids, token_type_ids)
        mlm_logits = self._mlm(sequence_output)
        nsp_logits = self.nsp_output(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(mlm_logits.view(-1, self.vocab_size), labels.view(-1))
        if token_type_labels is not None:
            loss += self.loss_fn(nsp_logits.view(-1, 2), token_type_labels.view(-1))

        return {"loss": loss, "logits": mlm_logits, "nsp_logits": nsp_logits}


class FNetForSequenceClassification(nn.Module):
    def __init__(self, config, num_labels=2, cls_dropout_p=0.1):
        super().__init__()
        self.config_dict = config

        self.embeddings = FNetEmbeddings(config)
        self.encoder = FNetEncoder(config)
        self.pooler = FNetPooler(config)
        self.dropout = nn.Dropout(cls_dropout_p)
        self.classifier = nn.Linear(config["hidden_size"], num_labels)
        self.loss_fn = (
            nn.CrossEntropyLoss() if num_labels > 1 else nn.BCEWithLogitsLoss()
        )

    def forward(self, input_ids, token_type_ids, labels=None, attention_mask=None):
        embedding_output = self.embeddings(input_ids, token_type_ids)
        sequence_output = self.encoder(embedding_output)
        pooled_output = self.pooler(sequence_output)
        logits = self.classifier(pooled_output)
        loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


class FNetForTokenClassification(nn.Module):
    def __init__(self, config, num_labels=2, cls_dropout_p=0.1):
        super().__init__()
        self.config_dict = config
        self.num_labels = num_labels

        self.embeddings = FNetEmbeddings(config)
        self.encoder = FNetEncoder(config)
        self.dropout = nn.Dropout(cls_dropout_p)
        self.classifier = nn.Linear(config["hidden_size"], num_labels)
        self.loss_fn = (
            nn.CrossEntropyLoss() if num_labels > 1 else nn.BCEWithLogitsLoss()
        )

    def forward(self, input_ids, token_type_ids, labels=None, attention_mask=None):
        embedding_output = self.embeddings(input_ids, token_type_ids)
        sequence_output = self.encoder(embedding_output)
        logits = self.classifier(sequence_output).view(-1, self.num_labels)
        loss = self.loss_fn(logits, labels.view(-1))

        return {"loss": loss, "logits": logits}
