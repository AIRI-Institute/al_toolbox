"""
CNN model for text classification and utils
Adapted from https://github.com/chriskhanhtran/CNN-Sentence-Classification-PyTorch
"""
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from ..utils.embeddings import load_embeddings
from ..utils.general import DictWithGetattr

log = logging.getLogger()


def init_text_classification_cnn(
    model_cfg, embeddings, word2idx, num_labels, only_tokenizer=False
):
    # create tokenizer
    tokenizer_model = WordLevel(word2idx, "[UNK]")
    tokenizer = Tokenizer(tokenizer_model)
    tokenizer.normalizer = Lowercase()
    tokenizer.pre_tokenizer = Whitespace()
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]"
    )
    if only_tokenizer:
        return None, hf_tokenizer
    # build CNN for text classification
    if embeddings is None and model_cfg.embeddings_path is not None:
        embeddings, word2idx = load_embeddings(
            model_cfg.embeddings_path, model_cfg.embeddings_cache_dir
        )
    model = TextClassificationCNN(
        pretrained_embedding=embeddings,
        freeze_embedding=model_cfg.freeze_embedding,
        vocab_size=model_cfg.vocab_size,
        embed_dim=model_cfg.embed_dim,
        filter_sizes=model_cfg.filter_sizes,
        num_filters=model_cfg.num_filters,
        num_classes=num_labels,
        dropout=model_cfg.classifier_dropout,
    )
    return model, hf_tokenizer


class TextClassificationCNN(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification."""

    def __init__(
        self,
        pretrained_embedding=None,
        freeze_embedding=False,
        vocab_size=None,
        embed_dim=300,
        filter_sizes=[3, 4, 5],
        num_filters=[100, 100, 100],
        num_classes=2,
        dropout=0.5,
    ):
        """
        The constructor for CNN_NLP class.

        Args:
            pretrained_embedding (torch.Tensor): Pretrained embeddings with
                shape (vocab_size, embed_dim)
            freeze_embedding (bool): Set to False to fine-tune pretraiend
                vectors. Default: False
            vocab_size (int): Need to be specified when not pretrained word
                embeddings are not used.
            embed_dim (int): Dimension of word vectors. Need to be specified
                when pretrained word embeddings are not used. Default: 300
            filter_sizes (List[int]): List of filter sizes. Default: [3, 4, 5]
            num_filters (List[int]): List of number of filters, has the same
                length as `filter_sizes`. Default: [100, 100, 100]
            n_classes (int): Number of classes. Default: 2
            dropout (float): Dropout rate. Default: 0.5
        """

        super(TextClassificationCNN, self).__init__()
        # Embedding layer
        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embedding,
                padding_idx=self.vocab_size - 1,
                freeze=freeze_embedding,
            )
        else:
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=self.embed_dim,
                padding_idx=0,
                max_norm=5.0,
            )
        # Conv Network
        self.conv1d_list = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=self.embed_dim,
                    out_channels=num_filters[i],
                    kernel_size=filter_sizes[i],
                )
                for i in range(len(filter_sizes))
            ]
        )
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # additional param for embeddings model
        self.return_embeddings = False
        self.to(self.device)

    def forward(self, input_ids, labels=None, output_hidden_states=False, **kwargs):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """

        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        x_embed = self.embedding(input_ids).float()

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [
            F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x_conv_list
        ]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)

        # Compute logits. Output shape: (b, n_classes)
        x_drop = self.dropout(x_fc)
        logits = self.fc(x_drop)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.view(-1))
        results = {"loss": loss, "logits": logits}
        if output_hidden_states or self.return_embeddings:
            results["last_hidden_state"] = torch.mean(x_embed, dim=1)
            results["hidden_states"] = [torch.mean(x_embed, dim=1)]
        results = DictWithGetattr(results)
        return results

    def save_pretrained(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load_pretrained(self, load_path):
        self.load_state_dict(torch.load(load_path))
