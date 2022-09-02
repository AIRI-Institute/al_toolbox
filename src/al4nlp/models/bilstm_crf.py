from typing import List

import flair
import numpy as np
import torch
import torch.nn as nn
from flair.data import Sentence, Dictionary
from flair.embeddings import TokenEmbeddings, StackedEmbeddings, WordEmbeddings
from flair.models import SequenceTagger
from transformers import set_seed


class CharEncoderCNN(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size,
        out_channels,
        kernel_width,
        pad_width,
        input_dropout_p=0.5,
        output_dropout_p=0,
        in_channels=1,
    ):

        super(CharEncoderCNN, self).__init__()

        self.out_channels = out_channels
        self.input_dropout = torch.nn.Dropout(input_dropout_p)
        self.output_dropout = torch.nn.Dropout(output_dropout_p)
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.cnn = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_width, embedding_size),
            padding=(pad_width, 0),
        )
        self._weight_init(self.embedding.weight)

    def _weight_init(self, input_embedding):
        """Make init like in Lipton's paper"""
        bias = np.sqrt(3.0 / input_embedding.size(1))
        torch.nn.init.uniform(input_embedding, -bias, bias)

    def forward(self, input_var, input_lengths=None):
        embedded = self.embedding(input_var).unsqueeze(1)
        embedded = self.input_dropout(embedded)
        output = self.cnn(embedded)
        output = torch.nn.functional.max_pool2d(output, kernel_size=(output.size(2), 1))
        output = output.squeeze(3).squeeze(2)

        return output


class CharacterCNNEmbeddings(TokenEmbeddings):
    def __init__(
        self,
        path_to_char_dict: str = None,
        char_embedding_dim: int = 25,
        hidden_size_char: int = 25,
        out_size_char: int = 25,
        input_dropout: float = 0.0,
        kernel_width: int = 3,
    ):

        super().__init__()
        self.name = "Char"
        self.static_embeddings = False

        # use list of common characters if none provided
        if path_to_char_dict is None:
            self.char_dictionary: Dictionary = Dictionary.load("common-chars")
        else:
            self.char_dictionary: Dictionary = Dictionary.load_from_file(
                path_to_char_dict
            )

        self.char_embedding_dim: int = char_embedding_dim
        self.hidden_size_char: int = hidden_size_char
        self.out_size_char: int = out_size_char
        self.char_embedding = torch.nn.Embedding(
            len(self.char_dictionary.item2idx), self.char_embedding_dim
        )

        """
        
        self.char_rnn = torch.nn.LSTM(
            self.char_embedding_dim,
            self.hidden_size_char,
            num_layers=1,
            bidirectional=True,
        )
        """

        self.char_cnn = CharEncoderCNN(
            len(self.char_dictionary.item2idx),
            self.char_embedding_dim,
            self.out_size_char,
            input_dropout_p=input_dropout,
            kernel_width=kernel_width,
            pad_width=1,
        )

        self.__embedding_length = self.out_size_char

        self.to(flair.device)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]):

        for sentence in sentences:

            tokens_char_indices = []

            # translate words in sentence into ints using dictionary
            for token in sentence.tokens:
                char_indices = [
                    self.char_dictionary.get_idx_for_item(char) for char in token.text
                ]
                tokens_char_indices.append(char_indices)

            # sort words by length, for batching and masking
            tokens_sorted_by_length = sorted(
                tokens_char_indices, key=lambda p: len(p), reverse=True
            )
            d = {}
            for i, ci in enumerate(tokens_char_indices):
                for j, cj in enumerate(tokens_sorted_by_length):
                    if ci == cj and j not in d:  # and i not in d.values()
                        d[j] = i
                        break
            chars2_length = [len(c) for c in tokens_sorted_by_length]
            longest_token_in_sentence = max(chars2_length)
            tokens_mask = torch.zeros(
                (len(tokens_sorted_by_length), longest_token_in_sentence),
                dtype=torch.long,
                device=flair.device,
            )

            for i, c in enumerate(tokens_sorted_by_length):
                tokens_mask[i, : chars2_length[i]] = torch.tensor(
                    c, dtype=torch.long, device=flair.device
                )

            # chars for rnn processing
            chars = tokens_mask

            """
            character_embeddings = self.char_embedding(chars).transpose(0, 1)

            packed = torch.nn.utils.rnn.pack_padded_sequence(
                character_embeddings, chars2_length
            )
            
            lstm_out, self.hidden = self.char_rnn(packed)
            """

            cnn_out = self.char_cnn(chars)
            # outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
            # print(outputs.shape)
            # outputs = outputs.transpose(0, 1)
            # print(outputs.shape)
            # print(output_lengths)
            character_embeddings = torch.zeros(
                (cnn_out.size(0), cnn_out.size(1)),
                dtype=torch.float,
                device=flair.device,
            )
            # print(character_embeddings.shape)
            # outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(cnn_out)
            """
            for i, index in enumerate(output_lengths):
                chars_embeds_temp[i] = outputs[i]
            """
            # character_embeddings = chars_embeds_temp.clone()
            for i in range(character_embeddings.size(0)):
                character_embeddings[d[i]] = cnn_out[i]

            for token_number, token in enumerate(sentence.tokens):
                token.set_embedding(self.name, character_embeddings[token_number])

    def __str__(self):
        return self.name


class FlairModelNER(SequenceTagger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def save_pretrained(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load_pretrained(self, load_path):
        self.load_state_dict(torch.load(load_path))


def init_lstm(input_lstm):
    for ind in range(0, input_lstm.num_layers):
        weight = eval("input_lstm.weight_ih_l" + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)
        weight = eval("input_lstm.weight_hh_l" + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)

    if input_lstm.bidirectional:
        for ind in range(0, input_lstm.num_layers):
            weight = eval("input_lstm.weight_ih_l" + str(ind) + "_reverse")
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform(weight, -bias, bias)
            weight = eval("input_lstm.weight_hh_l" + str(ind) + "_reverse")
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform(weight, -bias, bias)

    if input_lstm.bias:

        for ind in range(0, input_lstm.num_layers):
            weight = eval("input_lstm.bias_ih_l" + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size : 2 * input_lstm.hidden_size] = 1
            weight = eval("input_lstm.bias_hh_l" + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size : 2 * input_lstm.hidden_size] = 1

        if input_lstm.bidirectional:
            for ind in range(0, input_lstm.num_layers):
                weight = eval("input_lstm.bias_ih_l" + str(ind) + "_reverse")
                weight.data.zero_()
                weight.data[input_lstm.hidden_size : 2 * input_lstm.hidden_size] = 1
                weight = eval("input_lstm.bias_hh_l" + str(ind) + "_reverse")
                weight.data.zero_()
                weight.data[input_lstm.hidden_size : 2 * input_lstm.hidden_size] = 1


def init_linear(input_linear):
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


def create_flair_model_tokenizer(
    model_cfg,
    idx_to_tag: dict = None,
    label_type: str = "ner_tags",
    seed: int = 42,
    cache_dir=None,
):
    assert (
        model_cfg.checkpoint == "bilstm-crf"
    ), "For now Flair support only BiLSTM-CRF model!"

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed(seed)
    # create embedding
    # uncomment if we want to load custom embeddings
    """
    url = model_cfg.embedding_file
    save_path = os.path.join(cache_dir, 'embeddings')
    if not(os.path.isfile(save_path)):
        print("File doesn't found")
        save_path = wget.download(url, out=save_path)
    """
    save_path = model_cfg.embedding_file
    embedding_types = [
        WordEmbeddings(save_path),  # model_cfg.embedding_file),
    ]
    for kernel_width in model_cfg.char_emb.cnn_filters:
        embedding_types.append(
            CharacterCNNEmbeddings(
                char_embedding_dim=model_cfg.char_emb.emb_dim,
                hidden_size_char=model_cfg.char_emb.emb_dim,
                out_size_char=model_cfg.char_emb.emb_dim,
                input_dropout=model_cfg.char_emb.dropout,
                kernel_width=kernel_width,
            )
        )
    embeddings = StackedEmbeddings(embeddings=embedding_types)

    # convert idx_to_tag to flair dictionary
    tag_dict = Dictionary(add_unk=False)
    for value in idx_to_tag.values():
        tag_dict.add_item(value)
    # create tagger - BiLSTM-CRF model
    # TODO: check that dropout param use in each case
    tagger = FlairModelNER(
        hidden_size=model_cfg.lstm.hidden_size,
        embeddings=embeddings,
        tag_dictionary=tag_dict,
        tag_type=label_type,
        use_crf=True,
        use_rnn=True,
        rnn_layers=model_cfg.lstm.num_layers,
        dropout=model_cfg.lstm.dropout,
        word_dropout=model_cfg.lstm.layer_dropout_probability,
        locked_dropout=model_cfg.lstm.recurrent_dropout_probability,
        reproject_embeddings=model_cfg.lstm.reproject_embeddings,  # model_cfg.feedforward.hidden_size,
        train_initial_hidden_state=False,
        rnn_type="LSTM",
        beta=1.0,
    )
    # init rnn like in Lipton's paper
    init_lstm(tagger.rnn)
    init_linear(tagger.linear)

    # Here we also need a tokenizer, but we doesn't use it
    tokenizer = None

    return tagger, tokenizer
