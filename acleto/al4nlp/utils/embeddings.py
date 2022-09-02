import logging
import os
import shutil

import numpy as np
import torch
import wget
from gensim.models.keyedvectors import KeyedVectors
from tokenizers.pre_tokenizers import Whitespace

log = logging.getLogger()


def load_word_vectors(
    vectors_name, embeddings_path, model_cache_dir=None, n_vectors=None
):
    """Loads embeddings by url and name."""
    if vectors_name == "fasttext":
        embeddings_name = embeddings_path.split("/")[-1]
        save_path = os.path.join(model_cache_dir, embeddings_name)
        # check if this file already loaded
        os.makedirs(model_cache_dir, exist_ok=True)
        if not (os.path.isfile(save_path)):
            save_path = wget.download(embeddings_path, out=save_path)
        # unzip it and extract data to arrays
        if not (os.path.isfile(os.path.join(model_cache_dir, "crawl-300d-2M.vec"))):
            shutil.unpack_archive(save_path, model_cache_dir)
        fname = "crawl-300d-2M.vec"
    elif vectors_name == "word2vec":
        embeddings_name = "GoogleNews-vectors-negative300.bin.gz"
        save_path = os.path.join(model_cache_dir, embeddings_name)
        # check if this file already loaded
        os.makedirs(model_cache_dir, exist_ok=True)
        if not (os.path.isfile(save_path)):
            save_path = wget.download(embeddings_path, out=save_path)
        if not os.path.isfile(
            os.path.join(model_cache_dir, "GoogleNews-vectors-negative300.txt")
        ):
            model = KeyedVectors.load_word2vec_format(save_path, binary=True)
            model.save_word2vec_format(
                os.path.join(model_cache_dir, "GoogleNews-vectors-negative300.txt"),
                binary=False,
            )
        fname = "GoogleNews-vectors-negative300.txt"
    embeddings = []
    word2idx = {}
    log.info(f"Loading embeddings...")
    with open(os.path.join(model_cache_dir, fname), "r") as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0:
                # 0 line contains number of words
                continue
            tokens = line.rstrip().split(" ")
            word = tokens[0]
            vector = np.array(tokens[1:], dtype=np.float32)
            # here each word is unique, because we load pretrained embeddings
            word2idx[word] = idx
            embeddings.append(vector)
            if n_vectors is not None and idx > n_vectors:
                # use n_vectors parameter to load first n_vectors
                # mostly used for tests and debug
                break
    # add pad and unk tokens
    word2idx["[UNK]"] = 0
    embeddings.insert(0, np.mean(embeddings, axis=0))
    word2idx["[PAD]"] = idx + 1
    embeddings.append(np.zeros_like(embeddings[-1]))
    embeddings = torch.Tensor(np.asarray(embeddings))
    # After map words to ids and add pad and unk tokens
    # and also return np array of embeddings, not dict
    return embeddings, word2idx


def load_embeddings(embeddings_path, model_cache_dir=None, n_vectors=None):
    if embeddings_path == "fasttext":
        return load_word_vectors(
            "fasttext",
            "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip",
            model_cache_dir,
            n_vectors,
        )
    elif embeddings_path == "word2vec":
        return load_word_vectors(
            "word2vec",
            "https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&confirm=no_antivirus",
            model_cache_dir,
            n_vectors,
        )
    else:
        raise NotImplementedError


def check_models(config):
    """Check if any of a models is a cnn or a custom model, and get params of the model"""
    from ..models import PYTORCH_INIT_MODELS_DICT

    models_type = ["model", "acquisition_model", "successor_model", "target_model"]
    for model in models_type:
        if config.get(model, False):
            if config[model]["checkpoint"] in PYTORCH_INIT_MODELS_DICT.keys() or config[
                model
            ]["checkpoint"].endswith(".py"):
                return (
                    config[model]["embeddings_path"],
                    config[model]["embeddings_cache_dir"],
                )
    return None, None


def load_embeddings_with_text(
    data,
    embeddings_path="https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip",
    model_cache_dir=None,
    text_name="text",
    n_vectors=None,
):
    """Loads embeddings by url, uses fasttext by default.
    Now works only for fasttext. Also only use words from data for embeddings
    """
    embeddings, word2idx = load_embeddings(embeddings_path, model_cache_dir, n_vectors)
    # now reduce embeddings size
    pre_tokenizer = Whitespace()
    new_word2idx = {}
    new_embeddings = []
    new_word2idx["[UNK]"] = 0
    idx = 1
    for sentence in data:
        # split sentence by words with same pretokenizer
        if type(sentence[text_name]) == str:
            tokens = pre_tokenizer.pre_tokenize_str(sentence[text_name])
        else:
            tokens = sentence[text_name]
        tokens = [token[0].lower() for token in tokens]
        for token in tokens:
            if token not in new_word2idx and token in word2idx:
                new_word2idx[token] = idx
                new_embeddings.append(
                    np.array(embeddings[word2idx[token]], dtype=np.float32)
                )
                idx += 1
            elif token not in new_word2idx and token not in word2idx:
                # use random vector for embedding
                embedding = np.random.uniform(-0.1, 0.1, embeddings[0].shape)
                new_word2idx[token] = idx
                new_embeddings.append(np.array(embedding, dtype=np.float32))
                idx += 1
    new_embeddings.insert(0, np.mean(new_embeddings, axis=0))
    new_word2idx["[PAD]"] = idx
    new_embeddings.append(np.zeros_like(new_embeddings[-1]))
    new_embeddings = torch.Tensor(np.asarray(new_embeddings))
    log.info(f"Reduced embeddings size from {len(embeddings)} to {len(new_embeddings)}")
    return new_embeddings, new_word2idx
