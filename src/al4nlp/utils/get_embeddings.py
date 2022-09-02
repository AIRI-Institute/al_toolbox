from copy import deepcopy
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset as ArrowDataset
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformers import DataCollatorWithPadding

from .model_modifications import (
    get_model_without_cls_layer,
    _get_dim,
    ModelForFeaturesExtraction,
)
from .transformers_dataset import TransformersDataset
from ..model_wrappers.transformers.wrapper_cls import WrapperCls
from ..model_wrappers.transformers.wrapper_ner import WrapperNer

WRAPPERS = {"cls": WrapperCls, "ner": WrapperNer}


def get_embeddings(
    model,
    dataloader_or_data: Union[DataLoader, ArrowDataset, TransformersDataset],
    prepare_model: bool = True,
    use_activation: bool = False,
    use_spectralnorm: bool = False,
    to_eval_mode: bool = True,
    to_numpy: bool = False,
    data_is_tokenized: bool = False,
    batch_size: int = 100,
    use_automodel: bool = False,
    use_averaging: bool = False,
    **tokenization_kwargs,
):
    if prepare_model:
        use_pooling = tokenization_kwargs.get("task", "cls") != "ner"
        model_without_cls_layer = get_model_without_cls_layer(
            model, use_activation, use_spectralnorm, use_pooling=use_pooling
        )
    else:
        model_without_cls_layer = model

    device = next(model_without_cls_layer.parameters()).device

    wrapper = WRAPPERS[tokenization_kwargs.pop("task", "cls")]
    if not isinstance(dataloader_or_data, DataLoader):
        if not data_is_tokenized:
            if "label_name" not in tokenization_kwargs:
                tokenization_kwargs["test_mode"] = True
            tokenize_function = wrapper.get_tokenize_function(
                data=dataloader_or_data,
                **tokenization_kwargs,  # WTF: This should not be binded to the particular Wrapper
            )
            columns_to_remove = [
                x
                for x in dataloader_or_data.features.keys()
                if x not in ["labels", "weight"]
            ]
            # The last two arguments is a temporary fix for Kristofari
            dataloader_or_data = dataloader_or_data.map(
                tokenize_function,
                batched=True,
                remove_columns=columns_to_remove,
                load_from_cache_file=False,
                cache_file_name=f"tmp.data",
            )
            if wrapper == WrapperNer:
                idx_first_bpe = dataloader_or_data["idx_first_bpe"]
                dataloader_or_data = dataloader_or_data.remove_columns("idx_first_bpe")
        dataloader_or_data = DataLoader(
            dataloader_or_data,
            shuffle=False,
            batch_size=batch_size,
            collate_fn=wrapper.get_data_collator_class()(
                tokenizer=tokenization_kwargs["tokenizer"]
            ),
            pin_memory=(str(device).startswith("cuda")),
        )

    num_obs = len(dataloader_or_data.dataset)
    dim = _get_dim(model_without_cls_layer)
    if to_eval_mode:
        model_without_cls_layer.eval()

    embeddings = torch.empty((num_obs, dim), dtype=torch.float, device=device)
    if isinstance(model_without_cls_layer, ModelForFeaturesExtraction):
        possible_input_keys = model_without_cls_layer[
            0
        ].model.forward.__code__.co_varnames
    else:
        possible_input_keys = model_without_cls_layer.forward.__code__.co_varnames
    possible_input_keys = list(possible_input_keys) + ["input_ids", "attention_mask"]

    with torch.no_grad():
        start = 0
        for batch in tqdm(dataloader_or_data, desc="Embeddings created"):
            batch = {
                k: v.to(device) for k, v in batch.items() if k in possible_input_keys
            }
            predictions = model_without_cls_layer(**batch)
            if isinstance(model_without_cls_layer, ModelForFeaturesExtraction):
                batch_embeddings = predictions
            elif use_automodel and not use_activation:
                if use_averaging:
                    batch_embeddings = predictions.last_hidden_state
                    new_embeddings = []
                    for i, emb in enumerate(batch_embeddings):
                        # emb: shape seq_len x dim
                        non_padding_idx = batch["attention_mask"][i].bool()
                        new_embeddings.append(emb[non_padding_idx].mean(dim=0))
                    batch_embeddings = torch.stack(new_embeddings, dim=0)
                else:
                    batch_embeddings = predictions.last_hidden_state[:, 0]
            elif "pooler_output" in predictions.keys():
                batch_embeddings = predictions.pooler_output
            elif "last_hidden_state" in predictions.keys():
                batch_embeddings = predictions.last_hidden_state
            else:
                raise NotImplementedError

            end = start + len(batch["input_ids"])  # len(batch[list(batch.keys())[0]])
            # In this case we need to average batch_embeddings across idx_first_bpe
            if wrapper == WrapperNer:
                new_embeddings = []
                for i, emb in enumerate(batch_embeddings, start=start):
                    # emb: shape seq_len x dim
                    emb = emb[idx_first_bpe[i]].mean(dim=0)
                    new_embeddings.append(emb)
                batch_embeddings = torch.stack(new_embeddings, dim=0)
            embeddings[start:end].copy_(batch_embeddings, non_blocking=True)
            start = end

    if to_numpy:
        return embeddings.cpu().detach().numpy()
    return embeddings


def get_exp_grad_embedding(
    model,
    dataloader_or_data: Union[DataLoader, ArrowDataset, TransformersDataset],
    use_activation: bool = False,
    use_spectralnorm: bool = False,
    to_numpy: bool = False,
    data_is_tokenized: bool = False,
    batch_size: int = 100,
    **tokenization_kwargs,
):
    model_without_cls_layer = get_model_without_cls_layer(
        model, use_activation, use_spectralnorm
    )

    embDim = _get_dim(model_without_cls_layer)

    nLab = model.num_labels

    device = next(model.parameters()).device

    if not isinstance(dataloader_or_data, DataLoader):
        if not data_is_tokenized:
            if "label_name" not in tokenization_kwargs:
                tokenization_kwargs["test_mode"] = True
            tokenize_function = WRAPPERS[
                tokenization_kwargs.pop("task", "cls")
            ].get_tokenize_function(data=dataloader_or_data, **tokenization_kwargs,)
            columns_to_remove = [
                x
                for x in dataloader_or_data.features.keys()
                if x not in ["labels", "weight"]
            ]
            # The last two arguments is a temporary fix for Kristofari
            dataloader_or_data = dataloader_or_data.map(
                tokenize_function,
                batched=True,
                remove_columns=columns_to_remove,
                load_from_cache_file=False,
                cache_file_name=f"tmp.data",
            )

        dataloader_or_data = DataLoader(
            dataloader_or_data,
            shuffle=False,
            batch_size=batch_size,
            collate_fn=DataCollatorWithPadding(
                tokenizer=tokenization_kwargs["tokenizer"]
            ),
            pin_memory=(str(device).startswith("cuda")),
        )

    num_obs = len(dataloader_or_data.dataset)

    embedding = np.zeros([num_obs, nLab, embDim * nLab])

    for ind in range(nLab):
        with torch.no_grad():
            start = 0
            for batch in tqdm(dataloader_or_data, desc="Embeddings created"):
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if k in model.forward.__code__.co_varnames
                }
                tmp = model(**batch, output_hidden_states=True)
                cout, out = tmp["logits"], tmp["hidden_states"]
                out = out[0]
                out = out.data.cpu().numpy()
                out = np.mean(out, axis=1)

                end = start + len(batch["input_ids"])
                idxs = range(start, end)
                start = end
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                for j in range(batch_size):
                    if j >= len(out):
                        continue
                    for c in range(nLab):
                        if c == ind:
                            embedding[idxs[j]][ind][
                                embDim * c : embDim * (c + 1)
                            ] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][ind][
                                embDim * c : embDim * (c + 1)
                            ] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
                    embedding[idxs[j]][ind] = embedding[idxs[j]][ind] * np.sqrt(
                        batchProbs[j][ind]
                    )
    if to_numpy:
        return embedding

    return torch.Tensor(embedding)


def get_grad_embeddings(
    model,
    dataloader_or_data: Union[DataLoader, ArrowDataset, TransformersDataset],
    use_activation: bool = False,
    use_spectralnorm: bool = False,
    to_numpy: bool = False,
    data_is_tokenized: bool = False,
    batch_size: int = 100,
    **tokenization_kwargs,
):
    model_without_cls_layer = get_model_without_cls_layer(
        model, use_activation, use_spectralnorm
    )

    embDim = _get_dim(model_without_cls_layer)

    nLab = model.num_labels

    device = next(model.parameters()).device

    if not isinstance(dataloader_or_data, DataLoader):
        if not data_is_tokenized:
            if "label_name" not in tokenization_kwargs:
                tokenization_kwargs["test_mode"] = True
            tokenize_function = WRAPPERS[
                tokenization_kwargs.pop("task", "cls")
            ].get_tokenize_function(**tokenization_kwargs,)
            columns_to_remove = [
                x
                for x in dataloader_or_data.features.keys()
                if x not in ["labels", "weight"]
            ]
            # The last two arguments is a temporary fix for Kristofari
            dataloader_or_data = dataloader_or_data.map(
                tokenize_function,
                batched=True,
                remove_columns=columns_to_remove,
                load_from_cache_file=False,
                cache_file_name=f"tmp.data",
            )

        dataloader_or_data = DataLoader(
            dataloader_or_data,
            shuffle=False,
            batch_size=batch_size,
            collate_fn=DataCollatorWithPadding(
                tokenizer=tokenization_kwargs["tokenizer"]
            ),
            pin_memory=(str(device).startswith("cuda")),
        )

    num_obs = len(dataloader_or_data.dataset)
    embedding = np.zeros([num_obs, embDim * nLab])
    for ind in range(nLab):
        with torch.no_grad():
            start = 0
            for batch in tqdm(dataloader_or_data, desc="Embeddings created"):
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if k in model.forward.__code__.co_varnames
                }
                tmp = model(**batch, output_hidden_states=True)
                cout, out = tmp["logits"], tmp["hidden_states"]
                out = out[0]
                out = out.data.cpu().numpy()
                out = np.mean(out, axis=1)

                end = start + len(batch["input_ids"])
                idxs = range(start, end)
                start = end

                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)
                for j in range(batch_size):
                    if j >= len(out):
                        continue
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][
                                embDim * c : embDim * (c + 1)
                            ] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][
                                embDim * c : embDim * (c + 1)
                            ] = deepcopy(out[j]) * (-1 * batchProbs[j][c])

    if to_numpy:
        return embedding

    return torch.Tensor(embedding)
