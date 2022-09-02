import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from ..model_wrappers.transformers.wrapper_cls import WrapperCls
from ..model_wrappers.transformers.wrapper_ner import WrapperNer

WRAPPERS = {"cls": WrapperCls, "ner": WrapperNer}


def compute_egl_scores(
    model,
    dataloader_or_data,
    data_is_tokenized=False,
    batch_size=100,
    **tokenization_kwargs,
):
    device = next(model.model.parameters()).device
    criterion = torch.nn.CrossEntropyLoss(reduction="none").to(device)

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

    gradient_lengths = initialize_gradient_lengths_array(num_obs)

    offset = 0
    possible_input_keys = model.model.forward.__code__.co_varnames
    possible_input_keys = list(possible_input_keys) + ["input_ids", "attention_mask"]

    for batch in tqdm(dataloader_or_data):
        batch = {k: v.to(device) for k, v in batch.items() if k in possible_input_keys}
        compute_gradient_lengths(
            model, criterion, gradient_lengths, offset, batch, device
        )

        offset += len(batch["input_ids"])

    return gradient_lengths


def initialize_gradient_lengths_array(dim):
    return np.zeros(dim, dtype=np.float64)


def compute_gradient_lengths(model, criterion, gradient_lengths, offset, x, device):
    batch_len = len(x["input_ids"])
    all_classes = torch.LongTensor([batch_len * [i] for i in range(model.num_labels)])
    if device is not None and device != "cpu":
        all_classes = all_classes.to(device)

    gradients = initialize_gradients(batch_len, model.num_labels, device)

    # x = x.to(device, non_blocking=True)
    model.model.zero_grad()

    compute_gradient_lengths_batch(model, criterion, x, gradients, all_classes)
    aggregate_gradient_lengths_batch(batch_len, gradient_lengths, gradients, offset)


def initialize_gradients(batch_len, num_classes, device):
    return torch.zeros([num_classes, batch_len]).to(device, non_blocking=True)


def compute_gradient_lengths_batch(model, criterion, x, gradients, all_classes):

    batch_len = len(x["input_ids"])

    output = model.model(**x)["logits"]
    with torch.no_grad():
        sm = torch.nn.functional.softmax(output, dim=1)

    for j in range(model.num_labels):
        loss = criterion(output, all_classes[j])

        for k in range(batch_len):
            model.model.zero_grad()
            loss[k].backward(retain_graph=True)

            compute_gradient_length(model, sm, gradients, j, k)


def compute_gradient_length(model, sm, gradients, j, k):

    params = [
        param.grad.flatten()
        for param in model.model.parameters()
        if param.requires_grad
    ]
    params = torch.cat(params)

    gradients[j, k] = torch.sqrt(params.pow(2)).sum()
    gradients[j, k] *= sm[k, j].item()


def aggregate_gradient_lengths_batch(batch_len, gradient_lengths, gradients, offset):
    gradient_lengths[offset : offset + batch_len] = torch.sum(gradients, 0).to(
        "cpu", non_blocking=True
    )
