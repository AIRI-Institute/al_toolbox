import logging
import re

import numpy as np
import pandas as pd


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(rf'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


def convert_y_to_dict_format(X, y):
    dict_annots = []
    for sent_x, sent_y in zip(X, y):
        offsets = []
        curr_offset = 0
        for word in sent_x:
            offsets.append(curr_offset)
            curr_offset += len(word) + 1

        sent_dict_annots = []
        start_offset = -1
        last_offset = -1
        entity_tag = ""
        for i, tag in enumerate(sent_y):
            if tag.split("-")[0] == "O":
                if start_offset != -1:
                    sent_dict_annots.append(
                        {"tag": entity_tag, "start": start_offset, "end": last_offset}
                    )
                start_offset = -1

            if tag.split("-")[0] in {"B"}:
                if start_offset != -1:
                    sent_dict_annots.append(
                        {"tag": entity_tag, "start": start_offset, "end": last_offset}
                    )

                start_offset = offsets[i]
                entity_tag = tag.split("-")[1]
                last_offset = offsets[i] + len(sent_x[i])
            if tag.split("-")[0] == "I":
                last_offset = offsets[i] + len(sent_x[i])

        if start_offset != -1:
            sent_dict_annots.append(
                {"tag": entity_tag, "start": start_offset, "end": last_offset}
            )

        dict_annots.append(sent_dict_annots)

    return dict_annots


def create_helper(X_train): # TODO: UPD helper
    all_offsets = list()
    all_texts = list()
    for i in range(len(X_train)):
        tokenized_text = X_train[i]
        
        offsets = []
        text = ""
        curr_offset = 0
        for word in tokenized_text:
            offsets.append(curr_offset)
            text += word + " "
            curr_offset += len(word) + 1
        
        all_offsets.append(offsets)
        all_texts.append(text)
        
    return pd.DataFrame(all_texts, columns=["text"], index=list(range(len(all_texts)))), all_offsets


def format_entities(sentence, tags):
    annots = []
    curr_tag = {}

    for i, tag in enumerate(tags):
        real_tag = tag.split("-")

        if real_tag[0] == "B":
            if curr_tag:
                curr_tag["end"] = sentence[i - 1].end
                annots.append(curr_tag)

            curr_tag["tag"] = real_tag[1]
            curr_tag["begin"] = sentence[i].begin

        elif real_tag[0] == "O":
            if curr_tag:
                curr_tag["end"] = sentence[i - 1].end
                annots.append(curr_tag)
                curr_tag = {}

    return annots


def insert_tags(text, tags):
    html = ""
    end = 0
    for tag in tags:
        html += text[end : tag["offset"]] + tag["tag"]
        end = tag["offset"]

    html += text[end:]
    return html


def create_info_list_custom(save_path):
    list_with_full_ident = []
    a = np.load(save_path + "/custom_X.npy", allow_pickle=True)
    b = a.tolist()
    for i in b:
        if i is not None and i not in list_with_full_ident:
            list_with_full_ident.append(i)
    a = np.load(save_path + "/annotation.npy", allow_pickle=True)
    b = a.tolist()
    for i in b:
        if i is not None and i not in list_with_full_ident:
            list_with_full_ident.append(i)
    return list_with_full_ident
