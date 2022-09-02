def get_classifier_dropout_kwargs(
    pretrained_model_name: str, classifier_dropout: float
):
    if "distilbert" in pretrained_model_name:
        key = "seq_classif_dropout"
    elif "deberta" in pretrained_model_name:
        key = "pooler_dropout"
    elif "xlnet" in pretrained_model_name:
        key = "summary_last_dropout"
    elif "distilrubert" in pretrained_model_name:
        key = "dropout"
    elif "rubert-base" in pretrained_model_name:
        key = "hidden_dropout_prob"
    else:
        key = "classifier_dropout"
    return {key: classifier_dropout}


def get_tokenizer_kwargs(pretrained_model_name: str, task: str):
    if task == "ner" and "roberta" in pretrained_model_name:
        return dict(add_prefix_space=True)
    return {}
