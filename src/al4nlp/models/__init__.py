from al4nlp.models.fnet import init_fnet
from al4nlp.models.ner_bilstm import init_tagger, BilstmTagger
from al4nlp.models.text_classification_cnn import (
    init_text_classification_cnn,
    TextClassificationCNN,
)

INIT_MODELS_DICT = {"fnet": init_fnet}
PYTORCH_INIT_MODELS_DICT = {
    "cnn": {
        "model": init_text_classification_cnn,
        "model_class": TextClassificationCNN,
    },
    "bilstm": {"model": init_tagger, "model_class": BilstmTagger},
}
# add list of available models instead of dict to avoid flair import
FLAIR_MODELS = ["bilstm-crf"]
