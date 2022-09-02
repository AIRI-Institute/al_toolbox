# from allennlp.data.tokenizers import WhitespaceTokenizer
import logging
from collections import Counter
from copy import deepcopy

import nlpaug.augmenter.word as naw

log = logging.getLogger()


SYNONYM_AUG_P = 0.3
RANDOM_AUG_P = 0.05
NUM_AUGMENTATIONS = 10


def initialize_aug(aug, collection):
    counter = Counter(" ".join(collection).split(" "))
    words = [x[0] for x in counter.most_common(1000)]  # WTF:
    aug.target_words = words


def augment_instances(
    instances,
    path_to_augmentator="../../active_learning_nlp/acleto/aug/ppdb-2.0-tldr",
    include_original=True,
):

    texts = []
    for instance in instances:
        tokens = instance["tokens"].tokens
        text = " ".join([x.text for x in tokens])
        texts.append(text)

    collection = " ".join(texts)
    log.info("Collection created, start loading synonym augmentator...")
    synonym_aug = naw.SynonymAug(
        aug_src="ppdb", model_path=path_to_augmentator, aug_p=SYNONYM_AUG_P
    )
    log.info("Synonym augmentator loaded, start producing augmentations...")
    random_aug = naw.RandomWordAug("substitute", aug_p=RANDOM_AUG_P)
    initialize_aug(random_aug, collection)

    augmented_instances = []
    augmented_texts = []
    tokenizer = WhitespaceTokenizer()
    for _ in range(NUM_AUGMENTATIONS):
        new_texts = deepcopy(texts)
        new_texts = synonym_aug.augment(new_texts)
        new_texts = random_aug.augment(new_texts)
        augmented_texts.append(new_texts)

    for i, text in enumerate(texts):
        if include_original:
            augmented_instances.append(deepcopy(instances[i]))
        for i_aug in range(NUM_AUGMENTATIONS):
            text = augmented_texts[i_aug][i]
            instance = deepcopy(instances[i])
            instance["tokens"].tokens = tokenizer.tokenize(text)
            augmented_instances.append(instance)
    log.info("Augmentations produced.")

    return augmented_instances
