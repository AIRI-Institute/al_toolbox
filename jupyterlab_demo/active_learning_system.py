import warnings

warnings.filterwarnings("ignore")

import copy
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize, core
import matplotlib.pyplot as plt
from collections import OrderedDict

from annotator_tool.active_learner import ActiveLearner
from annotator_tool.active_learner_async import ActiveLearnerAsync
from annotator_tool.ui_widget import ActiveLearnerUiWidget
from annotator_tool.annotator_widget import AnnotatorWidget
from annotator_tool.visualizers.seq_annotation import SeqAnnotationVisualizer
from annotator_tool.al4nlp_adaptor.al4nlp_adaptor import AdaptorAl4Nlp
from annotator_tool.annotation_converter_bio import AnnotationConverterBio

from datasets import load_dataset

from acleto.al4nlp.constructors.construct_wrapper import construct_wrapper
from acleto.al4nlp.utils.transformers_dataset import TransformersDataset

from spacy import displacy
from nltk.tokenize import RegexpTokenizer

from utils_data import (
    convert_y_to_dict_format,
    create_helper,
    set_global_logging_level,
)

import logging

_log_format = f"%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - %(message)s"
_fh = logging.FileHandler("./logs/demo.log", mode='a')
_fh.setFormatter(logging.Formatter(_log_format))

logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
    handlers=[_fh],
)

logger = logging.getLogger(__name__)

set_global_logging_level(
    logging.CRITICAL, ["transformers", "nlp", "torch", "tensorflow"]
)


class ALSystem:
    def __init__(self, config, save_path, dataset_path):
        self.config = config
        self.evaluation_results = []
        self.recall_results = []
        self.pression_results = []
        self.save_path = save_path
        self._additional_X = None
        self._additional_y = None
        self.id2label = None
        self.dataset_path = str(dataset_path.parent)
        self.dataset_name = dataset_path.name
        self.al_widget = None
        self.active_learner = None
        self.dev_metrics = []
        self.test_metrics = []
        
        logger.info('\n\n\n\n================= Starting system ================')

    def load_annotations(self, annotation_path):
        labeled_data_path = Path(annotation_path) / "annotation.json"
        logger.info(f"Trying to load: {labeled_data_path}")
        
        y_labels = None
        if os.path.exists(labeled_data_path):
            with open(labeled_data_path) as f:
                y_labels = json.load(f)
        
        return y_labels

    def evaluate_learner(self, on_dev: bool = True, on_test: bool = True):
        if self.active_learner is not None:
            estimator = (
                self.active_learner._active_learn_alg.learner.estimator
            )
            metric_names = {"cls": "test_accuracy", "ner": "test_overall_f1"}
            metric_short_names = {"cls": "Accuracy", "ner": "Entity-F1"}
            
            metrics = estimator.evaluate(self.test_instances)
            target_metric_value = metrics[metric_names[self.task]]
            self.test_metrics.append(target_metric_value)
            print(
                f"Test Overall {metric_short_names[self.task]}: {target_metric_value:.2f}"
            )

            xlimit = max(20, len(self.dev_metrics), len(self.test_metrics))
            plt.plot(self.dev_metrics, "go--")
            plt.plot(self.test_metrics, "ro--")
            plt.xlim(0, xlimit)
            plt.ylim(0.0, 1.0)
            plt.xlabel("AL Iteration")
            plt.ylabel(f"{metric_short_names[self.task]} Score")
            plt.grid(True)
            plt.legend(["Dev", "Test"])
            plt.title("Active Learning evaluation")
        else:
            print("Not started yet")

    def convert_y_to_dict_format(self, dataset, id2label):
        X = dataset["tokens"]
        y = dataset["ner_tags"]
        y = [[id2label[tag] for tag in tags] for tags in y]
        converted = convert_y_to_dict_format(X, y)
        converted = list(map(lambda x: None if len(x) == 0 else x, converted))
        return converted
            
    def predict_tags(self, text: str):
        _tokenizer = RegexpTokenizer(r"\s+", gaps=True)
        tokens = _tokenizer.tokenize(text)
        estimator = (
            self.active_learner._active_learn_alg.learner.estimator
        )
        dataset = TransformersDataset(
            {"tokens": [tokens], "ner_tags": [[0] * len(tokens)]}
        )
        tags = np.argmax(estimator.predict_proba(dataset).squeeze(), -1)
        tags = list(map(self.id2label.__getitem__, tags))
        mapping = convert_y_to_dict_format([tokens], [tags])[0]
        for m in mapping:
            m["label"] = m["tag"]
            del m["tag"]
        displacy_input = [{"text": text, "ents": mapping, "title": None}]
        _ = displacy.render(displacy_input, style="ent", manual=True, jupyter=True)
        
    def create_active_learner(self):
        # Config for al4nlp
        core.global_hydra.GlobalHydra.instance().clear()
        initialize(config_path="./configs/")  # TODO: config
        default_config_name = "al_cls" if self.dataset_name != "conll2003" else "al_ner"
        config = compose(config_name=os.environ.get("CONFIG_NAME", default_config_name))
        config.data.dataset_name = self.dataset_name
        config.data.path = self.dataset_path
        self.task = config.task
        self._model_name = config.acquisition_model.checkpoint
        self.text_field_name = config.data.text_name
        self.label_name = config.data.label_name

        
        # Loading data
        dataset_dir = Path(config.data.path) / config.data.dataset_name
        labeled_dataset = load_dataset('json', data_files=str(dataset_dir / 'labeled.json'), split="train")
        unlabeled_dataset = load_dataset('json', data_files=str(dataset_dir / 'unlabeled.json'), split="train")
        
        with open(dataset_dir / 'tags.json') as f:
            labels = json.load(f)
        label2id = {v: k for k, v in enumerate(labels)}
        self.id2label = OrderedDict(list(sorted([(v, k) for k, v in label2id.items()], key=lambda e: e[1])))
        
        X_labeled = labeled_dataset[self.text_field_name]
        y_labeled = labeled_dataset[self.label_name]
        
        X_unlabeled = unlabeled_dataset[self.text_field_name]
        y_unlabeled = self.load_annotations(self.save_path)
        
        self.test_instances = load_dataset('json', data_files=str(dataset_dir / 'test.json'), split="train")

        if self.task == 'ner':
            self.X_helper, offsets = create_helper(X_unlabeled)
            self.annotation_converter = AnnotationConverterBio(offsets)
        else:
            self.X_helper = pd.DataFrame([e for e in X_unlabeled], columns=["texts"])
            self.annotation_converter = None

        # Create model
        model = construct_wrapper(
            config,
            config.acquisition_model,
            None,
            "transformers",
            self.id2label,
            "acquisition",
            time_dict_path=None,
            embeddings=None,
            word2idx=None,
        )

        # Create Active learner
        n_instances = config.al.step_p_or_n
        if isinstance(n_instances, float):
            n_instances = round(n_instances * len(X_labeled))
        
        np.random.seed(1) # For starting instances in the widget to be consistent
        al4ner_qs = AdaptorAl4Nlp(config, model, label2id, self.task, strategy_kwargs={'select_by_number_of_tokens': False})
        
        cl_learner = ActiveLearnerAsync if os.environ.get("USE_ASYNC", "True").lower() in {"true", "1", "t"} else ActiveLearner
        self.active_learner = cl_learner(
            active_learn_alg=al4ner_qs,
            X_labeled_dataset=X_labeled,
            y_labeled_dataset=y_labeled,
            X_unlabeled_dataset=X_unlabeled,
            y_unlabeled_dataset=y_unlabeled,
            rnd_start_steps=1,
            n_instances=n_instances
        )

        self.active_learner.start()

    def create_active_learning_widget(self):
        if self.al_widget is not None:
            return self.al_widget

        self.stop_ui()

        if self.task == "ner":
            tags = list(
                OrderedDict(
                    (
                        (tag.split("-")[1], None)
                        for tag in self.id2label.values()
                        if len(tag.split("-")) > 1
                    )
                ).keys()
            )
        else:
            tags = self.id2label.values()

        
        self.al_widget = ActiveLearnerUiWidget(
            active_learner=self.active_learner,
            X_helper=self.X_helper,
            annotation_converter=self.annotation_converter,
            display_feature_table=False,
            drop_labels=[],
            visualize_columns=["texts"],
            y_labels=None if self.task == "ner"
            else {label: i for i, label in enumerate(tags)},
            save_path=str(self.save_path / "annotation.json"), 
            save_time=120,
            visualizer=SeqAnnotationVisualizer(tags=tags)
            if self.task == "ner"
            else None,
        )

        return self.al_widget

    def save_model(self):
        seq_tagger = self.get_seq_tagger()
        seq_tagger = seq_tagger.cpu()
        torch.save(seq_tagger.state_dict(), f"{self.save_path}/model.pth")

    def add_custom_examples(self): # TODO: fix custom instances
        all_custom_examples = []
        for val, rep in self.custom_examples:
            for _ in range(rep):
                all_custom_examples.append(copy.deepcopy(val))
        logger.info(f"type of custom_examples: {type(self.custom_examples)}")
        logger.info(f"custom_examples: {self.custom_examples}")
        all_answers = []
        for answer, rep in zip(
            self.custom_annotation_widget.get_answers().tolist(),
            [e[1] for e in self.custom_examples],
        ):
            logger.info(f"answer: {answer}")
            logger.info(f"rep: {rep}")
            for _ in range(rep):
                all_answers.append(copy.deepcopy(answer))

        self.active_learner._active_learn_alg._libact_query_alg.impl.model._additional_X += (
            all_custom_examples
        )
        self.active_learner._active_learn_alg._libact_query_alg.impl.model._additional_y += (
            all_answers
        )

        np.save(
            Path(self.save_path) / "custom_X.npy",
            self._get_libact_nn()._additional_X,
            allow_pickle=True,
        )
        np.save(
            Path(self.save_path) / "custom_y.npy",
            self._get_libact_nn()._additional_y,
            allow_pickle=True,
        )

    def create_custom_annotator_widget(self, custom_examples):
        self.custom_examples = custom_examples
        self.custom_annotation_widget = AnnotatorWidget(
            pd.DataFrame([e[0] for e in custom_examples], columns=["text"]),
            answers=None,
            visualize_columns=[],
            drop_labels=[],
            visualizer=SeqAnnotationVisualizer(tags=self.tags),
            display_feature_table=False,
            y_labels=None,
        )

        return self.custom_annotation_widget

    def stop_ui(self):
        try:
            if active_learn_ui:
                self.active_learn_ui.stop()
        except NameError:
            pass
