import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NO_TQDM"] = "True"

import warnings

warnings.filterwarnings("ignore")

from tqdm import tqdm
from functools import partialmethod

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

import numpy as np
from yaspin import yaspin
from pathlib import Path
from tabulate import tabulate

import ipywidgets as widgets
from ipywidgets import Layout
from IPython.display import display

import transformers

transformers.utils.logging.set_verbosity_error()

from active_learning_system import ALSystem
from annotator_tool.path_selector_widget import PathSelectorWidget
from configs import default as MAIN_CONFIG

DATA_PATH = Path(os.environ.get("DATASET_PATH", "./data/NER"))
SAVE_PATH = Path(os.environ.get("OUTPUT_PATH", "./output"))


def create_custom_path(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

    return path


class CustomExamplesWidget:
    def __init__(self):
        add_to_custom_button = widgets.Button(
            description="Add results to custom annotation",
            disabled=False,
            button_style="",
            tooltip="Start Annotation",
            icon="check",
        )

        add_to_custom_button_add = widgets.Button(
            description="Add results to custom annotation",
            disabled=False,
            button_style="",
            tooltip="Start Annotation",
            icon="check",
        )


class DemoSystem:
    def __init__(self):
        self.data_selector = None

        self.activation_button = widgets.Button(
            description="Start active learner",
            disabled=False,
            button_style="",
            tooltip="Start active learner",
            icon="play",
            layout=Layout(width="180px"),
        )
        self.activation_button.on_click(self.on_button_clicked_start_annotation)

        self.info_widget_data = widgets.Text(
            value="",
            placeholder="Type something",
            description="Dataset path is:",
            disabled=False,
        )

        self.data_selector = PathSelectorWidget(DATA_PATH)
        self.load_data_button = widgets.Button(
            description="Load data",
            disabled=False,
            button_style="",
            tooltip="Load data",
            icon="check",
        )
        
        display(self.activation_button)

    def print_active_learner_info(self):
        unlab_len = (
            len(self.system.active_learner._X_unlabeled_dataset)
            if self.system.active_learner._y_unlabeled_dataset is None
            else len(
                np.where([(e is None) for e in self.system.active_learner._y_unlabeled_dataset])[0]
            )
        )
        
        labeled_len = len(self.system.active_learner._X_labeled_dataset) + len(self.system.active_learner._y_unlabeled_dataset) - unlab_len
        
        print(
            "\n",
            tabulate(
                [
                    ["Labeled", labeled_len], 
                    ["Unlabeled", unlab_len],
                    ["Test", len(self.system.test_instances)],
                ],
                headers=["Subset", "Number of instances"],
            )
        )

        print(f"\nAcquisition model: {self.system._model_name}")
        
    def on_button_clicked_start_annotation(self, _):
        self.activation_button.disabled = True
        
        self.dataset_path = DATA_PATH / self.data_selector.select_data_widget.value
        self.save_path = create_custom_path(SAVE_PATH / self.data_selector.select_data_widget.value)
        
        self.system = ALSystem(
            config=MAIN_CONFIG, save_path=self.save_path, dataset_path=self.dataset_path
        )
        
        with yaspin() as sp:
            sp.text = "Starting active learner..."
            self.system.load_annotations(self.save_path)
            self.system.create_active_learner()
            sp.ok()
            
        self.print_active_learner_info()

    def show(self):
        try:
            return self.system.create_active_learning_widget()
        except:
            raise Exception(
                'You should initialize active learner first. Please press the "Start active learner" button in the cell above.'
            )

    def evaluate_model(self):
        return self.system.evaluate_learner()

    def predict(self, text):
        return self.system.predict_tags(text)
