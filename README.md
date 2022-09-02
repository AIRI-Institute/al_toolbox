<!-- <p align="center">
<img height="200" acleto="https://cdn-icons-png.flaticon.com/512/2092/2092791.png" alt="ALToolbox" />
</p>
 -->
# 🛠 ALToolbox 
<!-- **🛠 ALToolbox 🛠**:  -->
ALToolbox is a framework for practical active learning in NLP.
<hr>

[Installation](#installation) | [Quick Start](#quick_start) | [Overview](#overview) | [Docs](#documentation) | [Citation](#citation)

<!-- ALToolbox provides a set of tools **Active Learning** for text classification and sequence tagging tasks: state-of-the-art query strategies, 
Several pre-implemented Query Strategies, Initialization Strategies, and Stopping Criterion are provided, 
which can be easily mixed and matched to build active learning applications or run experiments.
 -->
ALToolbox is a framework for **active learning** annotation in natural language processing. Currently, the framework supports text classification and sequence tagging tasks. ALToolbox provides state-of-the-art query strategies, serverless annotation tool for Jupyter IDE, and a set of tools that help to reduce computational overhead / duration of AL iterations and increase annotated data reusability.

<!-- computationally efficient and reusable -->


## <a name="installation"></a>⚙️ Installation 

```bash
pip install acleto
```
To annotate instances for active learning in Jupyter Notebook or Jupyter Lab one have to install additional widget after framework installation. In case of Jupyter Notebook usage run:
```bash
jupyter nbextension install --py --symlink --sys-prefix text_selector
jupyter nbextension enable --py --sys-prefix text_selector
```
In case of Jupyter Lab usage run:
```bash
jupyter labextension install js
jupyter labextension install text_selector
```

## <a name="quick_start"></a>💫 Quick Start 

For quick start, please see the examples of launching an active learning annotation or benchmarking a novel query stategy / unlabeled pool subsampling strategy for sequence tagging and text classification tasks:

| #   | Notebook                                                                                                                 |
|-----|--------------------------------------------------------------------------------------------------------------------------|
| 1   | [Launching Active Learning for Token Classification](jupyterlab_demo/ner_demo.ipynb)                                     |
| 2   | [Launching Active Learning for Text Classification](jupyterlab_demo/cls_demo.ipynb)                                      |
| 3   | [Benchmarking a novel AL query strategy / unlabeled pool subsampling strategy](examples/benchmark_custom_strategy.ipynb) |                   


## <a name="overview"></a>🔭 Overview 


### 1. Query Strategies 

| #   | Strategy                                                                                               | Citation                                                                                             |
|-----|--------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| 1   | [ALPS](al4nlp/query_strategies/alps_sampling.py)                                                       | [Citation](https://aclanthology.org/2020.emnlp-main.637/) | 
| 2   | [BADGE](al4nlp/query_strategies/badge_sampling.py)                                            | [Citation](https://openreview.net/forum?id=ryghZJBKPS) | 
| 3   | [BAIT](al4nlp/query_strategies/bait_sampling.py)                                              | [Citation](https://proceedings.neurips.cc/paper/2021/file/4afe044911ed2c247005912512ace23b-Paper.pdf) | 
| 4   | [BALD](al4nlp/query_strategies/bald_sampling.py)                                              | [Citation](https://arxiv.org/abs/1112.5745) | 
| 5   | [BatchBALD](al4nlp/query_strategies/batchbald_sampling.py)                                    | [Citation](https://proceedings.neurips.cc/paper/2019/file/95323660ed2124450caaac2c46b5ed90-Paper.pdf) | 
| 6   | [Breaking Ties (BT) (also Maximum Margin)](al4nlp/query_strategies/breaking_ties_sampling.py) | [Citation](https://ieeexplore.ieee.org/document/1334570) | 
| 7   | [Contrastive Active Learning (CAL)](al4nlp/query_strategies/cal_sampling.py)                  | [Citation](https://aclanthology.org/2021.emnlp-main.51/) | 
| 8   | [Cluster Margin](al4nlp/query_strategies/cluster_margin_sampling.py)                          | [Citation](https://arxiv.org/abs/2107.14263) | 
| 9   | [Coreset](al4nlp/query_strategies/coreset_sampling.py)                                        | [Citation](https://openreview.net/forum?id=H1aIuk-RW) | 
| 10  | [Expected Gradient Length (EGL)](al4nlp/query_strategies/egl_sampling.py)                     | [Citation](https://openreview.net/forum?id=ryghZJBKPS (?)) | 
| 11  | [Embeddings KM](al4nlp/query_strategies/embeddings_km_sampling.py)                            | [Citation](https://aclanthology.org/2020.emnlp-main.637/) | 
| 12  | [Entropy](al4nlp/query_strategies/entropy_sampling.py)                                        | [Citation](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.77.9855&rep=rep1&type=pdf) | 
| 13  | [Least Confidence (LC)](al4nlp/query_strategies/lc_sampling.py)                               | [Citation](https://arxiv.org/abs/cmp-lg/9407020) | 
| 14  | [Mahalanobis Distance](al4nlp/query_strategies/mahalanobis_sampling.py)                       | [Citation](https://proceedings.neurips.cc/paper/2018/file/abdeb6f575ac5c6676b747bca8d09cc2-Paper.pdf) | 
| 15  | [Maximum Normalized Log-Probability (MNLP)](al4nlp/query_strategies/mnlp_sampling.py)         | [Citation](https://aclanthology.org/W17-2630/) | 
| 16  | [Random (No AL)](al4nlp/query_strategies/random_sampling.py)                                  | -                                                                                                    |

### 3. Unlabeled Pool Subsampling Strategies

| #   | Strategy                                                                    | Citation                                                     |
|-----|-----------------------------------------------------------------------------|--------------------------------------------------------------|
| 1   | [UPS](al4nlp/pool_subsampling_strategies/ups_subsampling.py)       | [Citation](https://aclanthology.org/2022.findings-naacl.90/) |
| 2   | [Naïve](al4nlp/pool_subsampling_strategies/naive_subsampling.py)   | [Citation](https://aclanthology.org/2022.findings-naacl.90/) | 
| 3   | [Random](al4nlp/pool_subsampling_strategies/random_subsampling.py) | -                                                            |


### 4. Pipelines for postprocessing of annotated data and preparation of acquisition models

* PLASM postprocessing pipeline for annotated data reusability.
* Acquisition model distillation.
* Domain adaptation of acquisition models.


### 5. GUI Annotator tool in Jupyter IDE

Our framework provides a serverless GUI annotation tool integrated into the Jupyter IDE:
![GUI](gui.svg)

### 6. Extensible benchmark for query strategies

TODO:

## <a name="documentation"></a>📕 Documentation 

### Usage 
The `configs` folder contains config files with general settings. The `experiments` folder contains config files with experimental design. To run an experiment with a chosen configuration, specify config file name in `HYDRA_CONFIG_NAME` variable and run `train.sh` script (see `./examples/al` for details). 

For example to launch PLASM on AG-News with ELECTRA as a successor model:
```
cd PATH_TO_THIS_REPO
HYDRA_CONFIG_PATH=../experiments/ag_news HYDRA_EXP_CONFIG_NAME=ag_plasm python active_learning/run_tasks_on_multiple_gpus.py
```

### Config structure explanation 
- `cuda_devices`: list of CUDA devices to use: one experiment on one CUDA device. `cuda_devices=[0,1]` means using zero-th and first devices.
- `config_name`: name of config from **configs** folder with general settings: dataset, experiment setting (e.g. LC/ASM/PLASM), model checkpoints, hyperparameters etc.
- `config_path`: path to config with general settings.
- `command`: **.py** file to run. For AL experiments, use **run_active_learning.py**.
- `args`: arguments to modify from a general config in the current experiment. `acquisition_model.name=xlnet-base-cased` means that _xlnet-base-cased_ will be used as an acquisition model.
- `seeds`: random seeds to use. `seeds=[4837, 23419]` means that two separate experiments with the same settings (except for **seed**) will be run: one with **seed == 4837**, one with **seed == 23419**.

### Output Explanation 
By default, the results will be present in the folder `RUN_DIRECTORY/workdir_run_active_learning/DATE_OF_RUN/${TIME_OF_RUN}_${SEED}_${MODEL_CHECKPOINT}`. For instance, when launching from the repository folder: `al_nlp_feasible/workdir/run_active_learning/2022-06-11/15-59-31_23419_distilbert_base_uncased_bert_base_uncased`.

- When running a classic AL experiment (acquisition and successor models coincide, regardless of using UPS), the file with the model metrics is `acquisition_metrics.json`.
- When running an acquisition-successor mismatch experiment, the file with the model metrics is `successor_metrics.json`.
- When running a PLASM experiment, the file with the model metrics is `target_tracin_quantile_-1.0_metrics.json` (**-1.0** stands for the filtering value, meaning adaptive filtering rate; when using a deterministic filtering rate (e.g. **0.1**), the file will be named `target_tracin_quantile_0.1_metrics.json`). The file with the metrics of the model **without filtering** is `target_metrics.json`.

### Post-processing
Our framework provides tools for effective data post-processing for its re-usability and a possibility to build powerful models on it.
PLASM, which aims to alleviate the acquisition-successor mismatch problem and allow to build a model of an
arbitrary type using the labeled data without performance degradation, is implemented in `post_processing/pipeline_plasm`. 
It uses the config `cls_plasm` / `ner_plasm` (from `jupyterlab_demo/configs). A brief explanation of the config structure:
- pseudo-labeling model parameters are contained in the key `labeling_model`;
- successor model parameters are contained in the key `successor_model`;
- post-processing options are contained in the key `post_processing`:
  - `label_smoothing`: str / float / None, a parameter for label smoothing (LS) for pseudo-labeled instances. Accepts several options:
    - "adaptive": LS value equals the quality of the labeling model on the validation data.
    - float, 0 < value < 1: absolute value of label smoothing
    - None (default): no label smoothing is used
  - `labeled_weight`: int / float, weight for the labeled-by-human data. 1 < value < +inf 
  - `use_subsample_for_pl`: int / float / None, the size of the subsample used for pseudo-labeling
  (float means taking the share of the unlabeled data). None means that no subsampling is used.
  - `uncertainty_threshold`: float / None, the value of the threshold for filtering by uncertainty. If None,
  no filtering by uncertainty is used.
  - `filter_by_quantile`: bool, only used for classification, ignored if `uncertainty_threshold` is None. If True, `uncertainty_threshold`
  most uncertain instances are filtered. Otherwise, all instances whose (1 - max_prob) < `uncertainty_threshold` are filtered.
  - `tracin`:
    - `use`: bool, whether to use TracIn for filtering
    - `max_num_processes`: int, value > 0, maximum number of processes per one GPU
    - `quantile`: str / float (0 < value < 1), share of unlabeled data instances to filter using the TracIn score.
    - `num_model_checkpoints`: int, value > 0, how many model checkpoints to save and use for TracIn.
    - `nu`: float / int, value for TracIn algorithm.


### 🆕️ New strategies addition 
An AL query strategy should be designed as a function that:
   1) Receives 3 positional arguments and additional strategy kwargs:
     - `model` of inherited class `TransformersBaseWrapper` or `PytorchEncoderWrapper` or `FlairModelWrapper`: model wrapper;
     - `X_pool` of class `Dataset` or `TransformersDataset`: dataset with the unlabeled instances;
     - `n_instances` of class `int`: number of instances to query;
     - `kwargs`: additional strategy-specific arguments.
   2) Outputs 3 objects in the following order:
      - `query_idx` of class `array-like`: array with the indices of the queried instances;
      - `query` of class `Dataset` or `TransformersDataset`: dataset with the queried instances;
      - `uncertainty_estimates` of class `np.ndarray`: uncertainty estimates of the instances from `X_pool`. The higher the value - the more uncertain the model is in the instance.

The function with the strategy should be named the same as the file where it is placed (e.g. function `def my_strategy` inside a file `path_to_strategy/my_strategy.py`).
Use your strategy, setting `al.strategy=PATH_TO_FILE_YOUR_STRATEGY` in the experiment config.

The example is presented in `examples/benchmark_custom_strategy.ipynb`

### 🆕️ New pool subsampling strategies addition 
The addition of a new pool subsampling query strategy is similar to the addition of an AL query strategy. A subsampling strategy should be designed as a function that:
   1) It must receive 2 positional arguments and additional subsampling strategy kwargs:
     - `uncertainty_estimates` of class `np.ndarray`: uncertainty estimates of the instances in the order they are stored in the unlabeled data;
     - `gamma_or_k_confident_to_save` of class `float` or `int`: either a share / number of instances to save (as in random / naive subsampling) or an internal parameter (as in UPS);
     - `kwargs`: additional subsampling strategy specific arguments.
   2) It must output the indices of the instances to use (sampled indices) of class `np.ndarray`.

The function with the strategy should be named the same as the file where it is placed (e.g. function `def my_subsampling_strategy` inside a file `path_to_strategy/my_subsampling_strategy.py`).
Use your subsampling strategy, setting `al.sampling_type=PATH_TO_FILE_YOUR_SUBSAMPLING_STRATEGY` in the experiment config.

The example is presented in `examples/benchmark_custom_strategy.ipynb`

### Datasets 
The research has employed 2 Token Classification datasets (CoNLL-2003, OntoNotes-2012) and 2 Text Classification datasets (AG-News, IMDB). If one wants to launch an experiment on a custom dataset, they need to use one of the following ways to add it:

1) Upload to [Hugging Face datasets](https://huggingface.co/datasets) and set: `config.data.path=datasets, config.data.dataset_name=DATASET_NAME, config.data.text_name=COLUMN_WITH_TEXT_OR_TOKENS_NAME, config.data.label_name=COLUMN_WITH_LABELS_OR_NER_TAGS_NAME`
2) Upload to **data/DATASET_NAME** folder, create **train.csv** / **train.json** file with the dataset, and set: `config.data.path=PATH_TO_THIS_REPO/data, config.data.dataset_name=DATASET_NAME, config.data.text_name=COLUMN_WITH_TEXT_OR_TOKENS_NAME, config.data.label_name=COLUMN_WITH_LABELS_OR_NER_TAGS_NAME`
3) \* Upload to **data/DATASET_NAME** **train.txt**, **dev.txt**, and **test.txt** files and set the arguments as in the previous point.
4) \*\* Upload to **data/DATASET_NAME** with each folder for each class, where each file in the folder contains a text with the label of the folder. For details, please see the **bbc_news** dataset in **./data**. The arguments must be set as in the previous two points.

\* - only for Token Classification datasets

\*\* - only for Text Classification datasets

### Models 
The current version of the repository supports all models from [HuggingFace Transformers](https://huggingface.co/models), which can be used with `AutoModelForSequenceClassification` / `AutoModelForTokenClassification` classes (for Text / Token classification). For CNN-based / BiLSTM-CRF models, please see the **al_cls_cnn.yaml** / **al_ner_bilstm_crf_flair.yaml** configs from **./configs** folder for details.

### Testing 
By default, the tests will be run on the `cuda:0` device if CUDA is available or on CPU, otherwise. If one wants to manually specify the device for running the tests:

- On CPU: `CUDA_VISIBLE_DEVICES="" python -m pytest PATH_TO_REPO/tests`;
- On CUDA: `CUDA_VISIBLE_DEVICES="DEVICE_OR_DEVICES_NUMBER" python -m pytest PATH_TO_REPO/tests`.

We recommend to use CPU for the robustness of the results. The tests for CUDA are written under **Tesla V100-SXM3 32GB, CUDA V.10.1.243**. 

## 👯 Alternatives 

[FAMIE](https://github.com/nlp-uoregon/famie), [Small-Text](https://github.com/webis-de/small-text), [modAL](https://github.com/modAL-python/modAL), [ALiPy](https://github.com/NUAA-AL/ALiPy), [libact](https://github.com/ntucllab/libact)

## <a name="citation"></a>💬 Citation 

```

```

## 📄 License 
© 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research Institute" (AIRI). All rights reserved.

Licensed under the [MIT License](LICENSE).
