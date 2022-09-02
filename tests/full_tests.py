import os
import warnings
from pathlib import Path
from shutil import rmtree
from time import time

from hydra import initialize, compose

warnings.filterwarnings("ignore")


os.environ["HYDRA_CONFIG_PATH"] = ""
os.environ["HYDRA_CONFIG_NAME"] = ""
if (
    Path("/proc/driver/nvidia/version").exists()
    and os.environ.get("USE_CUDA_FOR_TESTS", True)
    and len(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) > 0
):
    USE_CUDA = True
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_ID_TO_USE", "0")
else:
    USE_CUDA = False
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

Path("test_output").mkdir(exist_ok=True)
os.chdir("test_output")

MODEL_OVERRIDES = lambda model_type: [
    f"{model_type}.training.trainer_args.fp16.training={USE_CUDA}",
    f"{model_type}.training.batch_size_args.min_num_gradient_steps=10",
    f"{model_type}.training.trainer_args.num_epochs=1",
    f"{model_type}.training.trainer_args.serialization_dir=output",
    f"{model_type}.training.batch_size_args.adjust_num_epochs=False",
    f"{model_type}.training.batch_size_args.adjust_batch_size=False",
]

AL_OVERRIDES = [
    "al.num_queries=1",
    "seed=42",
    "al.iters_to_recalc_scores=no",
    "al.sampling_type=random",
    "hydra.run.dir=.",
]

FULL_DATA_OVERRIDES = [
    "seed=42",
    "hydra.run.dir=.",
]

NER_TRANSFORMERS = [
    "data.dataset_name=ner_test",
    f"output_dir=test_ner_use_{'cuda' if USE_CUDA else 'cpu'}/",
    "al.step_p_or_n=0.01",
    "al.init_p_or_n=0.01",
    "al.gamma_or_k_confident_to_save=0.1",
]

CLS_TRANSFORMERS = [
    "data.dataset_name=bbc_news",
    "data.train_size_split=0.95",
    f"output_dir=test_cls_use_{'cuda' if USE_CUDA else 'cpu'}/",
    "al.step_p_or_n=0.002",
    "al.init_p_or_n=0.002",
    "al.gamma_or_k_confident_to_save=0.02",
]

AL_NER_TRANSFORMERS = (
    AL_OVERRIDES + MODEL_OVERRIDES("acquisition_model") + NER_TRANSFORMERS
)
AL_CLS_TRANSFORMERS = (
    AL_OVERRIDES + MODEL_OVERRIDES("acquisition_model") + CLS_TRANSFORMERS
)
FULL_NER_TRANSFORMERS = (
    FULL_DATA_OVERRIDES + MODEL_OVERRIDES("model") + NER_TRANSFORMERS[:2]
)
FULL_CLS_TRANSFORMERS = (
    FULL_DATA_OVERRIDES + MODEL_OVERRIDES("model") + CLS_TRANSFORMERS[:3]
)
SUCCESSOR_OVERRIDES = MODEL_OVERRIDES("successor_model")
TARGET_OVERRIDES = MODEL_OVERRIDES("target_model")


from al_benchmark.run_active_learning import run_active_learning
from al_benchmark.run_full_data import run_full_data

# TODO: add following tests:
# full data on ner/cls for all frameworks
# domain adaptation
# distillation
# real al? how?


# define some utils functions
def run_al_test_with_params(
    config_path, config_name, overrides, test_name, run_full=False
):
    start_time = time()
    init_dir = os.getcwd()
    # To be able to launch from both `tests` dir and root dir
    data_path = get_data_path()
    overrides = overrides + [
        f"data.path={data_path}",
    ]
    with initialize(config_path=config_path):
        config = compose(config_name=config_name, overrides=overrides,)
    work_dir = config.output_dir
    Path(work_dir).mkdir(exist_ok=True)
    if run_full:
        run_full_data(config)
    else:
        run_active_learning(config)
    print(f"Test {test_name} passed within {time() - start_time:.1f} seconds")
    os.chdir(init_dir)
    rmtree(work_dir, ignore_errors=True)


def get_data_path():
    data_path = "al_benchmark/data"
    if "al_benchmark" not in os.listdir():
        data_path = f"../{data_path}"
    return data_path


# block 1: test al and full data training for diff models and frameworks
# in this test we didn't check results but just check if experiments working
# without exceptions


def test_al_ner_transformers():
    config_path = "../al_benchmark/configs"
    config_name = "al_ner"
    test_name = "al ner transformers"
    run_al_test_with_params(config_path, config_name, AL_NER_TRANSFORMERS, test_name)


def test_full_ner_transformers():
    config_path = "../al_benchmark/configs"
    config_name = "full_data_ner"
    test_name = "full ner transformers"
    run_al_test_with_params(
        config_path, config_name, FULL_NER_TRANSFORMERS, test_name, True
    )


def test_al_cls_transformers():
    config_path = "../al_benchmark/configs"
    config_name = "al_cls"
    test_name = "al cls transformers"
    run_al_test_with_params(config_path, config_name, AL_CLS_TRANSFORMERS, test_name)


def test_full_cls_transformers():
    config_path = "../al_benchmark/configs"
    config_name = "full_data_cls"
    test_name = "full cls transformers"
    run_al_test_with_params(
        config_path, config_name, FULL_CLS_TRANSFORMERS, test_name, True
    )


def test_al_ner_plasm_transformers():
    config_path = "../al_benchmark/configs"
    config_name = "al_ner_plasm"
    test_name = "al ner plasm tracin transformers"
    plasm_tracin_overrides = (
        AL_NER_TRANSFORMERS + SUCCESSOR_OVERRIDES + TARGET_OVERRIDES
    )
    run_al_test_with_params(config_path, config_name, plasm_tracin_overrides, test_name)


def test_al_cls_plasm_transformers():
    config_path = "../al_benchmark/configs"
    config_name = "al_cls_plasm"
    test_name = "al cls plasm tracin transformers"
    plasm_tracin_overrides = (
        AL_CLS_TRANSFORMERS + SUCCESSOR_OVERRIDES + TARGET_OVERRIDES
    )
    run_al_test_with_params(config_path, config_name, plasm_tracin_overrides, test_name)


def test_al_ner_flair():
    config_path = "../al_benchmark/configs"
    config_name = "al_ner_bilstm_crf_flair"
    test_name = "al ner flair"
    flair_overrides = (
        AL_OVERRIDES + MODEL_OVERRIDES("acquisition_model")[1:] + NER_TRANSFORMERS
    )
    run_al_test_with_params(config_path, config_name, flair_overrides, test_name)


"""
def test_al_ner_pytorch():
    config_path = "../al_benchmark/configs"
    config_name = "al_ner_bilstm"
    test_name = "al ner pytorch"
    pytorch_ner_overrides = AL_NER_TRANSFORMERS + [
        "+data.n_vectors=30",
        f"acquisition_model.embeddings_cache_dir=test_ner_use_{'cuda' if USE_CUDA else 'cpu'}/embeddings",
    ]
    run_al_test_with_params(config_path, config_name, pytorch_ner_overrides, test_name)


def test_domain_adaptation():
    config_path = "../domain_adaptation/configs"
    config_name = "domain_adaptation"
    test_name = "domain adaptation"
    run_al_test_with_params(config_path, config_name, FULL_NER_TRANSFORMERS, test_name)
"""
