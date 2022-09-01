from pathlib import Path
from setuptools import setup, find_packages
import os


requirements = [
    "hydra-core==1.1.0",
    "datasets>=2.3.2",
    "rouge-score==0.0.4",
    "plotly==5.5.0",
    "psutil",
    "kaleido==0.2.1",
    "seqeval==1.2.2",
    "nlpaug>=1.1.10",
    "ruamel.yaml==0.17.20",
    "scikit-learn==1.0.2",
    "protobuf==3.19.4",
    "tqdm",
    "matplotlib",
    "pandas",
    "mlflow==1.7.2",
    "KDEpy==1.1.0",
    "hnswlib==0.6.0",
    "aioredis",
    "torch",
    "bs4",
    "ray==1.9.1",
    "transformers>=4.20.1",
    "nltk==3.6.5",
    "sacrebleu==1.5.0",
    "pytest==7.1.2",
    "pytest-cov==3.0.0",
    "toma==1.1.0",
    "pytest-runner",
    "hf-lfs>=0.0.3",
    "fastcluster",
    "thinc==8.0.12",
    "wget",
    "gensim",
    "pytreebank",
    "flair==0.10",
    "yargy",
    "yaspin",
    "rich",
    "spacy",
    "regex",
    "pybind11==2.10.0",
    "ipykernel==6.15.1",
    "small-text",
]

os.system("chmod a+x init.sh examples/*")

setup(
    name="acleto",
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests", "*.trainable_sampling*"]
    ),
    version="0.0.1",
    description="A Library for active learning. Supports text classification and sequence tagging tasks.",
    author="Tsvigun A., Sanochkin L., Kuzmin G., Larionov D., and Dr Shelmanov A.",
    license="MIT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="NLP active AL deep learning transformer pytorch PLASM UPS",
    install_requires=requirements,
    include_package_data=True,
    setup_requires=["pytest-runner"],
    tests_require=["pytest==7.1.2"],
    test_suite="tests",
)

os.system("./init.sh")
