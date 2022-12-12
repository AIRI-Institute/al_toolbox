.. _quick_start:

===========
Quick Start
===========

For quick start, please see the examples of launching an active learning annotation or benchmarking a novel query stategy / unlabeled pool subsampling strategy for sequence tagging and text classification tasks:

+-----+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| #   | Notebook                                                                                                                                                                          |
+-----+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 1   | `Launching Active Learning for Token Classification <https://github.com/AIRI-Institute/al_toolbox/blob/main/jupyterlab_demo/ner_demo.ipynb>`_                                     |
+-----+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 2   | `Launching Active Learning for Text Classification <https://github.com/AIRI-Institute/al_toolbox/blob/main/jupyterlab_demo/cls_demo.ipynb>`_                                      |
+-----+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 3   | `Benchmarking a novel AL query strategy / unlabeled pool subsampling strategy <https://github.com/AIRI-Institute/al_toolbox/blob/main/examples/benchmark_custom_strategy.ipynb>`_ |
+-----+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

By default all outputs save in the folder demo.nlpresearch.group/user/username/tree/acleto/jupyterlab_demo/output/dataset_name

- checkpoint_{num_itration} - folder with saved model on each iteration of active learning
- annotation.json - annotated data

If you don't want to save model on each iteration of active learning you need to
add a field al.save_checkpoints=False in the config file. All using configs locate in the current path: demo.nlpresearch.group/user/username/tree/acleto/jupyterlab_demo/configs

For more information check :ref:`config_structure`