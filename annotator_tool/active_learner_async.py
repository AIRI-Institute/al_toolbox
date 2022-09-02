import logging
import multiprocessing.dummy as mp
from multiprocessing import Array

import pandas as pd

from .active_learner import ActiveLearner

logger = logging.getLogger()


class ActiveLearnerAsync(ActiveLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_busy = False
        self.instance_priority = Array("i", list(range(len(self._y_unlabeled_dataset))))
        self._remove_labeled_from_priorities()
        
    def is_busy(self):
        return self._is_busy
        
    def choose_samples_for_annotation(self):
        if self._iteration_num < self._rnd_start_steps:
            return self.choose_random_sample_for_annotation(self.n_instances)
        
        message = {"msg_type": "choose_samples_to_annotate"}
        self.input_queue.put(message)
        return self.instance_priority[:self.n_instances]
        
    def make_iteration(self, indexes, answers):
        self.add_answers(indexes, answers)
        message = {"msg_type": "make_iteration", "indexes": indexes, "answers": answers}
        self.input_queue.put(message)
        self._remove_labeled_from_priorities()
        
    def _remove_labeled_from_priorities(self):
        y = pd.Series(self._y_unlabeled_dataset, index=list(range(len(self._y_unlabeled_dataset))))
        labeled_indexes = y[y.notnull()].index
        self.instance_priority = [e for e in self.instance_priority if e not in labeled_indexes]
        
    def start(self):
        def process_fn(input_queue, output_queue):
            logger.info("Starting backend loop")
            
            self._is_busy = True
            self._active_learn_alg.start(
                self._X_labeled_dataset, self._y_labeled_dataset, 
                self._X_unlabeled_dataset, self._y_unlabeled_dataset)
            self._is_busy = False
            
            self._iteration_num += 1

            while True:
                try:
                    message = input_queue.get(block=True)

                    self._is_busy = True

                    logger.info(f"Received scoring request with {message['msg_type']}")

                    msg_type = message["msg_type"]
                    if msg_type == "choose_samples_to_annotate":
                        indexes = self._active_learn_alg.choose_samples_for_annotation(len(self.instance_priority))
                        self.instance_priority = indexes
                        response = {'indexes' : indexes}

                    elif msg_type == "make_iteration":
                        indexes = message["indexes"]
                        answers = message["answers"]
                        self._active_learn_alg.make_iteration(indexes, answers)
                        response = {"res": 0}
                        self._iteration_num += 1

                        logger.info("Done make iteration. Results are put in queue")

                    else:
                        logger.info(f"Unmatched message type {msg_type}")

                    output_queue.put(response)
                    self._is_busy = False

                except Exception as e:
                    logger.info(f"Exception occurred = {e}")
                    
            logger.info("Process end-of-live")
            return
        
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()

        self.process = mp.Process(
            name="background_process",
            target=process_fn,
            args=(
                self.input_queue,
                self.output_queue,
            ),
        )
        self.process.start()