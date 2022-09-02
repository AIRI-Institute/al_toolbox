import logging

import numpy as np

from .annotation_converter import AnnotationConverter

logger = logging.getLogger()



class AnnotationConverterBio(AnnotationConverter):
    """ Supports span annotator for sequence tagging. """
    
    def __init__(self, offsets):
        """
        offsets: offests of starting positions of tokens
        """
        
        self._offsets = np.asarray(offsets)
    
    def __call__(self, indexes, answers):
        selected_offsets = self._offsets[indexes]
        
        res = []
        for i in range(selected_offsets.size):
            res.append(self.to_bio(answers[i], selected_offsets[i], len(selected_offsets[i])))
        
        return res
    
    def find_in_between(self, offsets, start, end):
        res = []
        for i, offset in enumerate(offsets):
            if start <= offset and offset <= end:
                res.append(i)
                
        return res
            
    def to_bio(self, y, offsets, n_tokens):
        good_ys = ["O"] * n_tokens
        
        if not y or y[0] == "None":
            return good_ys
        
        for w_y in y:
            positions = self.find_in_between(offsets, w_y["start"], w_y["end"])
            good_ys[positions[0]] = "B-" + w_y["tag"]

            for pos in positions[1:]:
                good_ys[pos] = "I-" + w_y["tag"]

        return good_ys
