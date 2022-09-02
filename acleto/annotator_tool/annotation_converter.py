from abc import abstractmethod


class AnnotationConverter:
    """ Converter from anntations in the widget to annotations for internal purpose. """
    
    @abstractmethod
    def __call__(self, indexes, answers):
        pass

    
class AnnotationConverterDefault(AnnotationConverter):
    def __call__(self, indexes, answers):
        return answers
