"""
This visualizer requires text_selector installed.
pip install git+https://github.com/IINemo/text_selector.git
"""

from text_selector.widget import Widget as TextSelectorWidget

from .colors import _COLOR_LIST


class SeqAnnotationVisualizer:
    def __init__(self, tags):
        self._tags = tags
        self._widgets = {}

    def _callback(self, index):
        self._answers[index] = self._widgets[index].res

    def init(self, dataframe, answers):
        self._dataframe = dataframe
        self._answers = answers

    def __call__(self, index):
        selected_colors = _COLOR_LIST[: len(self._tags)]
        widget = TextSelectorWidget(
            txt=self._dataframe.iloc[index, 0],
            tags=self._tags,
            callback=lambda *args, **kwargs: self._callback(index),
            colors=selected_colors,  # ["#33ff99", "#66b2ff", "#ff9999", "#ffff66", "#C0C0C0"],
        )
        if self._answers[index] is not None:
            widget.res = self._answers[index]

        self._widgets[index] = widget
        return widget,
