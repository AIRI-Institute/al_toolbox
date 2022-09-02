import os

import ipywidgets as widgets
from IPython.display import display
from ipywidgets import VBox


def select_data_folders(path):
    attributes = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            if "_checkpoints" not in os.path.join(root, name):
                attributes.append((os.path.join(root, name)))

    return attributes


class PathSelectorWidget(VBox):
    def __init__(self, data_path):
        style = {"description_width": "initial"}
        
        attr_list = select_data_folders(data_path)
        dataset_list = []
        for item in attr_list:
            dataset_list.append(item.split("/")[-1])  # TODO: use pathlib
        dataset_list = sorted(dataset_list)
        self.select_data_widget = widgets.Select(
            options=dataset_list,
            description="Choose dataset:",
            disabled=False,
            style=style,
        )
        
        self.info_widget_save = widgets.Text(
            value=self.select_data_widget.value,
            placeholder="",
            description="Save path is:",
            disabled=False,
            style=style,
        )
        
        self.select_data_widget.observe(self.on_select_change)

        display(self.info_widget_save, self.select_data_widget)
    
    def on_select_change(self, change):
        try:
            self.info_widget_save.value = change.owner.options[change.new['index']]
        except:
            pass
