from eflow._hidden.parent_objects import FileOutput

class JupyterWidget(FileOutput):
    def __init__(self,
                 widget_child_name):
        FileOutput.__init__(self,
                            f'_Extras/Pipeline Structure/JSON Files/Widgets/{widget_child_name}')