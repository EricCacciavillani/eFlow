from eflow._hidden.parent_objects import FileOutput

class PipelineJupyterWidget(FileOutput):
    def __init__(self,
                 widget_child_name):
        FileOutput.__init__(self,
                            f'_Extras/Pipeline Structure/Widgets/{widget_child_name}')