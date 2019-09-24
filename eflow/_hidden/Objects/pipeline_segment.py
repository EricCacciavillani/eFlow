from collections import deque
from eflow._hidden.objects import FileOutput

class PipelineSegment(FileOutput):
    def __init__(self,
                 project_name,
                 overwrite_full_path):
        FileOutput.__init__(project_name,
                            overwrite_full_path)
        self.__pipeline_segment = deque

    def __add_pipe(self,
                   feature,
                   option):
        self.__pipeline_segment.append(feature,
                                       option)

    def get_output_folder(self):
        return FileOutput.get_output_folder(self)