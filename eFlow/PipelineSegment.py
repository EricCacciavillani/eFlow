
from collections import deque

class PipelineSegment:
    def __init__(self):
        self.__pipeline_segment = deque

    def __add_pipe(self,
                   feature,
                   option):
        self.__pipeline_segment.append(feature,
                                       option)
        print(self.__pipeline_segment)