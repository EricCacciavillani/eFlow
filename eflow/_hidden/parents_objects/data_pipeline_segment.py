from eflow._hidden.parents_objects import FileOutput
from collections import deque


class DataPipelineSegment(FileOutput):
    def __init__(self,
                 project_name):
        self.__project_name = project_name

        self.__function_pipe = deque()

    def __add_function_to_que(self,
                              function_name,
                              param_vals):

        self.__function_pipe.append((function_name,
                                     param_vals))
        self.__print_que()

        if len(self.__function_pipe) == 1:
            FileOutput.__init__(self,
                                self.__project_name)
            del self.__project_name

    def __print_que(self):
        print(self.__function_pipe)