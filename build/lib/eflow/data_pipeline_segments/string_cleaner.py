from eflow._hidden.parent_objects import DataPipelineSegment

__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"

class StringCleaner(DataPipelineSegment):

    def __init__(self,
                 segment_id=None,
                 create_file=True):
        DataPipelineSegment.__init__(self,
                                     object_type=self.__class__.__name__,
                                     segment_id=segment_id,
                                     create_file=create_file)

    def run_widget(self):
        pass

    def filter_to_alphanumeric(self):
        pass

    def filter_to_alphabetical(self):
        pass

    def filter_to_numerical(self):
        pass

    def filter_to_ascii_characters(self):
        pass

    def remove_if_specfic_char_is_found(self):
        pass

    def remove_string(self):
        pass

    def feature_only_should_contain_these_values(self):
        pass
