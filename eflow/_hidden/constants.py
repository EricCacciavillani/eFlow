__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"

def enum(**enums):
    return type('Enum', (), enums)


SYS_CONSTANTS = enum(PARENT_OUTPUT_FOLDER_NAME="eflow Data")
GRAPH_DEFAULTS = enum(FIGSIZE=(10, 8),
                      DEFAULT_NULL_COLOR="#072F5F")
# FORMATED_STRINGS = enum()
# STRING_CONSTANTS = enum()
