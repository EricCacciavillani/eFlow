def enum(**enums):
    return type('Enum', (), enums)

SYS_CONSTANTS = enum(PARENT_OUTPUT_FOLDER_NAME="eflow Data")
GRAPH_DEFAULTS = enum(FIGSIZE=(10, 8))
# FORMATED_STRINGS = enum()
# STRING_CONSTANTS = enum()
