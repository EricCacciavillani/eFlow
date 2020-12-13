__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"

def enum(**enums):
    return type('Enum', (), enums)


SYS_CONSTANTS = enum(PARENT_OUTPUT_FOLDER_NAME="eflow Data")


DEFINED_LIST_OF_RANDOM_COLORS = ["#766604","#61396b","#fe1ce2","#0b03ce","#5b3461",
                                 "#93cff6","#d1ff1c","#3d0126","#1b8535","#241c68",
                                 "#55afc4","#d45261","#0b2547","#20944c","#70a68a",
                                 "#601970","#0e12b5","#8e600d","#af4e47","#114d13",
                                 "#71517b","#5142ce","#52cebd","#a3d833","#1af08f",
                                 "#4ff7e4","#8620f1","#51fae2","#edf3ee","#ae6524"]

GRAPH_DEFAULTS = enum(FIGSIZE=(13, 10),
                      NULL_FIGSIZE=(24, 10),
                      NULL_COLOR="#072F5F",
                      DEFINED_LIST_OF_RANDOM_COLORS=DEFINED_LIST_OF_RANDOM_COLORS)


BOOL_STRINGS = enum(TRUE_STRINGS={"y",
                                  "yes",
                                  "okay",
                                  "t",
                                  "true",
                                  "approve",
                                  "approved",
                                  "accepted",
                                  "alright"},
                    FALSE_STRINGS={"n",
                                   "no",
                                   "f",
                                   "false",
                                   "denined",
                                   "deny",
                                   "refuse",
                                   "abnegate",
                                   "never"})


# FORMATED_STRINGS = enum()
# STRING_CONSTANTS = enum()
