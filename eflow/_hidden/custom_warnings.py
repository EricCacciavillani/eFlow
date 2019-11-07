import warnings

__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"



def custom_formatwarning(msg,
                         *args,
                         **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'

class EflowWarning(UserWarning):
    warnings.formatwarning = custom_formatwarning
