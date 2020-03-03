import random

__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"

def convert_to_filename(filename,
                        remove_file_extension=True):
    """
    Desc:
        Attempts to make the filename string valid.

    Args:
        filename: string
           String of a potential filename.

        remove_file_extension: bool
            Removes everything after the first found value of "." found in the
            string if set to true.

    Returns:
        A string that is valid for saving.
    """
    if remove_file_extension:
        filename = filename.split(".")[0]
    return "".join(x for x in str(
        filename) if (x.isalnum() or x.isascii()) and x != ":")


def correct_directory_path(directory_path):
    """
    Desc:
        Attempts to convert the directory path to a proper one by removing
        any double slashes next to one another.

    Args:
        directory_path:
            String of a potential directory path.

    Returns:
        Returns the fixed path.
    """
    last_char = None
    new_string = ""
    for char in directory_path:
        if last_char and (last_char == "/" and char == "/"):
            pass
        else:
            new_string += char

        last_char = char

    if new_string[-1] != "/":
        new_string += "/"

    return new_string

def create_hex_decimal_string(string_len=10):
    """
    Desc:
        Creates a string of a random Hexadecimal value.

    Args:
        string_len:
            Length of the Hexadecimal string.

    Returns:
        Returns the Hexadecimal string
    """
    return f'%0{string_len}x' % random.randrange(16 ** string_len)