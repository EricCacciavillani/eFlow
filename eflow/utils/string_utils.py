import random
def convert_to_filename(filename):
    """
    filename:
       String of a potential filename.

    Returns/Desc:
        Attempts to ensure the filename is valid for saving.
    """

    filename = filename.split(".")[0]
    return "".join(x for x in str(
        filename) if x.isalnum() or x == "_" or x == "("
            or x == ")" or x == " " or x == "-")


def correct_directory_path(directory_pth):
    """
    directory_pth:
        String of a potential directory path.

    Returns/Desc:
        Attempts to convert the directory path to a proper one.
    """
    last_char = None
    new_string = ""
    for char in directory_pth:
        if last_char and (last_char == "/" and char == "/"):
            pass
        else:
            new_string += char

        last_char = char

    if new_string[-1] != "/":
        new_string += "/"

    return new_string

def create_hex_decimal_string(string_len=10):
    return f'%0{string_len}x' % random.randrange(16 ** string_len)