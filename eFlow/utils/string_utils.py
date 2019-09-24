def convert_to_filename(filename):
    return "".join(x for x in str(
        filename) if x.isalnum() or x == "_" or x == "("
            or x == ")" or x == " " or x == "-")


def correct_directory_path(directory_pth):
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