import copy
import os

def get_all_files_from_path(directory_path,
                            file_extension=None):
    """

        Gets all filenames with the provided path.

    Args:
        directory_path: string
            Given path that already exists.

        file_extension: string
            Only return files that have a given extension.

    Returns:
        Returns back a set a filenames with the provided path.
    """

    files_in_paths = []
    for (dirpath, dirnames, filenames) in os.walk(directory_path):

        if file_extension:
            file_extension = file_extension.replace(".","")
            for file in filenames:
                if file.endswith(f'.{file_extension}'):
                    files_in_paths.append(file)
        else:
            files_in_paths.extend(filenames)
        break

    return set(files_in_paths)

rst_files = get_all_files_from_path(os.getcwd() + "/_autosummary_old/")
print(rst_files)

for file in rst_files:

    f = open(f"_autosummary_old/{file}", "r")
    import_string = f.readline().replace("\\","").strip()
    sphinx_string = copy.deepcopy(import_string)
    print(sphinx_string)

    class_name = "".join([x.capitalize() for x in import_string.split(".")[-1].split("_")])
    f.close()
    import_string = f"from {import_string} import {class_name}"
    import_string = import_string.replace("_","\\_")

    f = open(f"_autosummary_new/{file}", "a")
    f.write(class_name)
    f.write("\n=================================\n")
    f.write(f"**{import_string}**\n")
    f.write(f"\n.. automodule:: {sphinx_string}\n")

    f.close()
