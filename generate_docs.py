# Import libs
import os

return getmarkdown(mod)

# Taken from utils.sys_utils
def get_all_directories_from_path(directory_pth):
    """
    directory_pth:
        Given path that already exists.

    Returns/Desc:
        Returns back a set a directories with the provided path.
    """

    dirs_in_paths = []
    for (dirpath, dirnames, filenames) in os.walk(directory_pth):
        dirs_in_paths.extend(dirnames)
        break

    return set(dirs_in_paths)


def get_all_files_from_path(directory_pth,
                            file_extension=None):
    """
    directory_pth:
        Given path that already exists.

    file_extension:
        Only return files that have a given extension.

    Returns/Desc:
        Returns back a set a filenames with the provided path.
    """

    files_in_paths = []
    for (dirpath, dirnames, filenames) in os.walk(directory_pth):

        if file_extension:
            file_extension = file_extension.replace(".","")
            for file in filenames:
                if file.endswith(f'.{file_extension}'):
                    files_in_paths.append(file)
        else:
            files_in_paths.extend(filenames)
        break

    return set(files_in_paths)

# Get the current working directory
current_work_dir = os.getcwd()
project_dir = current_work_dir[:current_work_dir.rfind('/')] + "/eflow/"

# Get all directories from project
all_dirs = get_all_directories_from_path(project_dir)

for dir_name in all_dirs:

    # Ignore any hidden files
    if dir_name[0] == "_":
        continue

    # Ignore utils for now
    if dir_name == "utils":
        continue

    dir_files = get_all_files_from_path(project_dir + dir_name,
                                        "py")
    print(dir_files)
    for file_name in dir_files:

        print(file_name)
        # Ignore hidden file
        if file_name[0] == "_":
            continue

        def_start = False
        with open(f'{project_dir}{dir_name}/{file_name}') as fp:
            line = fp.readline()
            while line:
                line = fp.readline()

                if line == "":


                # Create template
                if "# def " in line or "#def ":
                    continue

                if ("def " in line and "def _" not in line) or def_start:
                    def_start = True
                    if "):" in line:
                        def_start = False
                    print(line)
        break
    break