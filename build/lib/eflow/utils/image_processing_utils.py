from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageEnhance
import copy

from eflow.utils.sys_utils import create_dir_structure
from eflow.utils.string_utils import correct_directory_path

__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"



def create_plt_png(directory_path,
                   sub_dir,
                   filename,
                   sharpness=1.7):

    """
    Desc:
        Saves the plt based image in the correct directory.

    Args:
        directory_path:
            Already existing directory path.

        sub_dir:
            Directory structure to create on top of the already generated path of
            'directory_path'.

        filename:
            Filename to save into the full path of 'directory_path' + 'sub_dir'.

        sharpness:
            Changes the image's sharpness to look better.
    """
    directory_path = correct_directory_path(directory_path)

    # Ensure directory structure is init correctly
    abs_path = create_dir_structure(directory_path,
                                    sub_dir)

    # Ensure file ext is on the file.
    if filename[-4:] != ".png":
        filename += ".png"

    # plt.show()

    plt.savefig(abs_path + "/" + filename, bbox_inches='tight')

    if sharpness:
        full_path = directory_path + sub_dir + "/" + filename
        adjust_sharpness(full_path,
                         full_path,
                         sharpness)

# Taken from: http://tinyurl.com/y6x7nh7t
def adjust_brightness(input_image_path,
                      output_image_path,
                      factor=1.7):
    """
    Desc:
        Adjust brightness of a saved image and save them to a give image path
        with filename.

    Args:
        input_image_path:
            Path to the image file to apply changes.

        output_image:
            Path to output the modified image file.

        factor:
            A float based value to determine the level of effect on the image.
    """

    image = Image.open(input_image_path)
    enhancer_object = ImageEnhance.Brightness(image)
    out = enhancer_object.enhance(factor)
    out.save(output_image_path)

# Taken from: http://tinyurl.com/y6x7nh7t
def adjust_contrast(input_image_path,
                    output_image_path,
                    factor=1.7):
    """
    Desc:
        Adjust contrast of a saved image and save them to a give image path
        with filename.

    Args:
        input_image_path:
            Path to the image file to apply changes.

        output_image:
            Path to output the modified image file.

        factor:
            A float based value to determine the level of effect on the image.
    """
    image = Image.open(input_image_path)
    enhancer_object = ImageEnhance.Contrast(image)
    out = enhancer_object.enhance(factor)
    out.save(output_image_path)

# Taken from: http://tinyurl.com/y6x7nh7t
def adjust_sharpness(input_image_path,
                     output_image_path,
                     factor=1.7):
    """
    Desc:
        Adjust sharpness of a saved image and save them to a give image path
        with filename.

    Args:
        input_image_path:
            Path to the image file to apply changes.

        output_image:
            Path to output the modified image file.

        factor:
            A float based value to determine the level of effect on the image.
    """
    image = Image.open(input_image_path)
    enhancer_object = ImageEnhance.Sharpness(image)
    out = enhancer_object.enhance(factor)
    out.save(output_image_path)