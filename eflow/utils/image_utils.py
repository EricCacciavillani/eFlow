import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageEnhance
import copy
import six

from eflow.utils.sys_utils import check_create_dir_structure
from eflow.utils.string_utils import correct_directory_path

def create_plt_png(directory_pth,
                   sub_dir,
                   filename,
                   sharpness=1.7):
    """
        Saves the plt based image in the correct directory.
    """
    directory_pth = correct_directory_path(directory_pth)

    # Ensure directory structure is init correctly
    abs_path = check_create_dir_structure(directory_pth,
                                          sub_dir)

    # Ensure file ext is on the file.
    if filename[-4:] != ".png":
        filename += ".png"

    fig = plt.figure(1)
    fig.savefig(abs_path + "/" + filename, bbox_inches='tight')

    if sharpness:
        full_path = directory_pth + sub_dir + "/" + filename
        adjust_sharpness(full_path,
                         full_path,
                         sharpness)


def df_to_image(df,
                directory_pth,
                sub_dir,
                filename,
                sharpness=1.7,
                col_width=5.0,
                row_height=0.625,
                font_size=14,
                header_color='#40466e',
                row_colors=['#f1f1f2', 'w'],
                edge_color='w',
                bbox=[0, 0, 1, 1],
                header_columns=0,
                ax=None,
                show_index=False,
                index_color="#add8e6",
                format_float_pos=None,
                **kwargs):

    directory_pth = correct_directory_path(directory_pth)
    df = copy.deepcopy(df)

    if format_float_pos and format_float_pos > 1:
        float_format = '{:,.' + str(2) + 'f}'
        for col_feature in set(df.select_dtypes(include=["float"]).columns):
            df[col_feature] = df[col_feature].map(float_format.format)

    if ax is None:
        size = (np.array(df.shape[::-1]) + np.array([0, 1])) * np.array(
            [col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    if show_index:
        df.reset_index(inplace=True)

    mpl_table = ax.table(cellText=df.values, bbox=bbox,
                         colLabels=df.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            if index_color and show_index and k[1] == 0:
                cell.set_facecolor(index_color)
            else:
                cell.set_facecolor(row_colors[k[0] % len(row_colors)])

    if not sub_dir:
        sub_dir = ""

    create_plt_png(directory_pth,
                   sub_dir,
                   filename,
                   sharpness)

    plt.close()


# Taken from: http://tinyurl.com/y6x7nh7t
def adjust_brightness(input_image,
                      output_image,
                      factor=1.7):
    """
    input_image:

    output_image:

    factor:
    """

    image = Image.open(input_image)
    enhancer_object = ImageEnhance.Brightness(image)
    out = enhancer_object.enhance(factor)
    out.save(output_image)

# Taken from: http://tinyurl.com/y6x7nh7t
def adjust_contrast(input_image,
                    output_image,
                    factor=1.7):
    image = Image.open(input_image)
    enhancer_object = ImageEnhance.Contrast(image)
    out = enhancer_object.enhance(factor)
    out.save(output_image)

# Taken from: http://tinyurl.com/y6x7nh7t
def adjust_sharpness(input_image,
                     output_image,
                     factor=1.7):
    image = Image.open(input_image)
    enhancer_object = ImageEnhance.Sharpness(image)
    out = enhancer_object.enhance(factor)
    out.save(output_image)