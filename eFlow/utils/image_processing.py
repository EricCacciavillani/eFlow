from PIL import Image
from PIL import ImageEnhance

# Taken from: http://tinyurl.com/y6x7nh7t


def adjust_brightness(input_image,
                      output_image,
                      factor=1.7):
    image = Image.open(input_image)
    enhancer_object = ImageEnhance.Brightness(image)
    out = enhancer_object.enhance(factor)
    out.save(output_image)


def adjust_contrast(input_image,
                    output_image,
                    factor=1.7):
    image = Image.open(input_image)
    enhancer_object = ImageEnhance.Contrast(image)
    out = enhancer_object.enhance(factor)
    out.save(output_image)


def adjust_sharpness(input_image,
                     output_image,
                     factor=1.7):
    image = Image.open(input_image)
    enhancer_object = ImageEnhance.Sharpness(image)
    out = enhancer_object.enhance(factor)
    out.save(output_image)