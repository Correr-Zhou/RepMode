# Author: Evan Wiederspan <evanw@alleninstitute.org>
import numpy as np
from scipy.ndimage.measurements import center_of_mass


def get_flips(img, sec, axes=(-3, -2, -1)):
    """
    Calculates which axes to flip in order to have the center of mass of the image
    be located in the desired sector. Meant to be passed to flip()
    :param img: image as an n-dimensional numpy array to perform the calculations on.
    The image will not be modified by this function
    :param sec: String containing '+' and '-', same length as 'axes'. Tells the function
    which side of each axis the center of mass should be on, '+' meaning the upper half and
    '-' meaning the lower half
    >>> get_flips(img, "++-", axes=(-3, -2, -1))
    This, for example, would mean to have the center of mass be on the upper half of the z axis
    (index -3 for a CZYX image), the upper half of the y axis, and the lower half of the x axis
    :param axes: List or tuple of integers, specifies which axes to calculate the needed flips for.
    Default is the last three axes, meant to be the 3 spatial dimensions for a ZYX, CZYX, or TCZYX image.
    Must be the same length as 'sec' parameter
    :return: A list of integers, representing the indices of the axes to flip the image along
    Should be passed to flip()
    """
    if not isinstance(img, np.ndarray):
        raise ValueError("img must be a numpy array")
    com = center_of_mass(img)
    if len(sec) != len(axes):
        raise ValueError("sec and axes must be the same length")
    # return object, list of axis indices to flip on
    flips = []
    for side, axis in zip(sec, axes):
        try:
            # if we want the center of mass on the upper half
            if side == '+':
                if com[axis] < (img.shape[axis] // 2):
                    flips.append(axis)
            # if we want it on the lower half
            elif side == '-':
                if com[axis] > (img.shape[axis] // 2):
                    flips.append(axis)
            else:
                raise ValueError("Invalid sector char '{}', must be '+' or '-'".format(side))
        except IndexError:
            raise ValueError("Out of range axis value " + str(axis))
        except TypeError:
            raise ValueError("Invalid axis value " + str(axis) + ", must be an integer")
    return flips


def flip(images, flips):
    """
    Flips images based on the calculations from get_flips()
    :param images: Either a single n-dimensional image as a numpy array or a list of them.
    The images to flip
    :param flips: The output from get_flips(), tells the function which axes to flip the images along
    All images will be flipped the same way
    :return: Either a single flipped copy of the input image, or a list of them in the same order that they
    were passed in, depending on whether the 'images' parameter was a single picture or a list
    """
    if isinstance(images, (list, tuple)):
        return_list = True
        image_list = images
    else:
        return_list = False
        image_list = [images]
    out = []
    for img in image_list:
        # probably the most I've type 'flip' in my life
        flipped = img
        for flip_axis in flips:
            flipped = np.flip(flipped, flip_axis)
        out.append(flipped.copy())
    if return_list:
        return out
    else:
        return out[0]
