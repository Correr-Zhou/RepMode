# Author: Evan Wiederspan <evanw@alleninstitute.org>
import numpy as np


def get_edges(img, bg_val=0, axis=(-3, -2, -1)):
    """
    Returns the indices of the edges of the structure in the image
    :param img: CZYX image as a 4d numpy array
    :param bg_val: value to use for background
    :param axis: axis to get the edges for. Output length will be equal to axis length.
    Default is the last three axis
    :return: tuple of the same length as axis parameter. Contains lists the contain the left and right edges
    for each axis specified. 
    """
    try:
        ndim = img.ndim
    except AttributeError:
        raise ValueError("img must be a numpy array")
    # turn negative numbers in axis into positive
    try:
        axis = tuple(a if a >= 0 else ndim + a for a in axis)
        ends_list = tuple([0, img.shape[a]] for a in axis)
    except TypeError:
        raise ValueError("All values in axis must be integers")
    except IndexError:
        raise ValueError("All axis must be integers in the range of {} to {}".format(-ndim, ndim-1))
    for a_i, a in enumerate(axis):
        axis_slice = [slice(None, None)] * ndim
        axis_length = img.shape[a] - 1
        # loop from front to find min
        for s_i in range(axis_length):
            axis_slice[a] = s_i
            # iterate through until we find a slice that contains values other than bg_val,
            if not np.all(np.isnan(img[axis_slice]) if np.isnan(bg_val) else img[axis_slice] == bg_val):
                ends_list[a_i][0] = s_i
                break
        # loop from back to find max
        for s_i in range(axis_length, 0, -1):
            axis_slice[a] = s_i
            if not np.all(np.isnan(img[axis_slice]) if np.isnan(bg_val) else img[axis_slice] == bg_val):
                ends_list[a_i][1] = s_i + 1
                break
    return ends_list


def crop(img, bg_val=0, axis=(-3, -2, -1), padding=0, get_slices=False):
    """
    Crops an image to remove the background color bg_val along arbitrary axis
    :param img: numpy array to crop
    :param bg_val: value to crop out. Default is 0
    :param axis: tuple or list of axis indices to crop along. Can be either positive or negative values.
    Negative values will be from the end of the array as opposed to the start. By default, it crops along the last
    three axes
    :param padding: integer. Specifies how much of the background value to leave in the output. Will be applied 
    on all axis that are being cropped
    :param get_slices: boolean. If True, will return the slice indices that were taken out of the original image
    along with the cropped image. Default is False
    :return: either the cropped numpy array, or a tuple containing the cropped array and a tuple of slices taken
    out of the original data
    """
    # check that padding is a positive integer
    if not isinstance(padding, int) or padding < 0:
        raise ValueError('padding must be a positive integer')
    # get_edges will raise ValueErrors if parameters are bad
    edges = get_edges(img, bg_val=bg_val, axis=axis)
    # list of lists representing slice endpoints. Item i refers to the
    # slice for axis i
    ends_list = [[0, img.shape[a]] for a in range(img.ndim)]
    # merge edges into ends_list
    for a, edge in zip(axis, edges):
        ends_list[a] = edge
    # add in padding
    ends_list = [[max(0, ends[0] - padding), min(length, ends[1] + padding)]
                 for length, ends in zip(img.shape, ends_list)]
    crop_slices = tuple(slice(*axis_slice) for axis_slice in ends_list)
    if get_slices:
        return (img[crop_slices].copy(), tuple(ends_list))
    else:
        return img[crop_slices].copy()
