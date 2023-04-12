# Author: Evan Wiederspan <evanw@alleninstitute.org>
import numpy as np
from scipy.stats import mode


def _mean(img):
    """
    Subtract the mean value from the whole image
    """
    res = img - np.mean(img)
    res[res < 0] = 0
    return res


def _most_common(img):
    """
    Subtract the most common value from the whole image
    """
    common = mode(img, axis=None).mode[0]
    res = img - common
    res[res < 0] = 0
    return res


def _median(img):
    """
    Subtract the median value
    """
    res = img - np.sort(img.flatten())[img.size // 2]
    res[res < 0] = 0
    return res


def background_sub(img, mask=None, method="mean"):
    """
    Performs background subtraction on image using chosen method with optional mask
    :param img: numpy array, image to perform subtraction on
    :param mask: numpy mask, subtraction is calculated and performed on the area specified by the mask
    i.e. the mask should specify the background of the image
    :param method: string, selects the subtraction method to use. Default is 'mean'. Options are: mean, median, common.
    :return: numpy array, copy of input image with background subtracted out
    """
    # apply mask if there is one
    func_map = {'mean': _mean, 'common': _most_common, 'median': _median}
    if method not in func_map:
        raise ValueError("Invalid method")
    sub_method = func_map[method]
    if mask is not None:
        if mask.size != img.size:
            raise ValueError("Invalid mask shape " + mask.shape)
        res = img.copy()
        res[mask] = sub_method(img[mask])
        return res
    else:
        return sub_method(img)
