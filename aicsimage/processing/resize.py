# Author: Evan Wiederspan <evanw@alleninstitute.org>
from scipy.ndimage.interpolation import zoom


def resize(orig, factor, method="nearest"):
    """
    Scales a numpy array to a new size using a specified scaling method
    :param orig: n-dimen numpy array to resize
    :param factor: integer, double, or n-tuple to scale orig by
    :param method: string, interpolation method to use when resizing. Options are "nearest",
    "bilinear", and "cubic". Default is "nearest"
    :return: n-dimen numpy array
    """
    method_dict = {'nearest': 0, 'bilinear': 1, 'cubic': 2}
    if method.lower() not in method_dict:
        raise ValueError("Invalid interpolation method. Options are: " + ", ".join(method_dict.keys()))
    try:
        return zoom(orig, factor, order=method_dict[method.lower()])
    except RuntimeError:
        # raised by zoom when factor length does not match orig.shape length
        raise ValueError("Factor sequence length does not match input length")


def resize_to(orig, out_size, method="nearest"):
    """
    Scales a numpy array to fit within a specified output size
    :param orig: n-dimen numpy array to resize
    :param out_size: n-tuple, will be the shape of the output array.
    :param method: string, interpolation method to use when resizing. Options are "nearest",
    "bilinear", and "cubic". Default is "nearest"
    :return: n-dimen numpy array
    """
    try:
        if len(orig.shape) != len(out_size):
            raise ValueError("Factor sequence length does not match input length")
        factors = tuple(0 if in_length == 0 else float(out_length) / in_length
                        for out_length, in_length in zip(out_size, orig.shape))
    except TypeError:
        # thrown if out_size is not an iterable or doesn't contain numbers
        raise ValueError("Invalid type for out_size")
    # resize takes care of most of the input validation
    return resize(orig, factors, method)
