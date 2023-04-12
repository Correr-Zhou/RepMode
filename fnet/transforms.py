import numpy as np
import os
import pdb
import scipy
import warnings
import pdb


def normalize(img):
    """Subtract mean, set STD to 1.0"""
    result = img.astype(np.float64)
    result -= np.mean(result)
    result /= np.std(result)
    return result


def do_nothing(img):
    return img.astype(np.float)


class Propper(object):
    """Padder + Cropper"""

    def __init__(self, action='-', **kwargs):
        assert action in ('+', '-')

        self.action = action
        if self.action == '+':
            self.transformer = Padder('+', **kwargs)
        else:
            self.transformer = Cropper('-', **kwargs)

    def __repr__(self):
        return 'Propper({})'.format(self.action)

    def __str__(self):
        return '{} => transformer: {}'.format(self.__repr__(), self.transformer)

    def __call__(self, x_in):
        return self.transformer(x_in)

    def undo_last(self, x_in):
        return self.transformer.undo_last(x_in)


class Padder(object):
    def __init__(self, padding='+', by=16, mode='constant'):
        """
        padding: '+', int, sequence
          '+': pad dimensions up to multiple of "by"
          int: pad each dimension by this value
          sequence: pad each dimensions by corresponding value in sequence
        by: int
          for use with '+' padding option
        mode: str
          passed to numpy.pad function
        """
        self.padding = padding
        self.by = by
        self.mode = mode

        self.pads = {}
        self.last_pad = None

    def __repr__(self):
        return 'Padder{}'.format((self.padding, self.by, self.mode))

    def _calc_pad_width(self, shape_in):
        if isinstance(self.padding, (str, int)):
            paddings = (self.padding, )*len(shape_in)
        else:
            paddings = self.padding
        pad_width = []
        for i in range(len(shape_in)):
            if isinstance(paddings[i], int):
                pad_width.append((paddings[i],)*2)
            elif paddings[i] == '+':
                padding_total = int(np.ceil(1.*shape_in[i]/self.by)*self.by) - shape_in[i]
                pad_left = padding_total//2
                pad_right = padding_total - pad_left
                pad_width.append((pad_left, pad_right))
        assert len(pad_width) == len(shape_in)
        return pad_width

    def undo_last(self, x_in):
        """Crops input so its dimensions matches dimensions of last input to __call__."""
        assert x_in.shape == self.last_pad['shape_out']
        slices = [slice(a, -b) if (a, b) != (0, 0) else slice(None) for a, b in self.last_pad['pad_width']]
        return x_in[slices].copy()

    def __call__(self, x_in):
        shape_in = x_in.shape
        pad_width = self.pads.get(shape_in, self._calc_pad_width(shape_in))
        x_out = np.pad(x_in, pad_width, mode=self.mode)
        if shape_in not in self.pads:
            self.pads[shape_in] = pad_width
        self.last_pad = {'shape_in': shape_in, 'pad_width': pad_width, 'shape_out': x_out.shape}
        return x_out


class Cropper(object):
    def __init__(self, cropping, by=16, offset='mid', n_max_pixels=9732096):
        """Crop input array to given shape."""
        self.cropping = cropping
        self.offset = offset
        self.by = by
        self.n_max_pixels = n_max_pixels

        self.crops = {}
        self.last_crop = None

    def __repr__(self):
        return 'Cropper{}'.format((self.cropping, self.by, self.offset, self.n_max_pixels))

    def _adjust_shape_crop(self, shape_crop):
        key = tuple(shape_crop)
        shape_crop_new = list(shape_crop)
        prod_shape = np.prod(shape_crop_new)
        idx_dim_reduce = 0
        order_dim_reduce = list(range(len(shape_crop))[-2:])  # alternate between last two dimensions
        while prod_shape > self.n_max_pixels:
            dim = order_dim_reduce[idx_dim_reduce]
            if not (dim == 0 and shape_crop_new[dim] <= 64):
                shape_crop_new[dim] -= self.by
                prod_shape = np.prod(shape_crop_new)
            idx_dim_reduce += 1
            if idx_dim_reduce >= len(order_dim_reduce):
                idx_dim_reduce = 0
        value = tuple(shape_crop_new)
        print('DEBUG: cropper shape change', shape_crop, 'becomes', value)
        return value

    def _calc_shape_crop(self, shape_in):
        croppings = (self.cropping, )*len(shape_in) if isinstance(self.cropping, (str, int)) else self.cropping
        shape_crop = []
        for i in range(len(shape_in)):
            if croppings[i] is None:
                shape_crop.append(shape_in[i])
            elif isinstance(croppings[i], int):
                shape_crop.append(shape_in[i] - croppings[i])
            elif croppings[i] == '-':
                shape_crop.append(shape_in[i]//self.by*self.by)
            else:
                raise NotImplementedError
        if self.n_max_pixels is not None:
            shape_crop = self._adjust_shape_crop(shape_crop)
        self.crops[shape_in]['shape_crop'] = shape_crop
        return shape_crop

    def _calc_offsets_crop(self, shape_in, shape_crop):
        offsets = (self.offset, )*len(shape_in) if isinstance(self.offset, (str, int)) else self.offset
        offsets_crop = []
        for i in range(len(shape_in)):
            offset = (shape_in[i] - shape_crop[i])//2 if offsets[i] == 'mid' else offsets[i]
            if offset + shape_crop[i] > shape_in[i]:
                warnings.warn('Cannot crop outsize image dimensions ({}:{} for dim {}).'.format(offset, offset + shape_crop[i], i))
                raise AttributeError
            offsets_crop.append(offset)
        self.crops[shape_in]['offsets_crop'] = offsets_crop
        return offsets_crop

    def _calc_slices(self, shape_in):
        shape_crop = self._calc_shape_crop(shape_in)
        offsets_crop = self._calc_offsets_crop(shape_in, shape_crop)
        slices = [slice(offsets_crop[i], offsets_crop[i] + shape_crop[i]) for i in range(len(shape_in))]
        self.crops[shape_in]['slices'] = slices
        return slices

    def __call__(self, x_in):
        shape_in = x_in.shape
        if shape_in in self.crops:
            slices = self.crops[shape_in]['slices']
        else:
            self.crops[shape_in] = {}
            slices = self._calc_slices(shape_in)
        x_out = x_in[tuple(slices)].copy()
        self.last_crop = {'shape_in': shape_in, 'slices': slices, 'shape_out': x_out.shape}
        return x_out

    def undo_last(self, x_in):
        """Pads input with zeros so its dimensions matches dimensions of last input to __call__."""
        assert x_in.shape == self.last_crop['shape_out']
        shape_out = self.last_crop['shape_in']
        slices = self.last_crop['slices']
        x_out = np.zeros(shape_out, dtype=x_in.dtype)
        x_out[tuple(slices)] = x_in
        return x_out


class Resizer(object):
    def __init__(self, factors):
        """
        factors - tuple of resizing factors for each dimension of the input array"""
        self.factors = factors

    def __call__(self, x):
        return scipy.ndimage.zoom(x, (self.factors), mode='nearest')  # NOTE scale images in the axes of X and Y

    def __repr__(self):
        return 'Resizer({:s})'.format(str(self.factors))


class ReflectionPadder3d(object):
    def __init__(self, padding):
        """Return padded 3D numpy array by mirroring/reflection.

        Parameters:
        padding - (int or tuple) size of the padding. If padding is an int, pad all dimensions by the same value. If
        padding is a tuple, pad the (z, y, z) dimensions by values specified in the tuple."""
        self._padding = None

        if isinstance(padding, int):
            self._padding = (padding, )*3
        elif isinstance(padding, tuple):
            self._padding = padding
        if (self._padding == None) or any(i < 0 for i in self._padding):
            raise AttributeError

    def __call__(self, ar):
        return pad_mirror(ar, self._padding)


class Capper(object):
    def __init__(self, low=None, hi=None):
        self._low = low
        self._hi = hi

    def __call__(self, ar):
        result = ar.copy()
        if self._hi is not None:
            result[result > self._hi] = self._hi
        if self._low is not None:
            result[result < self._low] = self._low
        return result

    def __repr__(self):
        return 'Capper({}, {})'.format(self._low, self._hi)


def pad_mirror(ar, padding):
    """Pad 3d array using mirroring.

    Parameters:
    ar - (numpy.array) array to be padded
    padding - (tuple) per-dimension padding values
    """
    shape = tuple((ar.shape[i] + 2*padding[i]) for i in range(3))
    result = np.zeros(shape, dtype=ar.dtype)
    slices_center = tuple(slice(padding[i], padding[i] + ar.shape[i]) for i in range(3))
    result[slices_center] = ar
    # z-axis, centers
    if padding[0] > 0:
        result[0:padding[0], slices_center[1] , slices_center[2]] = np.flip(ar[0:padding[0], :, :], axis=0)
        result[ar.shape[0] + padding[0]:, slices_center[1] , slices_center[2]] = np.flip(ar[-padding[0]:, :, :], axis=0)
    # y-axis
    result[:, 0:padding[1], :] = np.flip(result[:, padding[1]:2*padding[1], :], axis=1)
    result[:, padding[1] + ar.shape[1]:, :] = np.flip(result[:, ar.shape[1]:ar.shape[1] + padding[1], :], axis=1)
    # x-axis
    result[:, :, 0:padding[2]] = np.flip(result[:, :, padding[2]:2*padding[2]], axis=2)
    result[:, :, padding[2] + ar.shape[2]:] = np.flip(result[:, :, ar.shape[2]:ar.shape[2] + padding[2]], axis=2)
    return result
