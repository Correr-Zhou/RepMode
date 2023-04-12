# Author: Evan Wiederspan

import numpy as np
import matplotlib.pyplot as pplot


def matproj(im, dim, method='max', slice_index=0):
    if method == 'max':
        im = np.max(im, dim)
    elif method == 'mean':
        im = np.mean(im, dim)
    elif method == 'sum':
        im = np.sum(im, dim)
    elif method == 'slice':
        im = im[slice_index]
    else:
        raise ValueError("Invalid projection method")
    return im


def imgtoprojection(im1, proj_all=False, proj_method='max', colors=lambda i: [1, 1, 1], global_adjust=False, local_adjust=False):
    """
    Outputs projections of a 4d CZYX numpy array into a CYX numpy array, allowing for color masks for each input channel
    as well as adjustment options
    :param im1: Either a 4d numpy array or a list of 3D or 2D numpy arrays. The input that will be projected
    :param proj_all: boolean. True outputs XY, YZ, and XZ projections in a grid, False just outputs XY. False by default
    :param proj_method: string. Method by which to do projections. 'Max' by default
    :param colors: Can be either a string which corresponds to a cmap function in matplotlib, a function that
    takes in the channel index and returns a list of numbers, or a list of lists containing the color multipliers.
    :param global_adjust: boolean. If true, scales each color channel to set its max to be 255
    after combining all channels. False by default
    :param local_adjust: boolean. If true, performs contrast adjustment on each channel individually. False by default
    :return: a CYX numpy array containing the requested projections
    """

    # turn list of 2d or 3d arrays into single 4d array if needed
    try:
        if isinstance(im1, (list, tuple)):
            # if only YX, add a single Z dimen
            if im1[0].ndim == 2:
                im1 = [np.expand_dims(c, axis=0) for c in im1]
            elif im1[0].ndim != 3:
                raise ValueError("im1 must be a list of 2d or 3d arrays")
            # combine list into 4d array
            im = np.stack(im1)
        else:
            if im1.ndim != 4:
                raise ValueError("Invalid dimensions for im1")
            im = im1

    except (AttributeError, IndexError):
        # its not a list of np arrays
        raise ValueError("im1 must be either a 4d numpy array or a list of numpy arrays")

    # color processing code
    if isinstance(colors, str):
        # pass it in to matplotlib
        try:
            colors = pplot.get_cmap(colors)(np.linspace(0, 1, im.shape[0]))
        except ValueError:
            # thrown when string is not valid function
            raise ValueError("Invalid cmap string")
    elif callable(colors):
        # if its a function
        try:
            colors = [colors(i) for i in range(im.shape[0])]
        except:
            raise ValueError("Invalid color function")

    # else, were assuming it's a list
    # scale colors down to 0-1 range if they're bigger than 1
    if any(v > 1 for v in np.array(colors).flatten()):
        colors = [[v / 255.0 for v in c] for c in colors]

    # create final image
    if not proj_all:
        img_final = np.zeros((3, im.shape[2], im.shape[3]))
    else:
        #                                 y + z,                     x + z
        img_final = np.zeros((3, im.shape[2] + im.shape[1], im.shape[3] + im.shape[1]))
    img_piece = np.zeros(img_final.shape)
    # loop through all channels
    for i, img_c in enumerate(im):
        try:
            proj_z = matproj(img_c, 0, proj_method, img_c.shape[0] // 2)
            if proj_all:
                proj_y, proj_x = (matproj(img_c, axis, proj_method, img_c.shape[axis] // 2) for axis in range(1, 3))
                # flipping to get them facing the right way
                proj_x = np.transpose(proj_x, (1, 0))
                proj_y = np.flipud(proj_y)
                sx, sy, sz = proj_z.shape[1], proj_z.shape[0], proj_y.shape[0]
                img_piece[:, :sy, :sz] = proj_x
                img_piece[:, :sy, sz:] = proj_z
                img_piece[:, sy:, sz:] = proj_y
            else:
                img_piece[:] = proj_z
        except ValueError:
            raise ValueError("Invalid projection function")

        for c in range(3):
            img_piece[c] *= colors[i][c]

        # local contrast adjustment, minus the min, divide the max
        if local_adjust:
            img_piece -= np.min(img_piece)
            img_max = np.max(img_piece)
            if img_max > 0:
                img_piece /= img_max
        # img_final += img_piece
        img_final += (1 - img_final) * img_piece

    # color range adjustment, ensure that max value is 255
    if global_adjust:
        # scale color channels independently
        for c in range(3):
            max_val = np.max(img_final[c].flatten())
            if max_val > 0:
                img_final[c] *= (255.0 / max_val)

    return img_final
