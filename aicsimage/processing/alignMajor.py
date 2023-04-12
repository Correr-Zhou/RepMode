# Author Evan Wiederspan <evanw@alleninstitute.org>

import numpy as np
from scipy.ndimage.interpolation import rotate
from math import ceil
from .backgroundCrop import crop


def get_major_minor_axis(img):
    """
    Finds the major and minor axis as 3d vectors of the passed in image
    :param img: CZYX numpy array
    :return: tuple containing two numpy arrays representing the major and minor axis as 3d vectors
    """
    # do a mean projection if more than 3 axes
    if img.ndim > 3:
        z, y, x = np.nonzero(np.mean(img, axis=tuple(range(img.ndim - 3))))
    else:
        z, y, x = np.nonzero(img)
    coords = np.stack([x - np.mean(x), y - np.mean(y), z - np.mean(z)])
    # eigenvectors and values of the covariance matrix
    evals, evecs = np.linalg.eig(np.cov(coords))
    # return largest and smallest eigenvectors (major and minor axis)
    order = np.argsort(evals)
    return (evecs[:, order[-1]], evecs[:, order[0]])


def _get_rotation_matrix(axis, angle):
    """
    Helper function to generate a rotation matrix for an x, y, or z axis
    Used in get_major_angles
    """
    cos = np.cos
    sin = np.sin
    angle = np.radians(angle)
    if axis == 2:
        # z axis
        return np.array([[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]])
    if axis == 1:
        # y axis
        return np.array([[cos(angle), 0, sin(angle)], [0, 1, 0], [-sin(angle), 0, cos(angle)]])
    else:
        # x axis
        return np.array([[1, 0, 0], [0, cos(angle), -sin(angle)], [0, sin(angle), cos(angle)]])


def unit_vector(v):
    """
    Return unit vector of v
    :param v: vector as numpy array
    :return: unit vector of same length as v
    """
    try:
        return v / np.linalg.norm(v)
    except ZeroDivisionError:
        return np.array([0] * v.ndim)


def angle_between(v1, v2):
    """
    Finds angle between two 2d vectors
    :param v1: first vector as a numpy array
    :param v2: second vector as a numpy array
    :return: angle between v1 and v2 in degrees
    """
    if getattr(v1, 'ndim', 0) != 1 or getattr(v2, 'ndim', 0) != 1:
        raise ValueError("v1 and v2 must be 1d numpy arrays")
    dot_prod = np.dot(unit_vector(v1), unit_vector(v2))
    # happens if of one the passed in vectors has length 0
    if np.isnan(dot_prod):
        return 0
    return np.degrees(np.arccos(dot_prod))


def get_align_angles(img, axes="zyx"):
    """
    Returns the angles needed to rotate an image to align it with the specified axes
    :param img: A CZYX image as a 4d numpy array. The image that will be measured to get the
    alignment angles. The image will not be altered by this function
    :param axes: string, that must be an arrangement of 'xyz'
    The major axis will be aligned with the first one, the minor with the last one.
    'zyx' by default
    :return: A list of tuple pairs, containing the axis indices and angles to rotate along the paired
    axis. Meant to be passed into align_major
    """
    if getattr(img, 'ndim', 0) < 3:
        raise ValueError('img must be at least a 3d numpy array')
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if not isinstance(axes, str) or len(axes) != 3 or not all(a in axis_map for a in axes):
        raise ValueError("axes must be an arrangement of 'xyz'")
    # axes parameter string turned into a list of indices
    axis_list = [axis_map[a] for a in axes]
    maj_axis_i = axis_list[0]
    # unit vectors for x, y, and z axis
    axis_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # slices for selecting yz, xz, and xy components from vectors
    slices = (slice(1, 3), slice(0, 3, 2), slice(0, 2))
    # index of the major axis (0, 1, or 2)
    maj_axis = axis_vectors[axis_list[0]]
    min_axis = axis_vectors[axis_list[-1]]
    img_maj_axis, img_min_axis = get_major_minor_axis(img)
    angles = []
    for a in range(3):
        if a != maj_axis_i:
            # rotate around other two axis (e.g if aligning major to Z axis, rotate around Y and X to get there)
            angle = angle_between(maj_axis[slices[a]], img_maj_axis[slices[a]])
            angles.append([a, -angle])
            img_maj_axis = np.dot(_get_rotation_matrix(a, angle), img_maj_axis)
            img_min_axis = np.dot(_get_rotation_matrix(a, angle), img_min_axis)
    # final rotation goes around major axis to align the minor axis properly
    # has to be done last
    angles.append([maj_axis_i, angle_between(min_axis[slices[maj_axis_i]], img_min_axis[slices[maj_axis_i]])])
    return angles


def align_major(images, angles, reshape=True):
    """
    Rotates images based on the angles passed in
    :param images: Either a single image or a list of them. Must be at least 3d
    numpy arrays, ordered as TCZYX
    :param angles: The tuple returned by get_align_angles. Tells the function how to rotate the images
    :param reshape: boolean. If True, the output will be resized to ensure that no data
    from img is lost. If False, the output will be the same size as the input, with potential to
    lose data that lies outside of the input shape after rotation. Default is True
    :return: If a single image was passed in, it will will return a rotated copy of that image. If a list was
    passed in, it will return a list of rotated images in the same order that they were passed in
    """
    if isinstance(images, (list, tuple)):
        return_list = True
        image_list = images
    else:
        # turn it into a single element list for easier code
        return_list = False
        image_list = [images]
    if not all(getattr(img, "ndim", 0) >= 3 for img in image_list):
        raise ValueError("All images must be at least 3d")
    rotate_axes = ((-3, -2), (-3, -1), (-2, -1))
    # output list
    out_list = []
    for img in image_list:
        out = img.copy()
        for axis, angle in angles:
            out = rotate(out, angle, reshape=reshape, order=1, axes=rotate_axes[axis], cval=(np.nan if reshape else 0))
        out_list.append(out)

    if reshape:
        # cropping necessary as each resize makes the image bigger
        # np.nan used as fill value when reshaping in order to make cropping easy
        for i in range(len(out_list)):
            out_list[i] = crop(out_list[i], np.nan)
            out_list[i][np.isnan(out_list[i])] = 0
    if return_list:
        return out_list
    else:
        # turn it back from a single element list to a single image
        return out_list[0]
