# author: Zach Crabtree zacharyc@alleninstitute.org
from __future__ import print_function
import numpy as np
import math as m
from scipy.ndimage.filters import gaussian_filter
from skimage.filters import threshold_otsu
from scipy.ndimage.measurements import label
from skimage.measure import regionprops
from skimage import morphology

def keep_connected_components(image, low_threshold, high_threshold=None):
    """
    This will keep components that have a larger volume than low_threshold
    and a smaller volume than high_threshold

    :param image: n-dimensional boolean array
    :param low_threshold: if any component is smaller than this value, it will be removed
    :param high_threshold: if any component is bigger than this value, it will be removed
    :return: n-dimensional boolean array with same size as image
    """
    if high_threshold is None:
        high_threshold = np.prod(image.shape)

    # label objects in image and get total number of objects detected
    labels, num_objects = label(image)
    output = np.zeros(image.shape)

    if num_objects > 1:
        # get the area of each object
        components = regionprops(labels)

        for component in components:
            if low_threshold < component.area <= high_threshold:
                # if the component has a volume within the two thresholds,
                # set the output image to 1 for every pixel of the component
                output[labels == component.label] = 1
    else:
        output = image.copy()

    return output

def fill_nucleus_segmentation(cell_index_img, nuc_original_img):
    """
    This function is built to fill in the holes of the nucleus segmentation channel
    :param cell_index_img: A ZYX ndarray - represents the segmented image of all cell bodies
    :param nuc_original_img: A ZYX ndarray - represents the original image of the nuclei channel
    :return: A ZYX ndarray - represents a corrected segmented image of the nuclei
    """
    if cell_index_img.ndim != 3 or nuc_original_img.ndim != 3:
        raise ValueError("fill_nucleus_segmentation only accepts ZYX ndarrays!")

    # cast as float and normalize the input image
    nuc_original_img = nuc_original_img.astype(np.float64)
    original_max = np.max(nuc_original_img)
    original_min = np.min(nuc_original_img)
    nuc_original_img = (nuc_original_img - original_min) / (original_max - original_min) * 255
    total_out = np.zeros(nuc_original_img.shape)

    for cell_value in range(1, cell_index_img.max() + 1):
        # get indices of cell with cell_value as its ID
        cell_indices = np.where(cell_index_img == cell_value)
        # if a cell exists with this cell_value
        if len(cell_indices[0]) > 0:

            z_indices, y_indices, x_indices = cell_indices[0], cell_indices[1], cell_indices[2]
            # creates buffer of 10 pixels/slices around the cell, or stops at the boundaries of the original image
            x_slice = slice(max(1, -10+min(x_indices)), min(10+max(x_indices), cell_index_img.shape[2]))
            y_slice = slice(max(1, -10+min(y_indices)), min(10+max(y_indices), cell_index_img.shape[1]))
            z_slice = slice(max(1, -10+min(z_indices)), min(10+max(z_indices), cell_index_img.shape[0]))
            # crop and mask the whole cell segmentation
            cropped_cell_seg = cell_index_img[z_slice, y_slice, x_slice].astype(np.float64).copy()
            cropped_cell_seg[cropped_cell_seg != cell_value] = 0
            nucleus_mask = cropped_cell_seg.copy()

            # crop and mask the nucleus channel
            output = nuc_original_img[z_slice, y_slice, x_slice].copy()
            output[nucleus_mask != cell_value] = 0

            # filter the membrane segmentation channel
            sigma = np.divide([51, 51, 21], (4*m.sqrt(2*m.log(2))))
            cropped_cell_seg = gaussian_filter(cropped_cell_seg, sigma)
            # filter the nuclear channel
            output = gaussian_filter(output, sigma)
            # this indexing assures that no values in output are divided by zero
            output[cropped_cell_seg > 0] /= cropped_cell_seg[cropped_cell_seg > 0]
            # threshold and mask to get the new nuclear segmentation
            output[cropped_cell_seg == 0] = 0


            if len(output[output > 0]) > 0:
                otsu_threshold = threshold_otsu(output[output > 0])
                output[output <= otsu_threshold] = 0
                output[output > otsu_threshold] = 1

                # clean the images of objects and holes
                output = morphology.remove_small_objects(output.astype(np.int))
                output = morphology.remove_small_holes(output.astype(np.int))
                # clean each slice of objects and holes
                for z in range(output.shape[0]):
                    output[z] = morphology.remove_small_objects(output[z].astype(np.int))
                    output[z] = morphology.remove_small_holes(output[z].astype(np.int))
                # output needs to be recast as int, because the morphology methods above return boolean arrays
                output = output.astype(np.int)
                # get the total volume and ignore components less than a quarter of that volume
                total_volume = np.count_nonzero(output)
                output = keep_connected_components(output,  total_volume // 4, total_volume * 2)
                # change boolean value back to original segmentation value
                output *= cell_value
                # mask the nucleus inside the cell membrane segmentation again
                output[nucleus_mask != cell_value] = 0
                total_out[z_slice, y_slice, x_slice] += output

    return total_out

