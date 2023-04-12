#!/usr/bin/env python

# authors: Dan Toloudis danielt@alleninstitute.org
#          Zach Crabtree zacharyc@alleninstitute.org

from __future__ import print_function
import numpy as np
import skimage.transform as t
import math as m

z_axis_index = 0
_cmy = [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]


def get_thresholds(image, **kwargs):
    """
    This function finds thresholds for an image in order to reduce noise and bring up the peak contrast

    :param image: CYX-dimensioned image
    :param kwargs:
        "border_percent" : how much of the corners to ignore when calculating the threshold. Sometimes corners can be unnecessarily bright
                           default = .1
        "max_percent" : how much to ignore from the top intensities of the image
                        default = .998
        "min_percent" : what proportion of the bottom intensities of the image will be factored out
                        default = .40
    :return: tuple of float values, the lower and upper thresholds of the image
    """
    border_percent = kwargs.get("border_percent", .1)
    max_percent = kwargs.get("max_percent", .998)
    min_percent = kwargs.get("min_percent", .4)

    # expects CYX
    # using this allows us to ignore the bright corners of a cell image
    im_width = image.shape[2]
    im_height = image.shape[1]
    left_bound = int(m.floor(border_percent * im_width))
    right_bound = int(m.ceil((1 - border_percent) * im_width)) + 1
    bottom_bound = int(m.floor(border_percent * im_height))
    top_bound = int(m.ceil((1 - border_percent) * im_height)) + 1
    cut_border = image[:, left_bound:right_bound, bottom_bound:top_bound]

    immin = cut_border.min()
    immax = cut_border.max()
    histogram, bin_edges = np.histogram(image, bins=256, range=(immin, immax))
    total_cut = 0
    total_pixels = sum(histogram)
    lower_threshold = 0
    for i in range(len(histogram)):
        column = histogram[i]
        total_cut += column
        if total_cut >= total_pixels * min_percent:
            lower_threshold = bin_edges[i]
            break

    upper_threshold = np.max(cut_border) * max_percent

    return lower_threshold, upper_threshold


def resize_cyx_image(image, new_size):
    """
    This function resizes a CYX image.

    :param image: CYX ndarray
    :param new_size: tuple of shape of desired image dimensions in CYX
    :return: image with shape of new_size
    """
    scaling = float(image.shape[1]) / float(new_size[1])
    # get the shape of the image that is resized by the scaling factor
    test_shape = np.ceil(np.divide(image.shape, [1, scaling, scaling]))
    # sometimes the scaling can be rounded incorrectly and scale the image to
    # one pixel too high or too low
    if not np.array_equal(test_shape, new_size):
        # getting the scaling from the other dimension solves this rounding problem
        scaling = float(image.shape[2]) / float(new_size[2])
        test_shape = np.ceil(np.divide(image.shape, [1, scaling, scaling]))
        # if neither scaling factors achieve the desired shape, then the aspect ratio of the image
        # is different than the aspect ratio of new_size
        if not np.array_equal(test_shape, new_size):
            raise ValueError("This image does not have the same aspect ratio as new_size")

    image = image.transpose((2, 1, 0))

    # im_out = t.resize(image, new_size)

    if scaling < 1:
        scaling = 1.0 / scaling
        im_out = t.pyramid_expand(image, upscale=scaling)
    elif scaling > 1:
        im_out = t.pyramid_reduce(image, downscale=scaling)
    else:
        im_out = image

    im_out = im_out.transpose((2, 1, 0))
    assert im_out.shape == new_size

    return im_out


def create_projection(image, axis, method='max', **kwargs):
    """
    This function creates a 2D projection out of an n-dimensional image.

    :param image: ZCYX array
    :param axis: the axis that the projection should be performed along
    :param method: the method that will be used to create the projection
                   Options: ["max", "mean", "sum", "slice", "sections"]
                   - max will look through each axis-slice, and determine the max value for each pixel
                   - mean will look through each axis-slice, and determine the mean value for each pixel
                   - sum will look through each axis-slice, and sum all pixels together
                   - slice will take the pixel values from the middle slice of the stack
                   - sections will split the stack into proj_sections number of sections, and take a
                   max projection for each.
    :param kwargs:
    :return:
    """
    slice_index = kwargs.get("slice_index", 0)
    sections = kwargs.get("sections", 3)

    if method == 'max':
        image = np.max(image, axis)
    elif method == 'mean':
        image = np.mean(image, axis)
    elif method == 'sum':
        image = np.sum(image, axis)
    elif method == 'slice':
        image = image[slice_index]
    elif method == 'sections':
        separator = int(m.floor(image.shape[0] / sections))
        # stack is a 2D YX im
        stack = np.zeros(image[0].shape)
        for i in range(sections - 1):
            bottom_bound = separator * i
            top_bound = separator + bottom_bound
            # TODO: this line assumes the stack is separated through the z-axis, instead of the designated axis param
            section = np.max(image[bottom_bound:top_bound], axis)
            stack += section
        stack += np.max(image[separator * sections - 1:])

        return stack
    # returns 2D image, YX
    return image


def subtract_noise_floor(image, bins=256):
    # image is a 3D ZYX image
    immin = image.min()
    immax = image.max()
    hi, bin_edges = np.histogram(image, bins=bins, range=(immin, immax))
    # index of tallest peak in histogram
    peakind = np.argmax(hi)
    # subtract this out
    subtracted = image.astype(np.float32)
    subtracted -= bin_edges[peakind]
    # don't go negative
    subtracted[subtracted < 0] = 0
    return subtracted


class ThumbnailGenerator:
    """

    This class is used to generate thumbnails for 4D CZYX images.

    Example:
        generator = ThumbnailGenerator()
        for image in image_array:
            thumbnail = generator.make_thumbnail(image)

    """

    def __init__(self, colors=_cmy, size=128,
                 channel_indices=None, channel_thresholds=None, channel_multipliers=None,
                 mask_channel_index=5, **kwargs):
        """
        :param colors: The color palette that will be used to color each channel. The default palette
                       colors the membrane channel cyan, structure with magenta, and nucleus with yellow.
                       Keep color-blind acccessibility in mind.

        :param size: This constrains the image to have the X or Y dims max out at this value, but keep
                     the original aspect ratio of the image.

        :param channel_indices: An array of channel indices to represent the three main channels of the cell

        :param mask_channel_index: The index for the segmentation channel in image that will be used to mask the thumbnail

        :param kwargs:
            "layering" : The method that will be used to layer each channel's projection over each other.
                         Options: ["superimpose", "alpha-blend"]
                         - superimpose will overwrite pixels on the final image as it layers each channel
                         - alpha-blend will blend the final image's pixels with each new channel layer

            "projection" : The method that will be used to generate each channel's projection. This is done
                           for each pixel, through the z-axis
                           Options: ["max", "slice", "sections"]
                           - max will look through each z-slice, and determine the max value for each pixel
                           - slice will take the pixel values from the middle slice of the z-stack
                           - sections will split the zstack into proj_sections number of sections, and take a
                             max projection for each.

            "proj_sections" : The number of sections that will be used to determine projections, if projection="sections"
            "old_alg" : Use the old algorithm for generating thumbnails.
                    False -> use new parameters
                    True -> use old algorithm
        """

        if channel_indices is None:
            channel_indices = [0, 1, 2]
        if channel_thresholds is None:
            channel_thresholds = [.65, .65, .65]
        if channel_multipliers is None:
            channel_multipliers = [1, 1, 1]

        self.layering = kwargs.get("layering", "alpha-blend")
        self.projection = kwargs.get("projection", "max")
        self.proj_sections = kwargs.get("proj_sections", 3)
        self.old_alg = kwargs.get("old_alg", False)

        assert self.layering == "superimpose" or self.layering == "alpha-blend"
        assert self.projection == "slice" or self.projection == "max" or self.projection == "sections"

        assert len(colors) == 3 and len(colors[0]) == 3
        self.colors = colors

        self.size = size

        assert len(colors) == len(channel_indices)
        assert min(channel_indices) >= 0
        self.channel_indices = channel_indices

        assert len(channel_thresholds) == len(channel_indices)
        self.channel_thresholds = channel_thresholds

        assert len(channel_multipliers) == len(channel_indices)
        self.channel_multipliers = channel_multipliers

        self.mask_channel_index = mask_channel_index

    def _old_algorithm(self, image, new_size, apply_cell_mask=False):
        if apply_cell_mask:
            shape_out_rgb = new_size

            # apply the cell segmentation mask.  bye bye to data outside the cell
            # for i in range(len(self.channel_indices)):
            #     image[:, i] = np.multiply(image[:, i], image[:, self.mask_channel_index] > 0)

            num_noise_floor_bins = 32
            composite = np.zeros(shape_out_rgb)
            for i in range(3):
                ch = self.channel_indices[i]
                # try to subtract out the noise floor.
                # range is chosen to ignore zeros due to masking.  alternative is to pass mask image as weights=im1[-1]
                thumb = subtract_noise_floor(image[:, i], bins=num_noise_floor_bins)
                # apply mask
                thumb = np.multiply(thumb, image[:, self.mask_channel_index] > 0)

                # renormalize
                thmax = thumb.max()
                thumb /= thmax

                # resize before projection?
                rgbproj = np.asarray(thumb)
                rgbproj = create_projection(rgbproj, 0, self.projection, slice_index=rgbproj.shape[1] // 2)
                rgb_out = np.expand_dims(rgbproj, 2)
                rgb_out = np.repeat(rgb_out, 3, 2)

                # inject color.  careful of type mismatches.
                rgb_out *= self.colors[i]

                rgb_out /= np.max(rgb_out)

                rgb_out = resize_cyx_image(rgb_out.transpose((2, 1, 0)), shape_out_rgb).astype(np.float32)
                composite += rgb_out
            # renormalize
            composite /= composite.max()
            # return as cyx for pngwriter
            return composite.transpose((0, 2, 1))
        else:
            image = image.transpose((1, 0, 2, 3))
            shape_out_rgb = new_size

            num_noise_floor_bins = 16
            composite = np.zeros(shape_out_rgb)
            channel_indices = self.channel_indices
            rgb_image = image[:, 0].astype('float')
            for i in channel_indices:
                # subtract out the noise floor.
                immin = image[i].min()
                immax = image[i].max()
                hi, bin_edges = np.histogram(image[i], bins=num_noise_floor_bins, range=(max(1, immin), immax))
                # index of tallest peak in histogram
                peakind = np.argmax(hi)
                # subtract this out
                thumb = image[i].astype(np.float32)
                thumb -= bin_edges[peakind]
                # don't go negative
                thumb[thumb < 0] = 0
                # renormalize
                thmax = thumb.max()
                thumb /= thmax

                imdbl = np.asarray(thumb).astype('double')
                im_proj = create_projection(imdbl, 0, 'slice', slice_index=int(thumb.shape[0] // 2))

                rgb_image[i] = im_proj

            for i in range(len(channel_indices)):
                # turn into RGB
                rgb_out = np.expand_dims(rgb_image[i], 2)
                rgb_out = np.repeat(rgb_out, 3, 2).astype('float')

                # inject color.  careful of type mismatches.
                rgb_out *= self.colors[i]

                rgb_out /= np.max(rgb_out)

                rgb_out = resize_cyx_image(rgb_out.transpose((2, 1, 0)), shape_out_rgb)
                composite += rgb_out

            # returns a CYX array for the pngwriter
            return composite.transpose((0, 2, 1))

    def _get_output_shape(self, im_size):
        """
        This method will take in a 3D ZYX shape and return a 3D XYC of the final thumbnail

        :param im_size: 3D ZYX shape of original image
        :return: CYX dims for a resized thumbnail where the maximum X or Y dimension is the one specified in the constructor.
        """
        # size down to this edge size, maintaining aspect ratio.
        max_edge = self.size
        # keep same number of z slices.
        shape_out = np.hstack((im_size[0],
                               max_edge if im_size[1] > im_size[2] else max_edge * (float(im_size[1]) / im_size[2]),
                               max_edge if im_size[1] < im_size[2] else max_edge * (float(im_size[2]) / im_size[1])
                               ))
        return 4 if not self.old_alg else 3, int(np.ceil(shape_out[2])), int(np.ceil(shape_out[1]))

    def _layer_projections(self, projection_array, mask_array):
        """
        This method will take in a list of 2D XY projections and layer them according to the method specified in the constructor

        :param projection_array: list of 2D XY projections (for each channel of a cell image)
        :return: single 3D XYC image where C is RGB values for each pixel
        """
        # array cannot be empty or have more channels than the color array
        assert projection_array
        assert len(projection_array) == len(self.colors)
        layered_image = np.zeros((4, projection_array[0].shape[1], projection_array[0].shape[0]))

        for i in range(len(projection_array)):

            projection = projection_array[i]
            # normalize channel projection
            projection /= np.max(projection)
            assert projection.shape == projection_array[0].shape

            projection *= self.channel_multipliers[i]
            projection[projection > 1] = 1

            # 4 channels - rgba
            rgb_out = np.expand_dims(projection, 2)
            rgb_out = np.repeat(rgb_out, 4, 2).astype('float')
            # inject color.  careful of type mismatches.
            rgb_out = (rgb_out * (self.colors[i] + [1.0])).transpose((2, 1, 0))
            # since there is a projection for each channel, there will be a threshold for each projection.
            min_percent = self.channel_thresholds[i]
            lower_threshold, upper_threshold = get_thresholds(rgb_out, min_percent=min_percent)

            def superimpose(source_pixel, dest_pixel):
                pixel_weight = np.mean(source_pixel)
                if lower_threshold < pixel_weight < upper_threshold:
                    return source_pixel
                else:
                    return dest_pixel

            def alpha_blend(source_pixel, dest_pixel):
                pixel_weight = np.mean(source_pixel)
                if lower_threshold < pixel_weight < upper_threshold:
                    # this alpha value is based on the intensity of the pixel in the channel's original projection
                    alpha = projection[x, y]
                    # premultiplied alpha
                    return source_pixel + (1 - alpha) * dest_pixel
                else:
                    return dest_pixel

            layering_method = superimpose if self.layering == "superimpose" else alpha_blend

            for x in range(rgb_out.shape[2]):
                for y in range(rgb_out.shape[1]):
                    # these slicing methods in C channel are getting the rgb data *only* and ignoring the alpha values
                    src_px = rgb_out[0:3, y, x]
                    dest_px = layered_image[0:3, y, x]
                    layered_image[0:3, y, x] = layering_method(source_pixel=src_px, dest_pixel=dest_px)
                    # if mask_array has elements and the pixel is 0
                    if mask_array and mask_array[i][x, y] == 0.0:
                        layered_image[3, y, x] = 0.0
                    else:
                        layered_image[3, y, x] = 1.0

        return layered_image

    def make_thumbnail(self, image, apply_cell_mask=False):
        """
        This method is the primary interface with the ThumbnailGenerator. It can be used many times with different
        images in order to save the configuration that was specified at the beginning of the generator.

        :param image: ZCYX image that is the source of the thumbnail
        :param apply_cell_mask: boolean value that designates whether the image is a fullfield or segmented cell
                                False -> fullfield, True -> segmented cell
        :return: a CYX image, scaled down to the size designated in the constructor
        """

        image = image.astype(np.float32)
        # check to make sure there are 6 or more channels
        assert image.shape[1] >= 6
        assert image.shape[2] > 1 and image.shape[3] > 1
        assert self.mask_channel_index <= image.shape[1]
        assert max(self.channel_indices) <= image.shape[1] - 1

        im_size = np.array(image[:, 0].shape)
        assert len(im_size) == 3
        shape_out_rgb = self._get_output_shape(im_size)

        if self.old_alg:
            return self._old_algorithm(image, shape_out_rgb, apply_cell_mask=apply_cell_mask)

        if apply_cell_mask:
            for i in range(len(self.channel_indices)):
                image[:, i] = np.multiply(image[:, i], image[:, self.mask_channel_index] > 0)

        num_noise_floor_bins = 256
        projection_array = []
        mask_array = []
        projection_type = self.projection
        for i in self.channel_indices:
            # don't use max projections on the fullfield images... they get too messy
            if not apply_cell_mask:
                projection_type = 'slice'
            # subtract out the noise floor.
            image /= np.max(image)
            thumb = subtract_noise_floor(image[:, i], bins=num_noise_floor_bins)
            thumb = np.asarray(thumb).astype('double')
            im_proj = create_projection(thumb, 0, projection_type, slice_index=int(thumb.shape[0] // 2),
                                        sections=self.proj_sections)
            if apply_cell_mask:
                mask_proj = create_projection(image[:, self.mask_channel_index], 0, method="max")
                mask_array.append(mask_proj)
            projection_array.append(im_proj)

        layered_image = self._layer_projections(projection_array, mask_array)
        comp = resize_cyx_image(layered_image, shape_out_rgb)
        comp /= np.max(comp)
        comp[comp < 0] = 0
        # returns a CYX array for the png writer
        return comp
