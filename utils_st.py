#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : PyCharm
#   File name   : utils.py
#   Author      : Hume
#   Created date: 2019-1-16 09:04:00
#   Description : Useful tools for processing, analysis and metrics.
#
# ================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import logging
import copy
import colorsys
import random
import math

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


##########################################################################
# log function
##########################################################################

def log(level):
    """
    Log some useful information.
    :param level: string.
    :return: None
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(__name__)
            logger.info("[{level}]: call function \"{func}\"".format(level=level, func=func.__name__))
            return func(*args, **kwargs)
        return wrapper
    return decorator


def is_type(dtype):
    """
    :param dtype: type or tuple of type.
    :return: is_{type}
    """
    def is_a_type(obj):
        """
        :param obj: object.
        :return: True or Flase
        """
        return isinstance(obj, dtype)

    return is_a_type


##########################################################################
# Processing tools
##########################################################################


def function_caller(func):
    """
    Help call inner function.
    :param func: function.
    :return: inner function.
    """
    @functools.wraps(func)
    def inner_function(*args, **kwargs):
        return func(**kwargs)(*args)
    return inner_function


def batch_caller(func):
    """
    Batch call function.
    :param func: function.
    :return: list of result.
    """
    @functools.wraps(func)
    def wrapper(*args):
        is_type_list = is_type((list, tuple))
        return valid_output(list(map(func, *args))) if is_type_list(args[0]) else func(*args)
    return wrapper


def batch_caller_multi_output(func):
    """
    Batch call function.
    :param func: function.
    :return: list of result.
    """
    @functools.wraps(func)
    def wrapper(*args):
        is_type_list = is_type((list, tuple))
        return zip(*list(map(func, *args))) if is_type_list(args[0]) else func(*args)
    return wrapper


def valid_output(images):
    """
    Escape those elements which is None.
    :param images: list of ndarray, input images, [[height, width], ...].
    :return: list of ndarray, valid images, [[height, width], ...].
    """
    if isinstance(images, (list, tuple)):
        valid_indices = list(map(lambda x: x is not None, images))
        return list(np.asarray(images)[valid_indices])


@log('DEBUG')
@function_caller
def open_image(shape=(2292, 2804)):
    """
    Read image.
    :param shape: shape of image.
    :return: list of ndarray, [[height, width, channels], ...] or [[height, width], ...].
    """
    @log('DEBUG')
    @batch_caller
    def _open_image(file_path):
        """
        :param file_path: tuple or list of string, list of file path of the image data.
        :param file_path:
        :return:
        """

        # Ensure file_path is string and ended with ".raw"
        assert isinstance(file_path, str), "Excepted type of file_path is string, but got {}".format(type(file_path))

        assert file_path.endswith('.raw'), "Excepted file_path is ended with \".raw\""

        # (Height, Width) of image.
        try:
            raw_image = np.reshape(np.fromfile(file_path, dtype=np.uint16), shape)
            return raw_image
        except ValueError:
            logging.warning("Cannot reshape a array read from \"{name}\" into shape {shape}".
                            format(name=file_path, shape=shape))
            pass

    return _open_image


@log('DEBUG')
@function_caller
def save_image(output_dir='./', output_type='jpg', parameters=None):
    """
    Save image(s) to the given directory as .jpg or .raw file
    :param output_dir: string
    :param output_type: string, 'jpg' or 'raw'
    :param file_path: the full path of the file
    :param parameters: the parameters used to processed the images
    :return:
    """

    @log('DEBUG')
    @batch_caller
    def save_image_operator(image, file_path):

        assert parameters, 'No parameters are provided.'
        try:
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
        except:
            assert True, 'Can not make directory at {}'.format(output_dir)

        assert output_type in ('jpg', 'raw'), 'Expect the type of output file to be jpg or raw, but provided ' \
                                              '{}'.format(output_type)

        scales, border_width, aux_function, enhance_factor_low, enhance_factor_high = parameters
        folder = 'St_scale{}_{}_{}_low{}_high{}/'.format(scales, border_width, aux_function.__name__,
                                                         enhance_factor_low, enhance_factor_high)

        filename = file_path.split('/')[-1]
        if not os.path.exists(output_dir + folder):
            os.mkdir(output_dir + folder)
        output_file_path = output_dir + folder + filename.replace('.raw', '.' + output_type)

        cv2.imwrite(output_file_path, image)
    return save_image_operator


@log('DEBUG')
@function_caller
def pre_processing(gamma=2.2):
    """
    Pre process the image data.
    :param gamma: modulus of gamma correction.
    :return: gamma corrector.
    """
    assert isinstance(gamma, (float, int)), "Excepted type of gamma is float or int, but got {type}".\
        format(type=type(gamma))

    @log('DEBUG')
    @batch_caller
    def gamma_correction(image):
        """
        Perform gamma correction on origin images.
        :param image: ndarray, origin image,
            [height, width] or [height, width, channels].
        :return: pre-processed image, ndarray,
            [height, width] or [height, width, channels].
        """
        assert isinstance(image, np.ndarray), "Excepted type of image is numpy.ndarray, but got {type}".\
            format(type=type(image))

        if len(image.shape) == 3:
            img = []
            for i, single_image in enumerate(np.squeeze(np.split(image, 3, axis=-1))):
                min_value = np.min(single_image)
                max_value = np.max(single_image)
                single_image = ((((single_image - min_value) / (max_value - min_value)) ** (
                        1. / gamma)) + min_value) * (max_value - min_value)
                img.append(single_image)
            return np.concatenate(np.expand_dims(img, axis=-1), axis=-1)

        else:
            min_value = np.min(image)
            max_value = np.max(image)

            # Gamma correction.
            image = ((image / (max_value - min_value)) ** (1. / gamma)) * (max_value - min_value)

            return image

    return gamma_correction


@log('DEBUG')
@batch_caller
def cut(origin_image, enhanced_image):
    """
    Cut those intensity of enhanced image pixel whose intensity exceed
        the origin maximal intensity for better preserving the detail.
    :param origin_image: ndarray, origin image, [height, width].
    :param enhanced_image: ndarray, enhanced image, [height, width].
    :return: cutted-processed image, ndarrary, [height, width].
    """
    is_ndarray = is_type(np.ndarray)
    assert all(list(map(is_ndarray, [origin_image, enhanced_image]))), "Excepted type of all origin_images are " \
                                                                       "numpy.ndarray, but got something else."

    max_value = np.max(origin_image)
    _enhanced_image = copy.deepcopy(enhanced_image)
    enhanced_image[_enhanced_image > max_value] = max_value
    return enhanced_image


@log('DEBUG')
@function_caller
def complementary(dtype=None):
    """
    Get complementary function.
    :param dtype: dtype, must be specified when image has been transformed to float.
    :return: complementary function.
    """
    assert dtype is None or dtype in (np.uint8, np.uint16), "Excepted dtype is either None, np.uint8 or" \
                                                            "np.uint16, but got {type}".\
        format(type=type(dtype))

    @log('DEBUG')
    @batch_caller
    def _complementary(img):
        """
        Get complementary image.
        :param img: dict, input image and the , [height, width].
        :return: ndarray, complementary image, [height, width].
        """
        assert img.dtype in [np.uint8, np.uint16, np.float32, np.float64], \
            "Excepted dtype must be in [np.uint8, np.uint16, np.float32, np.float64], " \
            "but got {}".format(img.dtype)

        def __complementary(constant):
            return constant - img

        com_image = __complementary(255) if img.dtype == np.uint8 else __complementary(65535) if \
            img.dtype in [np.uint8, np.uint16] \
            else __complementary(255.) if dtype == np.uint8 else __complementary(65535.)

        return com_image

    return _complementary


@log('DEBUG')
@function_caller
def boundary_processing(dtype=None):
    """
    Get boundary processor.
    :param dtype: dtype, must be specified when image has been transformed to float.
    :return: boundary processor function.
    """
    assert dtype is None or isinstance(dtype, (np.uint8, np.uint16)), "Excepted dtype is either None, np.uint8 or" \
                                                                      "np.uint16, but got {type}". \
        format(type=type(dtype))

    @log('DEBUG')
    @batch_caller
    def boundary_processor(img):
        """
        Limit the intensity of each pixel in a certain range which is determined by dtype.
        :param img: ndarray, input image, [height, width].
        :return: ndarray, bounded image, [height, width].
        """
        if img.dtype in [np.uint8, np.uint16]:
            if img.dtype == np.uint8:
                img[img > 255] = 255
                img[img < 0] = 0
            else:
                img[img > 65535] = 65535
                img[img < 0] = 0

        else:
            assert img.dtype in [np.float32, np.float64], \
                "Excepted dtype must be in [np.uint8, np.uint16, np.float32, np.float64], " \
                "but got {}".format(img.dtype)
            if dtype == np.uint8:
                img[img > 255.] = 255.
                img[img < 0.] = 0.
            else:
                img[img > 65535.] = 65535.
                img[img < 0.] = 0.

        return img

    return boundary_processor


@log('DEBUG')
@function_caller
def unsharp_mask(**unsharp_mask_modulus):
    """
    Get unsharp mask function
    :param unsharp_mask_modulus: dict, modulus used by unsharp mask.
    :return: list of ndarray, unsharped_images, [[height, width], ...].
    """
    @log('DEBUG')
    @batch_caller
    def _unsharpe_mask(image):
        """
        Unsharp mask algorithm.
        :param image: ndarray, input images, [height, width].
        :return:
        """
        assert isinstance(image, np.ndarray), "Excepted type of all images is numpy.ndarray, but got {}".\
            format(type(image))

        sigma = unsharp_mask_modulus['sigma'] or 1
        alpha = unsharp_mask_modulus['alpha'] or 1

        filter_size = 1 + 2 * math.ceil(3 * sigma)
        stride = (filter_size - 1) / 2
        x = np.expand_dims(np.linspace(start=-stride, stop=stride, num=filter_size), axis=-1)
        y = np.transpose(x, [1, 0])

        gx = np.exp(-(x ** 2) / (2 * sigma * sigma))
        gy = np.transpose(gx, [1, 0])

        # Canny filter on x and y direction
        canny_filter_dx = functools.partial(cv2.filter2D, ddepth=-1, kernel=x*gx)
        canny_filter_dy = functools.partial(cv2.filter2D, ddepth=-1, kernel=y*gy)
        canny_filter_x = functools.partial(cv2.filter2D, ddepth=-1, kernel=gx)
        canny_filter_y = functools.partial(cv2.filter2D, ddepth=-1, kernel=gy)

        image_x = canny_filter_dx(image)
        image_x = canny_filter_x(image_x)
        image_y = canny_filter_dy(image)
        image_y = canny_filter_y(image_y)

        mag = np.sqrt(image_x ** 2 + image_y ** 2).astype(np.float32)

        unsharped_image = image + alpha * mag

        return boundary_processing(unsharped_image, dtype=np.uint8)

    return _unsharpe_mask


@log('DEBUG')
@batch_caller
def set_min_max_window(image):
    """
    Set min-max-window on image data.
    :param image: input image data.
    :return: output image.
    """
    assert isinstance(image, np.ndarray), "Excepted type of image is numpy.ndarray, but got {}".\
        format(type(image))

    min_value = np.min(image)
    max_value = np.max(image)
    return 255. * ((image - min_value) / (max_value - min_value))


@log('DEBUG')
@function_caller
def post_processing(**post_modulus):
    """
    Get post-processor.
    :param post_modulus: unsharp mask modulus.
    :return: post processor.
    """

    @log('DEBUG')
    @batch_caller
    def post_processor(enhanced_image):
        """
        Post-process the image data.
        :param enhanced_image: ndarray, enhanced images, [height, width].
        :return: ndarray, post-processed images, [height, width].
        """
        assert isinstance(enhanced_image, np.ndarray), \
            "Excepted type of all images is numpy.ndarray, but got {}".format(type(enhanced_image))

        # Set min-max windows.
        #windowed_image = set_min_max_window(enhanced_image)
        windowed_image = 255. * ((enhanced_image - np.min(enhanced_image)) /
                                 (np.max(enhanced_image) - np.min(enhanced_image)))

        # Get complementary function.
        if post_modulus and post_modulus['using_unsharp_mask']:

            # Perform unsharp mask algorithm on enhanced images.
            post_processed_image = unsharp_mask(complementary(windowed_image))

        else:
            # Obtain post-processed images.
            post_processed_image = complementary(windowed_image)

        post_processed_image = post_processed_image.astype(np.uint8)

        return post_processed_image

    return post_processor


@log('DEBUG')
@function_caller
def set_min_max_window_with_gamma_correction(gamma=2.2):
    """
    Get min-max window function
    :param gamma: gamma correction factor.
    :return: min-max window function.
    """
    assert isinstance(gamma, (int, float)), "Excepted type of gamma is int or float, but got {}".\
        format(type(gamma))

    @log('DEBUG')
    @batch_caller
    def _set_min_max_window_with_gamma_correction(img):
        """
        Set min-max window with gamma correction.
        :param img: ndarray, input image, [height, width].
        :return: ndarray, output image, [height, width].
        """
        assert isinstance(img, np.ndarray), "Excepted type of gamma is numpy.ndarray, but got {}". \
            format(type(img))

        max_value = np.max(img)
        min_value = np.min(img)
        return 255. * ((img / (max_value - min_value)) ** (1.0 / gamma))

    return set_min_max_window_with_gamma_correction


@log('DEBUG')
@batch_caller_multi_output
def find_roi(image):
    """
    Find the region of interest of input images.
    :param image: ndarray, input images, [origin_height, origin_width].
    :return: ndarray, roi image, [interest_height, interest_width].
    """
    assert isinstance(image, np.ndarray), \
        "Excepted type of all images is numpy.ndarray, but got {}".format(type(image))

    # Get the origin height, width
    height, width = image.shape

    # Otsu algorithm accept ndarray with dtype=np.uint8 as input only.
    img_ = copy.deepcopy(image / 65535. * 255.).astype(np.uint8)

    # Smooth the image for better performance.
    blur = cv2.GaussianBlur(img_, (5, 5), 0)

    # Otsu binary segmentation.
    ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Get the left top and down right coordinates.
    horizontal_indicies = np.where(np.any(th, axis=0))[0]
    vertical_indicies = np.where(np.any(th, axis=1))[0]
    x1, x2 = horizontal_indicies[[0, -1]]
    y1, y2 = vertical_indicies[[0, -1]]

    if x1 == 0 or x2 == width:
        x1, x2 = 0, width
    if y1 == 0 or y2 == height:
        y1, y2 = 0, height
    if (width / 2 - x1) > (x2 - width / 2) and abs((width / 2 - x1) / (x2 - width / 2) - 1) > 1:
        x2 = width - x1
    elif (width / 2 - x1) < (x2 - width / 2) and abs((x2 - width / 2) / (width / 2 - x1) - 1) > 1:
        x1 = width - x2
    if (height / 2 - y1) > (y2 - height / 2) and abs((height / 2 - y1) / (y2 - height / 2) - 1) > 1:
        y2 = height - y1
    elif (height / 2 - y1) < (y2 - height / 2) and abs((y2 - height / 2) / (height / 2 - y1) - 1) > 1:
        y1 = height - y2
    return image[y1: y2 + (
        1 if y2 < height else 0), x1: x2 + (
        1 if x2 < width else 0)], y2 - y1 + (
        1 if y2 < height else 0), x2 - x1 + (
        1 if x2 < width else 0)


@log('DEBUG')
@batch_caller_multi_output
def find_roi_coordinates(image):
    """
    Find the region of interest of input images.
    :param image: ndarray, input images, [origin_height, origin_width].
    :return: ndarray, roi image, [interest_height, interest_width].
    """
    assert isinstance(image, np.ndarray), \
        "Excepted type of all images is numpy.ndarray, but got {}".format(type(image))

    # Get the origin height, width
    height, width = image.shape

    # Otsu algorithm accept ndarray with dtype=np.uint8 as input only.
    img_ = copy.deepcopy(image / 65535. * 255.).astype(np.uint8)

    # Smooth the image for better performance.
    blur = cv2.GaussianBlur(img_, (5, 5), 0)

    # Otsu binary segmentation.
    ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Get the left top and down right coordinates.
    horizontal_indicies = np.where(np.any(th, axis=0))[0]
    vertical_indicies = np.where(np.any(th, axis=1))[0]
    x1, x2 = horizontal_indicies[[0, -1]]
    y1, y2 = vertical_indicies[[0, -1]]

    if x1 == 0 or x2 == width:
        x1, x2 = 0, width
    if y1 == 0 or y2 == height:
        y1, y2 = 0, height
    if (width / 2 - x1) > (x2 - width / 2) and abs((width / 2 - x1) / (x2 - width / 2) - 1) > 1:
        x2 = width - x1
    elif (width / 2 - x1) < (x2 - width / 2) and abs((x2 - width / 2) / (width / 2 - x1) - 1) > 1:
        x1 = width - x2
    if (height / 2 - y1) > (y2 - height / 2) and abs((height / 2 - y1) / (y2 - height / 2) - 1) > 1:
        y2 = height - y1
    elif (height / 2 - y1) < (y2 - height / 2) and abs((y2 - height / 2) / (height / 2 - y1) - 1) > 1:
        y1 = height - y2
    return y1, y2+1, x1, x2+1


@log('DEBUG')
@batch_caller
def get_roi_image(original_image, h1, h2, w1, w2):
    """
    Segment the ROI of the image from the given image using h1, h2, w1, w2
    :param original_image: np.ndarray, the image to be segmented
    :param h1: uint
    :param h2: uint
    :param w1: uint
    :param w2: uint
    :return: the ROI of the input image
    """
    return original_image[h1: h2, w1: w2]


@log('DEBUG')
def prime_factor_decomposition(num):
    """
    Prime factor decomposition of a number.
    :param num: int, input number.
    :return: list, all prime factors of input number.
    """
    primes = [2]
    for i in range(3, num):
        if i * i > num:
            break
        flag = True
        for j in primes:
            if i % j == 0:
                flag = False
                break
            if j * j > i:
                break
        if flag:
            primes.append(i)
    factor = []
    for i in primes:
        if i * i > num:
            break
        while num % i == 0:
            factor.append(i)
            num = num // i
    if num != 1:
        factor.append(num)
    return factor


@log('DEBUG')
def cnr_based_noise_reduce(image, cnr):
    """
    CNR based noise reduction.
    :param image: input image.
    :param cnr: contrast-to-noise ratio of the image.
    :return:
    """
    noise = cnr <= 3
    subtle_app = (cnr > 3) == (cnr <= 9)
    excess = cnr > 9
    image[noise] = image[noise] * 0.6
    image[subtle_app] = image[subtle_app] * (0.1 * cnr[subtle_app] + 0.3)
    image[excess] = image[excess] * 1.2
    return image

##########################################################################
# Visualize tools
##########################################################################


@log('DEBUG')
def density_along_center_line(image):
    """
    Density along center line.
    :param image: input image.
    :return:
    """
    height, width = image.shape
    center_line = image[height // 2, ...]
    x = list(range(width))
    return x, center_line


@log('DEBUG')
def visualize_center_line_density(images, titles=None):
    """
    Visualize center line density.
    :param images: images to be shown.
    :param titles: titles of input images.
    :return:
    """
    images, titles, num_images, num_titles = deal_with_inputs(images, titles)
    colors = random_colors(num_images)

    if num_images % 3 != 0:
        num_rows = num_images // 3 + 1
    else:
        num_rows = num_images // 3

    if num_rows == 1:
        num_cols = num_images
    else:
        num_cols = 3

    fig = plt.figure()

    for i, image in enumerate(images):
        if len(image.shape) == 3:
            r, g, b = np.split(image, 3, axis=-1)
            y_image = np.squeeze(0.3 * r + 0.59 * g + 0.11 * b)

        else:
            y_image = image

        ax = fig.add_subplot(num_rows + 1, num_cols, i + 1)
        if len(image.shape) == 3:
            ax.imshow(image, interpolation='nearest')
        else:
            ax.imshow(image, interpolation='nearest', cmap='gray')
        title = titles[i]
        if title:
            if isinstance(title, str):
                title = str(title)
            ax.set_title(title, fontsize=10)

        ax.set_yticks([])
        ax.set_xticks([])

        x, center_line = density_along_center_line(y_image)

        ax = fig.add_subplot(2, 1, 2)
        ax.plot(x, center_line, color=colors[i], label=title)
        ax.set_title('density along center line')
        ax.legend(loc='upper left')
        ax.set_xlabel('location')
        ax.set_ylabel('density')


@log('DEBUG')
def visualize_images(images, titles=None):
    """
    Show all images with respective titles.
    :param images:
    :param titles:
    :return:
    """
    images, titles, num_images, num_titles = deal_with_inputs(images, titles)

    if num_images % 3 != 0:
        num_rows = num_images // 3 + 1
    else:
        num_rows = num_images // 3

    if num_rows == 1:
        num_cols = num_images
    else:
        num_cols = 3

    fig = plt.figure()

    for i, image in enumerate(images):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        ax.imshow(image, 'gray')
        title = titles[i]
        if title:
            if isinstance(title, str):
                title = str(title)
            ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])


@log('DEBUG')
def show_histogram(images, titles=None, bins=255):
    """
    Show the histogram of images
    :param images: input images
    :param titles:
    :param bins:
    :return:
    """
    images, titles, num_images, num_titles = deal_with_inputs(images, titles)
    if isinstance(bins, tuple):
        bins = list(bins)

    if not isinstance(bins, list):
        bins = [bins]

    if not bins:
        bins = [255] * num_images

    num_bins = len(bins)
    if num_bins > num_images:
        bins = bins[:num_images]
    elif num_bins < num_images:
        bins += (num_images - num_bins) * [255]

    if num_images % 2 != 0:
        num_rows = num_images // 2 + 1
    else:
        num_rows = num_images // 2

    if num_rows == 1:
        num_cols = num_images
    else:
        num_cols = 2

    fig = plt.figure()

    ns = []
    binses = []

    for i, image in enumerate(images):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        n, _bins, _ = ax.hist(np.reshape(image, [-1]), bins=bins[i])
        ax.plot(_bins[1:], n, color='r')
        title = titles[i]
        if title:
            if isinstance(title, str):
                title = str(title)
            ax.set_title(title, fontsize=10)
        ns.append(n)
        binses.append(_bins[1:])
    return ns, binses


@log('DEBUG')
def deal_with_inputs(ipt1, ipt2):
    if isinstance(ipt1, tuple):
        ipt1 = list(ipt1)

    if not isinstance(ipt1, list):
        ipt1 = [ipt1]

    num_ipt1 = len(ipt1)

    if isinstance(ipt2, tuple):
        ipt2 = list(ipt2)

    if not isinstance(ipt2, list):
        ipt2 = [ipt2]

    if not ipt2:
        ipt2 = [None] * num_ipt1

    num_ipt2 = len(ipt2)

    if num_ipt2 > num_ipt1:
        ipt2 = ipt2[:num_ipt1]
    elif num_ipt2 < num_ipt1:
        ipt2 += (num_ipt1 - num_ipt2) * [None]

    return ipt1, ipt2, num_ipt1, num_ipt2


@log('DEBUG')
def random_colors(n, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / n, 1, brightness) for i in range(n)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

##########################################################################
# Metrics method
##########################################################################


@log('DEBUG')
def calc_entropy(images):
    """
    Calculate entropy of given images.
    :param images: list, input images.
    :return: list, list of entropy
    """
    origin_image = images[0]
    y_images = []
    if len(origin_image.shape) == 3:
        for image in images:
            r, g, b = np.split(image, 3, axis=2)
            y_images.append(np.squeeze(0.3 * r + 0.59 * g + 0.11 * b))
    else:
        y_images = images

    ns, binses = show_histogram(y_images)
    entropys = []
    for n, bins in zip(ns, binses):
        sum_n = np.sum(n)
        entropy = 0
        for i in n:
            entropy -= i / sum_n * math.log(i / sum_n + 1e-10)
        entropys.append(entropy)

    return entropys


@log('DEBUG')
def calc_mse_psnr(images, image_type='uint16'):
    """
    Calculate PSNR between each enhanced image and origin image.
    :param images: ndarray, enhanced image and origin image.
    :param image_type: string, could be uint16 or uint8.
    :return: list, list of psnrs
    """

    assert image_type in ['uint8', 'uint16'], "Image type can only be 'uint16' or 'uint8'"
    psnrs = []
    origin_image = images[0]
    if len(origin_image.shape) == 3:
        r, g, b = np.split(origin_image, 3, axis=2)
        y_origin = np.squeeze(0.3 * r + 0.59 * g + 0.11 * b)
        y_images = []
        for image in images[1:]:
            r, g, b = np.split(image, 3, axis=2)
            y_images.append(np.squeeze(0.3 * r + 0.59 * g + 0.11 * b))
    else:
        y_origin = origin_image
        y_images = images[1:]
    for y_image in y_images:
        mse = ((y_origin - y_image) ** 2).mean()
        if image_type == 'uint16':
            psnr = np.mean(20 * np.log10(65535. / np.sqrt(mse + 1e-10)))
        else:
            psnr = np.mean(20 * np.log10(255. / np.sqrt(mse + 1e-10)))
        psnrs.append(psnr)
    return psnrs


@log('DEBUG')
def calc_ssim(images, image_type='uint16'):
    """
    Calculate SSIM between each enhanced image and origin image.
    :param images: enhanced image and origin image.
    :param image_type: string, could be uint16 or uint8.
    :return: list, list of ssims
    """
    assert image_type in ['uint8', 'uint16'], "Image type can only be 'uint16' or 'uint8'"
    ssims = []
    origin_image = images[0]
    image_size = origin_image.shape[0] * origin_image.shape[1]
    avg_origin = origin_image.mean(keepdims=True)
    std_origin = origin_image.std(ddof=1)
    k1 = 0.01
    k2 = 0.03
    c1 = (k1 * 65535) ** 2
    c2 = (k2 * 65535) ** 2
    c3 = c2 / 2
    for image in images[1:]:
        avg_enhanced = image.mean(keepdims=True)
        std_enhanced = image.std(ddof=1)
        cov = ((origin_image - avg_origin) * (image - avg_enhanced)).mean() * image_size / (image_size - 1)
        if image_type == 'uint16':
            ssims.append(np.mean(
                (2 * avg_origin * avg_enhanced + c1) * 2 * (cov + c3) / (
                        avg_origin ** 2 + avg_enhanced ** 2 + c1
                ) / (std_origin ** 2 + std_enhanced ** 2 + c2)
            ))
    return ssims


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    origin_image = open_image('../raw_data/raw/1.2.276.0.7230010.3.0.3.5.1.10163059.109444229Pre.raw')
    origin_image = find_roi(origin_image)
    plt.imshow(origin_image[0], 'gray')
    plt.show()


if __name__ == '__main__':
    main()
