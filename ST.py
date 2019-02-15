"""
This script is the main function of the Shearlet Transformation for X-Ray image enhancement.
Coding = utf-8
Copyright (C) 2019 * Ltd. All rights reserved.
Editor      : PyCharm
File name   : utils.py
Author      : Yu Sun
Date modified: 2019-2-14 15:46:00
"""

import time
import os
import numpy as np
import utils_st as utils
import logging
import warnings
import matplotlib.pyplot as plt
from meyerShearlet import meyeraux, kutyniokaux, gaussian
from support import scalesShearsAndSpectra, allinone, enhance_Psi

@utils.log('DEBUG')
@utils.function_caller
def Shearlet_one_part(Psi=None, numOfScales=3, border_width=160, enhance_factors=(1.05, 3.5)):
    assert isinstance(Psi, np.ndarray), 'Expect Psi to be np.ndarray, but provided {}'.format(type(Psi))
    assert (0 < numOfScales < 5), 'Expect numOfScale belongs to [1, 4], but provided {}'.format(numOfScales)
    assert isinstance(border_width, int), 'Expect border_width to be int, but provided {}'.format(type(border_width))
    assert len(enhance_factors) == 2, 'Expect the length of enhance_factor to be 2, but provided {}'\
        .format(len(enhance_factors))
    assert isinstance(enhance_factors[0], (int, float)), 'Expect the type of each element of enhance_factors ' \
                                                         'to be int or float, but ' \
                                                         'provided {}'.format(type(enhance_factors[0]))

    @utils.log('DEBUG')
    @utils.batch_caller
    def Shearlet_one_part_operator(float_image):

        float_image_padded = np.pad(float_image, ((border_width, border_width), (border_width, border_width)), mode='reflect')

        img_shape = float_image_padded.shape
        # print('ROI shape:{}'.format(img_shape))

        # the shearlet transform only support that both the height and the width of the image to be even.
        part_image = float_image_padded[:img_shape[0] // 2 * 2, :img_shape[1] // 2 * 2]

        result = allinone(part_image, Psi, numOfScales, enhance_factor=enhance_factors)

        roi_image_max = np.max(float_image)
        result[result > roi_image_max] = roi_image_max

        result = result[border_width: -border_width, border_width: -border_width]

        return result
    return Shearlet_one_part_operator


@utils.log('DEBUG')
@utils.function_caller
def Fast_Shearlet_Transformation_Enhancement(numOfScales=3, border_width=160, aux_function=meyeraux,
                                             enhance_factors=(1.05, 3.5), image_shape=(2804, 2292)):

    assert isinstance(numOfScales, int), 'Expect numOfScale to be int, but provided {}'.format(type(numOfScales))
    if numOfScales >= 5:
        warnings.warn("'numOfScale (= {})' is too large, it may trigger MemoryError".format(numOfScales))
    assert numOfScales > 0, 'Expect numOfScale to be greater than 0, but provided {}'.format(numOfScales)
    assert isinstance(border_width, int), 'Expect border_width to be int, but provided {}'.format(type(border_width))
    assert aux_function in (meyeraux, kutyniokaux, gaussian), 'Expect aux_function to be one of "meyeraux, ' \
                                                                        'kutyniokaux, gaussian", but provided {}' \
                                                                        ''.format(aux_function.__name__)
    assert len(enhance_factors) == 2, 'Expect the length of enhance_factor to be 2, but provided {}' \
        .format(len(enhance_factors))
    assert isinstance(enhance_factors[0], (int, float)), 'Expect the type of each element of enhance_factors ' \
                                                         'to be int or float, but ' \
                                                         'provided {}'.format(type(enhance_factors[0]))
    Psi = 0
    h_size, v_size = image_shape


    try:
        Psi = np.load('./Psi_file/Psi_compact_file/Psi_scale{}_border{}_{}_low{}_high{}.npy'
                      ''.format(numOfScales, border_width, aux_function.__name__,
                                enhance_factors[0], enhance_factors[1]))
    except FileNotFoundError:
        psi_time_start = time.time()

        warnings.warn('Did not find ./Psi_file/Psi_compact_file/Psi_scale{}_border{}_{}_low{}_high{}.npy, start '
                      'calculating it...'.format(numOfScales, border_width, aux_function.__name__, enhance_factors[0],
                                                 enhance_factors[1]))
        try:
            Psi_fromfile = np.load('./Psi_file/Psi_scale{}_border{}_{}.npy'.format(numOfScales, border_width,
                                                                                   aux_function.__name__))
            Psi = enhance_Psi(Psi=Psi_fromfile, enhance_factors=enhance_factors)
            Psi_filename = './Psi_file/Psi_compact_file/Psi_scale{}_border{}_{}_low{}_high{}.npy' \
                           ''.format(numOfScales, border_width, aux_function.__name__,
                                     enhance_factors[0], enhance_factors[1])
            if not os.path.exists('./Psi_file/Psi_compact_file/'):
                os.makedirs('./Psi_file/Psi_compact_file/')

            np.save(Psi_filename, Psi)
            print('Time consumed for calculating Psi array: {}s'.format(time.time() - psi_time_start))
        except FileNotFoundError:
            warnings.warn('Did not find ./Psi_file/Psi_scale{}_border{}_{}.npy, start calculating it...'
                          .format(numOfScales, border_width, aux_function.__name__))

            Psi_from_calculating = scalesShearsAndSpectra((v_size + border_width * 2, h_size + border_width * 2),
                                                          numOfScales=numOfScales, shearletArg=aux_function)

            Psi_filename = './Psi_file/Psi_compact_file/Psi_scale{}_border{}_{}.npy'.format(numOfScales, border_width,
                                                                                            aux_function.__name__)

            if not os.path.exists('./Psi_file/'):
                os.makedirs('./Psi_file/')

            np.save(Psi_filename, Psi_from_calculating)

            Psi = enhance_Psi(Psi=Psi_from_calculating, enhance_factors=enhance_factors)
            Psi_filename = './Psi_file/Psi_scale{}_border{}_{}_low{}_high{}.npy' \
                           ''.format(numOfScales, border_width, aux_function.__name__,
                                     enhance_factors[0], enhance_factors[1])
            if not os.path.exists('./Psi_file/Psi_compact_file/'):
                os.makedirs('./Psi_file/Psi_compact_file/')

            np.save(Psi_filename, Psi)
            print('Time consumed for calculating Psi array: {}s'.format(time.time() - psi_time_start))

    @utils.log('DEBUG')
    @utils.batch_caller
    def Fast_Shearlet_Transformation_Enhancement_operator(roi_image_uint16):

        float_image = roi_image_uint16 / 65535.0
        float_image = float_image[:v_size, : h_size]

        ################################################################################################################
        # the original Shearlet_one_part() function
        ################################################################################################################

        float_image_padded = np.pad(float_image, ((border_width, border_width), (border_width, border_width)),
                                    mode='reflect')

        img_shape = float_image_padded.shape
        # print('ROI shape:{}'.format(img_shape))

        # the shearlet transform only support that both the height and the width of the image to be even.
        part_image = float_image_padded[:img_shape[0] // 2 * 2, :img_shape[1] // 2 * 2]

        result = allinone(part_image, Psi, numOfScales, enhance_factor=enhance_factors)

        roi_image_max = np.max(float_image)
        result[result > roi_image_max] = roi_image_max

        result = result[border_width: -border_width, border_width: -border_width]
        ################################################################################################################
        # result = Shearlet_one_part(roi_image[:v_size, : h_size], Psi=Psi, numOfScales=numOfScales,
        #                            border_width=border_width, enhance_factors=enhance_factors)

        return result
    return Fast_Shearlet_Transformation_Enhancement_operator


def myappend(one_element, base_data_array):
    result = []
    if isinstance(one_element, list):
        for base_element in base_data_array:
            result.append(one_element + [base_element])
    else:
        for base_element in base_data_array:
            result.append([one_element, base_element])
    return result


def loopinloop(base_data_array, loop_numbers):
    ini_array = [[element] for element in base_data_array]
    while loop_numbers > 1:
        temp_result = []
        for one_data in ini_array:
            temp_data = myappend(one_data, base_data_array)
            for one_temp_data in temp_data:
                temp_result.append(one_temp_data)

        ini_array = temp_result

        loop_numbers -= 1

    return ini_array


def generate_parameters():
    numOfScales = [3, 4]
    border_widths = [50]
    aux_functions = [meyeraux, kutyniokaux, gaussian]
    #enhance_factors_low = [0.9, 1, 1.05, 1.1, 1.2, 1.5]
    enhance_factors_low = [1.05, 1.15, 1.5]
    #enhance_factors_base = [0.9, 1.05, 1.1, 1.25, 1.5, 3.0, 5.0]
    enhance_factors_base = [3.0, 1.25, 1.05, 0.9, 0.1]
    enhance_factors_high_list = [loopinloop(enhance_factors_base, numOfScale) for numOfScale in numOfScales]

    parameters = [[numOfScales[i], border_width, aux_function, enhance_factor_low, enhance_factor_high]
                  for i in range(len(numOfScales)) for border_width in border_widths for aux_function in aux_functions
                  for enhance_factor_low in enhance_factors_low
                  for enhance_factor_high in enhance_factors_high_list[i]]

    return parameters

def is_useful_file(filename):
    if filename[-7:] == 'Pre.raw':
        return True
    else:
        return False


def main():
    #logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    datadir = '/home/yu/PycharmProjects/dataset/xiaozhi_cloud/raw_data/'
    file_list = os.listdir(datadir)
    batch_size = 20
    test_images_amount = 772

    filenames = list(map(lambda x, y: x+y, [datadir] * len(file_list), file_list))
    useful_file_collection = list(filter(is_useful_file, filenames))

    # Only enhance 1/5 of files for fine tune the parameters
    useful_file = useful_file_collection[::(len(useful_file_collection) // test_images_amount)]

    if len(useful_file) > test_images_amount:
        useful_file = useful_file[:test_images_amount]

    parameter_list = generate_parameters()

    print("Start enhancing by Shearlet Transformation algorithm ...")
    start_time = time.time()

    parameter_index = 0
    for parameter in parameter_list:
        print('=======================================================================================================')
        print('Processing {}/{} parameters...'.format(parameter_index, len(parameter_list)))
        parameter_start_time = time.time()

        numOfScales, border_width, aux_function, enhance_factors_low, enhance_factors_high = parameter

        # =================================================================================================
        # remove the following 6 lines when fine-tune the parameters.
        numOfScales = 4
        border_width = 50
        aux_function = meyeraux
        enhance_factors_low = 1.15
        enhance_factors_high = (1.25, 3.0, 3.0, 1.25)
        parameter = numOfScales, border_width, aux_function, enhance_factors_low, enhance_factors_high
        # =================================================================================================

        if parameter_index >= 1:
            parameter_index += 1
            continue

        for i in range(0, len(useful_file), batch_size):

            batch_start = time.time()
            batch_files = 0
            if len(useful_file) - i < batch_size:
                batch_files = useful_file[i:]
            else:
                batch_files = useful_file[i: i + batch_size]

            raw_image = utils.open_image(batch_files)

            hight1, hight2, width1, width2 = utils.find_roi_coordinates(raw_image)

            pre_processed = utils.pre_processing(raw_image)
            enhanced_image = Fast_Shearlet_Transformation_Enhancement(pre_processed, numOfScales=numOfScales,
                                                                      border_width=border_width,
                                                                      aux_function=aux_function,
                                                                      enhance_factors=(enhance_factors_low,
                                                                                       enhance_factors_high))
            enhanced_image_roi = utils.get_roi_image(enhanced_image, hight1, hight2, width1, width2)

            enhanced_image_roi = utils.boundary_processing(enhanced_image_roi)

            enhanced_image_post = utils.post_processing(enhanced_image_roi)

            utils.save_image(enhanced_image_post, batch_files,
                             output_dir='../../dataset/xiaozhi_cloud/enhanced_image/Shearlet_Transformation/',
                             output_type='jpg', parameters=parameter)
            print('Finished {}/{} batch , time elapsed: {}s.'.format(i // batch_size, len(useful_file) // batch_size -
                                                                     1, time.time() - batch_start))
        os.remove('./Psi_file/Psi_compact_file/Psi_scale{}_border{}_{}_low{}_high{}.npy'
                  .format(numOfScales, border_width, aux_function.__name__, enhance_factors_low, enhance_factors_high))
        print('Finished {}/{} parameters, time elapsed: {}s'.format(parameter_index, len(parameter_list),
                                                                    time.time() - parameter_start_time))

        parameter_index += 1
    print('Finished processing, time elapsed: {}s'.format(time.time() - start_time))


def main1():
    start_time = time.time()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    datadir = '/home/yu/PycharmProjects/dataset/xiaozhi_cloud/raw_data/'
    file_list = os.listdir(datadir)

    filenames = list(map(lambda x, y: x+y, [datadir] * len(file_list), file_list))
    useful_file = list(filter(is_useful_file, filenames))

    parameter_list = generate_parameters()
    parameter = parameter_list[0]
    filename = useful_file[0]
    filename = datadir + '1.2.276.0.7230010.3.0.3.5.1.10421207.3948650831Pre.raw'

    raw_image = utils.open_image(filename)

    hight1, hight2, width1, width2 = utils.find_roi_coordinates(raw_image)

    pre_processed = utils.pre_processing(raw_image)

    numOfScales, border_width, aux_function, enhance_factors_low, enhance_factors_high = parameter

    numOfScales = 3
    enhance_factors_low = 1.05
    aux_function = gaussian

    if numOfScales == 3:
        enhance_factors_high = (1.05, 1.05, 3)
    else:
        enhance_factors_high = (2.5, 2.5, 2.5, 1)
    # for debug, use different numberOfScales (3, 4) and different aux_function (meyer, kutyniokaux, gaussian)
    enhanced_image = Fast_Shearlet_Transformation_Enhancement(pre_processed, numOfScales=numOfScales,
                                                              border_width=border_width,
                                                              aux_function=aux_function,
                                                              enhance_factors=(enhance_factors_low,
                                                                               enhance_factors_high))
    enhanced_image_roi = utils.get_roi_image(enhanced_image, hight1, hight2, width1, width2)

    enhanced_image_roi = utils.boundary_processing(enhanced_image_roi)

    enhanced_image_post = utils.post_processing(enhanced_image_roi)

    print('Parameters: numOfScales = {}, border_width = {}, aux_function = {}, enhance_factors = {}'
          .format(numOfScales, border_width, aux_function.__name__, (enhance_factors_low, enhance_factors_high)))

    print('Finished enhancing image, time elapsed: {}s.'.format(time.time() - start_time))

    utils.save_image(enhanced_image_post, filename,
                     output_dir='./',
                     output_type='jpg', parameters=parameter)

    plt.imshow(enhanced_image_post, cmap='gray')
    plt.show()



if __name__ == '__main__':
    main()