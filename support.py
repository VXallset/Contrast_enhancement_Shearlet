import numpy as np
import warnings
import cv2
import time

def Gamma_correction(low_frequency, alpha=0.5):
    """
    Adjust the coefficient of low frequency component using Gamma correction.
    :param low_frequency: the low frequency component of the image calculated with Shearlet transformation.
    :param alpha: adjustment factor with a range from 0 to 1, in addition, it lies between 0.4 and 0.6.
    :return: the result after correction.
    """
    x_max = np.max(low_frequency)
    x_min = np.min(low_frequency)

    phi_x = (low_frequency - x_min) / (x_max - x_min)
    f_x = np.pi * phi_x
    gamma_x = 1.0 + alpha * np.cos(f_x)
    output = x_max * phi_x ** (1.0 / gamma_x)
    return output


def nonuniform_correction(low_frequancy):
    #low_frequancy = low_frequancy[..., 0]
    shape_x, shape_y = low_frequancy.shape
    kernel1_size = np.int64(np.max([shape_x, shape_y]) / 150)
    kernel2_size = np.int64(np.max([shape_x, shape_y]) / 75)
    print('kernel1_size = {}, kernel2_size = {}'.format(kernel1_size, kernel2_size))

    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel1_size, kernel1_size))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel2_size, kernel2_size))
    SCl1 = cv2.morphologyEx(low_frequancy, cv2.MORPH_OPEN, kernel1)
    SCl2 = cv2.morphologyEx(low_frequancy, cv2.MORPH_OPEN, kernel2)

    SCcor = SCl2 / SCl1 * low_frequancy
    return SCcor


def linearly_enhancing_details(alphas, numOfScales=2, window_scale=3):
    """

    :param alphas: the high frequency component of the image calculated with Shearlet transformation.
                           i * j * (kl) array, where i, j are equal to the height and width of the original image, kl is
                           depend by the number of scales and the number of directions per scale. kl = 4 + 8 + 16 + ...
                           when there are 1, 2, 3... scales.
           numOfScales: the number of scales of high frequency coefficients.
    :return:
    """
    # the gain factor of each scale
    if numOfScales > 3 or numOfScales < 1:
        print('Number of Scale mast be in the set of {1, 2, 3} !!!')
    gain_weight = np.array([0.9, 1.0, 1.1, 1.2, 1.5, 3.0])
    lam = 1000
    # calculate the gain factor vector W
    w_array = []
    for i in range(1, numOfScales + 1):
        w_array = np.concatenate((w_array, gain_weight[i] * np.ones(2 ** (i + 1))))

    betas = alphas * w_array

    a_times_b = alphas * betas
    kernel = np.ones((window_scale, window_scale))
    numerator_ab_part = 1.0 / (window_scale**2) * cv2.filter2D(a_times_b, -1, kernel=kernel)
    mu0 = cv2.filter2D(alphas, -1, kernel=kernel)
    mu1 = cv2.filter2D(betas, -1, kernel=kernel)
    #mu1 = mu0 * w_array

    # fast method to calculate the local standard deviation (mean(img^2) - mean(img)^2)
    # instead of using the original method std = 1/n x sum(Xn - mean)
    sigma2 = cv2.filter2D(alphas**2, -1, kernel) - (cv2.filter2D(alphas, -1, kernel)) ** 2

    pm = (numerator_ab_part - mu0 * mu1) / (lam + sigma2)
    qm = mu1 - pm * mu0
    gammai = pm * alphas + qm

    return gammai


def normalization(image):
    """
    Normalizing the input image to [0.0, 1.0]
    :param image:
    :return:
    """
    return (image - np.min(image)) / (np.max(image) - np.min(image))


# ===========================================================================================
#                           Shearlet Transformation stuff
# ===========================================================================================



from meyerShearlet import (meyerShearletSpect, meyeraux, kutyniokaux, gaussian)


def _defaultNumberOfScales(l):
    numOfScales = int(np.floor(0.5 * np.log2(np.max(l))))
    if numOfScales < 1:
        raise ValueError('image too small!')
    return numOfScales


def scalesShearsAndSpectra(shape, numOfScales=None,
                           realCoefficients=True, maxScale='max',
                           shearletSpect=meyerShearletSpect,
                           shearletArg=kutyniokaux, realReal=True,
                           fftshift_spectra=True):
    """  Compute the shearlet spectra of a given shape and number of scales.

    The number of scales and a boolean indicating real or complex shearlets
    are optional parameters.

    Parameters
    ----------
    shape : array-like
        dimensions of the image
    numOfScales : int
        number of scales
    realCoefficients : bool
        Controls whether real or complex shearlets are generated.
    shearletSpect : string or handle
        shearlet spectrum
    shearletArg : ???
        further parameters for shearlet
    realReal : bool
        guarantee truly real shearlets
    maxScale : {'max', 'min'}, optional
        maximal or minimal finest scale

    Returns
    -------
    Psi : ndarray
        Shearlets in the Fourier domain.
    """
    if len(shape) != 2:
        raise ValueError("2D image dimensions required")

    if numOfScales is None:
        numOfScales = _defaultNumberOfScales(shape)

    # rectangular images
    if shape[1] != shape[0]:
        rectangular = True
    else:
        rectangular = False

    # for better symmetry each dimensions of the array should be odd
    shape = np.asarray(shape)
    shape_orig = shape.copy()
    shapem = np.mod(shape, 2) == 0  # True for even sized axes
    both_even = np.all(np.equal(shapem, False))
    both_odd = np.all(np.equal(shapem, True))
    shape[shapem] += 1

    if not realCoefficients:
        warnings.warn("Complex shearlet case may be buggy.  Doesn't "
                      "currently give perfect reconstruction.")

    if not (both_even or both_odd):
        # for some reason reconstruction is not exact in this case, so don't
        # allow it for now.
        raise ValueError("Mixture of odd and even axis sizes is currently "
                         "unsupported.")

    # create meshgrid
    # largest value where psi_1 is equal to 1
    maxScale = maxScale.lower()
    if maxScale == 'max':
        X = 2**(2 * (numOfScales - 1) + 1)
    elif maxScale == 'min':
        X = 2**(2 * (numOfScales - 1))
    else:
        raise ValueError('Wrong option for maxScale, must be "max" or "min"')

    xi_x_init = np.linspace(0, X, (shape[1] + 1) // 2)
    xi_x_init = np.concatenate((-xi_x_init[-1:0:-1], xi_x_init), axis=0)
    if rectangular:
        xi_y_init = np.linspace(0, X, (shape[0] + 1) // 2)
        xi_y_init = np.concatenate((-xi_y_init[-1:0:-1], xi_y_init), axis=0)
    else:
        xi_y_init = xi_x_init

    # create grid, from left to right, bottom to top
    [xi_x, xi_y] = np.meshgrid(xi_x_init, xi_y_init[::-1], indexing='xy')

    # cones
    C_hor = np.abs(xi_x) >= np.abs(xi_y)  # with diag
    C_ver = np.abs(xi_x) < np.abs(xi_y)

    # number of shears: |-2^j,...,0,...,2^j| = 2 * 2^j + 1
    # now: inner shears for both cones:
    # |-(2^j-1),...,0,...,2^j-1|
    # = 2 * (2^j - 1) + 1
    # = 2^(j+1) - 2 + 1 = 2^(j+1) - 1
    # outer scales: 2 ("one" for each cone)
    # shears for each scale: hor: 2^(j+1) - 1, ver: 2^(j+1) - 1, diag: 2
    #  -> hor + ver + diag = 2*(2^(j+1) - 1) +2 = 2^(j + 2)
    #  + 1 for low-pass
    shearsPerScale = 2**(np.arange(numOfScales) + 2)
    numOfAllShears = 1 + shearsPerScale.sum()

    # init
    Psi = np.zeros(tuple(shape) + (numOfAllShears, ))
    # frequency domain:
    # k  2^j 0 -2^j
    #
    #     4  3  2  -2^j
    #      \ | /
    #   (5)- x -1  0
    #      / | \
    #              2^j
    #
    #        [0:-1:-2^j][-2^j:1:2^j][2^j:-1:1] (not 0)
    #           hor          ver        hor
    #
    # start with shear -2^j (insert in index 2^j+1 (with transposed
    # added)) then continue with increasing scale. Save to index 2^j+1 +- k,
    # if + k save transposed. If shear 0 is reached save -k starting from
    # the end (thus modulo). For + k just continue.
    #
    # then in time domain:
    #
    #  2  1  8
    #   \ | /
    #  3- x -7
    #   / | \
    #  4  5  6
    #

    # lowpass
    Psi[:, :, 0] = shearletSpect(xi_x, xi_y, np.NaN, np.NaN, realCoefficients,
                                 shearletArg, scaling_only=True)

    # loop for each scale
    for j in range(numOfScales):
        # starting index
        idx = 2**j
        start_index = 1 + shearsPerScale[:j].sum()
        shift = 1
        for k in range(-2**j, 2**j + 1):
            # shearlet spectrum
            P_hor = shearletSpect(xi_x, xi_y, 2**(-2 * j), k * 2**(-j),
                                  realCoefficients, shearletArg)
            if rectangular:
                P_ver = shearletSpect(xi_y, xi_x, 2**(-2 * j), k * 2**(-j),
                                      realCoefficients, shearletArg)
            else:
                # the matrix is supposed to be mirrored at the counter
                # diagonal
                # P_ver = fliplr(flipud(P_hor'))
                P_ver = np.rot90(P_hor, 2).T  # TODO: np.conj here too?
            if not realCoefficients:
                # workaround to cover left-upper part
                P_ver = np.rot90(P_ver, 2)

            if k == -2**j:
                Psi[:, :, start_index + idx] = P_hor * C_hor + P_ver * C_ver
            elif k == 2**j:
                Psi_idx = start_index + idx + shift
                Psi[:, :, Psi_idx] = P_hor * C_hor + P_ver * C_ver
            else:
                new_pos = np.mod(idx + 1 - shift, shearsPerScale[j]) - 1
                if(new_pos == -1):
                    new_pos = shearsPerScale[j] - 1
                Psi[:, :, start_index + new_pos] = P_hor
                Psi[:, :, start_index + idx + shift] = P_ver

                # update shift
                shift += 1

    # generate output with size shape_orig
    Psi = Psi[:shape_orig[0], :shape_orig[1], :]

    # modify spectra at finest scales to obtain really real shearlets
    # the modification has only to be done for dimensions with even length
    if realCoefficients and realReal and (shapem[0] or shapem[1]):
        idx_finest_scale = (1 + np.sum(shearsPerScale[:-1]))
        scale_idx = idx_finest_scale + np.concatenate(
            (np.arange(1, (idx_finest_scale + 1) / 2 + 1),
             np.arange((idx_finest_scale + 1) / 2 + 2, shearsPerScale[-1])),
            axis=0)
        scale_idx = scale_idx.astype(np.int)
        if shapem[0]:  # even number of rows -> modify first row:
            idx = slice(1, shape_orig[1])
            Psi[0, idx, scale_idx] = 1 / np.sqrt(2) * (
                Psi[0, idx, scale_idx] +
                Psi[0, shape_orig[1] - 1:0:-1, scale_idx])
        if shapem[1]:  # even number of columns -> modify first column:
            idx = slice(1, shape_orig[0])
            Psi[idx, 0, scale_idx] = 1 / np.sqrt(2) * (
                Psi[idx, 0, scale_idx] +
                Psi[shape_orig[0] - 1:0:-1, 0, scale_idx])

    if fftshift_spectra:
        # Note: changed to ifftshift so roundtrip tests pass for odd sized
        # arrays
        Psi = np.fft.ifftshift(Psi, axes=(0, 1))

    # Add the following two lines to calculate the spectra of the Shearlet to comprass the size of the .npy file
    #Psi[..., 1] = np.sum(Psi[..., 1:], axis=-1)
    #Psi = Psi[..., :2]
    return Psi


def fft2d(A):
    return cv2.dft(A, flags=cv2.DFT_COMPLEX_OUTPUT)


def ifft2d(A):
    return cv2.idft(A, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)


def enhance_Psi(Psi, enhance_factors):
    print('enhance_factors = {}'.format(enhance_factors))
    assert len(enhance_factors) == 2, 'Ecpect the length of enhance_factors to be 2, but provided {}.' \
                                      ''.format(len(enhance_factors))
    assert isinstance(enhance_factors[1], (list, tuple)), 'Expect the type of enhance_factors[1] to be list or tuple' \
                                                          ', but provided {}.'.format(type(enhance_factors[1]))
    enhance_factor_array_high = np.concatenate([[enhance_factors[1][i]] *
                                                (2 ** (i + 2)) for i in range(len(enhance_factors[1]))])

    enhance_factor_array = np.concatenate([[enhance_factors[0]], enhance_factor_array_high])
    Psi = Psi * enhance_factor_array
    Psi_sum = np.sum(Psi, axis=-1)
    return Psi_sum


def allinone(A, Psi=None, numOfScales=None, realCoefficients=True, maxScale='max', shearletSpect=meyerShearletSpect,
             realReal=True, enhance_factor=(1.05, (2.5, 2.5, 1.0))):

    A_FFT = fft2d(A)

    #enhanced_Psi = enhance_Psi(Psi=Psi, enhance_factors=enhance_factor)

    A_complex = A_FFT * Psi[..., np.newaxis]

    A = ifft2d(A_complex)

    return A


def allinone_original(A, Psi=None, numOfScales=None, realCoefficients=True, maxScale='max',
                      shearletSpect=meyerShearletSpect, realReal=True, enhance_factor=(1.05, 3.5)):

    A_FFT = fft2d(A)

    Psi[..., 1] = np.sum(Psi[..., 1:], axis=-1)
    Psi = Psi[..., :2]
    print('in allinone, enhancefactors = {}'.format(enhance_factor))

    Psi_low = Psi[..., 0] * enhance_factor[0]
    Psi_high =Psi[..., 1] * enhance_factor[1][0]

    A_FFT_real = A_FFT[..., 0]
    A_FFT_img = A_FFT[..., 1]


    A_low_real = Psi_low * A_FFT_real
    A_low_img = Psi_low * A_FFT_img
    A_high_real = Psi_high * A_FFT_real
    A_high_img = Psi_high * A_FFT_img

    A_real = A_high_real + A_low_real
    A_img = A_high_img + A_low_img

    A_complex = np.stack((A_real, A_img), axis=-1)

    A = ifft2d(A_complex)

    return A




