import numpy as np


def gaussian(x):
    y = -np.exp(-1. * x ** 2 / 0.45 **2) * (x >= 0) * (x <= 1) + 1
    #y = np.polyval([0.5, 0], x) * (x >= 0) * (x <= 1 / 2)
    #y += np.polyval([0.5, 0], x) * (x > 1 / 2) * (x <= 1)
    #y += (x > 1)
    return y


def kutyniokaux(x):
    """
    kutyniok wavelet auxiliary function.

    v(x) = 0                for x < 0
           2x^2             for 0 <= x < 1/2
           -2x^2 + 4x - 1   for 1/2 < x <= 1
           1                for x>1

    :param x: array, grid points
    :return: y : array, values at x
    """
    y = np.polyval([2, 0, 0], x) * (x >= 0) * (x <= 1/2)
    y += np.polyval([-2, 4, -1], x) * (x > 1/2) * (x <= 1)
    y += (x > 1)
    return y

def meyeraux(x):
    """meyer wavelet auxiliary function.

    v(x) = 35*x^4 - 84*x^5 + 70*x^6 - 20*x^7.

    Parameters
    ----------
    x : array
        grid points

    Returns
    -------
    y : array
        values at x
    """
    # Auxiliary def values.
    y = np.polyval([-20, 70, -84, 35, 0, 0, 0, 0], x) * (x >= 0) * (x <= 1)
    y += (x > 1)
    return y


def _meyerHelper(x, realCoefficients=True, meyeraux_func=meyeraux):
    if realCoefficients:
        xa = np.abs(x)
    else:
        # consider left and upper part of the image due to first row and column
        xa = -x

    int1 = ((xa >= 1) & (xa < 2))
    int2 = ((xa >= 2) & (xa < 4))

    psihat = int1 * np.sin(np.pi/2*meyeraux_func(xa-1))
    psihat = psihat + int2 * np.cos(np.pi/2*meyeraux_func(1/2*xa-1))

    y = psihat
    return y


def meyerWavelet(x, realCoefficients=True, meyeraux_func=meyeraux):
    """ compute Meyer Wavelet.

    Parameters
    ----------
    x : array
        grid points

    Returns
    -------
    y : phihat
        values at given points x

    """
    y = np.sqrt(np.abs(_meyerHelper(x, realCoefficients, meyeraux_func))**2 +
                np.abs(_meyerHelper(2*x, realCoefficients, meyeraux_func))**2)
    return y


def meyerScaling(x, meyeraux_func=meyeraux):
    """mother scaling def for meyer shearlet.

    Parameters
    ----------
    x : array
        grid points
    meyeraux_func : function
        auxiliary function

    Returns
    -------
    y : phihat
        values at given points x

    """
    xa = np.abs(x)

    # Compute support of Fourier transform of phi.
    int1 = ((xa < 1/2))
    int2 = ((xa >= 1/2) & (xa < 1))

    # Compute Fourier transform of phi.
    # phihat = int1 * np.ones_like(xa)
    # phihat = phihat + int2 * np.cos(np.pi/2*meyeraux_func(2*xa-1))
    phihat = int1 + int2 * np.cos(np.pi/2*meyeraux_func(2*xa-1))

    return phihat

def meyerBump(x, meyeraux_func=meyeraux):
    int1 = meyeraux_func(x) * (x >= 0) * (x <= 1)
    y = int1 + (x > 1)
    return y


def bump(x, meyeraux_func=meyeraux):
    """compute the def psi_2^ at given points x.

    Parameters
    ----------
    x : array
        grid points
    meyeraux_func : function
        auxiliary function

    Returns
    -------
    y : array
        values at given points x
    """
    y = meyerBump(1+x, meyeraux_func)*(x <= 0) + \
        meyerBump(1-x, meyeraux_func)*(x > 0)
    y = np.sqrt(y)
    return y

def meyerShearletSpect(x, y, a, s, realCoefficients=True,
                       meyeraux_func=meyeraux, scaling_only=False):
    """Returns the spectrum of the shearlet "meyerShearlet".

    Parameters
    ----------
    x : array
        meshgrid for the x-axis
    y : array
        meshgrid for the y-axis
    a : float
        scale
    s : float
        shear
    realCoefficients : bool, optional
        enforce real-valued coefficients
    meyeraux_func : function
        auxiliary function
    scaling_only : bool, optional
        return the scalings instead of computing the spectrum

    Returns
    -------
    y : Psi
        The shearlet spectrum

    """
    if scaling_only:
        # cones
        C_hor = np.abs(x) >= np.abs(y)  # with diag
        C_ver = np.abs(x) < np.abs(y)
        Psi = (meyerScaling(x, meyeraux_func) * C_hor +
               meyerScaling(y, meyeraux_func) * C_ver)
        return Psi

    # compute scaling and shearing
    y = s * np.sqrt(a) * x + np.sqrt(a) * y
    x = a * x

    # set values with x=0 to 1 (for division)
    xx = (np.abs(x) == 0) + (np.abs(x) > 0)*x

    # compute spectrum
    Psi = meyerWavelet(x, realCoefficients, meyeraux_func) * \
        bump(y/xx, meyeraux_func)
    return Psi