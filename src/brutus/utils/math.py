#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mathematical utility functions for brutus.

This module contains mathematical utility functions including matrix operations,
statistical distributions, and numerical utilities.
"""

import numpy as np
from math import erf, gamma, log, sqrt

__all__ = [
    "_function_wrapper",
    "adjoint3",
    "dot3",
    "inverse_transpose3",
    "inverse3",
    "isPSD",
    "chisquare_logpdf",
    "truncnorm_pdf",
    "truncnorm_logpdf",
]


class _function_wrapper(object):
    """
    A hack to make functions pickleable when `args` or `kwargs` are
    also included. Based on the implementation in
    `emcee <http://dan.iel.fm/emcee/>`_.

    Parameters
    ----------
    func : callable
        The function to wrap.
    args : tuple
        Additional positional arguments to pass to the function.
    kwargs : dict
        Additional keyword arguments to pass to the function.
    name : str, optional
        Name for the function (used in error messages).
    """

    def __init__(self, func, args, kwargs, name="input"):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.name = name

    def __call__(self, x):
        try:
            return self.func(x, *self.args, **self.kwargs)
        except:
            import traceback

            print("Exception while calling {0} function:".format(self.name))
            print("  params:", x)
            print("  args:", self.args)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise


def adjoint3(A):
    """
    Compute the adjoint of a series of 3x3 matrices without division
    by the determinant.

    The adjoint (or adjugate) matrix is the transpose of the cofactor matrix.
    For a 3x3 matrix, this can be computed efficiently using cross products.

    Parameters
    ----------
    A : `~numpy.ndarray` of shape `(..., 3, 3)`
        Array of 3x3 matrices.

    Returns
    -------
    AI : `~numpy.ndarray` of shape `(..., 3, 3)`
        Adjoint matrices.
    """
    AI = np.empty_like(A)

    for i in range(3):
        AI[..., i, :] = np.cross(A[..., i - 2, :], A[..., i - 1, :])

    return AI


def dot3(A, B):
    """
    Take the dot product of arrays of vectors, contracting over the
    last indices.

    This is equivalent to np.sum(A * B, axis=-1) but uses einsum
    for clarity and potential performance benefits.

    Parameters
    ----------
    A : `~numpy.ndarray` of shape `(..., N)`
        First array of vectors.
    B : `~numpy.ndarray` of shape `(..., N)`
        Second array of vectors.

    Returns
    -------
    result : `~numpy.ndarray` of shape `(...,)`
        Dot products.
    """
    return np.einsum("...i,...i->...", A, B)


def inverse_transpose3(A):
    """
    Compute the inverse-transpose of a series of 3x3 matrices.

    This computes (A^T)^(-1) = (A^(-1))^T efficiently using the adjoint method.

    Parameters
    ----------
    A : `~numpy.ndarray` of shape `(..., 3, 3)`
        Array of 3x3 matrices.

    Returns
    -------
    AI_T : `~numpy.ndarray` of shape `(..., 3, 3)`
        Inverse-transpose matrices.
    """
    Id = adjoint3(A)
    det = dot3(Id, A).mean(axis=-1)

    return Id / det[..., None, None]


def inverse3(A):
    """
    Compute the inverse of a series of 3x3 matrices using adjoints.

    This method is often more numerically stable than LU decomposition
    for small matrices and avoids the overhead of calling scipy.linalg.inv
    in a loop.

    Parameters
    ----------
    A : `~numpy.ndarray` of shape `(..., 3, 3)`
        Array of 3x3 matrices.

    Returns
    -------
    A_inv : `~numpy.ndarray` of shape `(..., 3, 3)`
        Inverse matrices.
    """
    return np.swapaxes(inverse_transpose3(A), -1, -2)


def isPSD(A):
    """
    Check if `A` is a positive semidefinite matrix.

    A matrix is positive semidefinite if all its eigenvalues are non-negative.
    This function checks this by attempting a Cholesky decomposition, which
    only succeeds for positive semidefinite matrices.

    Parameters
    ----------
    A : `~numpy.ndarray` of shape `(N, N)`
        Square matrix to test.

    Returns
    -------
    is_psd : bool
        True if the matrix is positive semidefinite, False otherwise.
    """
    try:
        _ = np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def chisquare_logpdf(x, df, loc=0, scale=1):
    """
    Compute log-PDF of a chi-square distribution.

    `chisquare_logpdf(x, df, loc, scale)` is equal to
    `chisquare_logpdf(y, df) - ln(scale)`, where `y = (x-loc)/scale`.
    NOTE: This function replicates `~scipy.stats.chi2.logpdf`.

    Parameters
    ----------
    x : `~numpy.ndarray` of shape `(N)` or float
        Input values.

    df : float
        Degrees of freedom.

    loc : float, optional
        Offset of distribution. Default is 0.

    scale : float, optional
        Scaling of distribution. Default is 1.

    Returns
    -------
    ans : `~numpy.ndarray` of shape `(N)` or float
        The natural log of the PDF values.
    """
    if isinstance(x, list):
        x = np.asarray(x)

    y = (x - loc) / scale
    is_scalar = isinstance(y, (float, int))

    if is_scalar:
        if y <= 0:
            return -np.inf
    else:
        keys = y <= 0
        y = np.where(keys, 0.1, y)  # temporary value, will be set to -inf below

    # Compute log-pdf
    ans = -log(2 ** (df / 2.0) * gamma(df / 2.0))
    ans = ans + (df / 2.0 - 1.0) * np.log(y) - y / 2.0 - log(scale)

    if not is_scalar:
        ans = np.where(keys, -np.inf, ans)

    return ans


def truncnorm_pdf(x, a, b, loc=0.0, scale=1.0):
    """
    Compute PDF of a truncated normal distribution.

    The parent normal distribution has a mean of `loc` and
    standard deviation of `scale`. The distribution is cut off at `a` and `b`.
    NOTE: This function replicates `~scipy.stats.truncnorm.pdf`.

    Parameters
    ----------
    x : `~numpy.ndarray` of shape `(N)` or float
        Input values.

    a : float
        Lower cutoff of normal distribution.

    b : float
        Upper cutoff of normal distribution.

    loc : float, optional
        Mean of normal distribution. Default is 0.0.

    scale : float, optional
        Standard deviation of normal distribution. Default is 1.0.

    Returns
    -------
    ans : `~numpy.ndarray` of shape `(N)` or float
        The PDF values.
    """
    _a = scale * a + loc
    _b = scale * b + loc
    xi = (x - loc) / scale
    alpha = (_a - loc) / scale
    beta = (_b - loc) / scale

    phix = np.exp(-0.5 * xi**2) / np.sqrt(2.0 * np.pi)
    Phia = 0.5 * (1 + erf(alpha / np.sqrt(2)))
    Phib = 0.5 * (1 + erf(beta / np.sqrt(2)))

    ans = phix / (scale * (Phib - Phia))

    if not isinstance(x, (float, int)):
        keys = np.logical_or(x < _a, x > _b)
        ans[keys] = 0
    else:
        if x < _a or x > _b:
            ans = 0

    return ans


def truncnorm_logpdf(x, a, b, loc=0.0, scale=1.0):
    """
    Compute log-PDF of a truncated normal distribution.

    The parent normal distribution has a mean of `loc` and
    standard deviation of `scale`. The distribution is cut off at `a` and `b`.
    NOTE: This function replicates `~scipy.stats.truncnorm.logpdf`.

    Parameters
    ----------
    x : `~numpy.ndarray` of shape `(N)` or float
        Input values.

    a : float
        Lower cutoff of normal distribution.

    b : float
        Upper cutoff of normal distribution.

    loc : float, optional
        Mean of normal distribution. Default is 0.0.

    scale : float, optional
        Standard deviation of normal distribution. Default is 1.0.

    Returns
    -------
    ans : `~numpy.ndarray` of shape `(N)` or float
        The natural log PDF values.
    """
    _a = scale * a + loc
    _b = scale * b + loc

    xi = (x - loc) / scale
    alpha = (_a - loc) / scale
    beta = (_b - loc) / scale

    lnphi = -log(sqrt(2 * np.pi)) - 0.5 * np.square(xi)
    lndenom = log(scale / 2.0) + log(erf(beta / np.sqrt(2)) - erf(alpha / sqrt(2)))

    ans = np.subtract(lnphi, lndenom)

    if not isinstance(x, (float, int)):
        keys = np.logical_or(x < _a, x > _b)
        ans[keys] = -np.inf
    else:
        if x < _a or x > _b:
            ans = -np.inf

    return ans
