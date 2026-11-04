"""Tensor-type dispatch for migrated contraction kernels, without densifying."""
import numpy as np

from . import lasyten_backend as lb


def is_lasyten_tensor(tensor):
    return (isinstance(tensor, lb.LasyTenArray)
            or type(tensor).__module__.split('.')[0] == 'lasyten')


def tensordot(a, b, axes=2):
    if is_lasyten_tensor(a) or is_lasyten_tensor(b):
        return lb.tensordot(a, b, axes=axes)
    return np.tensordot(a, b, axes=axes)


def transpose(a, axes=None):
    if is_lasyten_tensor(a):
        return lb.transpose(a, axes=axes)
    return np.transpose(a, axes=axes)


def conjugate(a):
    return lb.conjugate(a) if is_lasyten_tensor(a) else np.conjugate(a)
