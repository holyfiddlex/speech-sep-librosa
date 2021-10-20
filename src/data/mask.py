"""Mask object definitions"""

import numpy as np
from fastai.torch_basics import Tensor
from src.data.audio import SpecArray, SpecTensor

class MaskObject():
    """Mask filter object"""

    data_type: callable = None
    spec_type: type = None

    # pylint: disable=not-callable
    def __init__(self, data):
        self.data = self.data_type(data)

    @property
    def shape(self):
        """Return shape of mask data"""
        return self.data.shape

    def mult(self, spec):
        """Multiplication rule to be overridden in case of new mask type"""
        return self.data*spec.data

    def __mul__(self, spec):
        """Multiply spec and mask, applying filter"""
        data_filt = self.mult(spec.data)
        return self.spec_type(data_filt, spec.sr, spec.fn)

    def __rmul__(self, spec):
        """Multiply spec and mask, applying filter"""
        return self*spec

class MaskArray(MaskObject):
    """MaskArray object"""

    data_type: callable = np.array
    spec_type: type = SpecArray

class MaskTensor(MaskObject):
    """MaskTensor object"""

    data_type: callable = Tensor
    spec_type: type = SpecTensor
