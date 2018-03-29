__author__ = 'drenkng1'

import numpy as np
import SimpleITK as sitk
import logging
import math

dtype_map = {
    'uint8': sitk.sitkUInt8,
    'int16': sitk.sitkInt16,
    'uint16': sitk.sitkUInt16,
    'int32': sitk.sitkInt32,
    'uint32': sitk.sitkUInt32,
    'float32': sitk.sitkFloat32,
    'float64': sitk.sitkFloat64
}

