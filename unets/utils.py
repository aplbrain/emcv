"""
Copyright 2018 The Johns Hopkins University Applied Physics Laboratory.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

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

