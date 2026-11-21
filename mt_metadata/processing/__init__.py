"""
processing metadata
======================

:mod:`mt_metadata.processing` is a package that contains classes for
describing processing metadata. Including calculating Fourier transforms
and decimating time series data, and a specific module for processing
using `Aurora <https://github.com/simpeg/aurora>`__.

"""

from .prewhitening import Prewhitening, PrewhiteningTypeEnum
from .short_time_fourier_transform import ShortTimeFourierTransform
from .time_series_decimation import TimeSeriesDecimation
from .window import Window

__all__ = [
    "Prewhitening",
    "PrewhiteningTypeEnum",
    "ShortTimeFourierTransform",
    "TimeSeriesDecimation",
    "Window",
]
