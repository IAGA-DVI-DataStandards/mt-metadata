# =====================================================
# Imports
# =====================================================
from typing import Annotated, Union

from pydantic import Field

from mt_metadata.base import MetadataBase
from mt_metadata.common.enumerations import StrEnumerationBase
from mt_metadata.processing.window import Window


# =====================================================
class MethodEnum(StrEnumerationBase):
    fft = "fft"
    wavelet = "wavelet"
    other = "other"


class PerWindowDetrendTypeEnum(StrEnumerationBase):
    linear = "linear"
    constant = "constant"
    null = ""


class PreFftDetrendTypeEnum(StrEnumerationBase):
    linear = "linear"
    other = "other"
    null = ""


class PrewhiteningTypeEnum(StrEnumerationBase):
    none = ""
    first_difference = "first difference"
    arma = "ARMA"
    other = "other"


class ShortTimeFourierTransform(MetadataBase):
    harmonic_indices: Annotated[
        Union[int, list[int], None],
        Field(
            default=None,
            description="List of harmonics indices kept, if all use -1",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": [[0, 4, 8]],
            },
        ),
    ]

    method: Annotated[
        MethodEnum,
        Field(
            default=MethodEnum.fft,
            description="Fourier transform method",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["fft"],
            },
        ),
    ]

    min_num_stft_windows: Annotated[
        int,
        Field(
            default=0,
            description="How many FFT windows must be available for the time series to valid for STFT.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": [4],
            },
        ),
    ]

    per_window_detrend_type: Annotated[
        PerWindowDetrendTypeEnum,
        Field(
            default=PerWindowDetrendTypeEnum.null,
            description="Additional detrending applied per window.  Not available for standard scipy spectrogram -- placholder for ARMA prewhitening.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["linear"],
            },
        ),
    ]

    pre_fft_detrend_type: Annotated[
        PreFftDetrendTypeEnum,
        Field(
            default=PreFftDetrendTypeEnum.linear,
            description="Pre FFT detrend method to be applied",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["linear"],
            },
        ),
    ]

    prewhitening_type: Annotated[
        PrewhiteningTypeEnum,
        Field(
            default=PrewhiteningTypeEnum.first_difference,
            description="Prewhitening method to be applied",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["first difference"],
            },
        ),
    ]

    prewhitening_ar_order: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            description="Order of the autoregressive prewhitening model",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": [3],
            },
        ),
    ]

    prewhitening_ma_order: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            description="Order of the moving-average prewhitening model",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": [0],
            },
        ),
    ]

    recoloring: Annotated[
        bool,
        Field(
            default=True,
            description="Whether the data are recolored [True] or not [False].",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": [True],
            },
        ),
    ]

    window: Annotated[
        Window,
        Field(
            default_factory=Window,  # type: ignore
            description="Window settings",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["Window()"],
            },
        ),
    ]


# what is the point of this main function?
# def main():
#     stft = ShortTimeFourierTransform()


# if __name__ == "__main__":
#     main()
