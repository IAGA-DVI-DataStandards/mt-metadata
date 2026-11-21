# =====================================================
# Imports
# =====================================================
from typing import Annotated, Union

from pydantic import Field, model_validator

from mt_metadata.base import MetadataBase
from mt_metadata.common.enumerations import StrEnumerationBase
from mt_metadata.processing.prewhitening import Prewhitening, PrewhiteningTypeEnum
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

    prewhitening: Annotated[
        Prewhitening,
        Field(
            default_factory=Prewhitening,
            description="Prewhitening configuration",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["Prewhitening()"],
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

    @model_validator(mode="before")
    @classmethod
    def map_legacy_prewhitening_fields(cls, values):
        """Map legacy flat fields to the composed prewhitening metadata."""
        if not isinstance(values, dict):
            return values

        values = values.copy()
        legacy_values = {
            "type": values.pop("prewhitening_type", None),
            "ar_order": values.pop("prewhitening_ar_order", None),
            "ma_order": values.pop("prewhitening_ma_order", None),
        }
        legacy_values = {
            key: value for key, value in legacy_values.items() if value is not None
        }
        prewhitening = values.get("prewhitening")
        if legacy_values:
            if prewhitening is None:
                prewhitening = {}
            elif isinstance(prewhitening, Prewhitening):
                prewhitening = prewhitening.model_dump()
            else:
                prewhitening = prewhitening.copy()
            prewhitening.update(legacy_values)
            values["prewhitening"] = prewhitening
        return values

    @property
    def prewhitening_type(self) -> PrewhiteningTypeEnum:
        """Legacy compatibility property; use ``prewhitening.type`` instead."""
        return self.prewhitening.type

    @prewhitening_type.setter
    def prewhitening_type(self, value: PrewhiteningTypeEnum) -> None:
        self.prewhitening.type = value

    @property
    def prewhitening_ar_order(self) -> int:
        """Legacy compatibility property; use ``prewhitening.ar_order`` instead."""
        return self.prewhitening.ar_order

    @prewhitening_ar_order.setter
    def prewhitening_ar_order(self, value: int) -> None:
        self.prewhitening.ar_order = value

    @property
    def prewhitening_ma_order(self) -> int:
        """Legacy compatibility property; use ``prewhitening.ma_order`` instead."""
        return self.prewhitening.ma_order

    @prewhitening_ma_order.setter
    def prewhitening_ma_order(self, value: int) -> None:
        self.prewhitening.ma_order = value
