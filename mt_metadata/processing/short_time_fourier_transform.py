# =====================================================
# Imports
# =====================================================
from typing import Annotated, Union

from pydantic import Field, field_validator, model_validator

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

    @model_validator(mode="before")
    @classmethod
    def populate_prewhitening(cls, values):
        """Map legacy flat fields to the composed prewhitening metadata."""
        if not isinstance(values, dict):
            return values

        values = values.copy()
        prewhitening = values.get("prewhitening")
        if prewhitening is None:
            values["prewhitening"] = {
                "type": values.get(
                    "prewhitening_type",
                    PrewhiteningTypeEnum.first_difference,
                ),
                "ar_order": values.get("prewhitening_ar_order", 0),
                "ma_order": values.get("prewhitening_ma_order", 0),
            }
        else:
            if isinstance(prewhitening, dict):
                get_value = prewhitening.get
            else:
                get_value = lambda name: getattr(prewhitening, name)
            values.setdefault(
                "prewhitening_type",
                get_value("type"),
            )
            values.setdefault(
                "prewhitening_ar_order",
                get_value("ar_order"),
            )
            values.setdefault(
                "prewhitening_ma_order",
                get_value("ma_order"),
            )
        return values

    @field_validator("prewhitening_type", mode="after")
    @classmethod
    def sync_prewhitening_type(cls, value, info):
        """Keep the composed metadata synchronized with the legacy field."""
        if "prewhitening" in info.data:
            info.data["prewhitening"].type = value
        return value

    @field_validator("prewhitening_ar_order", "prewhitening_ma_order", mode="after")
    @classmethod
    def sync_prewhitening_orders(cls, value, info):
        """Keep the composed metadata synchronized with legacy order fields."""
        if "prewhitening" in info.data:
            field_name = info.field_name.removeprefix("prewhitening_")
            setattr(info.data["prewhitening"], field_name, value)
        return value

    @model_validator(mode="after")
    def sync_legacy_prewhitening_fields(self):
        """Expose composed metadata through the legacy flat STFT fields."""
        object.__setattr__(self, "prewhitening_type", self.prewhitening.type)
        object.__setattr__(self, "prewhitening_ar_order", self.prewhitening.ar_order)
        object.__setattr__(self, "prewhitening_ma_order", self.prewhitening.ma_order)
        return self


# what is the point of this main function?
# def main():
#     stft = ShortTimeFourierTransform()


# if __name__ == "__main__":
#     main()
