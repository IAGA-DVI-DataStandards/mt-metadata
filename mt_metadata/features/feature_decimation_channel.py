# =====================================================
# Imports
# =====================================================
import json
from typing import Annotated

from pydantic import Field, field_validator

from mt_metadata.base import MetadataBase
from mt_metadata.common import TimePeriod
from mt_metadata.common.units import get_unit_object


# =====================================================
class FeatureDecimationChannel(MetadataBase):
    name: Annotated[
        str,
        Field(
            default="",
            description="Name of channel",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["ex"],
            },
        ),
    ]

    frequency_max: Annotated[
        float,
        Field(
            default=0.0,
            description="Highest frequency present in the sprectrogam data.",
            alias=None,
            json_schema_extra={
                "units": "samples per second",
                "required": True,
                "examples": [77.0],
            },
        ),
    ]

    frequency_min: Annotated[
        float,
        Field(
            default=0.0,
            description="Lowest frequency present in the sprectrogam data.",
            alias=None,
            json_schema_extra={
                "units": "samples per second",
                "required": True,
                "examples": [99.0],
            },
        ),
    ]

    sample_rate_decimation_level: Annotated[
        float,
        Field(
            default=1.0,
            description="Sample rate of the time series that was Fourier transformed to generate the FC decimation level.",
            alias=None,
            json_schema_extra={
                "units": "samples per second",
                "required": True,
                "examples": [60],
            },
        ),
    ]

    sample_rate_window_step: Annotated[
        float,
        Field(
            default=1.0,
            description="Sample rate of the windows.",
            alias=None,
            json_schema_extra={
                "units": "samples per second",
                "required": True,
                "examples": [4],
            },
        ),
    ]

    units: Annotated[
        str,
        Field(
            default="count",
            description="Units of the channel",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["milliVolts"],
            },
        ),
    ]

    time_period: Annotated[
        TimePeriod,
        Field(
            default_factory=TimePeriod,
            description="Time period of the channel",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": [{"start": "2020-01-01", "end": "2020-01-02"}],
            },
        ),
    ]

    prewhitening_type: Annotated[
        str,
        Field(
            default="",
            description="Prewhitening method applied before Fourier transformation",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["ARMA"],
            },
        ),
    ]

    prewhitening_fit_domain: Annotated[
        str,
        Field(
            default="",
            description="Time-series domain over which prewhitening was fit",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["full run"],
            },
        ),
    ]

    prewhitening_response_convention: Annotated[
        str,
        Field(
            default="",
            description="Discrete-time transfer-function convention for prewhitening",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["A(z) = 1 - sum(phi_k z^-k)"],
            },
        ),
    ]

    prewhitening_ar_coefficients: Annotated[
        list[float],
        Field(
            default_factory=list,
            description="Fitted autoregressive prewhitening coefficients",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": [[0.8, -0.2]],
            },
        ),
    ]

    prewhitening_ma_coefficients: Annotated[
        list[float],
        Field(
            default_factory=list,
            description="Fitted moving-average prewhitening coefficients",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": [[]],
            },
        ),
    ]

    @field_validator("units", mode="before")
    @classmethod
    def validate_units(cls, value: str) -> str:
        if value in [None, ""]:
            return ""
        try:
            unit_object = get_unit_object(value)
            return unit_object.name
        except ValueError as error:
            raise KeyError(error)
        except KeyError as error:
            raise KeyError(error)

    @field_validator(
        "prewhitening_ar_coefficients",
        "prewhitening_ma_coefficients",
        mode="before",
    )
    @classmethod
    def validate_prewhitening_coefficients(cls, value):
        """Deserialize coefficient lists read from HDF5 attributes."""
        if isinstance(value, str):
            return json.loads(value)
        return value
