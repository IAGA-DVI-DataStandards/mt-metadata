# =====================================================
# Imports
# =====================================================
from typing import Annotated

import numpy as np
from pydantic import Field, field_validator

from mt_metadata.base import MetadataBase
from mt_metadata.common.enumerations import ArrayDTypeEnum
from mt_metadata.common.units import get_unit_object


# =====================================================


class StatisticalEstimate(MetadataBase):
    name: Annotated[
        str,
        Field(
            default="",
            description="Name of the statistical estimate",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["transfer function"],
            },
        ),
    ]

    data_type: Annotated[
        ArrayDTypeEnum,
        Field(
            default="complex",
            description="Type of number contained in the estimate",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["real"],
            },
        ),
    ]

    description: Annotated[
        str,
        Field(
            default="",
            description="Description of the statistical estimate",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["this is an estimate"],
            },
        ),
    ]

    input_channels: Annotated[
        list[str] | str | np.ndarray,
        Field(
            default=[],
            description="List of input channels (sources)",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["hx, hy", ["hx", "hy"]],
            },
        ),
    ]

    output_channels: Annotated[
        list[str] | str | np.ndarray,
        Field(
            default=[],
            description="List of output channels (response).",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["hx, hy", ["hx", "hy"]],
            },
        ),
    ]

    units: Annotated[
        str,
        Field(
            default="",
            description="Units of the estimate.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["millivolts per kilometer per nanotesla"],
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

    @field_validator("input_channels", "output_channels", mode="before")
    @classmethod
    def validate_channels(cls, value: list[str] | str | np.ndarray) -> list[str]:
        """convert channels to a list of single channels"""
        if isinstance(value, str):
            value = [v.strip() for v in value.split(",")]
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        elif not isinstance(value, list):
            raise TypeError(
                f"Input channels must be a list of channels, not {type(value)}."
            )
        return value
