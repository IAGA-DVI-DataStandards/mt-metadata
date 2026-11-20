from typing import Annotated

from pydantic import Field

from mt_metadata.base import MetadataBase
from mt_metadata.common.enumerations import StrEnumerationBase


class PrewhiteningTypeEnum(StrEnumerationBase):
    none = ""
    first_difference = "first difference"
    arma = "ARMA"
    other = "other"


class Prewhitening(MetadataBase):
    """Configuration metadata for time-series prewhitening."""

    type: Annotated[
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

    ar_order: Annotated[
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

    ma_order: Annotated[
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
