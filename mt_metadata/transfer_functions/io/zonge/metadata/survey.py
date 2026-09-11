# =====================================================
# Imports
# =====================================================
from enum import Enum
from typing import Annotated

from pydantic import Field, field_validator

from mt_metadata.base import MetadataBase
from mt_metadata.common.enumerations import DataTypeEnum
from mt_metadata.utils.location_helpers import validate_datum

# =====================================================


class ArrayEnum(str, Enum):
    tensor = "tensor"


class ProjEnum(str, Enum):
    UTM = "UTM"
    other = "other"


class Survey(MetadataBase):
    type: Annotated[
        DataTypeEnum,
        Field(
            default=DataTypeEnum.NSAMT,
            description="Type of EM survey",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["nsamt"],
            },
        ),
    ]

    array: Annotated[
        ArrayEnum,
        Field(
            default=ArrayEnum.tensor,
            description="Type of array",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["tensor"],
            },
        ),
    ]

    datum: Annotated[
        str,
        Field(
            default="WGS84",
            description="Datum of the location",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["WGS84"],
            },
        ),
    ]

    u_t_m_zone: Annotated[
        int,
        Field(
            default=0,
            description="UTM zone of location",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["12"],
            },
        ),
    ]

    proj: Annotated[
        ProjEnum,
        Field(
            default=ProjEnum.UTM,
            description="Projection of the location coordinates",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["UTM"],
            },
        ),
    ]

    @field_validator("datum", mode="before")
    @classmethod
    def validate_datum(cls, value: str | int) -> str:
        """
        Validate the datum value and convert it to the appropriate enum type.
        """
        return validate_datum(value)
