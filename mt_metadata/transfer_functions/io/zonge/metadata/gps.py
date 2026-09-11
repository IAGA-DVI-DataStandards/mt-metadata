# =====================================================
# Imports
# =====================================================
from typing import Annotated

from pydantic import Field, field_validator, ValidationInfo

from mt_metadata.base import MetadataBase
from mt_metadata.utils import location_helpers

# =====================================================


class GPS(MetadataBase):
    lat: Annotated[
        float,
        Field(
            default=0.0,
            description="latitude",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["10.3"],
            },
        ),
    ]

    lon: Annotated[
        float,
        Field(
            default=0.0,
            description="longitude",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["10.3"],
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

    @field_validator("datum", mode="before")
    @classmethod
    def validate_datum(cls, value: str | int) -> str:
        """
        Validate the datum value and convert it to the appropriate enum type.
        """
        return location_helpers.validate_datum(value)

    @field_validator("lat", "lon", mode="before")
    @classmethod
    def validate_position(cls, value, info: ValidationInfo):
        if info.field_name in ["lat"]:
            field_name = "latitude"
        elif info.field_name in ["lon"]:
            field_name = "longitude"
        return location_helpers.validate_position(value, field_name)
