# =====================================================
# Imports
# =====================================================
from typing import Annotated

from pydantic import AliasChoices, Field

from mt_metadata.base import MetadataBase


# =====================================================
class Instrument(MetadataBase):
    id: Annotated[
        str | None,
        Field(
            default="",
            description="Instrument ID number can be serial number or a designated ID.",
            validation_alias=AliasChoices("id", "serial"),
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["mt01"],
            },
        ),
    ]

    manufacturer: Annotated[
        str | None,
        Field(
            default="",
            description="Who manufactured the instrument.",
            validation_alias=AliasChoices("manufacturer", "make"),
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["mt gurus"],
            },
        ),
    ]

    type: Annotated[
        str | None,
        Field(
            default="",
            description="Description of the instrument type.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["broadband 32-bit"],
            },
        ),
    ]

    model: Annotated[
        str | None,
        Field(
            default=None,
            description="Model version of the instrument.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["falcon5"],
            },
        ),
    ]

    name: Annotated[
        str | None,
        Field(
            default=None,
            description="Standard marketing name of the instrument.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["falcon5"],
            },
        ),
    ]
