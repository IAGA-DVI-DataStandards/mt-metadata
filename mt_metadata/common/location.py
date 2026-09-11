# =====================================================
# Imports
# =====================================================
from typing import Annotated

from pydantic import AliasChoices, Field, field_validator, ValidationInfo

from mt_metadata.base import MetadataBase
from mt_metadata.common import Declination, GeographicLocation
from mt_metadata.utils.location_helpers import validate_datum, validate_position

# =====================================================


class BasicLocationNoDatum(MetadataBase):
    """
    A partial location class that only includes the latitude, longitude, and elevation.
    This is used to avoid circular imports.
    """

    latitude: Annotated[
        float | None,
        Field(
            default=0.0,
            description="Latitude of the location.",
            validation_alias=AliasChoices("latitude", "lat"),
            json_schema_extra={
                "units": "degrees",
                "required": False,
                "examples": ["12.324"],
            },
        ),
    ]

    longitude: Annotated[
        float | None,
        Field(
            default=0.0,
            description="Longitude of the location.",
            validation_alias=AliasChoices("longitude", "lon", "long"),
            json_schema_extra={
                "units": "degrees",
                "required": False,
                "examples": ["12.324"],
            },
        ),
    ]

    elevation: Annotated[
        float,
        Field(
            default=0.0,
            description="Elevation of the location.",
            alias=None,
            json_schema_extra={
                "units": "meters",
                "required": False,
                "examples": ["1234.0"],
            },
        ),
    ]

    @field_validator("latitude", "longitude", mode="before")
    @classmethod
    def validate_position(cls, value, info: ValidationInfo):
        return validate_position(value, info.field_name)


class BasicLocation(BasicLocationNoDatum):
    datum: Annotated[
        str | int,
        Field(
            default="WGS 84",
            description="Datum of the location values.  Usually a well known datum like WGS84.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["WGS 84"],
            },
        ),
    ]

    x: Annotated[
        float,
        Field(
            default=0.0,
            description="relative distance to the center of the station",
            validation_alias=AliasChoices("x", "easting", "east"),
            json_schema_extra={
                "units": "meters",
                "required": False,
                "examples": ["10.0"],
            },
        ),
    ]

    y: Annotated[
        float,
        Field(
            default=0.0,
            description="relative distance to the center of the station",
            validation_alias=AliasChoices("y", "north", "northing"),
            json_schema_extra={
                "units": "meters",
                "required": False,
                "examples": ["10.0"],
            },
        ),
    ]

    z: Annotated[
        float,
        Field(
            default=0.0,
            description="relative elevation to the center of the station",
            alias=None,
            json_schema_extra={
                "units": "meters",
                "required": False,
                "examples": ["10.0"],
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


class Location(BasicLocation):
    """
    Positional location of a geographic point
    """

    latitude_uncertainty: Annotated[
        float,
        Field(
            default=0.0,
            description="uncertainty in latitude estimation in degrees",
            alias=None,
            json_schema_extra={
                "units": "degrees",
                "required": False,
                "examples": ["0.01"],
            },
        ),
    ]

    longitude_uncertainty: Annotated[
        float,
        Field(
            default=0.0,
            description="uncertainty in longitude estimation in degrees",
            alias=None,
            json_schema_extra={
                "units": "degrees",
                "required": False,
                "examples": ["0.01"],
            },
        ),
    ]

    elevation_uncertainty: Annotated[
        float,
        Field(
            default=0.0,
            description="uncertainty in elevation estimation",
            alias=None,
            json_schema_extra={
                "units": "meters",
                "required": False,
                "examples": ["0.01"],
            },
        ),
    ]

    x2: Annotated[
        float,
        Field(
            default=0.0,
            description="relative distance to the center of the station",
            validation_alias=AliasChoices("x2", "east", "easting"),
            json_schema_extra={
                "units": "meters",
                "required": False,
                "examples": ["10.0"],
            },
        ),
    ]

    y2: Annotated[
        float,
        Field(
            default=0.0,
            description="relative distance to the center of the station",
            validation_alias=AliasChoices("y2", "north", "northing"),
            json_schema_extra={
                "units": "meters",
                "required": False,
                "examples": ["10.0"],
            },
        ),
    ]

    z2: Annotated[
        float,
        Field(
            default=0.0,
            description="relative elevation to the center of the station",
            alias=None,
            json_schema_extra={
                "units": "meters",
                "required": False,
                "examples": ["10.0"],
            },
        ),
    ]

    x_uncertainty: Annotated[
        float,
        Field(
            default=0.0,
            description="uncertainty in longitude estimation in x-direction",
            alias=None,
            json_schema_extra={
                "units": "meters",
                "required": False,
                "examples": ["0.01"],
            },
        ),
    ]

    y_uncertainty: Annotated[
        float,
        Field(
            default=0.0,
            description="uncertainty in longitude estimation in y-direction",
            alias=None,
            json_schema_extra={
                "units": "meters",
                "required": False,
                "examples": ["0.01"],
            },
        ),
    ]

    z_uncertainty: Annotated[
        float,
        Field(
            default=0.0,
            description="uncertainty in longitude estimation in z-direction",
            alias=None,
            json_schema_extra={
                "units": "meters",
                "required": False,
                "examples": ["0.01"],
            },
        ),
    ]


class StationLocation(Location):
    """
    A class that represents the location of a station. It includes latitude, longitude, elevation,
    and other related attributes.
    """

    declination: Annotated[
        Declination,
        Field(
            default_factory=Declination,  # type: ignore
            description="Declination of the location.",
            alias=None,
            json_schema_extra={
                "units": "degrees",
                "required": False,
                "examples": ["Declination(10.0)"],
            },
        ),
    ]

    geographic_location: Annotated[
        GeographicLocation,
        Field(
            default_factory=GeographicLocation,  # type: ignore
            description="Geographic location of the station.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["GeographicLocation(latitude=12.34, longitude=56.78)"],
            },
        ),
    ]
