# =====================================================
# Imports
# =====================================================
from typing import Annotated

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import Field, field_validator

from mt_metadata import __version__
from mt_metadata.common import (
    BasicLocation,
    Declination,
    GeographicLocation,
    GeographicReferenceFrameEnum,
    StdEDIversionsEnum,
)
from mt_metadata.common.mttime import get_now_utc, MTime
from mt_metadata.common.units import get_unit_object
from mt_metadata.utils.location_helpers import convert_position_float2str
from mt_metadata.utils.validators import validate_station_name

# =====================================================


class Header(BasicLocation, GeographicLocation):
    acqby: Annotated[
        str | None,
        Field(
            default=None,
            description="person, group, company, university that collected the data",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["mt experts"],
            },
        ),
    ]

    acqdate: Annotated[
        MTime | str | float | int | np.datetime64 | pd.Timestamp,
        Field(
            default_factory=lambda: MTime(time_stamp=None),
            description="Start date the time series data were collected",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["2020-01-01"],
            },
        ),
    ]

    coordinate_system: Annotated[
        GeographicReferenceFrameEnum,
        Field(
            default="geographic",
            description="coordinate system the transfer function is currently in. Its preferred the transfer function be in a geographic coordinate system for archiving and sharing.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["geopgraphic"],
            },
        ),
    ]

    dataid: Annotated[
        str,
        Field(
            default="",
            description="station ID.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["mt001"],
            },
        ),
    ]

    enddate: Annotated[
        MTime | str | float | int | np.datetime64 | pd.Timestamp | None,
        Field(
            default_factory=lambda: MTime(time_stamp=None),
            description="End date the time series data were collected",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["2020-01-01"],
            },
        ),
    ]

    empty: Annotated[
        float,
        Field(
            default=1e32,
            description="null data values, usually a large number",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["1E+32"],
            },
        ),
    ]

    fileby: Annotated[
        str,
        Field(
            default="",
            description="person, group, company, university that made the file",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["mt experts"],
            },
        ),
    ]

    filedate: Annotated[
        MTime | str | float | int | np.datetime64 | pd.Timestamp,
        Field(
            default_factory=lambda: MTime(time_stamp=None),
            description="Date the file was made",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["2020-01-01"],
            },
        ),
    ]

    progdate: Annotated[
        MTime | str | float | int | np.datetime64 | pd.Timestamp,
        Field(
            default_factory=lambda: MTime(time_stamp=None),
            description="Date of the most recent update of the program used to make the file",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["2020-01-01"],
            },
        ),
    ]

    progname: Annotated[
        str,
        Field(
            default="mt_metadata",
            description="Name of the program used to make the file.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["mt_metadata"],
            },
        ),
    ]

    progvers: Annotated[
        str,
        Field(
            default="0.1.6",
            description="Version of the program used to make the file.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["0.1.6"],
            },
        ),
    ]

    project: Annotated[
        str | None,
        Field(
            default=None,
            description="Name of the project the data was collected for, usually a short description or acronym of the project name.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["iMUSH"],
            },
        ),
    ]

    prospect: Annotated[
        str | None,
        Field(
            default=None,
            description="Name of the prospect the data was collected for, usually a short description of the location",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["Benton"],
            },
        ),
    ]

    loc: Annotated[
        str | None,
        Field(
            default=None,
            description="Usually a short description of the location",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["Benton, CA"],
            },
        ),
    ]

    declination: Annotated[
        Declination,
        Field(
            default_factory=lambda: Declination(value=0.0),  # type: ignore
            description="Declination of the station in degrees",
            alias=None,
            json_schema_extra={
                "units": "degrees",
                "required": True,
                "examples": ["Declination(10.0)"],
            },
        ),
    ]

    stdvers: Annotated[
        StdEDIversionsEnum,
        Field(
            default="SEG 1.0",
            description="EDI standards version SEG 1.0",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["SEG 1.0"],
            },
        ),
    ]

    survey: Annotated[
        str | None,
        Field(
            default=None,
            description="Name of the survey",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["CONUS"],
            },
        ),
    ]

    units: Annotated[
        str | None,
        Field(
            default="milliVolt per kilometer per nanoTesla",
            description="In the EDI standards this is the elevation units, in newer versions this should be units of the transfer function.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["milliVolt per kilometer per nanoTesla"],
            },
        ),
    ]

    @field_validator("acqdate", "enddate", "filedate", "progdate", mode="before")
    @classmethod
    def validate_acqdate(
        cls, field_value: MTime | float | int | np.datetime64 | pd.Timestamp | str
    ):
        if isinstance(field_value, MTime):
            return field_value
        return MTime(time_stamp=field_value)

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

    def __str__(self):
        return "".join(self.write_header())

    def __repr__(self):
        return self.__str__()

    def get_header_list(self, edi_lines: list[str]) -> list[str]:
        """
        Get the header information from the .edi file in the form of a list.

        Extracts header lines from an EDI file, returning each key-value pair
        as a formatted string.

        Parameters
        ----------
        edi_lines : list of str
            List of lines from an EDI file to parse for header information.

        Returns
        -------
        list of str
            List of header key-value pairs in the format 'key=value'.

        Examples
        --------
        >>> header = Header()
        >>> edi_lines = ['>HEAD', 'DATAID=MT001', 'LAT=45.5', '>']
        >>> header.get_header_list(edi_lines)
        ['DATAID=MT001', 'LAT=45.5']

        """

        header_list = []
        head_find = False

        # read in list line by line and then truncate
        for line in edi_lines:
            # check for header label
            if ">" in line and "head" in line.lower():
                head_find = True
            # if the header line has been found then the next >
            # should be the next section so stop
            elif ">" in line:
                if head_find is True:
                    break
                else:
                    pass
            # get the header information into a list
            elif head_find:
                # skip any blank lines
                if len(line.strip()) > 2:
                    line = line.strip().replace('"', "")
                    h_list = line.split("=")
                    if len(h_list) == 2:
                        key = h_list[0].strip()
                        value = h_list[1].strip()
                        header_list.append(f"{key}={value}")
        return header_list

    def read_header(self, edi_lines: list[str]) -> None:
        """
        Read and parse header information from EDI file lines.

        Parses header lines from an EDI file and populates the Header object
        attributes with the corresponding values. Handles special cases like
        Phoenix-formatted EDI files and coordinate system normalization.

        Parameters
        ----------
        edi_lines : list of str
            List of lines from an EDI file containing header information.

        Returns
        -------
        None
            Updates the object's attributes in place.

        Examples
        --------
        >>> header = Header()
        >>> edi_lines = ['>HEAD', 'DATAID=MT001', 'LAT=45:30:00', 'LON=-122:30:00', '>']
        >>> header.read_header(edi_lines)
        >>> header.dataid
        'mt001'
        >>> header.latitude
        45.5

        Notes
        -----
        - Station IDs are automatically validated and normalized to lowercase
        - Coordinate systems are normalized to 'geographic', 'geomagnetic', or 'station'
        - Phoenix MT-Editor format is automatically detected

        """

        for h_line in self.get_header_list(edi_lines):
            h_list = h_line.split("=")
            key = h_list[0].lower()
            value = h_list[1]
            # test if its a phoenix formated .edi file
            if key in ["progvers"]:
                if value.lower().find("mt-editor") != -1:
                    self.phoenix_edi = True
            elif key in ["coordinate_system"]:
                value = value.lower()
                if "geomagnetic" in value:
                    value = "geomagnetic"
                elif "geographic" in value:
                    value = "geographic"
                elif "station" in value:
                    value = "station"
            elif key in ["stdvers"]:
                if value in ["N/A", "None", "null"]:
                    value = "SEG 1.0"
            elif key in ["units"]:
                if value in ["m", "M"]:
                    value = "m"
                elif value in ["millivolts_per_kilometer_per_nanotesla"]:
                    value = "milliVolt per kilometer per nanoTesla"

            if "declination" in key:
                if key.find(".") < 0:
                    if value in ["None", "none", None, "null"]:
                        value = 0.0
                    try:
                        setattr(self.declination, "value", value)
                    except Exception as error:
                        logger.warning(
                            f"Could not set declination value with {value}, cause the following error {error}"
                        )
                    continue
                else:
                    name, attr = key.split(".", 1)
                    try:
                        setattr(getattr(self, name), attr, value)
                    except Exception as error:
                        logger.warning(
                            f"Could not set {name}.{attr} with {value}, cause the following error {error}"
                        )
                    continue

            elif key in ["long", "lon", "lonigutde"]:
                key = "longitude"
            elif key in ["lat", "latitude"]:
                key = "latitude"
            elif key in ["elev", "elevation"]:
                key = "elevation"
            else:
                if key in ["dataid"]:
                    value = validate_station_name(value)
            try:
                setattr(self, key, value)
            except Exception as error:
                logger.warning(
                    f"Could not set attribute {key} with value {value}, cause the following error {error}"
                )

    def write_header(
        self,
        longitude_format: str = "LON",
        latlon_format: str = "dms",
        required: bool = False,
    ) -> list[str]:
        """
        Write header information to a list of formatted lines for EDI output.

        Formats all header attributes as EDI-compliant key-value pairs. Automatically
        updates file metadata (filedate, progvers, progname) to current values.

        Parameters
        ----------
        longitude_format : {'LON', 'LONG'}, default 'LON'
            Format for longitude field name in output.
        latlon_format : {'dms', 'dd'}, default 'dms'
            Format for latitude/longitude values.
            - 'dms': degrees:minutes:seconds (e.g., '45:30:00')
            - 'dd': decimal degrees (e.g., '45.500000')
        required : bool, default False
            If True, only include required fields in output.

        Returns
        -------
        list of str
            List of formatted header lines starting with '>HEAD' and ending with
            a blank line. Each data line follows the format 'KEY=value'.

        Examples
        --------
        >>> header = Header(dataid='mt001', latitude=45.5, longitude=-122.5)
        >>> lines = header.write_header(latlon_format='dd')
        >>> print(lines[0])
        >HEAD
        >>> print(lines[1])
        \tDATAID=mt001

        >>> # Write with degrees-minutes-seconds format
        >>> lines_dms = header.write_header(latlon_format='dms')
        >>> # Latitude will be formatted as 45:30:00

        Notes
        -----
        - filedate is automatically set to current UTC time
        - progvers is set to mt_metadata version
        - Zero declination values are omitted from output
        - None values are skipped

        """

        self.filedate = get_now_utc()
        self.progvers = __version__
        self.progname = "mt_metadata"
        self.progdate = "2021-12-01"

        header_lines = [">HEAD\n"]
        for key, value in self.to_dict(single=True, required=required).items():
            if key in ["x", "x2", "y", "y2", "z", "z2"]:
                continue
            if value in [None, "None"]:
                continue
            if key in ["latitude"]:
                key = "lat"
            elif key in ["longitude"]:
                key = longitude_format.lower()
            elif key in ["elevation"]:
                key = "elev"
            if "declination" in key:
                if self.declination.value == 0.0:
                    continue
            if key in ["lat", "lon", "long"] and value is not None:
                if latlon_format.lower() == "dd":
                    value = f"{value:.6f}"
                else:
                    value = convert_position_float2str(value)
            if key in ["elev"] and value is not None:
                value = "{0:.3f}".format(value)
            if isinstance(value, list):
                value = ",".join(value)
            header_lines.append(f"\t{key.upper()}={value}\n")
        header_lines.append("\n")
        return header_lines

    def _validate_header_list(self, header_list: list[str] | None) -> list[str] | None:
        """
        Validate and normalize header list format.

        Cleans header list by removing quotes, extra whitespace, and ensuring
        proper key=value format for each line.

        Parameters
        ----------
        header_list : list of str or None
            Raw header lines to validate and normalize.

        Returns
        -------
        list of str or None
            Normalized header lines in 'key=value' format, or None if input is None.

        Examples
        --------
        >>> header = Header()
        >>> raw_list = ['  DATAID = "MT001" ', 'LAT=45.5', '', 'INVALID']
        >>> header._validate_header_list(raw_list)
        ['dataid=MT001', 'lat=45.5']

        >>> header._validate_header_list(None)
        None

        Notes
        -----
        - Empty lines are removed
        - Keys are converted to lowercase
        - Lines without '=' are skipped
        - Double quotes are removed from values

        """

        if header_list is None:
            logger.info("No header information to read")
            return None
        new_header_list = []
        for h_line in header_list:
            h_line = h_line.strip().replace('"', "")
            if len(h_line) > 1:
                h_list = h_line.split("=")
                if len(h_list) == 2:
                    key = h_list[0].strip().lower()
                    value = h_list[1].strip()
                    new_header_list.append("{0}={1}".format(key, value))
        return new_header_list
