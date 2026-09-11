# =====================================================
# Imports
# =====================================================
from collections import OrderedDict
from typing import Annotated

from loguru import logger
from pydantic import computed_field, Field, field_validator, ValidationInfo

from mt_metadata.base import MetadataBase
from mt_metadata.common import (
    AuthorPerson,
    BasicLocationNoDatum,
    Citation,
    Comment,
    Copyright,
    Fdsn,
    FundingSource,
    TimePeriodDate,
)
from mt_metadata.common.list_dict import ListDict
from mt_metadata.timeseries import Station
from mt_metadata.timeseries.filters import (
    CoefficientFilter,
    FIRFilter,
    FrequencyResponseTableFilter,
    PoleZeroFilter,
    TimeDelayFilter,
)
from mt_metadata.utils.location_helpers import validate_datum

# =====================================================


class Survey(MetadataBase):
    id: Annotated[
        str,
        Field(
            default="",
            description="Alpha numeric ID that will be unique for archiving.",
            alias=None,
            pattern=r"^[a-zA-Z0-9_\- ]*$",  # Allow empty string (zero or more chars)
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["EMT20"],
            },
        ),
    ]

    comments: Annotated[
        Comment,
        Field(
            default_factory=lambda: Comment(),
            description="Any comments about the survey.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["long survey"],
            },
        ),
    ]

    datum: Annotated[
        str | int,
        Field(
            default="WGS 84",
            description="Datum of latitude and longitude coordinates. Should be a well-known datum, such as WGS84, and will be the reference datum for all locations.  This is important for the user, they need to make sure all coordinates in the survey and child items (i.e. stations, channels) are referenced to this datum.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["WGS 84"],
            },
        ),
    ]

    geographic_name: Annotated[
        str,
        Field(
            default="",
            description="Closest geographic reference to survey, usually a city but could be a landmark or some other common geographic reference point.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["Yukon"],
            },
        ),
    ]

    name: Annotated[
        str,
        Field(
            default="",
            description="Descriptive name of the survey.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["MT Characterization of Yukon Terrane"],
            },
        ),
    ]

    project: Annotated[
        str,
        Field(
            default="",
            description="Alpha numeric name for the project e.g USGS-GEOMAG.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["YUTOO"],
            },
        ),
    ]

    stations: Annotated[
        ListDict | list | dict | OrderedDict | tuple,
        Field(
            default_factory=ListDict,
            description="List of stations recorded in the survey.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["ListDict[Station(id=id)]"],
            },
        ),
    ]

    filters: Annotated[
        ListDict | list | dict | OrderedDict | tuple,
        Field(
            default_factory=ListDict,
            description="List of filters for channel responses.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["ListDict[Filter()]"],
            },
        ),
    ]

    summary: Annotated[
        str,
        Field(
            default="",
            description="Summary paragraph of survey including the purpose; difficulties; data quality; summary of outcomes if the data have been processed and modeled.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": [
                    "long project of characterizing mineral resources in Yukon"
                ],
            },
        ),
    ]

    time_period: Annotated[
        TimePeriodDate,
        Field(
            default_factory=TimePeriodDate,
            description="End date of the survey in UTC.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": [
                    "TimePeriodDate(start_date='2000-01-01', end_date='2000-01-31')"
                ],
            },
        ),
    ]

    fdsn: Annotated[
        Fdsn,
        Field(
            default_factory=Fdsn,
            description="FDSN web service information.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["Fdsn()"],
            },
        ),
    ]

    acquired_by: Annotated[
        AuthorPerson,
        Field(
            default_factory=AuthorPerson,
            description="Person or group that acquired the data.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["Person()"],
            },
        ),
    ]

    funding_source: Annotated[
        FundingSource,
        Field(
            default_factory=FundingSource,
            description="Funding source for the survey.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["FundingSource()"],
            },
        ),
    ]

    citation_dataset: Annotated[
        Citation,
        Field(
            default_factory=Citation,
            description="Citation for the dataset.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["Citation()"],
            },
        ),
    ]

    citation_journal: Annotated[
        Citation,
        Field(
            default_factory=Citation,
            description="Citation for the journal.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["Citation()"],
            },
        ),
    ]

    northwest_corner: Annotated[
        BasicLocationNoDatum,
        Field(
            default_factory=BasicLocationNoDatum,
            description="Northwest corner of the survey area.",
            alias=None,
            json_schema_extra={
                "units": "degrees",
                "required": False,
                "examples": ["BasicLocationNoDatum()"],
            },
        ),
    ]

    southeast_corner: Annotated[
        BasicLocationNoDatum,
        Field(
            default_factory=BasicLocationNoDatum,
            description="Southeast corner of the survey area.",
            alias=None,
            json_schema_extra={
                "units": "degrees",
                "required": False,
                "examples": ["BasicLocationNoDatum()"],
            },
        ),
    ]

    country: Annotated[
        list[str] | str | None,
        Field(
            default=None,
            description="Country where the survey was conducted.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["Canada"],
            },
        ),
    ]

    state: Annotated[
        list[str] | str | None,
        Field(
            default=None,
            description="State or province where the survey was conducted.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["Yukon"],
            },
        ),
    ]

    project_lead: Annotated[
        AuthorPerson,
        Field(
            default_factory=AuthorPerson,
            description="Person or group that led the project.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["Person()"],
            },
        ),
    ]

    release_license: Annotated[
        str,
        Field(
            default="CC-BY-4.0",
            description="Release license for the data.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["CC-BY-4.0"],
            },
        ),
    ]

    @field_validator("comments", mode="before")
    @classmethod
    def validate_comments(cls, value, info: ValidationInfo) -> Comment:
        if isinstance(value, str):
            return Comment(value=value)
        return value

    @field_validator("datum", mode="before")
    @classmethod
    def validate_datum(cls, value: str | int) -> str:
        """
        Validate the datum value and convert it to the appropriate enum type.
        """
        return validate_datum(value)

    @field_validator("release_license", mode="before")
    @classmethod
    def validate_release_license(cls, value: str, info: ValidationInfo) -> str:
        """
        Validate that the value is a valid license.
        """
        if isinstance(value, str):
            copyright_object = Copyright(release_license=value)
            return copyright_object.release_license

    @field_validator("country", "state", mode="before")
    @classmethod
    def validate_areas(cls, value) -> list[str]:
        """validate country and state to be a list"""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",")]
        elif isinstance(value, (list, tuple)):
            return list(value)
        elif value == None:
            return None
        else:
            raise TypeError(f"Cannot make a list from types {type(value)}.")

    @field_validator("stations", mode="before")
    @classmethod
    def validate_stations(cls, value, info: ValidationInfo) -> ListDict:
        if not isinstance(value, (list, tuple, dict, ListDict, OrderedDict)):
            msg = (
                "input stations must be an iterable, should be a list or dict "
                f"not {type(value)}"
            )
            logger.error(msg)
            raise TypeError(msg)

        fails = []
        stations = ListDict()
        if isinstance(value, (dict, ListDict, OrderedDict)):
            value_list = value.values()

        elif isinstance(value, (list, tuple)):
            value_list = value

        for ii, station_entry in enumerate(value_list):
            if isinstance(station_entry, (dict, OrderedDict)):
                try:
                    station = Station()
                    station.from_dict(station_entry)
                    stations.append(station)
                except KeyError:
                    msg = f"Item {ii} is not type(Station); type={type(station_entry)}"
                    fails.append(msg)
                    logger.error(msg)
            elif not isinstance(station_entry, (Station)):
                msg = f"Item {ii} is not type(Run); type={type(station_entry)}"
                fails.append(msg)
                logger.error(msg)
            else:
                stations.append(station_entry)
        if len(fails) > 0:
            raise TypeError("\n".join(fails))

        return stations

    @field_validator("filters", mode="before")
    @classmethod
    def validate_filters(
        cls, value: str | list | ListDict, info: ValidationInfo
    ) -> ListDict:
        """

        Parameters
        ----------
        value : _type_
            _description_
        info : ValidationInfo
            _description_

        Returns
        -------
        ListDict
            _description_
        """
        filters = ListDict()
        fails = []
        if value is None:
            return

        if isinstance(value, list):
            if len(value) > 0:
                for ff in value:
                    if isinstance(ff, (dict, OrderedDict, ListDict)):
                        f_type = ff["type"]
                        if f_type is None:
                            msg = (
                                "filter type is None do not know how to read the filter"
                            )
                            fails.append(msg)
                            logger.error(msg)
                        if f_type.lower() in ["zpk"]:
                            f = PoleZeroFilter()
                        elif f_type.lower() in ["coefficient"]:
                            f = CoefficientFilter()
                        elif f_type.lower() in ["time delay"]:
                            f = TimeDelayFilter()
                        elif f_type.lower() in ["fir"]:
                            f = FIRFilter()
                        elif f_type.lower() in ["frequency response table"]:
                            f = FrequencyResponseTableFilter()
                        else:
                            msg = f"filter type {f_type} not supported."
                            fails.append(msg)
                            logger.error(msg)

                        f.from_dict(ff)
                        filters[f.name] = f
                    elif isinstance(
                        ff,
                        (
                            PoleZeroFilter,
                            CoefficientFilter,
                            FrequencyResponseTableFilter,
                            TimeDelayFilter,
                            FIRFilter,
                        ),
                    ):
                        filters[ff.name] = ff
                    else:
                        msg = f"Item {ff} is not Filter type; type={type(ff)}"
                        fails.append(msg)
                        logger.error(msg)

        elif not isinstance(value, (dict, OrderedDict, ListDict)):
            msg = (
                "Filters must be a dictionary with keys = names of filters, "
                f"not {type(value)}"
            )
            logger.error(msg)
            raise TypeError(msg)
        else:
            for k, v in value.items():
                if not isinstance(
                    v,
                    (
                        PoleZeroFilter,
                        CoefficientFilter,
                        TimeDelayFilter,
                        FrequencyResponseTableFilter,
                        FIRFilter,
                    ),
                ):
                    msg = f"Item {k} is not Filter type; type={type(v)}"
                    fails.append(msg)
                    logger.error(msg)
                else:
                    filters[k.lower()] = v
        if len(fails) > 0:
            raise TypeError("\n".join(fails))

        return filters

    @computed_field
    @property
    def survey_extent(self) -> dict:
        """
        Return the survey extent as a dictionary with keys 'northwest' and 'southeast'.
        """
        return {
            "latitude": {
                "min": self.southeast_corner.latitude,
                "max": self.northwest_corner.latitude,
            },
            "longitude": {
                "min": self.northwest_corner.longitude,
                "max": self.southeast_corner.longitude,
            },
        }

    def merge(self, other: "Survey", inplace=False) -> "Survey":
        """
        Merge surveys together using the original metadata but adding other's stations.

        Parameters
        ----------
        other : Survey
            Survey object
        inplace : bool, optional
            merge in place, by default False

        Returns
        -------
        Survey
            merged surveys

        Raises
        ------
        TypeError
            If items cannot be merged.
        """
        if isinstance(other, Survey):
            self.stations.extend(other.stations)
            self.update_all()
            if not inplace:
                return self
        else:
            msg = f"Can only merge Survey objects, not {type(other)}"
            logger.error(msg)
            raise TypeError(msg)

    @property
    def n_stations(self) -> int:
        """
        Return the number of stations in the station.

        :return: number of runs in the station
        :rtype: int

        """
        return len(self.stations)

    @property
    def station_names(self):
        """Return names of station in survey"""
        return self.stations.keys()

    @property
    def filter_names(self):
        """return a list of filter names"""
        return list(self.filters.keys())

    def has_station(self, station_id):
        """
        Has station id

        :param station_id: station id verbatim
        :type station_id: string
        :return: True if exists or False if not
        :rtype: boolean

        """
        if station_id in self.station_names:
            return True
        return False

    def station_index(self, station_id):
        """
        Get station index

        :param station_id: station id verbatim
        :type station_id: string
        :return: index value if station is found
        :rtype: integer

        """

        if self.has_station(station_id):
            return self.station_names.index(station_id)
        return None

    def add_station(self, station_obj, update=True):
        """
        Add a station, if has the same name update that object.

        :param station_obj: station object to add
        :type station_obj: `:class:`mt_metadata.timeseries.Station`

        """

        if not isinstance(station_obj, Station):
            raise TypeError(
                f"Input must be a mt_metadata.timeseries.Station object not {type(station_obj)}"
            )

        if self.has_station(station_obj.id):
            self.stations[station_obj.id].update(station_obj)
            logger.warning(
                f"Station {station_obj.id} already exists, updating metadata"
            )
        else:
            self.stations.append(station_obj)

        if update:
            self.update_bounding_box()
            self.update_time_period()
            self.update_station_keys()

    def get_station(self, station_id):
        """
        Get a station from the station id

        :param station_id: station id verbatim
        :type station_id: string
        :return: station object
        :rtype: :class:`mt_metadata.timeseries.Station`

        """

        if self.has_station(station_id):
            return self.stations[station_id]
        else:
            logger.warning(f"Could not find station {station_id}")
            return None

    def remove_station(self, station_id, update=True):
        """
        remove a station from the survey

        :param station_id: station id verbatim
        :type station_id: string

        """

        if self.has_station(station_id):
            self.stations.remove(station_id)
            if update:
                self.update_bounding_box()
                self.update_time_period()
        else:
            logger.warning(f"Could not find {station_id} to remove.")

    def update_station_keys(self):
        """
        Update the keys in the stations ListDict to match current station IDs.

        This is useful when station IDs have been modified after stations were
        added to the survey, ensuring that stations can be accessed by their
        current ID values.

        :returns: mapping of old keys to new keys
        :rtype: dict

        Example:
            >>> survey = Survey()
            >>> station = Station()
            >>> station.id = ""  # empty ID initially
            >>> survey.add_station(station)
            >>> station.id = "MT001"  # update the ID
            >>> key_mapping = survey.update_station_keys()
            >>> print(key_mapping)  # {'': 'MT001'}
            >>> # Now station can be accessed as survey.stations['MT001']
        """
        return self.stations.update_keys()

    def update_bounding_box(self):
        """
        Update the bounding box of the survey from the station information

        """
        if self.n_stations > 0:
            lat = []
            lon = []
            for station in self.stations:
                if station.location.latitude is not None:
                    lat.append(station.location.latitude)
                if station.location.longitude is not None:
                    lon.append(station.location.longitude)

            if not len(lat) == 0:
                self.southeast_corner.latitude = min(lat)
                self.northwest_corner.latitude = max(lat)
            if not len(lon) == 0:
                self.southeast_corner.longitude = max(lon)
                self.northwest_corner.longitude = min(lon)

    def update_time_period(self):
        """
        Update the start and end time of the survey based on the stations
        """
        if self.__len__() > 0:
            start = []
            end = []
            for station in self.stations:
                if not station.time_period.start_is_default():
                    start.append(station.time_period.start)
                if not station.time_period.end_is_default():
                    end.append(station.time_period.end)

            if start:
                if self.time_period.start_is_default():
                    self.time_period.start_date = min(start)
                else:
                    if self.time_period.start_date > min(start):
                        self.time_period.start_date = min(start)

            if end:
                if self.time_period.end_is_default():
                    self.time_period.end_date = max(end)
                else:
                    if self.time_period.end_date < max(end):
                        self.time_period.end_date = max(end)

    def update_all(self):
        """
        Update time period and bounding box
        """
        self.update_time_period()
        self.update_bounding_box()
