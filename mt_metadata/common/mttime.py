# -*- coding: utf-8 -*-
"""
Created on Wed May 13 19:10:46 2020

For dealing with obsy.core.UTCDatetime and an soft requirement imports
have a look at https://github.com/pydantic/pydantic/discussions/3673.

@author: jpeacock
"""

import datetime

# =============================================================================
# IMPORTS
# =============================================================================
from typing import Annotated, Optional

import numpy as np
import pandas as pd
from dateutil.parser import parse as dtparser
from loguru import logger
from pandas._libs.tslibs import OutOfBoundsDatetime


try:
    from obspy.core.utcdatetime import UTCDateTime  # for type hinting

    from_obspy = True
except ImportError:
    from_obspy = False

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    PrivateAttr,
    ValidationInfo,
)


# =============================================================================
#  Get leap seconds
# =============================================================================
leap_second_dict = {
    0: {"min": datetime.date(1980, 1, 1), "max": datetime.date(1981, 7, 1)},
    1: {"min": datetime.date(1981, 7, 1), "max": datetime.date(1982, 7, 1)},
    2: {"min": datetime.date(1982, 7, 1), "max": datetime.date(1983, 7, 1)},
    3: {"min": datetime.date(1983, 7, 1), "max": datetime.date(1985, 7, 1)},
    4: {"min": datetime.date(1985, 7, 1), "max": datetime.date(1988, 1, 1)},
    5: {"min": datetime.date(1988, 1, 1), "max": datetime.date(1990, 1, 1)},
    6: {"min": datetime.date(1990, 1, 1), "max": datetime.date(1991, 1, 1)},
    7: {"min": datetime.date(1991, 1, 1), "max": datetime.date(1992, 7, 1)},
    8: {"min": datetime.date(1992, 7, 1), "max": datetime.date(1993, 7, 1)},
    9: {"min": datetime.date(1993, 7, 1), "max": datetime.date(1994, 7, 1)},
    10: {"min": datetime.date(1994, 7, 1), "max": datetime.date(1996, 1, 1)},
    11: {"min": datetime.date(1996, 1, 1), "max": datetime.date(1997, 7, 1)},
    12: {"min": datetime.date(1997, 7, 1), "max": datetime.date(1999, 1, 1)},
    13: {"min": datetime.date(1999, 1, 1), "max": datetime.date(2006, 1, 1)},
    14: {"min": datetime.date(2006, 1, 1), "max": datetime.date(2009, 1, 1)},
    15: {"min": datetime.date(2009, 1, 1), "max": datetime.date(2012, 6, 30)},
    16: {"min": datetime.date(2012, 7, 1), "max": datetime.date(2015, 7, 1)},
    17: {"min": datetime.date(2015, 7, 1), "max": datetime.date(2016, 12, 31)},
    18: {"min": datetime.date(2017, 1, 1), "max": datetime.date(2026, 7, 1)},
}


def calculate_leap_seconds(year: int, month: int, day: int) -> int:
    """
    Get the leap seconds for the given year to convert GPS time to UTC time.

    GPS time started in 1980. GPS time is leap seconds ahead of UTC time,
    therefore you should subtract leap seconds from GPS time to get UTC time.

    Parameters
    ----------
    year : int
        Year of the date.
    month : int
        Month of the date (1-12).
    day : int
        Day of the date (1-31).

    Returns
    -------
    int
        Number of leap seconds for the given date.

    Raises
    ------
    ValueError
        If the date is outside the defined leap second range
        (1981-07-01 to 2026-07-01).

    Notes
    -----
    Leap seconds are defined for the following date ranges:

    =========================== ===============================================
    Date Range                  Leap Seconds
    =========================== ===============================================
    1981-07-01 - 1982-07-01     1
    1982-07-01 - 1983-07-01     2
    1983-07-01 - 1985-07-01     3
    1985-07-01 - 1988-01-01     4
    1988-01-01 - 1990-01-01     5
    1990-01-01 - 1991-01-01     6
    1991-01-01 - 1992-07-01     7
    1992-07-01 - 1993-07-01     8
    1993-07-01 - 1994-07-01     9
    1994-07-01 - 1996-01-01     10
    1996-01-01 - 1997-07-01     11
    1997-07-01 - 1999-01-01     12
    1999-01-01 - 2006-01-01     13
    2006-01-01 - 2009-01-01     14
    2009-01-01 - 2012-07-01     15
    2012-07-01 - 2015-07-01     16
    2015-07-01 - 2017-01-01     17
    2017-01-01 - ????-??-??     18
    =========================== ===============================================

    """

    # make the date a datetime object, easier to test
    given_date = datetime.date(int(year), int(month), int(day))

    # made an executive decision that the date can be equal to the min, but
    # not the max, otherwise get an error.
    for leap_key in sorted(leap_second_dict.keys()):
        if (
            given_date < leap_second_dict[leap_key]["max"]
            and given_date >= leap_second_dict[leap_key]["min"]
        ):
            return int(leap_key)

    raise ValueError(
        f"Leap seconds not defined for date {year}-{month:02d}-{day:02d}. "
        "Leap seconds are defined from 1981-07-01 to 2026-07-01."
    )


# =============================================================================
#  Functions for parsing time stamps
# =============================================================================
def _localize_utc(stamp: pd.Timestamp) -> pd.Timestamp:
    """
    Localize a timestamp to UTC timezone.

    Forces a timestamp to have a timezone of UTC. If the timestamp is
    timezone-naive, it will be localized to UTC.

    Parameters
    ----------
    stamp : pd.Timestamp
        Input timestamp to be localized.

    Returns
    -------
    pd.Timestamp
        Timestamp with UTC timezone.
    """

    # check time zone and enforce UTC
    if stamp.tz is None:
        stamp = stamp.tz_localize("UTC").tz_convert("UTC")

    return stamp


# minimum and maximum time stamps for pandas
# these are the minimum and maximum time stamps for pandas
TMIN = _localize_utc(pd.Timestamp.min)
TMAX = _localize_utc(pd.Timestamp.max)


def _parse_string(dt_str: str) -> datetime.datetime:
    """
    Parse a datetime string with flexible day/month order handling.

    Attempts to parse a datetime string using dateutil.parser. If parsing
    fails due to 24-hour format (T24:00:00), it converts to T23:00:00 and
    adds one hour. Also tries day-first parsing as a fallback.

    Parameters
    ----------
    dt_str : str
        Datetime string to parse (e.g., "2020-01-15T12:30:45").

    Returns
    -------
    datetime.datetime
        Parsed datetime object.

    Raises
    ------
    ValueError
        If unable to parse the string using any of the attempted methods.
        Error message suggests proper formatting as YYYY-MM-DDThh:mm:ss.ns.
    """

    if dt_str in ["", "none", "None", "NONE", "null", "Null", "NULL"]:
        default_time = "1980-01-01T00:00:00+00:00"
        msg = f"Input is an null value {dt_str}, setting to default {default_time}."
        return dtparser(default_time)
    try:
        return dtparser(dt_str)
    except ValueError as ve:
        error_24h = "hour must be in 0..23"
        if error_24h in ve.args[0]:
            # hh=24 was supplied -- this is legal if it is midnight
            one_hour_earlier_dt_str = dt_str.replace("T24", "T23")
            one_hour_earlier_result = dtparser(one_hour_earlier_dt_str)
            result = one_hour_earlier_result + datetime.timedelta(hours=1)
            return result
        else:
            try:
                return dtparser(dt_str, dayfirst=True)
            except ValueError:
                msg = (
                    f"Could not parse string {dt_str} check formatting, "
                    "should be YYYY-MM-DDThh:mm:ss.ns"
                )
                logger.error(msg)
                raise ValueError(msg)


def _fix_out_of_bounds_time_stamp(dt):
    """
    Fix timestamps that are outside pandas Timestamp bounds.

    Checks if a datetime is outside the valid pandas timestamp range
    (approximately 1900-2200) and clamps it to the minimum or maximum
    allowed values.

    Parameters
    ----------
    dt : datetime.datetime
        Input datetime object to check.

    Returns
    -------
    stamp : pd.Timestamp
        Corrected timestamp within valid bounds.
    t_min_max : bool
        True if the timestamp was clamped to bounds, False otherwise.

    Notes
    -----
    If the year is greater than 2200, the timestamp is set to TMAX.
    If the year is less than 1900, the timestamp is set to TMIN.
    Otherwise, a UTC timezone is applied to the timestamp.
    """
    t_min_max = False

    if dt.year > 2200:
        logger.info(f"{dt} is too large setting to {TMAX}")
        stamp = TMAX
        t_min_max = True
    elif dt.year < 1900:
        logger.info(f"{dt} is too small setting to {TMIN}")
        stamp = TMIN
        t_min_max = True
    else:
        stamp = pd.Timestamp(dt, tz="UTC")

    return stamp, t_min_max


def _check_timestamp(pd_timestamp):
    """
    Check if timestamp is within allowed bounds and clamp if necessary.

    Verifies that a pandas timestamp is within the minimum and maximum
    allowed time range. If outside bounds, clamps to the nearest boundary.

    Parameters
    ----------
    pd_timestamp : pd.Timestamp
        Input timestamp to validate.

    Returns
    -------
    t_min_max : bool
        True if the timestamp was clamped to bounds, False otherwise.
    pd_timestamp : pd.Timestamp
        Validated timestamp, potentially clamped to bounds and
        localized to UTC.
    """

    t_min_max = False
    pd_timestamp = _localize_utc(pd_timestamp)

    if pd_timestamp <= TMIN:
        t_min_max = True
        pd_timestamp = TMIN
    elif pd_timestamp >= TMAX:
        t_min_max = True
        pd_timestamp = TMAX

    return t_min_max, pd_timestamp


def parse(
    dt_str: Optional[
        float | int | np.number | np.datetime64 | pd.Timestamp | str | dict
    ] = None,
    gps_time: bool = False,
) -> pd.Timestamp:
    """
    Parse a datetime input into a pandas Timestamp with UTC timezone.

    Accepts various input types and converts them to a standardized pandas
    Timestamp object. Handles special cases like GPS time conversion,
    nanosecond timestamps, and out-of-bounds dates.

    Parameters
    ----------
    dt_str : float, int, np.number, np.datetime64, pd.Timestamp, str, dict, or None, optional
        Input to be parsed. Can be:
        - None, empty string, or "none" variants: returns default 1980-01-01
        - float/int: interpreted as epoch seconds (or nanoseconds if large)
        - numpy numeric types: converted to standard Python types first
        - pd.Timestamp: validated and timezone-corrected
        - str: parsed using dateutil.parser with flexible formatting
        - dict: extracted time_stamp and gps_time fields
        Default is None.
    gps_time : bool, optional
        If True, converts GPS time to UTC by subtracting leap seconds.
        Default is False.

    Returns
    -------
    pd.Timestamp
        UTC-localized timestamp object.

    Raises
    ------
    ValueError
        If input is before GPS start time when gps_time=True.
        If string parsing fails with invalid format.

    Notes
    -----
    - Large numeric inputs (ratio > 1000 vs 3e8) are assumed to be nanoseconds
    - Timestamps outside pandas bounds are clamped to min/max values
    - GPS time conversion uses calculated leap seconds for the date
    - All outputs are forced to UTC timezone regardless of input timezone
    """
    t_min_max = False
    if dt_str in [None, "", "none", "None", "NONE", "Na", {}]:
        logger.debug("Time string is None, setting to 1980-01-01:00:00:00")
        stamp = pd.Timestamp("1980-01-01T00:00:00+00:00", tz="UTC")

    elif isinstance(dt_str, pd.Timestamp):
        t_min_max, stamp = _check_timestamp(dt_str)

    elif isinstance(dt_str, (float, int, np.number)):
        # Convert numpy numbers to standard Python float/int for consistent handling
        if isinstance(dt_str, np.number):
            dt_str = float(dt_str)

        # using 3E8 which is about the start of GPS time
        ratio = dt_str / 3e8
        if ratio < 1 and gps_time:
            raise ValueError(
                "Input is before GPS start time '1980/01/06', check value."
            )
        elif ratio > 1e3:
            dt_str = dt_str / 1e9
            logger.debug(
                "Assuming input float/int is in nanoseconds, converting to seconds."
            )
        # Use explicit UTC conversion to avoid local timezone issues.
        t_min_max, stamp = _check_timestamp(
            pd.Timestamp.fromtimestamp(dt_str, tz="UTC")
        )

    elif hasattr(dt_str, "isoformat"):
        try:
            t_min_max, stamp = _check_timestamp(pd.Timestamp(dt_str.isoformat()))
        except OutOfBoundsDatetime:
            stamp, t_min_max = _fix_out_of_bounds_time_stamp(
                _parse_string(dt_str.isoformat())
            )
    else:
        try:
            if isinstance(dt_str, dict):
                gps_time = dt_str.get("gps_time", gps_time)
                dt_str = dt_str.get("time_stamp", "1980-01-01T00:00:00+00:00")

            t_min_max, stamp = _check_timestamp(pd.Timestamp(dt_str))
        except (ValueError, TypeError, OutOfBoundsDatetime, OverflowError):
            dt = _parse_string(dt_str)
            stamp, t_min_max = _fix_out_of_bounds_time_stamp(dt)

    if isinstance(stamp, (type(pd.NaT), type(None))):
        logger.debug("Time string is None, setting to 1980-01-01:00:00:00")
        stamp = pd.Timestamp("1980-01-01T00:00:00+00:00", tz="UTC")

    # check time zone and enforce UTC
    stamp = _localize_utc(stamp)

    # there can be a machine round off error, if it is close to 1 round to
    # microseconds
    if round(stamp.nanosecond / 1000) == 1 and not t_min_max:
        stamp = stamp.round(freq="us")

    if gps_time:
        leap_seconds = calculate_leap_seconds(stamp.year, stamp.month, stamp.day)
        logger.debug("Converting GPS time to UTC with %s leap seconds", leap_seconds)
        stamp -= pd.Timedelta(seconds=leap_seconds)

    return _localize_utc(stamp)


# ==============================================================================
# convenience date-time container
# ==============================================================================
class MTime(BaseModel):
    """
    Date and time container based on pandas.Timestamp with UTC enforcement.

    A flexible datetime container that accepts various input formats and
    converts them to a UTC-localized pandas.Timestamp object. Provides
    convenient access to date/time components and handles nanosecond precision.

    Parameters
    ----------
    time_stamp : float, int, np.number, np.datetime64, pd.Timestamp, str, or None, optional
        Input timestamp in various formats:
        - float/int: epoch seconds
        - np.number: numpy numeric types (converted to Python types)
        - np.datetime64: numpy datetime
        - pd.Timestamp: pandas timestamp (will be UTC-localized)
        - str: ISO format or parseable date string
        - None: defaults to 1980-01-01T00:00:00+00:00
        Default is None.
    gps_time : bool, optional
        If True, interprets time_stamp as GPS time and converts to UTC.
        Default is False.

    Attributes
    ----------
    time_stamp : pd.Timestamp
        The stored timestamp, always UTC-localized.
    gps_time : bool
        Whether GPS time conversion was applied.

    Notes
    -----
    The pandas.Timestamp backend allows nanosecond precision timing.

    Input values outside pandas timestamp bounds are automatically clamped:
    - Values > 2200: set to pandas.Timestamp.max (2262-04-11 23:47:16.854775807)
    - Values < 1900: set to pandas.Timestamp.min (1677-09-21 00:12:43.145224193)

    All timestamps are forced to UTC timezone regardless of input timezone.

    Examples
    --------
    Create from various input types:

    >>> t = MTime()  # Default time
    >>> t.isoformat()
    '1980-01-01T00:00:00+00:00'

    >>> t = MTime(time_stamp="2020-01-15T12:30:45")
    >>> t.year
    2020

    >>> t = MTime(time_stamp=1579095045.0)  # Epoch seconds
    >>> t.isoformat()
    '2020-01-15T12:30:45+00:00'

    Access and modify components:

    >>> t.year = 2025
    >>> t.month = 12
    >>> t.day = 31
    >>> t.epoch_seconds
    1767225045.0
    """

    _default_time: str = PrivateAttr("1980-01-01T00:00:00+00:00")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    gps_time: Annotated[
        bool,
        Field(
            description="Defines if the time give in GPS time [True] or UTC [False]",
            default=False,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": [True, False],
            },
        ),
    ] = False

    time_stamp: Annotated[
        float | int | np.number | np.datetime64 | pd.Timestamp | str | None,
        Field(
            default_factory=lambda: pd.Timestamp(MTime._default_time.default),
            # default_factory=lambda: pd.Timestamp("1980-01-01T00:00:00+00:00"),
            description="Time in UTC format",
            examples=["1980-01-01T00:00:00+00:00"],
        ),
    ]

    @field_validator("time_stamp", mode="before")
    @classmethod
    def validate_time_stamp(
        cls,
        field_value: float | int | np.datetime64 | pd.Timestamp | str,
        validation_info: ValidationInfo,
    ) -> pd.Timestamp:
        """
        Validate and convert input timestamp to pandas Timestamp.

        Pydantic field validator that processes various timestamp input formats
        and converts them to a standardized UTC pandas Timestamp object.

        Parameters
        ----------
        field_value : float, int, np.datetime64, pd.Timestamp, str, or UTCDateTime
            Input timestamp value in any supported format.
        validation_info : ValidationInfo
            Pydantic validation context containing model data including gps_time setting.

        Returns
        -------
        pd.Timestamp
            UTC-localized timestamp object, clamped to pandas bounds if necessary.

        Notes
        -----
        This method is automatically called during model instantiation.
        GPS time conversion is applied if gps_time=True in the model data.
        Out-of-bounds timestamps are automatically clamped to valid ranges.
        """
        # Check if the time_stamp is a string and parse it
        return parse(field_value, gps_time=validation_info.data["gps_time"])

    @model_serializer
    def _serialize_model(self):
        """
        Custom serializer to handle pandas.Timestamp serialization.

        Returns
        -------
        dict
            Serialized model with Timestamp as ISO format string.
        """
        return {
            "time_stamp": (
                self.time_stamp.isoformat()
                if isinstance(self.time_stamp, pd.Timestamp)
                else self.time_stamp
            ),
            "gps_time": self.gps_time,
        }

    def __str__(self) -> str:
        """
        Represents the object as a string in ISO format.

        Returns
        -------
        str
            ISO formatted string of the time stamp.
        """
        return self.isoformat()

    def __repr__(self) -> str:
        """
        Represents the object as a string in ISO format.

        Returns
        -------
        str
            ISO formatted string of the time stamp.
        """
        return self.isoformat()

    def __eq__(
        self,
        other: float | int | np.datetime64 | pd.Timestamp | str,  # | UTCDateTime
    ) -> bool:
        """
        Checks if the time stamp is equal to another time stamp.
        This function is used to compare two time stamps and check if they are
        equal.

        The input will be parsed first into a pd.Timestamp object and then
        compared to the current time stamp.

        Parameters
        ----------
        other : float | int | np.datetime64 | pd.Timestamp | str | UTCDateTime
            other time stamp to compare to

        Returns
        -------
        bool
            if equal return True, otherwise False
        """
        if not isinstance(other, MTime):
            try:
                other = MTime(time_stamp=other)
            except Exception as e:
                logger.debug(f"Failed to convert {other} to MTime: {e}")
                return False

        epoch_seconds = bool(self.time_stamp.value == other.time_stamp.value)

        tz = bool(self.time_stamp.tz == other.time_stamp.tz)

        if epoch_seconds and tz:
            return True
        elif epoch_seconds and not tz:
            logger.info(
                f"Time zones are not equal {self.time_stamp.tz} != "
                f"{other.time_stamp.tz}"
            )
            return False
        elif not epoch_seconds:
            return False

    def __ne__(
        self,
        other: float | int | np.datetime64 | pd.Timestamp | str,  # | UTCDateTime
    ) -> bool:
        """
        Checks if the time stamp is not equal to another time stamp.

        Parameters
        ----------
        other : float | int | np.datetime64 | pd.Timestamp | str | UTCDateTime
            other time stamp to compare to

        Returns
        -------
        bool
            True if not equal, otherwise False
        """
        return not self.__eq__(other)

    def __lt__(
        self,
        other: float | int | np.datetime64 | pd.Timestamp | str,  # | UTCDateTime
    ) -> bool:
        """
        Checks if the other is less than the current time stamp.

        Parameters
        ----------
        other : float | int | np.datetime64 | pd.Timestamp | str | UTCDateTime
            other time stamp to compare to

        Returns
        -------
        bool
            _True if other is less than the current time stamp, otherwise False
        """
        if not isinstance(other, MTime):
            other = MTime(time_stamp=other)

        return bool(self.time_stamp < other.time_stamp)

    def __le__(
        self,
        other: float | int | np.datetime64 | pd.Timestamp | str,  # | UTCDateTime
    ) -> bool:
        """
        Checks if the other is less than or equal to the current time stamp.

        Parameters
        ----------
        other : float | int | np.datetime64 | pd.Timestamp | str | UTCDateTime
            other time stamp to compare to

        Returns
        -------
        _type_
            True if other is less than or equal to the current time stamp,
            otherwise False
        """
        if not isinstance(other, MTime):
            other = MTime(time_stamp=other)

        return bool(self.time_stamp <= other.time_stamp)

    def __gt__(
        self,
        other: float | int | np.datetime64 | pd.Timestamp | str,  # | UTCDateTime
    ) -> bool:
        """
        Checks if the other is greater than the current time stamp.

        Parameters
        ----------
        other : float | int | np.datetime64 | pd.Timestamp | str | UTCDateTime
            other time stamp to compare to

        Returns
        -------
        bool
            True if other is greater than the current time stamp, otherwise False
        """
        return not self.__lt__(other)

    def __ge__(
        self,
        other: float | int | np.datetime64 | pd.Timestamp | str,  # | UTCDateTime
    ) -> bool:
        """
        Checks if the other is greater than or equal to the current time stamp.

        Parameters
        ----------
        other : float | int | np.datetime64 | pd.Timestamp | str | UTCDateTime
            other time stamp to compare to

        Returns
        -------
        bool
            True if other is greater than or equal to the current time stamp,
            otherwise False
        """
        if not isinstance(other, MTime):
            other = MTime(time_stamp=other)

        return bool(self.time_stamp >= other.time_stamp)

    def __add__(
        self, other: int | float | datetime.timedelta | np.timedelta64
    ) -> "MTime":
        """
        Add time to the existing time stamp.  Must be a time delta object
        or a number in seconds.

        .. note:: Adding two time stamps does not make sense, use either
                 pd.Timedelta or seconds as a float or int.

        """
        if isinstance(other, (int, float)):
            other = pd.Timedelta(seconds=other)
            logger.debug("Assuming other time is in seconds")

        elif isinstance(other, (datetime.timedelta, np.timedelta64)):
            other = pd.Timedelta(other)

        if not isinstance(other, (pd.Timedelta)):
            msg = (
                "Adding times stamps does not make sense, use either "
                "pd.Timedelta or seconds as a float or int."
            )
            logger.error(msg)
            raise ValueError(msg)

        return MTime(time_stamp=self.time_stamp + other)

    def __sub__(
        self, other: int | float | datetime.timedelta | np.timedelta64
    ) -> "MTime":
        """
        Get the time difference between to times in seconds.

        :param other: other time value
        :type other: [ str | float | int | datetime.datetime | np.datetime64 ]
        :return: time difference in seconds
        :rtype: float

        """

        if isinstance(other, (int, float)):
            other = pd.Timedelta(seconds=other)
            logger.info("Assuming other time is in seconds and not epoch seconds.")

        elif isinstance(other, (datetime.timedelta, np.timedelta64)):
            other = pd.Timedelta(other)

        else:
            try:
                other = MTime(time_stamp=other)
            except ValueError as error:
                raise TypeError(error)

        if not isinstance(other, (pd.Timedelta, MTime)):
            msg = "Subtracting times must be either timedelta or another time."
            logger.error(msg)
            raise ValueError(msg)

        if isinstance(other, MTime):
            other = MTime(time_stamp=other)

            return (self.time_stamp - other.time_stamp).total_seconds()

        elif isinstance(other, pd.Timedelta):
            return MTime(time_stamp=self.time_stamp - other)

    def __hash__(self) -> int:
        return hash(self.isoformat())

    def is_default(self) -> bool:
        """
        Test if the time_stamp value is the default value
        """
        return self.time_stamp == pd.Timestamp(self._default_time)

    def to_dict(self, nested=False, single=False, required=True) -> str:
        """
        Convert the time stamp to a dictionary with the ISO format string.

        Returns
        -------
        str
            The ISO format string.
        """
        return self.isoformat()

    def from_dict(
        self,
        value: str | int | float | np.datetime64 | pd.Timestamp,
        skip_none=False,
    ) -> None:
        """
        This will have to accept just a single value, not a dict.
        This is to keep original functionality.

        Parameters
        ----------
        value : str | int | float | np.datetime64 | pd.Timestamp
            time stamp value
        """
        self.time_stamp = value

    @property
    def iso_str(self) -> str:
        """

        Returns
        -------
        str
            ISO formatted string of the time stamp.
        """
        logger.warning(
            "iso_str will be deprecated in the future. Use isoformat() instead"
        )
        return self.time_stamp.isoformat()

    @property
    def iso_no_tz(self) -> str:
        """
        ISO formatted string of the time stamp without the timezone.
        This is useful for storing the time stamp in a database or other
        format where the timezone is not needed.

        Returns
        -------
        str
            ISO formatted string of the time stamp without the timezone.
        """
        return self.time_stamp.isoformat().split("+", 1)[0]

    @property
    def epoch_seconds(self) -> float:
        """
        Epoch seconds of the time stamp.  This is the number of seconds
        since the epoch (1970-01-01 00:00:00 UTC).

        Returns
        -------
        float
            epoch seconds of the time stamp.
        """
        return self.time_stamp.timestamp()

    @epoch_seconds.setter
    def epoch_seconds(self, seconds: float | int) -> None:
        """
        Sets the time stamp to the given epoch seconds.  This is the number of
        seconds since the epoch (1970-01-01 00:00:00 UTC).

        Parameters
        ----------
        seconds : float | int
            epoch seconds for the time stamp.
        """
        logger.debug("reading time from epoch seconds, assuming UTC time zone")
        # has to be seconds
        self.time_stamp = pd.Timestamp(seconds, tz="UTC", unit="s")

    @property
    def date(self) -> str:
        """
        Date in ISO format.  This is the date part of the time stamp
        without the time part.  This is useful for storing the date in a
        database or other format where the time is not needed.
        The date is in the format YYYY-MM-DD.

        Returns
        -------
        str
            ISO formatted date string of the time stamp.
        """
        return self.time_stamp.date().isoformat()

    @property
    def year(self) -> int:
        """
        Year of the time stamp

        Returns
        -------
        int
            year of the time stamp
        """
        return self.time_stamp.year

    @year.setter
    def year(self, value: int) -> None:
        """
        Sets the year of the time stamp to the given value.  This is the
        year part of the time stamp.  This is useful for setting the year
        of the time stamp to a specific value.
        The year is in the format YYYY.

        Parameters
        ----------
        value : int
            New year value for the time stamp.
        """
        self.time_stamp = self.time_stamp.replace(year=value)

    @property
    def month(self) -> int:
        """
        Month of the time stamp. This is the month part of the time stamp
        without the time part.  This is useful for storing the month in a
        database or other format where the time is not needed.

        Returns
        -------
        int
            month of time stamp
        """
        return self.time_stamp.month

    @month.setter
    def month(self, value: int) -> None:
        """
        Sets the month of the time stamp to the given value.  This is the
        month part of the time stamp.  This is useful for setting the month


        Parameters
        ----------
        value : int
            new month value for the time stamp.
        """
        self.time_stamp = self.time_stamp.replace(month=value)

    @property
    def day(self) -> int:
        """
        Day of the time stamp. This is the day part of the time stamp
        without the time part.

        Returns
        -------
        int
            Day of the time stamp
        """
        return self.time_stamp.day

    @day.setter
    def day(self, value: int) -> None:
        """
        Sets the day of the time stamp to the given value.  This is the
        day part of the time stamp.

        Parameters
        ----------
        value : int
            new day of the time stamp
        """
        self.time_stamp = self.time_stamp.replace(day=value)

    @property
    def hour(self) -> int:
        """
        Hour of the time stamp.

        Returns
        -------
        int
            hour of the time stamp
        """
        return self.time_stamp.hour

    @hour.setter
    def hour(self, value: int) -> None:
        """
        Sets the hour of the time stamp to the given value.

        Parameters
        ----------
        value : int
            new hour of the time stamp
        """
        self.time_stamp = self.time_stamp.replace(hour=value)

    @property
    def minutes(self) -> int:
        return self.time_stamp.minute

    @minutes.setter
    def minutes(self, value: int) -> None:
        self.time_stamp = self.time_stamp.replace(minute=value)

    @property
    def seconds(self) -> int:
        return self.time_stamp.second

    @seconds.setter
    def seconds(self, value: int) -> None:
        self.time_stamp = self.time_stamp.replace(second=value)

    @property
    def microseconds(self) -> int:
        return self.time_stamp.microsecond

    @microseconds.setter
    def microseconds(self, value: int) -> None:
        self.time_stamp = self.time_stamp.replace(microsecond=value)

    @property
    def nanoseconds(self) -> int:
        return self.time_stamp.nanosecond

    @nanoseconds.setter
    def nanoseconds(self, value: int) -> None:
        self.time_stamp = self.time_stamp.replace(nanosecond=value)

    def now(self) -> "MTime":
        """
        The current time in UTC format.

        Returns
        -------
        MTime
            The current time as an MTime object.
        """
        self.time_stamp = pd.Timestamp.now(tz="UTC")

        return self

    def copy(self) -> "MTime":
        """make a copy of the time"""
        return self.model_copy(deep=True)

    def isoformat(self) -> str:
        """
        ISO formatted string of the time stamp.  This is the ISO format
        string of the time stamp.

        formatted as: YYYY-MM-DDThh:mm:ss.ssssss+00:00

        Returns
        -------
        str
            ISO formatted date time string
        """
        return self.time_stamp.isoformat()

    def isodate(self) -> str:
        """
        ISO formatted date string of the time stamp.  This is the ISO format
        string of the date part of the time stamp.

        formatted as: YYYY-MM-DD

        Returns
        -------
        str
            _description_
        """
        return self.time_stamp.date().isoformat()

    def isocalendar(self) -> str:
        """
        ISO formatted calendar string of the time stamp.  This is the ISO
        format string of the calendar part of the time stamp.

        Formatted as: YYYY-WW-D
        where YYYY is the year, WW is the week number, and D is the day of
        the week.

        Returns
        -------
        str
            ISO formatted calendar string of the time stamp.
        """
        return self.time_stamp.isocalendar()


def get_now_utc() -> "MTime":
    """
    Get the current UTC time as an MTime object.

    Creates an MTime instance set to the current UTC time.

    Returns
    -------
    str
        ISO format string of the current UTC time.

    Notes
    -----
    Despite the return type annotation suggesting MTime, this function
    actually returns an ISO format string from the MTime object.
    """

    m_obj = MTime()
    m_obj.now()
    return m_obj.isoformat()


class MDate(MTime):
    def __str__(self) -> str:
        """
        Represents the object as a string in ISO format.

        Returns
        -------
        str
            ISO formatted string of the time stamp.
        """
        return self.isodate()

    def __repr__(self) -> str:
        """
        Represents the object as a string in ISO format.

        Returns
        -------
        str
            ISO formatted string of the time stamp.
        """
        return self.isodate()

    def __add__(
        self, other: int | float | datetime.timedelta | np.timedelta64
    ) -> "MTime":
        """
        Add time to the existing time stamp.  Must be a time delta object
        or a number in seconds.

        .. note:: Adding two time stamps does not make sense, use either
                 pd.Timedelta or seconds as a float or int.

        """
        if isinstance(other, (int, float)):
            other = pd.Timedelta(days=other)
            logger.debug("Assuming other time is in days")

        elif isinstance(other, (datetime.timedelta, np.timedelta64)):
            other = pd.Timedelta(other)

        if not isinstance(other, (pd.Timedelta)):
            msg = (
                "Adding times stamps does not make sense, use either "
                "pd.Timedelta or seconds as a float or int."
            )
            logger.error(msg)
            raise ValueError(msg)

        return MDate(time_stamp=self.time_stamp + other)

    def __sub__(
        self, other: int | float | datetime.timedelta | np.timedelta64
    ) -> "MTime":
        """
        Get the time difference between to times in seconds.

        :param other: other time value
        :type other: [ str | float | int | datetime.datetime | np.datetime64 ]
        :return: time difference in seconds
        :rtype: float

        """

        if isinstance(other, (int, float)):
            other = pd.Timedelta(days=other)
            logger.info("Assuming other time is in seconds and not epoch seconds.")

        elif isinstance(other, (datetime.timedelta, np.timedelta64)):
            other = pd.Timedelta(other)

        else:
            try:
                other = MTime(time_stamp=other)
            except ValueError as error:
                raise TypeError(error)

        if not isinstance(other, (pd.Timedelta, MDate, MTime)):
            msg = "Subtracting times must be either timedelta or another time."
            logger.error(msg)
            raise ValueError(msg)

        if isinstance(other, (MDate, MTime)):
            other = MDate(time_stamp=other)

            return (self.time_stamp - other.time_stamp).total_seconds() / 86400

        elif isinstance(other, pd.Timedelta):
            return MDate(time_stamp=self.time_stamp - other)

    def __hash__(self) -> int:
        return hash(self.isodate())

    def is_default(self):
        """
        Test if time_stamp is the default value
        """
        return self.isoformat() == pd.Timestamp(self._default_time).date().isoformat()

    def isoformat(self) -> str:
        """
        ISO formatted string of the time stamp.  This is the ISO format
        string of the time stamp.

        formatted as: YYYY-MM-DDThh:mm:ss.ssssss+00:00

        Returns
        -------
        str
            ISO formatted date time string
        """
        return self.isodate()

    def to_dict(self, nested=False, single=False, required=True) -> str:
        """
        Convert the time stamp to a dictionary with the ISO format string.

        Returns
        -------
        str
            The ISO format string.
        """
        return self.isodate()

    def from_dict(
        self,
        value: str | int | float | np.datetime64 | pd.Timestamp,
        skip_none=False,
    ) -> None:
        """
        This will have to accept just a single value, not a dict.
        This is to keep original functionality.

        Parameters
        ----------
        value : str | int | float | np.datetime64 | pd.Timestamp
            time stamp value
        """
        self.time_stamp = value
