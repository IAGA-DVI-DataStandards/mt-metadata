# ===============================================================
# imports
# ===============================================================
from functools import lru_cache

import numpy as np
from loguru import logger
from pyproj import CRS

# ===============================================================


def convert_position_float2str(position: float | int) -> str:
    """
    Convert position float to a string in the format of DD:MM:SS.ms

    Parameters
    ----------
    position : float | int
        decimal degrees of latitude or longitude

    Returns
    -------
    str
        latitude or longitude in format of DD:MM:SS.ms
    """
    if not isinstance(position, (float, int)):
        msg = f"position must be a float or int, not {type(position)}"
        logger.error(msg)
        raise TypeError(msg)

    deg = int(position)
    sign = 1
    if deg < 0:
        sign = -1

    deg = abs(deg)
    minutes = (abs(position) - deg) * 60.0
    # need to round seconds to 4 decimal places otherwise machine precision
    # keeps the 60 second roll over and the string is incorrect.
    sec = np.round((minutes - int(minutes)) * 60.0, 4)
    if sec >= 60.0:
        minutes += 1
        sec = 0

    if int(minutes) == 60:
        deg += 1
        minutes = 0

    position_str = f"{sign * int(deg)}:{int(minutes):02.0f}:{sec:05.6f}"
    logger.debug(f"Converted {position} to {position_str}")

    return position_str


def convert_position_str2float(position_str: str) -> float:
    """
    Convert a position string in the format of DD:MM:SS to decimal degrees

    :param position: latitude or longitude om DD:MM:SS.ms
    :type position: float

    :returns: latitude or longitude as a float

    Parameters
    ----------
    position_str : str
        latitude or longitude om DD:MM:SS.ms

    Returns
    -------
    float
        latitude or longitude as a float

    Raises
    ------
    ValueError
        If position string cannot be converted to a float or
        if the format is incorrect.
    """

    if position_str in [None, "None"]:
        return 0.0

    p_list = position_str.split(":")
    if len(p_list) != 3:
        msg = f"{position_str} not correct format, should be DD:MM:SS"
        logger.error(msg)
        raise ValueError(msg)

    deg = float(p_list[0])
    minutes = _assert_minutes(float(p_list[1]))
    sec = _assert_seconds(float(p_list[2]))

    # get the sign of the position so that when all are added together the
    # position is in the correct place
    sign = 1
    if deg < 0:
        sign = -1

    position_value = sign * (abs(deg) + minutes / 60.0 + sec / 3600.0)

    logger.debug(f"Converted {position_str} to {position_value}")

    return position_value


def _assert_minutes(minutes: float | int) -> float:
    """
    Assert minutes are between 0 and 60

    Parameters
    ----------
    minutes : float | int
        number of minutesto be checked

    Returns
    -------
    float
        number of minutes if valid
    """
    if not 0 <= minutes < 60.0:
        msg = (
            f"minutes should be 0 < > 60, currently {minutes:.0f} "
            "conversion will account for non-uniform "
            "time. Be sure to check accuracy."
        )
        logger.warning(msg)

    return minutes


def _assert_seconds(seconds: float | int) -> float:
    """
    Assert seconds are between 0 and 60

    Parameters
    ----------
    seconds : float | int
        number of seconds to be checked

    Returns
    -------
    float
        number of seconds if valid
    """
    if not 0 <= seconds < 60.0:
        msg = (
            f"seconds should be 0 < > 60, currently {seconds:.0f} "
            "conversion will account for non-uniform "
            "time. Be sure to check accuracy."
        )
        logger.warning(msg)

    return seconds


def validate_position(value: str | float, position_type: str) -> float:
    """
    Validate position value (latitude or longitude) and convert to float.

    Parameters
    ----------
    value : str | float
        The position value to validate and convert.
    position_type : str
        The type of position ('latitude' or 'longitude').

    Returns
    -------
    float
        The validated and converted position value.

    Raises
    ------
    ValueError
        If the value is not a valid latitude or longitude.
    """
    if position_type not in ["latitude", "longitude"]:
        raise ValueError("position_type must be 'latitude' or 'longitude'")

    if value in [None, "None"]:
        return 0.0

    elif isinstance(value, str) and ":" in value:
        value = convert_position_str2float(value)
    else:
        try:
            value = float(value)
        except ValueError:
            raise ValueError("latitude and longitude must be float or str")

    if not (abs(value) <= 90) and position_type in ["latitude", "lat"]:
        raise ValueError("latitude must be between -90 and 90 degrees")
    if not (abs(value) <= 180) and position_type in ["longitude", "lon"]:
        raise ValueError("longitude must be between -180 and 180 degrees")
    return value


@lru_cache(maxsize=None)
def _crs_name(value: str | int) -> str:
    """
    Get the name of the CRS a datum describes, keeping the CRS built for each
    datum so the same one is not built over and over.

    Parameters
    ----------
    value : str | int
        The datum value to look up.

    Returns
    -------
    str
        The name of the CRS.
    """
    return CRS.from_user_input(value).name


def validate_datum(value: str | int) -> str:
    """
    Validate the datum value and convert it to the name of the CRS.

    Parameters
    ----------
    value : str | int
        The datum value to validate.

    Returns
    -------
    str
        The name of the CRS the datum describes.

    Raises
    ------
    ValueError
        If the value is not a valid CRS string or identifier.
    """
    try:
        if isinstance(value, (str, int)):
            return _crs_name(value)
        return CRS.from_user_input(value).name
    except Exception:
        raise ValueError(
            f"Invalid datum value: {value}. Must be a valid CRS string or identifier."
        )
