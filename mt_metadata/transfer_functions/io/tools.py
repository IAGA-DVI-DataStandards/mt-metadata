# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 17:44:51 2021

@author: jpeacock
"""

import json

# =============================================================================
# imports
# =============================================================================
import urllib.request as url_request
from functools import lru_cache

from loguru import logger

# =============================================================================


def _validate_str_with_equals(input_string):
    """
    make sure an input string is of the format {0}={1} {2}={3} {4}={5} ...
    Some software programs put spaces after the equals sign and that's not
    cool.  So we make the string into a readable format

    :param input_string: input string from an edi file
    :type input_string: string

    :returns line_list: list of lines as ['key_00=value_00',
                                          'key_01=value_01']
    :rtype line_list: list
    """
    input_string = input_string.strip()
    # remove the first >XXXXX
    if ">" in input_string:
        input_string = input_string[input_string.find(" ") :]

    # check if there is a // at the end of the line
    if input_string.find("//") > 0:
        input_string = input_string[0 : input_string.find("//")]

    # split the line by =
    l_list = input_string.strip().split("=")

    # split the remaining strings
    str_list = []
    for line in l_list:
        s_list = line.strip().split()
        for l_str in s_list:
            str_list.append(l_str.strip())

    # probably not a good return
    if len(str_list) % 2 != 0:
        # _logger.info(
        #     'The number of entries in {0} is not even'.format(str_list))
        return str_list

    line_list = [
        "{0}={1}".format(str_list[ii], str_list[ii + 1])
        for ii in range(0, len(str_list), 2)
    ]

    return line_list


# ==============================================================================
# Index finder
# ==============================================================================
class index_locator(object):
    def __init__(self, component_list):
        self.ex = None
        self.ey = None
        self.hx = None
        self.hy = None
        self.hz = None
        self.rhx = None
        self.rhy = None
        self.rhz = None
        for ii, comp in enumerate(component_list):
            setattr(self, comp, ii)
        if self.rhx is None:
            self.rhx = self.hx
        if self.rhy is None:
            self.rhy = self.hy

    def __str__(self):
        lines = ["Index Values"]
        for k, v in self.__dict__.items():
            if v is not None:
                lines.append(f"\t{k} = {v}")
        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

    @property
    def n_channels(self):
        count = 0
        for k, v in self.__dict__.items():
            if "r" in k:
                continue
            if v is not None:
                count += 1

        return count

    @property
    def has_tipper(self):
        if self.hz is not None:
            return True
        return False

    @property
    def has_electric(self):
        if self.ex != None or self.ey != None:
            return True
        return False

    @property
    def input_channels(self):
        return [self.hx, self.hy]

    @property
    def output_channels(self):
        if self.has_tipper:
            if self.has_electric:
                return [self.hz, self.ex, self.ey]
            return [self.hz]
        return [self.ex, self.ey]

    @property
    def n_inputs(self):
        return len(self.input_channels)

    @property
    def n_outputs(self):
        return len(self.output_channels)


def _validate_edi_lines(edi_lines) -> list[str]:
    """
    check for carriage returns or hard returns

    :param edi_lines: list of edi lines
    :type edi_lines: list

    :returns: list of edi lines
    :rtype: list
    """

    if len(edi_lines) == 1:
        edi_lines = (
            edi_lines[0]
            .replace('"', "")
            .replace("\r", "\n")
            .replace("'", "")
            .replace("[", "")
            .replace("]", "")
            .split("\n")
        )
        if len(edi_lines) > 1:
            return edi_lines
        else:
            raise ValueError("*** EDI format not correct check file ***")
    else:
        return [
            line.replace('"', "").replace("'", "").replace("[", "").replace("]", "")
            for line in edi_lines
        ]


def get_nm_elev(latitude, longitude):
    """
    Get national map elevation for a given lat and lon.

    Queries the national map website for the elevation value.

    :param lat: latitude in decimal degrees
    :type lat: float

    :param lon: longitude in decimal degrees
    :type lon: float

    :return: elevation (meters)
    :rtype: float

    :Example: ::

        >>> import mtpy.usgs.usgs_archive as archive
        >>> archive.get_nm_elev(35.467, -115.3355)
        >>> 809.12

    .. note:: Needs an internet connection to work.

    """
    # the service only holds the US 3DEP model, so a point outside its
    # footprint is a wasted round trip
    if not (17.0 <= latitude <= 72.0 and (longitude <= -65.0 or longitude >= 172.0)):
        logger.debug(
            f"({latitude}, {longitude}) is outside the US National Map, "
            "skipping elevation request."
        )
        return 0.0
    return _request_nm_elev(round(latitude, 5), round(longitude, 5))


@lru_cache(maxsize=256)
def _request_nm_elev(latitude, longitude):
    """Query the national map for one point, cached per rounded coordinate."""
    nm_url = (
        r"https://epqs.nationalmap.gov/v1/json?"
        f"x={longitude}&y={latitude}&units=Meters&wkid=4326&includeDate=False"
    )
    # call the url and get the response
    try:
        response = url_request.urlopen(nm_url, timeout=5)
    except Exception:
        logger.error("Could not connect to internet to get elevation data.")
        return 0.0

    # read the xml response and convert to a float
    try:
        # unpack the response to a dictionary
        info = json.loads(response.read())

    except json.JSONDecodeError:
        logger.error(
            f"Input values (latitude={latitude}, longitude={longitude}) "
            "could not be found on US National Map."
        )
        return 0.0
    try:
        return float(info["value"])
    except KeyError:
        logger.warning("Could not find elevation data")
        return 0.0
    except ValueError:
        logger.warning(f"Could not convert elevation {info['value']} to float")
        return 0.0
