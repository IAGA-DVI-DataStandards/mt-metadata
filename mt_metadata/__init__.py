# -*- coding: utf-8 -*-
"""
==================
metadata
==================

This module deals with metadata as defined by the MT metadata standards.

There are multiple containers for each type of metadata, named appropriately.

Each container will be able to read and write:
    * dictionary
    * json
    * xml
    * csv?
    * pandas.Series
    * anything else?


Each container has an attribute called _attr_dict which dictates if the
attribute is included in output objects, the data type, whether it is a
required parameter, and the style of output.  This should help down the road
with validation and keeping the data types consistent.  And if things change
you should only have to changes these dictionaries.

:copyright:
    Jared Peacock (jpeacock@usgs.gov)

:license:
    MIT


"""

# =============================================================================
# Package details
# =============================================================================


__author__ = """Jared Peacock"""
__email__ = "jpeacock@usgs.gov"
__version__ = "1.0.8"

# =============================================================================
# Imports
# =============================================================================
import sys
from pathlib import Path

from loguru import logger

# =============================================================================
# Global Variables
# =============================================================================


ACCEPTED_STYLES = [
    "name",
    "url",
    "email",
    "number",
    "date",
    "free form",
    "time",
    "date time",
    "name list",
    "number list",
    "controlled vocabulary",
    "alpha numeric",
]

REQUIRED_KEYS = [
    "attribute",
    "type",
    "required",
    "units",
    "style",
    "description",
    "options",
    "alias",
    "example",
    "default",
]

DEFAULT_CHANNEL_NOMENCLATURE = {
    "hx": "hx",
    "hy": "hy",
    "hz": "hz",
    "ex": "ex",
    "ey": "ey",
}

NULL_VALUES = [
    None,
    "",
    "null",
    "None",
    "NONE",
    "NULL",
    "Null",
    "none",
    "1980-01-01T00:00:00",
    "1980-01-01T00:00:00+00:00",
]

# =============================================================================
# Initiate loggers
# =============================================================================
config = {
    "handlers": [
        {
            "sink": sys.stdout,
            "level": "INFO",
            "colorize": True,
            "format": "<level>{time} | {level: <3} | {name} | {function} | line: {line} | {message}</level>",
        },
    ],
    "extra": {"user": "someone"},
}
logger.configure(**config)
# logger.disable("mt_metadata")


from mt_metadata.data import *
