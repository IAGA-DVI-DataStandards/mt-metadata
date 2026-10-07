# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 21:30:36 2020

:copyright:
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""

from typing import Annotated

# =============================================================================
# Imports
# =============================================================================
from xml.etree import cElementTree as et

from pydantic import Field

from mt_metadata import NULL_VALUES
from mt_metadata.base.helpers import element_to_string
from mt_metadata.common import Instrument as CommonInstrument


# =============================================================================
class Instrument(CommonInstrument):
    settings: Annotated[
        str | None,
        Field(
            default=None,
            description="Settings for the instrument, such as configuration or calibration details.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["calibration settings", "configuration details"],
            },
        ),
    ] = None

    def to_xml(self, string: bool = False, required: bool = False) -> str | et.Element:
        """
        Convert the Instrument object to an XML element or string.
        :param string: If True, return as a string; otherwise, return as an XML element.
        :type string: bool, optional
        :param required: If True, include only required fields; otherwise, include all fields.
        :type required: bool, optional
        :return: XML representation of the Instrument object.
        :rtype: str | et.Element

        """

        root = et.Element(self.__class__.__name__)
        if self.type not in [None, ""]:
            root.attrib["type"] = self.type

        for key in ["manufacturer", "name", "id"]:
            value = getattr(self, key)
            if value not in NULL_VALUES:
                et.SubElement(root, key).text = value

        et.SubElement(root, "settings").text = getattr(self, "settings")

        if string:
            return element_to_string(root)
        return root
