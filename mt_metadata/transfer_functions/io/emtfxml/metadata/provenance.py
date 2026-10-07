# =====================================================
# Imports
# =====================================================
from typing import Annotated
from xml.etree import cElementTree as et

import numpy as np
import pandas as pd
from pydantic import Field, field_validator

from mt_metadata import __version__
from mt_metadata.base import MetadataBase
from mt_metadata.base.helpers import dict_to_xml, element_to_string
from mt_metadata.common.mttime import MTime
from mt_metadata.transfer_functions.io.emtfxml.metadata import helpers, Person


# =====================================================
class Provenance(MetadataBase):
    create_time: Annotated[
        MTime | str | float | int | np.datetime64 | pd.Timestamp,
        Field(
            default_factory=lambda: MTime(time_stamp=None),
            description="date and time the file was created",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["2020-02-08T12:23:40.324600+00:00"],
            },
        ),
    ]

    creating_application: Annotated[
        str,
        Field(
            default="mt_metadata",
            description="name of the application that created the XML file",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["EMTF File Conversion Utilities 4.0"],
            },
        ),
    ]

    creator: Annotated[
        Person,
        Field(
            default_factory=Person,  # type: ignore
            description="Person or group responsible for creating the data",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["Person(name='John Doe', email='john.doe@example.com')"],
            },
        ),
    ]
    submitter: Annotated[
        Person,
        Field(
            default_factory=Person,  # type: ignore
            description="Person or group responsible for submitting the data",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": [
                    "Person(name='Jane Smith', email='jane.smith@example.com')"
                ],
            },
        ),
    ]

    @field_validator("create_time", mode="before")
    @classmethod
    def validate_create_time(
        cls, field_value: MTime | float | int | np.datetime64 | pd.Timestamp | str
    ):
        if isinstance(field_value, MTime):
            return field_value
        return MTime(time_stamp=field_value)

    def read_dict(self, input_dict: dict) -> None:
        """
        Read the Provenance object from a dictionary.

        Parameters
        ----------
        input_dict : dict
            The input dictionary containing the Provenance data.
        """
        helpers._read_element(self, input_dict, "provenance")

    def to_xml(self, string: bool = False, required: bool = True) -> str | et.Element:
        """
        Convert the Provenance object to XML format.

        Parameters
        ----------
        string : bool, optional
            Whether to return the XML as a string, by default False
        required : bool, optional
            Whether all required fields must be present, by default True

        Returns
        -------
        str | et.Element
            The XML representation of the Provenance object
        """

        self.creating_application = f"mt_metadata {__version__}"
        self.create_time = MTime(time_stamp=None).now()

        element = dict_to_xml(self.to_dict(nested=True, required=required))
        if not string:
            return element
        else:
            return element_to_string(element)
