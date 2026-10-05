# =====================================================
# Imports
# =====================================================
from typing import Annotated
from xml.etree import ElementTree as et

import numpy as np
import pandas as pd
from pydantic import Field, field_validator

from mt_metadata.common import Software
from mt_metadata.common.mttime import MTime
from mt_metadata.transfer_functions.io.emtfxml.metadata import helpers


# =====================================================
class ProcessingSoftware(Software):
    last_mod: Annotated[
        MTime | str | float | int | np.datetime64 | pd.Timestamp | None,
        Field(
            default_factory=lambda: MTime(time_stamp=None),
            description="Date the software was last modified",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["2020-01-01"],
            },
        ),
    ]

    @field_validator("last_mod", mode="before")
    @classmethod
    def validate_last_mod(
        cls, field_value: MTime | float | int | np.datetime64 | pd.Timestamp | str
    ):
        if isinstance(field_value, MTime):
            return field_value.date
        return MTime(time_stamp=field_value).date

    def read_dict(self, input_dict: dict) -> None:
        """
        Read processing software information from a dictionary.

        Parameters
        ----------
        input_dict : dict
            A dictionary containing processing software information.
        """
        helpers._read_element(self, input_dict, "processing_software")

    def to_xml(self, string: bool = False, required: bool = True) -> str | et.Element:
        """Convert the processing software information to XML format.

        Parameters
        ----------
        string : bool, optional
            If True, return the XML as a string. If False, return an ElementTree element.
        required : bool, optional
            If True, include all required fields in the XML.

        Returns
        -------
        str | et.Element
            The XML representation of the processing software information.
        """

        return helpers.to_xml(
            self,
            string=string,
            required=required,
            order=["name", "last_mod", "author"],
        )
