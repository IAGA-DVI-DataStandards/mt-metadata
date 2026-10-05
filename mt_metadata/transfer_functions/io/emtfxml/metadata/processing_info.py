# =====================================================
# Imports
# =====================================================
from typing import Annotated
from xml.etree import ElementTree as et

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import Field, field_validator

from mt_metadata.base import MetadataBase
from mt_metadata.common.enumerations import SignConventionEnum
from mt_metadata.common.mttime import MTime

from . import helpers, ProcessingSoftware, RemoteInfo, RemoteRef

# =====================================================


class ProcessingInfo(MetadataBase):
    sign_convention: Annotated[
        SignConventionEnum,
        Field(
            default=r"exp(+ i\omega t)",
            description="Sign convention of the processing software output",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["exp(+ i\\omega t)"],
            },
        ),
    ]

    processed_by: Annotated[
        str | None,
        Field(
            default=None,
            description="Names of people who processed the data",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["MT Guru"],
            },
        ),
    ]

    process_date: Annotated[
        MTime | str | float | int | np.datetime64 | pd.Timestamp | None,
        Field(
            default_factory=lambda: MTime(time_stamp=None),
            description="Date the data were processed",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["2020-01-01"],
            },
        ),
    ]

    processing_tag: Annotated[
        str | None,
        Field(
            default=None,
            description="List of remote references",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["mt001-mt002"],
            },
        ),
    ]

    processing_software: Annotated[
        ProcessingSoftware,
        Field(
            default_factory=lambda: ProcessingSoftware(),  # type: ignore
            description="Information about the processing software used",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": [
                    "ProcessingSoftware(name='MT Processing Software', version='1.0')"
                ],
            },
        ),
    ]

    remote_info: Annotated[
        RemoteInfo,
        Field(
            default_factory=RemoteInfo,  # type: ignore
            description="Information about remote data sources",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["RemoteInfo(name='MT Remote')"],
            },
        ),
    ]

    remote_ref: Annotated[
        RemoteRef,
        Field(
            default_factory=RemoteRef,  # type: ignore
            description="List of remote references",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": [["MT001a", "MT001b"]],
            },
        ),
    ]

    _order: list[str] = [
        "sign_convention",
        "remote_ref",
        "remote_info",
        "processed_by",
        "process_date",
        "processing_software",
        "processing_tag",
    ]

    @field_validator("process_date", mode="before")
    @classmethod
    def validate_process_date(
        cls, field_value: MTime | float | int | np.datetime64 | pd.Timestamp | str
    ):
        return MTime(time_stamp=field_value).date

    def read_dict(self, input_dict: dict) -> None:
        """
        Read processing information from a dictionary.

        Parameters
        ----------
        input_dict : dict
            A dictionary containing processing information.
        """
        try:
            processing_dict = input_dict["processing_info"]
        except KeyError:
            return

        if "field_notes" in processing_dict.keys():
            processing_dict.pop("field_notes")

        for key in ["remote_ref", "remote_info", "processing_software"]:
            try:
                pop_dict = {key: processing_dict.pop(key)}
                getattr(self, key).read_dict(pop_dict)
            except KeyError:
                logger.debug(f"No {key} information in xml.")

        helpers._read_element(self, input_dict, "processing_info")

    def to_xml(self, string: bool = False, required: bool = True) -> str | et.Element:
        """
        Convert the processing information to XML format.

        Parameters
        ----------
        string : bool, optional
            Whether to return the XML as a string, by default False
        required : bool, optional
            Whether the XML is required, by default True

        Returns
        -------
        str | et.Element
            The XML representation of the processing information
        """

        return helpers.to_xml(
            self,
            string=string,
            required=required,
            order=self._order,
        )
