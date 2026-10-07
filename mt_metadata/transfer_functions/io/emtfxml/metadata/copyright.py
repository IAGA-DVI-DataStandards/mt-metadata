# =====================================================
# Imports
# =====================================================
from typing import Annotated

from pydantic import Field, field_validator

from mt_metadata.base import MetadataBase
from mt_metadata.common.enumerations import ReleaseStatusEnum
from mt_metadata.transfer_functions.io.emtfxml.metadata import helpers

from . import Citation

# =====================================================


class Copyright(MetadataBase):
    citation: Annotated[
        Citation,
        Field(
            default_factory=Citation,  # type: ignore
            description="The citation information for the data",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": [
                    "Citation(authors='Doe, J.', year='2023', title='Title of the paper', journal='Journal Name', volume='45', pages='123-145')"
                ],
            },
        ),
    ]
    selected_publications: Annotated[
        str | None,
        Field(
            default=None,
            description="Any publications that use this data",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["my paper"],
            },
        ),
    ]

    release_status: Annotated[
        ReleaseStatusEnum,
        Field(
            default="Unrestricted Release",
            description="the release status of the data",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["Unrestricted release"],
            },
        ),
    ]

    conditions_of_use: Annotated[
        str,
        Field(
            default="All data and metadata for this survey are available free of charge and may be copied freely, duplicated and further distributed provided this data set is cited as the reference. While the author(s) strive to provide data and metadata of best possible quality, neither the author(s) of this data set, not Earthscope make any claims, promises, or guarantees about the accuracy, completeness, or adequacy of this information, and expressly disclaim liability for errors and omissions in the contents of this file. Guidelines about the quality or limitations of the data and metadata, as obtained from the author(s), are included for informational purposes only.",
            description="Any notes on conditions of use",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["Cite data upon usage."],
            },
        ),
    ]

    acknowledgement: Annotated[
        str | None,
        Field(
            default=None,
            description="any acknowledgments the transfer function should have.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["This project was funded by x."],
            },
        ),
    ]

    additional_info: Annotated[
        str | None,
        Field(
            default=None,
            description="any additional information about the data.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["This purpose of this project is ..."],
            },
        ),
    ]

    @field_validator("citation", mode="before")
    @classmethod
    def validate_citation(cls, value) -> Citation:
        """Ensure citation is an instance of Citation, an empty element is None."""
        if value is None:
            return Citation()  # type: ignore
        return value

    def read_dict(self, input_dict):
        """

        :param input_dict: DESCRIPTION
        :type input_dict: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        helpers._read_element(self, input_dict, "copyright")

    def to_xml(self, string=False, required=True):
        """ """
        # Create a shallow copy to avoid mutating the original object
        import copy

        xml_copy = copy.copy(self)
        # Set the title-cased release_status on the copy using object.__setattr__
        # to bypass Pydantic validation
        object.__setattr__(xml_copy, "release_status", self.release_status.title())

        return helpers.to_xml(
            xml_copy,
            string=string,
            required=required,
            order=[
                "citation",
                "selected_publications",
                "acknowledgement",
                "release_status",
                "conditions_of_use",
                "additional_info",
            ],
        )
