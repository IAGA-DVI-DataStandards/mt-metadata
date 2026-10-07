# =====================================================
# Imports
# =====================================================
from typing import Annotated
from xml.etree import ElementTree as et

from pydantic import (
    AliasChoices,
    AnyUrl,
    EmailStr,
    Field,
    field_validator,
    ValidationInfo,
)

from mt_metadata import NULL_VALUES
from mt_metadata.base import MetadataBase
from mt_metadata.common import Comment
from mt_metadata.transfer_functions.io.emtfxml.metadata import helpers


# =====================================================
class Person(MetadataBase):
    name: Annotated[
        str | None,
        Field(
            default="",
            description="Persons name, should be full first and last name.",
            validation_alias=AliasChoices("name", "author"),
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["person name"],
            },
        ),
    ]
    org: Annotated[
        str | None,
        Field(
            default=None,
            description="Organization full name",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["mt gurus"],
            },
        ),
    ]

    email: Annotated[
        EmailStr | AnyUrl | None,
        Field(
            default=None,
            description="Email of the contact person",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["mt.guru@em.org"],
            },
        ),
    ]

    org_url: Annotated[
        AnyUrl | None | str,
        Field(
            default=None,
            description="URL of the contact person",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["https://em.org"],
            },
        ),
    ]

    comments: Annotated[
        Comment,
        Field(
            default_factory=Comment,  # type: ignore[return-value]
            description="Any comments about the person",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["expert digger"],
            },
        ),
    ]

    @field_validator("comments", mode="before")
    @classmethod
    def validate_comments(cls, value, info: ValidationInfo) -> Comment:
        """
        Validate that the value is a valid comment.
        """
        if isinstance(value, str | None):
            return Comment(value=value)  # type: ignore[return-value]
        return value

    @field_validator("org_url", mode="before")
    @classmethod
    def validate_url(
        cls, value: AnyUrl | None | str, info: ValidationInfo
    ) -> AnyUrl | None:
        """
        Validate that the value is a valid URL.
        """
        if isinstance(value, str):
            if value in NULL_VALUES:
                return None
            return AnyUrl(value)
        elif isinstance(value, AnyUrl):
            return value
        elif value is None:
            return None

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
            order=["name", "email", "org", "org_url"],
        )
