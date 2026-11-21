import json
from typing import Annotated

from pydantic import Field, field_validator, model_validator

from mt_metadata.base import MetadataBase
from mt_metadata.common.enumerations import StrEnumerationBase


class PrewhiteningTypeEnum(StrEnumerationBase):
    none = ""
    first_difference = "first difference"
    arma = "ARMA"
    other = "other"


class Prewhitening(MetadataBase):
    """Configuration and applied-filter metadata for prewhitening."""

    type: Annotated[
        PrewhiteningTypeEnum,
        Field(
            default=PrewhiteningTypeEnum.first_difference,
            description="Prewhitening method to be applied",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["first difference"],
            },
        ),
    ]

    ar_order: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            description="Order of the autoregressive prewhitening model",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": [3],
            },
        ),
    ]

    ma_order: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            description="Order of the moving-average prewhitening model",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": [0],
            },
        ),
    ]

    fit_domain: Annotated[
        str,
        Field(
            default="",
            description="Time-series domain over which prewhitening was fit",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["full run"],
            },
        ),
    ]

    response_convention: Annotated[
        str,
        Field(
            default="",
            description="Discrete-time transfer-function convention for prewhitening",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["A(z) = 1 - sum(phi_k z^-k)"],
            },
        ),
    ]

    ar_coefficients: Annotated[
        list[float],
        Field(
            default_factory=list,
            description="Fitted autoregressive prewhitening coefficients",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": [[0.8, -0.2]],
            },
        ),
    ]

    ma_coefficients: Annotated[
        list[float],
        Field(
            default_factory=list,
            description="Fitted moving-average prewhitening coefficients",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": [[]],
            },
        ),
    ]

    @field_validator("ar_coefficients", "ma_coefficients", mode="before")
    @classmethod
    def validate_coefficients(cls, value):
        """Deserialize coefficient lists read from HDF5 attributes."""
        if isinstance(value, str):
            return json.loads(value)
        return value

    @model_validator(mode="after")
    def validate_arma_orders(self) -> "Prewhitening":
        """Validate the orders supported by the configured prewhitening type."""
        if self.type == PrewhiteningTypeEnum.arma and self.ar_order < 1:
            msg = "ARMA prewhitening requires ar_order to be at least 1"
            raise ValueError(msg)
        if self.type == PrewhiteningTypeEnum.arma and self.ma_order != 0:
            msg = "ARMA prewhitening currently supports only ma_order == 0"
            raise NotImplementedError(msg)
        return self
