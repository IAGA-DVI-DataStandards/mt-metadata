# =====================================================
# Imports
# =====================================================
from typing import Annotated

from pydantic import Field, field_validator

from mt_metadata.base import MetadataBase
from mt_metadata.common import TimePeriod
from mt_metadata.common.units import get_unit_object
from mt_metadata.processing.prewhitening import Prewhitening, PrewhiteningTypeEnum


# =====================================================
class FCChannel(MetadataBase):
    component: Annotated[
        str,
        Field(
            default="",
            description="Name of channel",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["ex"],
            },
        ),
    ]

    frequency_max: Annotated[
        float,
        Field(
            default=0.0,
            description="Highest frequency present in the sprectrogam data.",
            alias=None,
            json_schema_extra={
                "units": "samples per second",
                "required": True,
                "examples": [77.0],
            },
        ),
    ]

    frequency_min: Annotated[
        float,
        Field(
            default=0.0,
            description="Lowest frequency present in the sprectrogam data.",
            alias=None,
            json_schema_extra={
                "units": "samples per second",
                "required": True,
                "examples": [99.0],
            },
        ),
    ]

    sample_rate_decimation_level: Annotated[
        float,
        Field(
            default=1.0,
            description="Sample rate of the time series that was Fourier transformed to generate the FC decimation level.",
            alias=None,
            json_schema_extra={
                "units": "samples per second",
                "required": True,
                "examples": [60],
            },
        ),
    ]

    sample_rate_window_step: Annotated[
        float,
        Field(
            default=1.0,
            description="Sample rate of the windows.",
            alias=None,
            json_schema_extra={
                "units": "samples per second",
                "required": True,
                "examples": [4],
            },
        ),
    ]

    units: Annotated[
        str,
        Field(
            default="counts",
            description="Units of the channel",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["millivolts"],
            },
        ),
    ]

    time_period: Annotated[
        TimePeriod,
        Field(
            default_factory=TimePeriod,  # type: ignore
            description="Time period of the channel",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": [TimePeriod(start="2020-01-01", end="2020-01-02")],
            },
        ),
    ]

    prewhitening: Annotated[
        Prewhitening | None,
        Field(
            default=None,
            description="Prewhitening configuration and applied-filter metadata",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["Prewhitening()"],
            },
        ),
    ]

    @field_validator("units", mode="before")
    @classmethod
    def validate_units(cls, value: str) -> str:
        if value in [None, ""]:
            return ""
        try:
            unit_object = get_unit_object(value)
            return unit_object.name
        except ValueError as error:
            raise KeyError(error)
        except KeyError as error:
            raise KeyError(error)

    def _update_legacy_prewhitening(self, **updates) -> None:
        prewhitening = self.prewhitening or Prewhitening(type=PrewhiteningTypeEnum.none)
        object.__setattr__(
            self,
            "prewhitening",
            prewhitening.model_copy(update=updates),
        )

    @property
    def prewhitening_type(self) -> str:
        """Deprecated compatibility view of ``prewhitening.type``."""
        return self.prewhitening.type.value if self.prewhitening else ""

    @prewhitening_type.setter
    def prewhitening_type(self, value: str | PrewhiteningTypeEnum) -> None:
        self._update_legacy_prewhitening(type=PrewhiteningTypeEnum(value))

    @property
    def prewhitening_fit_domain(self) -> str:
        """Deprecated compatibility view of ``prewhitening.fit_domain``."""
        return self.prewhitening.fit_domain if self.prewhitening else ""

    @prewhitening_fit_domain.setter
    def prewhitening_fit_domain(self, value: str) -> None:
        self._update_legacy_prewhitening(fit_domain=value)

    @property
    def prewhitening_response_convention(self) -> str:
        """Deprecated compatibility view of ``prewhitening.response_convention``."""
        return self.prewhitening.response_convention if self.prewhitening else ""

    @prewhitening_response_convention.setter
    def prewhitening_response_convention(self, value: str) -> None:
        self._update_legacy_prewhitening(response_convention=value)

    @property
    def prewhitening_ar_coefficients(self) -> list[float]:
        """Deprecated compatibility view of ``prewhitening.ar_coefficients``."""
        return self.prewhitening.ar_coefficients if self.prewhitening else []

    @prewhitening_ar_coefficients.setter
    def prewhitening_ar_coefficients(self, value: list[float]) -> None:
        self._update_legacy_prewhitening(
            ar_order=len(value),
            ar_coefficients=value,
        )

    @property
    def prewhitening_ma_coefficients(self) -> list[float]:
        """Deprecated compatibility view of ``prewhitening.ma_coefficients``."""
        return self.prewhitening.ma_coefficients if self.prewhitening else []

    @prewhitening_ma_coefficients.setter
    def prewhitening_ma_coefficients(self, value: list[float]) -> None:
        self._update_legacy_prewhitening(
            ma_order=len(value),
            ma_coefficients=value,
        )
