import pytest
from pydantic import ValidationError

from mt_metadata.common import Instrument


def test_instrument_default_values():
    """
    Test the default values of the Instrument model.
    """
    instrument = Instrument()

    assert instrument.id == ""
    assert instrument.manufacturer == ""
    assert instrument.type == ""
    assert instrument.model is None
    assert instrument.name is None


def test_instrument_custom_values():
    """
    Test the Instrument model with custom values.
    """
    instrument = Instrument(
        id="mt01",
        manufacturer="MT Gurus",
        type="broadband 32-bit",
        model="falcon5",
        name="Falcon 5",
    )

    assert instrument.id == "mt01"
    assert instrument.manufacturer == "MT Gurus"
    assert instrument.type == "broadband 32-bit"
    assert instrument.model == "falcon5"
    assert instrument.name == "Falcon 5"


def test_instrument_partial_values():
    """
    Test the Instrument model with partial values.
    """
    instrument = Instrument(
        id="mt02",
        manufacturer="GeoTech",
    )

    assert instrument.id == "mt02"
    assert instrument.manufacturer == "GeoTech"
    assert instrument.type == ""
    assert instrument.model is None
    assert instrument.name is None


def test_instrument_invalid_id_type():
    """
    Test the Instrument model with an invalid id type.
    """
    with pytest.raises(ValidationError):
        Instrument(id=True)  # ID must be a string


def test_instrument_invalid_manufacturer_type():
    """
    Test the Instrument model with an invalid manufacturer type.
    """
    with pytest.raises(ValidationError):
        Instrument(manufacturer=[])  # Manufacturer must be a string


def test_instrument_invalid_type_type():
    """
    Test the Instrument model with an invalid type type.
    """
    with pytest.raises(ValidationError):
        Instrument(type=["broadband", "32-bit"])  # Type must be a string


def test_instrument_alias_validation():
    """
    Test the alias validation for the id field.
    """
    instrument = Instrument(serial="mt03")

    assert instrument.id == "mt03"


def test_instrument_manufacturer_alias_validation():
    """
    Test the alias validation for the manufacturer field.
    """
    instrument = Instrument(make="MT Gurus")

    assert instrument.manufacturer == "MT Gurus"
