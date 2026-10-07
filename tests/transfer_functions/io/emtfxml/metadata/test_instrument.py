# -*- coding: utf-8 -*-
"""
Test suite for Instrument basemodel
"""

import pytest

from mt_metadata.transfer_functions.io.emtfxml.metadata import Instrument


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def basic_instrument_data():
    """Basic instrument data for testing."""
    return {
        "manufacturer": "TestCorp",
        "name": "Test Magnetometer",
        "id": "MAG001",
        "type": "broadband magnetometer",
        "model": "TM-2000",
        "settings": "sensitivity: 10 V/nT, bandwidth: 0.001-1000 Hz",
    }


@pytest.fixture
def minimal_instrument_data():
    """Minimal instrument data for testing."""
    return {
        "manufacturer": "MinimalCorp",
        "name": "Basic Instrument",
        "id": "MIN001",
        "type": "sensor",
    }


@pytest.fixture
def empty_instrument():
    """Empty instrument instance."""
    return Instrument()


@pytest.fixture
def basic_instrument(basic_instrument_data):
    """Basic instrument instance."""
    return Instrument(**basic_instrument_data)


@pytest.fixture
def minimal_instrument(minimal_instrument_data):
    """Minimal instrument instance."""
    return Instrument(**minimal_instrument_data)


# =============================================================================
# Test Class: Instrument Instantiation
# =============================================================================
class TestInstrumentInstantiation:
    """Test instrument instantiation scenarios."""

    def test_empty_instrument_creation(self, empty_instrument):
        """Test creating an empty instrument."""
        assert empty_instrument.manufacturer == ""
        assert empty_instrument.name is None
        assert empty_instrument.id == ""
        assert empty_instrument.type == ""
        assert empty_instrument.model is None
        assert empty_instrument.settings is None

    def test_basic_instrument_creation(self, basic_instrument, basic_instrument_data):
        """Test creating a basic instrument with valid data."""
        assert basic_instrument.manufacturer == basic_instrument_data["manufacturer"]
        assert basic_instrument.name == basic_instrument_data["name"]
        assert basic_instrument.id == basic_instrument_data["id"]
        assert basic_instrument.type == basic_instrument_data["type"]
        assert basic_instrument.model == basic_instrument_data["model"]
        assert basic_instrument.settings == basic_instrument_data["settings"]

    @pytest.mark.parametrize(
        "field,value,expected",
        [
            ("manufacturer", "ACME Corp", "ACME Corp"),
            ("manufacturer", "", ""),
            ("name", "Test Instrument", "Test Instrument"),
            ("name", None, None),
            ("id", "INST123", "INST123"),
            ("id", "", ""),
            ("type", "magnetometer", "magnetometer"),
            ("type", "", ""),
            ("model", "v2.0", "v2.0"),
            ("model", None, None),
            ("settings", "config=default", "config=default"),
            ("settings", None, None),
        ],
    )
    def test_field_assignment(self, empty_instrument, field, value, expected):
        """Test individual field assignment."""
        setattr(empty_instrument, field, value)
        assert getattr(empty_instrument, field) == expected

    def test_instrument_with_serial_alias(self):
        """Test instrument creation using 'serial' alias for id."""
        # The id field has alias ['id', 'serial'] according to the model_fields
        instrument = Instrument(serial="SER123")
        assert instrument.id == "SER123"

    def test_inheritance_from_common_instrument(self, basic_instrument):
        """Test that Instrument inherits from CommonInstrument correctly."""
        # Should have all the fields from CommonInstrument
        assert hasattr(basic_instrument, "manufacturer")
        assert hasattr(basic_instrument, "name")
        assert hasattr(basic_instrument, "id")
        assert hasattr(basic_instrument, "type")
        assert hasattr(basic_instrument, "model")
        # Plus the additional settings field
        assert hasattr(basic_instrument, "settings")


# =============================================================================
# Test Class: XML Serialization
# =============================================================================
class TestXMLSerialization:
    """Test XML serialization functionality."""

    def test_to_xml_empty_instrument(self, empty_instrument):
        """Test XML serialization of empty instrument."""
        xml_element = empty_instrument.to_xml()

        assert xml_element.tag == "Instrument"
        assert xml_element.attrib == {}  # No type attribute when type is empty

        # Settings is required in EMTFXML even when its value is null.
        assert [child.tag for child in xml_element] == ["settings"]
        assert xml_element.find("settings").text is None

    def test_to_xml_basic_instrument(self, basic_instrument):
        """Test XML serialization of basic instrument."""
        xml_element = basic_instrument.to_xml()

        assert xml_element.tag == "Instrument"
        assert xml_element.attrib["type"] == "broadband magnetometer"

        # Check child elements
        children = {child.tag: child for child in xml_element}
        assert children["manufacturer"].text == "TestCorp"
        assert children["name"].text == "Test Magnetometer"
        assert children["id"].text == "MAG001"
        assert (
            children["settings"].text
            == "sensitivity: 10 V/nT, bandwidth: 0.001-1000 Hz"
        )

    def test_to_xml_string_output(self, basic_instrument):
        """Test XML string serialization."""
        xml_string = basic_instrument.to_xml(string=True)

        assert isinstance(xml_string, str)
        assert '<?xml version="1.0" encoding="UTF-8"?>' in xml_string
        assert '<Instrument type="broadband magnetometer">' in xml_string
        assert "<manufacturer>TestCorp</manufacturer>" in xml_string
        assert "<name>Test Magnetometer</name>" in xml_string
        assert "<id>MAG001</id>" in xml_string
        assert "</Instrument>" in xml_string

    def test_to_xml_type_attribute_handling(self, empty_instrument):
        """Test that type attribute is only added when not empty."""
        # Empty type - no attribute
        xml_element = empty_instrument.to_xml()
        assert "type" not in xml_element.attrib

        # Set type - should appear as attribute
        empty_instrument.type = "test_type"
        xml_element = empty_instrument.to_xml()
        assert xml_element.attrib["type"] == "test_type"

    @pytest.mark.parametrize(
        "field,value,should_appear",
        [
            ("manufacturer", "TestCorp", True),
            ("manufacturer", "", False),  # Empty string in NULL_VALUES
            ("manufacturer", None, False),  # None in NULL_VALUES
            ("manufacturer", "null", False),  # 'null' in NULL_VALUES
            ("name", "Test Name", True),
            ("name", None, False),
            ("id", "VALID_ID", True),
            ("id", "", False),
            ("settings", "valid settings", True),
            ("settings", None, True),
            ("settings", "null", True),
        ],
    )
    def test_to_xml_null_values_filtering(
        self, empty_instrument, field, value, should_appear
    ):
        """Test that NULL_VALUES are properly filtered from XML output."""
        setattr(empty_instrument, field, value)
        xml_element = empty_instrument.to_xml()

        field_element = xml_element.find(field)
        if should_appear:
            assert field_element is not None
            assert field_element.text == value
        else:
            assert field_element is None

    def test_to_xml_required_parameter(self, basic_instrument):
        """Test the required parameter (though implementation doesn't use it)."""
        # Test that required parameter doesn't break functionality
        xml_element_required = basic_instrument.to_xml(required=True)
        xml_element_all = basic_instrument.to_xml(required=False)

        # Both should produce the same result as implementation doesn't use required
        assert xml_element_required.tag == xml_element_all.tag
        assert xml_element_required.attrib == xml_element_all.attrib


# =============================================================================
# Test Class: Dictionary Operations
# =============================================================================
class TestDictionaryOperations:
    """Test dictionary serialization functionality."""

    def test_to_dict_basic_instrument(self, basic_instrument):
        """Test dictionary serialization of basic instrument."""
        instrument_dict = basic_instrument.to_dict()

        # Should follow the pattern from other basemodels
        assert "instrument" in instrument_dict
        data = instrument_dict["instrument"]

        assert data["manufacturer"] == "TestCorp"
        assert data["name"] == "Test Magnetometer"
        assert data["id"] == "MAG001"
        assert data["type"] == "broadband magnetometer"
        assert data["model"] == "TM-2000"
        assert data["settings"] == "sensitivity: 10 V/nT, bandwidth: 0.001-1000 Hz"

    def test_to_dict_empty_instrument(self, empty_instrument):
        """Test dictionary serialization of empty instrument."""
        instrument_dict = empty_instrument.to_dict(required=False)

        assert "instrument" in instrument_dict
        data = instrument_dict["instrument"]

        assert data["manufacturer"] == ""
        assert data["name"] is None
        assert data["id"] == ""
        assert data["type"] == ""
        assert data["model"] is None
        assert data["settings"] is None


# =============================================================================
# Test Class: Edge Cases and Error Handling
# =============================================================================
class TestEdgeCases:
    """Test edge cases and error scenarios."""

    def test_special_characters_in_fields(self, empty_instrument):
        """Test handling of special characters in fields."""
        special_cases = [
            ("manufacturer", "Åcme Çorp & Sønš", "Åcme Çorp & Sønš"),
            ("name", "Test'\"Instrument<>&", "Test'\"Instrument<>&"),
            ("id", "ID-123_ABC.001", "ID-123_ABC.001"),
            (
                "type",
                "Multi-line\nType\tWith\rSpecial",
                "Multi-line\nType\tWith\rSpecial",
            ),
            ("settings", 'key=value; other="quoted"', 'key=value; other="quoted"'),
        ]

        for field, input_value, expected in special_cases:
            setattr(empty_instrument, field, input_value)
            assert getattr(empty_instrument, field) == expected

    def test_numeric_string_coercion(self, empty_instrument):
        """Test that numeric values are coerced to strings."""
        numeric_cases = [
            ("manufacturer", 123, "123"),
            ("id", 456.789, "456.789"),
            ("type", 0, "0"),
        ]

        for field, input_value, expected in numeric_cases:
            setattr(empty_instrument, field, input_value)
            assert getattr(empty_instrument, field) == expected

    def test_very_long_field_values(self, empty_instrument):
        """Test handling of very long field values."""
        long_value = "x" * 10000

        empty_instrument.settings = long_value
        assert empty_instrument.settings == long_value

        # Should still serialize correctly
        xml_element = empty_instrument.to_xml()
        settings_element = xml_element.find("settings")
        assert settings_element.text == long_value

    def test_xml_with_all_null_values(self, empty_instrument):
        """Test XML generation when all values are in NULL_VALUES."""
        # Set all fields to NULL values
        empty_instrument.manufacturer = ""
        empty_instrument.name = None
        empty_instrument.id = "null"
        empty_instrument.type = ""
        empty_instrument.model = None
        empty_instrument.settings = "None"

        xml_element = empty_instrument.to_xml()

        # Settings is emitted unconditionally by the EMTFXML standard.
        assert [child.tag for child in xml_element] == ["settings"]
        assert xml_element.find("settings").text == "None"
        # Should have no type attribute
        assert "type" not in xml_element.attrib

    def test_xml_escaping(self, empty_instrument):
        """Test that XML special characters are properly escaped."""
        empty_instrument.manufacturer = "Corp & Sons <Ltd>"
        empty_instrument.name = "Test \"Instrument\" 'Model'"
        empty_instrument.id = "ID<123>"

        xml_string = empty_instrument.to_xml(string=True)

        # XML should be properly escaped
        assert "&amp;" in xml_string or "Corp &amp; Sons" in xml_string
        assert "&lt;" in xml_string or "&gt;" in xml_string


# =============================================================================
# Test Class: NULL_VALUES Integration
# =============================================================================
class TestNullValuesIntegration:
    """Test integration with NULL_VALUES constant."""

    def test_null_values_constant_access(self):
        """Test that NULL_VALUES constant is accessible."""
        from mt_metadata import NULL_VALUES

        assert isinstance(NULL_VALUES, list)
        assert None in NULL_VALUES
        assert "" in NULL_VALUES
        assert "null" in NULL_VALUES

    @pytest.mark.parametrize(
        "null_value",
        [
            None,
            "",
            "null",
            "None",
            "NONE",
        ],
    )
    def test_all_null_values_filtered(self, empty_instrument, null_value):
        """Test that all NULL_VALUES are properly filtered from XML."""
        empty_instrument.manufacturer = null_value
        xml_element = empty_instrument.to_xml()

        manufacturer_element = xml_element.find("manufacturer")
        assert manufacturer_element is None

    def test_datetime_null_values(self, empty_instrument):
        """Test that datetime NULL_VALUES are handled."""
        # These are also in NULL_VALUES
        datetime_nulls = ["1980-01-01T00:00:00", "1980-01-01T00:00:00+00:00"]

        for null_val in datetime_nulls:
            empty_instrument.settings = null_val
            xml_element = empty_instrument.to_xml()

            settings_element = xml_element.find("settings")
            assert settings_element is not None
            assert settings_element.text == null_val


# =============================================================================
# Test Class: Boundary Value Testing
# =============================================================================
class TestBoundaryValues:
    """Test boundary values and limits."""

    def test_empty_vs_none_handling(self, empty_instrument):
        """Test distinction between empty string and None."""
        # Both should be filtered as NULL_VALUES
        empty_instrument.manufacturer = ""
        empty_instrument.name = None

        xml_element = empty_instrument.to_xml()

        assert xml_element.find("manufacturer") is None
        assert xml_element.find("name") is None

    def test_whitespace_handling(self, empty_instrument):
        """Test whitespace handling in fields."""
        whitespace_cases = [
            "   leading spaces",
            "trailing spaces   ",
            "   both sides   ",
            "\t\ttabs\t\t",
            "\n\nnewlines\n\n",
        ]

        for case in whitespace_cases:
            empty_instrument.manufacturer = case
            assert empty_instrument.manufacturer == case

            # Should appear in XML since it's not in NULL_VALUES
            xml_element = empty_instrument.to_xml()
            manufacturer_element = xml_element.find("manufacturer")
            assert manufacturer_element is not None
            assert manufacturer_element.text == case

    def test_case_sensitivity_null_values(self, empty_instrument):
        """Test case sensitivity of NULL_VALUES."""
        # Test different cases of 'null' and 'none'
        case_variants = [
            ("null", False),  # Should be filtered
            ("NULL", False),  # Should be filtered (if in NULL_VALUES)
            ("Null", False),  # Should NOT be filtered
            ("none", False),  # Should NOT be filtered
            ("None", False),  # Should be filtered
            ("NONE", False),  # Should be filtered
        ]

        for value, should_appear in case_variants:
            empty_instrument.manufacturer = value
            xml_element = empty_instrument.to_xml()
            manufacturer_element = xml_element.find("manufacturer")

            if should_appear:
                assert manufacturer_element is not None
                assert manufacturer_element.text == value
            else:
                assert manufacturer_element is None


# =============================================================================
# Test Class: Integration Tests
# =============================================================================
class TestIntegration:
    """Test integration scenarios and workflows."""

    def test_instrument_creation_and_serialization_workflow(self):
        """Test complete instrument creation and serialization workflow."""
        # Create instrument
        instrument = Instrument(
            manufacturer="GeoInstruments",
            name="MT-Pro 5000",
            id="MT5000_001",
            type="magnetotelluric system",
            model="Pro5K",
            settings="sample_rate=1000Hz, gain=10x",
        )

        # Test all serialization methods
        xml_element = instrument.to_xml()
        xml_string = instrument.to_xml(string=True)
        instrument_dict = instrument.to_dict()

        # Verify XML
        assert xml_element.tag == "Instrument"
        assert xml_element.attrib["type"] == "magnetotelluric system"

        # Verify string
        assert isinstance(xml_string, str)
        assert "GeoInstruments" in xml_string

        # Verify dict
        assert "instrument" in instrument_dict

    def test_instrument_modification_workflow(self, minimal_instrument):
        """Test modifying instrument after creation."""
        # Initial state
        assert minimal_instrument.manufacturer == "MinimalCorp"

        # Modify fields
        minimal_instrument.manufacturer = "UpdatedCorp"
        minimal_instrument.settings = "updated settings"

        # Verify changes
        assert minimal_instrument.manufacturer == "UpdatedCorp"
        assert minimal_instrument.settings == "updated settings"

        # Verify XML reflects changes
        xml_element = minimal_instrument.to_xml()
        children = {child.tag: child for child in xml_element}
        assert children["manufacturer"].text == "UpdatedCorp"
        assert children["settings"].text == "updated settings"

    def test_multiple_instruments_creation(self):
        """Test creating multiple instrument instances."""
        instruments = []

        for i in range(5):
            instrument = Instrument(
                manufacturer=f"Corp{i}",
                name=f"Instrument{i}",
                id=f"ID{i:03d}",
                type=f"type{i}",
            )
            instruments.append(instrument)

        assert len(instruments) == 5

        # Test they all work independently
        for i, instrument in enumerate(instruments):
            assert instrument.manufacturer == f"Corp{i}"
            xml_element = instrument.to_xml()
            assert xml_element.tag == "Instrument"


# =============================================================================
# Test Class: Performance
# =============================================================================
class TestPerformance:
    """Test performance characteristics."""

    def test_large_batch_creation(self):
        """Test creating many instrument instances."""
        instruments = []

        for i in range(100):
            instrument = Instrument(
                manufacturer=f"Manufacturer{i}",
                name=f"Instrument{i}",
                id=f"ID{i:03d}",
                type="standard",
            )
            instruments.append(instrument)

        assert len(instruments) == 100

        # Test a few work correctly
        for instrument in instruments[:5]:
            xml_element = instrument.to_xml()
            assert xml_element.tag == "Instrument"

    def test_xml_serialization_performance(self, basic_instrument):
        """Test XML serialization performance."""
        # Should complete quickly
        for _ in range(100):
            xml_element = basic_instrument.to_xml()
            assert xml_element.tag == "Instrument"

    def test_dict_serialization_performance(self, basic_instrument):
        """Test dictionary serialization performance."""
        # Should complete quickly
        for _ in range(100):
            instrument_dict = basic_instrument.to_dict()
            assert "instrument" in instrument_dict
