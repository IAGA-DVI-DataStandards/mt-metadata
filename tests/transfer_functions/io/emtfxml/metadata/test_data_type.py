# -*- coding: utf-8 -*-
"""
Comprehensive pytest test suite for DataType basemodel class.

This module tests the DataType basemodel class from the transfer_functions.io.emtfxml.metadata
module, including validation, XML generation, dictionary parsing, enum handling, and units validation.
"""

from unittest.mock import MagicMock, patch
from xml.etree import cElementTree as et

import pytest
from pydantic import HttpUrl, ValidationError

from mt_metadata.transfer_functions.io.emtfxml.metadata import DataType
from mt_metadata.transfer_functions.io.emtfxml.metadata.data_type import (
    ArrayDTypeEnum,
    EstimateIntentionEnum,
    InputEnum,
    OutputEnum,
)


@pytest.fixture(scope="module")
def minimal_data_type():
    """Return a DataType instance with minimal data."""
    return DataType()


@pytest.fixture(scope="module")
def basic_data_type():
    """Return a DataType instance with basic data."""
    return DataType(
        name="variance",
        type=ArrayDTypeEnum.real_type,
        description="Variance estimate",
        external_url="http://example.com",
        intention=EstimateIntentionEnum.error_estimate,
        tag="error",
        output=OutputEnum.E,
        input=InputEnum.H,
        units="[mV/km]/[nT]",
    )


@pytest.fixture(scope="module")
def complete_data_type():
    """Return a DataType instance with comprehensive data."""
    return DataType(
        name="impedance_tensor",
        type=ArrayDTypeEnum.complex_type,
        description="Complete magnetotelluric impedance tensor estimate",
        external_url="https://www.iris.edu/dms/products/emtf/impedance.html",
        intention=EstimateIntentionEnum.primary_data_type,
        tag="impedance",
        output=OutputEnum.E,
        input=InputEnum.H,
        units="[mV/km]/[nT]",
    )


@pytest.fixture(
    params=[ArrayDTypeEnum.real_type, ArrayDTypeEnum.complex_type, "real", "complex"]
)
def type_values(request):
    """Return various type enum values for testing."""
    return request.param


@pytest.fixture(
    params=[
        EstimateIntentionEnum.error_estimate,
        EstimateIntentionEnum.signal_coherence,
        EstimateIntentionEnum.signal_power_estimate,
        EstimateIntentionEnum.primary_data_type,
        EstimateIntentionEnum.derived_data_type,
        "error estimate",
        "signal coherence",
    ]
)
def intention_values(request):
    """Return various intention enum values for testing."""
    return request.param


@pytest.fixture(params=[OutputEnum.E, OutputEnum.H, "E", "H"])
def channel_values(request):
    """Return various channel enum values for testing."""
    return request.param


@pytest.fixture(params=["[mV/km]/[nT]", "mV/km/nT", "V/m/T", "dimensionless", ""])
def unit_values(request):
    """Return various unit values for testing."""
    return request.param


@pytest.fixture(
    params=[
        "http://example.com",
        "https://www.iris.edu/dms/products/emtf/",
        "https://mtnet.info/data/formats/edi.html",
        "http://test.com/path?param=value",
    ]
)
def url_values(request):
    """Return various URL values for testing."""
    return request.param


@pytest.fixture(
    params=[
        {"estimate": {"name": "variance", "type": "real"}},
        {"estimate": {"name": "coherence", "description": "Signal coherence"}},
        {
            "estimate": {
                "external_url": "http://example.com",
                "intention": "error estimate",
            }
        },
        {"estimate": {}},  # Empty dict
    ]
)
def valid_dict_inputs(request):
    """Return valid dictionary inputs for read_dict testing."""
    return request.param


class TestDataTypeBasicFunctionality:
    """Test basic DataType functionality."""

    def test_default_initialization(self, minimal_data_type):
        """Test default DataType initialization."""
        assert minimal_data_type.name == ""
        assert minimal_data_type.type == ArrayDTypeEnum.real_type
        assert minimal_data_type.description == ""
        assert minimal_data_type.external_url == ""
        assert minimal_data_type.intention == ""
        assert minimal_data_type.tag == ""
        assert minimal_data_type.output == ""
        assert minimal_data_type.input == ""
        assert minimal_data_type.units == "[mV/km]/[nT]"

    def test_basic_initialization(self, basic_data_type):
        """Test DataType initialization with basic data."""
        assert basic_data_type.name == "variance"
        assert basic_data_type.type == ArrayDTypeEnum.real_type
        assert basic_data_type.description == "Variance estimate"
        assert str(basic_data_type.external_url) == "http://example.com/"
        assert basic_data_type.intention == EstimateIntentionEnum.error_estimate
        assert basic_data_type.tag == "error"
        assert basic_data_type.output == OutputEnum.E
        assert basic_data_type.input == InputEnum.H
        assert basic_data_type.units == "[mV/km]/[nT]"

    def test_complete_initialization(self, complete_data_type):
        """Test DataType initialization with complete data."""
        assert complete_data_type.name == "impedance_tensor"
        assert complete_data_type.type == ArrayDTypeEnum.complex_type
        assert "magnetotelluric impedance" in complete_data_type.description
        assert "iris.edu" in str(complete_data_type.external_url)
        assert complete_data_type.intention == EstimateIntentionEnum.primary_data_type

    def test_field_assignment_after_creation(self):
        """Test field assignment after DataType creation."""
        dt = DataType()
        dt.name = "test_name"
        dt.type = ArrayDTypeEnum.complex_type
        dt.description = "Test description"

        assert dt.name == "test_name"
        assert dt.type == ArrayDTypeEnum.complex_type
        assert dt.description == "Test description"


class TestDataTypeValidation:
    """Test DataType field validation."""

    def test_type_enum_validation(self, type_values):
        """Test type field validation with various values."""
        dt = DataType(type=type_values)
        if isinstance(type_values, ArrayDTypeEnum):
            assert dt.type == type_values.value
        else:
            assert dt.type == type_values

    def test_intention_enum_validation(self, intention_values):
        """Test intention field validation with various values."""
        dt = DataType(intention=intention_values)
        if isinstance(intention_values, EstimateIntentionEnum):
            assert dt.intention == intention_values.value
        else:
            assert dt.intention == intention_values

    def test_output_channel_validation(self, channel_values):
        """Test output field validation with various values."""
        dt = DataType(output=channel_values)
        if isinstance(channel_values, (OutputEnum, InputEnum)):
            assert dt.output == channel_values.value
        else:
            assert dt.output == channel_values

    def test_input_channel_validation(self, channel_values):
        """Test input field validation with various values."""
        dt = DataType(input=channel_values)
        if isinstance(channel_values, (OutputEnum, InputEnum)):
            assert dt.input == channel_values.value
        else:
            assert dt.input == channel_values

    def test_units_validation(self, unit_values):
        """Test units field validation with various values."""
        if unit_values == "":
            dt = DataType(units=unit_values)
            assert dt.units == "[]"
        else:
            # Units validator should convert to standardized form
            dt = DataType(units=unit_values)
            assert isinstance(dt.units, str)
            # Check that it's been processed by the units validator
            assert dt.units != ""

    def test_external_url_validation(self, url_values):
        """Test external_url field validation with various values."""
        dt = DataType(external_url=url_values)
        assert isinstance(dt.external_url, HttpUrl)
        assert str(dt.external_url).startswith("http")

    def test_empty_url_validation(self):
        """Test that empty URL is rejected."""
        with pytest.raises(ValidationError):
            DataType(external_url="")

    def test_invalid_type_enum(self):
        """Test invalid type enum values."""
        with pytest.raises(ValidationError):
            DataType(type="invalid_type")

    def test_invalid_intention_enum(self):
        """Test invalid intention enum values."""
        with pytest.raises(ValidationError):
            DataType(intention="invalid_intention")

    def test_invalid_channel_enum(self):
        """Test invalid channel enum values."""
        with pytest.raises(ValidationError):
            DataType(output="X")

        with pytest.raises(ValidationError):
            DataType(input="Y")

    def test_invalid_url(self):
        """Test invalid URL values."""
        with pytest.raises(ValidationError):
            DataType(external_url="not_a_url")

    def test_units_validator_error_handling(self):
        """Test units validator error handling with invalid units."""
        # Invalid units use the standard's dimensionless representation.
        dt = DataType(units="invalid_unit_xyz")
        assert dt.units == "[]"


class TestDataTypeReadDict:
    """Test DataType read_dict functionality."""

    def test_read_dict_complete_input(self):
        """Test read_dict with complete dictionary input."""
        input_dict = {
            "estimate": {
                "name": "impedance",
                "type": "complex",
                "description": "Impedance tensor",
                "external_url": "http://example.com",
                "intention": "primary data type",
                "tag": "mt_tensor",
                "output": "E",
                "input": "H",
                "units": "[mV/km]/[nT]",
            }
        }

        with patch(
            "mt_metadata.transfer_functions.io.emtfxml.metadata.helpers._read_element"
        ) as mock_helper:
            dt = DataType()
            dt.read_dict(input_dict)
            mock_helper.assert_called_once_with(dt, input_dict, "estimate")

    def test_read_dict_partial_input(self):
        """Test read_dict with partial dictionary input."""
        input_dict = {"estimate": {"name": "variance", "type": "real"}}

        with patch(
            "mt_metadata.transfer_functions.io.emtfxml.metadata.helpers._read_element"
        ) as mock_helper:
            dt = DataType()
            dt.read_dict(input_dict)
            mock_helper.assert_called_once_with(dt, input_dict, "estimate")

    def test_read_dict_valid_inputs(self, valid_dict_inputs):
        """Test read_dict with various valid dictionary inputs."""
        with patch(
            "mt_metadata.transfer_functions.io.emtfxml.metadata.helpers._read_element"
        ) as mock_helper:
            dt = DataType()
            dt.read_dict(valid_dict_inputs)
            mock_helper.assert_called_once_with(dt, valid_dict_inputs, "estimate")

    def test_read_dict_calls_helper(self):
        """Test that read_dict calls the correct helper function."""
        input_dict = {"estimate": {"name": "test"}}

        with patch(
            "mt_metadata.transfer_functions.io.emtfxml.metadata.helpers._read_element"
        ) as mock_helper:
            dt = DataType()
            dt.read_dict(input_dict)
            mock_helper.assert_called_once_with(dt, input_dict, "estimate")


class TestDataTypeXMLGeneration:
    """Test DataType XML generation functionality."""

    def test_xml_generation_complete(self, complete_data_type):
        """Test XML generation with complete DataType data."""
        xml_element = complete_data_type.to_xml(string=False, required=True)

        assert isinstance(xml_element, et.Element)
        assert xml_element.attrib["name"] == "impedance_tensor"
        assert xml_element.attrib["type"] == "complex"
        assert xml_element.attrib["output"] == "E"
        assert xml_element.attrib["input"] == "H"
        assert xml_element.attrib["units"] == "[mV/km]/[nT]"

    def test_xml_generation_string_output(self, basic_data_type):
        """Test XML generation with string output."""
        with patch(
            "mt_metadata.transfer_functions.io.emtfxml.metadata.helpers.to_xml"
        ) as mock_helper:
            mock_helper.return_value = "<estimate>test</estimate>"

            xml_string = basic_data_type.to_xml(string=True, required=True)

            mock_helper.assert_called_once_with(
                basic_data_type,
                string=True,
                required=True,
                order=["description", "external_url", "intention", "tag"],
            )
            assert xml_string == "<estimate>test</estimate>"

    def test_xml_generation_minimal(self, minimal_data_type):
        """Test XML generation with minimal DataType data."""
        xml_element = minimal_data_type.to_xml(string=False, required=True)

        assert isinstance(xml_element, et.Element)
        assert xml_element.attrib["name"] == ""
        assert xml_element.attrib["type"] == "real"
        assert xml_element.attrib["output"] == ""
        assert xml_element.attrib["input"] == ""
        assert xml_element.attrib["units"] == "[mV/km]/[nT]"

    def test_xml_generation_parameters(self):
        """Test XML generation with different parameters."""
        dt = DataType(name="test", type=ArrayDTypeEnum.complex_type)

        # Test with string=False
        with patch(
            "mt_metadata.transfer_functions.io.emtfxml.metadata.helpers.to_xml"
        ) as mock_helper:
            mock_element = MagicMock()
            mock_element.attrib = {}
            mock_helper.return_value = mock_element

            result = dt.to_xml(string=False, required=False)

            mock_helper.assert_called_once_with(
                dt,
                string=False,
                required=False,
                order=["description", "external_url", "intention", "tag"],
            )

            # Check that attributes were set
            assert mock_element.attrib["name"] == "test"
            assert mock_element.attrib["type"] == "complex"

    def test_xml_generation_order(self):
        """Test that XML generation uses correct field order."""
        dt = DataType()

        with patch(
            "mt_metadata.transfer_functions.io.emtfxml.metadata.helpers.to_xml"
        ) as mock_helper:
            mock_helper.return_value = MagicMock()

            dt.to_xml()

            # Verify the order parameter
            call_args = mock_helper.call_args
            assert call_args[1]["order"] == [
                "description",
                "external_url",
                "intention",
                "tag",
            ]

    def test_xml_attributes_setting(self):
        """Test that XML attributes are set correctly."""
        dt = DataType(
            name="test_name",
            type=ArrayDTypeEnum.real_type,
            output=OutputEnum.E,
            input=InputEnum.H,
            units="mV/km",
        )

        with patch(
            "mt_metadata.transfer_functions.io.emtfxml.metadata.helpers.to_xml"
        ) as mock_helper:
            mock_element = MagicMock()
            mock_element.attrib = {}
            mock_helper.return_value = mock_element

            dt.to_xml(string=False)

            # Check all required attributes were set
            expected_attrs = {
                "name": "test_name",
                "type": "real",
                "output": "E",
                "input": "H",
                "units": "mV/km",
            }

            for key, value in expected_attrs.items():
                assert mock_element.attrib[key] == value


class TestDataTypeEdgeCases:
    """Test DataType edge cases and boundary conditions."""

    def test_empty_string_fields(self):
        """Test DataType with empty string fields."""
        dt = DataType(name="", description="", tag="", units="")

        assert dt.name == ""
        assert dt.description == ""
        assert dt.tag == ""
        assert dt.units == "[]"

    def test_long_text_fields(self):
        """Test DataType with very long text fields."""
        long_text = "A" * 1000
        dt = DataType(name=long_text, description=long_text, tag=long_text)

        assert dt.name == long_text
        assert dt.description == long_text
        assert dt.tag == long_text

    def test_special_characters_in_text(self):
        """Test DataType with special characters in text fields."""
        special_text = "Test with special chars: !@#$%^&*()[]{}|;:'\",.<>?/~`"
        dt = DataType(name=special_text, description=special_text, tag=special_text)

        assert dt.name == special_text
        assert dt.description == special_text
        assert dt.tag == special_text

    def test_unicode_characters(self):
        """Test DataType with unicode characters."""
        unicode_text = "Tëst with ünïcödë: αβγδε ☺☻♠♣♥♦"
        dt = DataType(name=unicode_text, description=unicode_text, tag=unicode_text)

        assert dt.name == unicode_text
        assert dt.description == unicode_text
        assert dt.tag == unicode_text

    def test_field_reassignment(self):
        """Test reassignment of DataType fields."""
        dt = DataType(name="original", type=ArrayDTypeEnum.real_type)

        # Reassign fields
        dt.name = "modified"
        dt.type = ArrayDTypeEnum.complex_type
        dt.intention = EstimateIntentionEnum.error_estimate

        assert dt.name == "modified"
        assert dt.type == ArrayDTypeEnum.complex_type
        assert dt.intention == EstimateIntentionEnum.error_estimate

    def test_none_values_handling(self):
        """Test DataType handling of None values where applicable."""
        # Most fields have defaults, so None should be converted
        dt = DataType()

        # Check that None values are handled appropriately
        assert dt.name == ""  # Default empty string
        assert dt.type == ArrayDTypeEnum.real_type  # Default value
        assert dt.description == ""


class TestDataTypePerformance:
    """Test DataType performance characteristics."""

    def test_bulk_creation_performance(self):
        """Test performance of creating many DataType instances."""
        import time

        start_time = time.time()
        data_types = []

        for i in range(100):
            dt = DataType(
                name=f"test_{i}",
                type=(
                    ArrayDTypeEnum.real_type
                    if i % 2 == 0
                    else ArrayDTypeEnum.complex_type
                ),
                description=f"Test description {i}",
                external_url="http://example.com",
                intention=EstimateIntentionEnum.error_estimate,
                tag=f"tag_{i}",
                output=OutputEnum.E,
                input=InputEnum.H,
                units="[mV/km]/[nT]",
            )
            data_types.append(dt)

        creation_time = time.time() - start_time

        # Should be able to create 100 instances reasonably quickly
        assert creation_time < 5.0  # 5 seconds should be more than enough
        assert len(data_types) == 100

    def test_xml_generation_performance(self):
        """Test performance of XML generation."""
        import time

        dt = DataType(
            name="performance_test",
            type=ArrayDTypeEnum.complex_type,
            description="Performance test data type",
            external_url="http://example.com",
            intention=EstimateIntentionEnum.primary_data_type,
            tag="performance",
            output=OutputEnum.E,
            input=InputEnum.H,
            units="[mV/km]/[nT]",
        )

        start_time = time.time()

        for _ in range(50):
            xml_element = dt.to_xml(string=False)
            assert isinstance(xml_element, et.Element)

        xml_generation_time = time.time() - start_time

        # XML generation should be fast
        assert xml_generation_time < 2.0  # 2 seconds for 50 generations

    def test_units_validation_performance(self):
        """Test performance of units validation."""
        import time

        valid_units = ["[mV/km]/[nT]", "mV/km/nT", "V/m/T", "dimensionless"]

        start_time = time.time()

        for _ in range(25):
            for unit in valid_units:
                dt = DataType(units=unit)
                assert isinstance(dt.units, str)

        validation_time = time.time() - start_time

        # Units validation should be reasonably fast
        assert validation_time < 3.0  # 3 seconds for 100 validations


class TestDataTypeIntegration:
    """Test DataType integration with other components."""

    def test_enum_integration(self):
        """Test integration with enum classes."""
        # Test that enums work correctly
        dt = DataType(
            type=ArrayDTypeEnum.complex_type,
            intention=EstimateIntentionEnum.signal_coherence,
            output=OutputEnum.E,
            input=InputEnum.H,
        )

        assert dt.type == ArrayDTypeEnum.complex_type
        assert dt.intention == EstimateIntentionEnum.signal_coherence
        assert dt.output == OutputEnum.E
        assert dt.input == InputEnum.H

        # Test string access
        assert dt.type == "complex"
        assert dt.intention == "signal coherence"

    def test_url_integration(self):
        """Test integration with HttpUrl validation."""
        urls = [
            "http://example.com",
            "https://www.iris.edu/dms/products/emtf/",
            "https://test.com/path?param=value#fragment",
        ]

        for url in urls:
            dt = DataType(external_url=url)
            assert isinstance(dt.external_url, HttpUrl)
            assert str(dt.external_url).startswith("http")

    def test_helpers_integration(self):
        """Test integration with helpers module."""
        dt = DataType()
        input_dict = {"estimate": {"name": "test"}}

        # Test that read_dict integrates with helpers
        with patch(
            "mt_metadata.transfer_functions.io.emtfxml.metadata.helpers._read_element"
        ) as mock_read:
            dt.read_dict(input_dict)
            mock_read.assert_called_once()

        # Test that to_xml integrates with helpers
        with patch(
            "mt_metadata.transfer_functions.io.emtfxml.metadata.helpers.to_xml"
        ) as mock_xml:
            mock_xml.return_value = MagicMock()
            dt.to_xml()
            mock_xml.assert_called_once()

    def test_inheritance_from_metadata_base(self):
        """Test that DataType properly inherits from MetadataBase."""
        dt = DataType()

        # Should have MetadataBase methods and attributes
        assert hasattr(dt, "model_fields")
        assert hasattr(dt, "model_validate")
        assert hasattr(dt, "model_dump")

        # Test serialization capabilities
        data_dict = dt.model_dump()
        assert isinstance(data_dict, dict)
        assert "name" in data_dict
        assert "type" in data_dict

    def test_units_validator_integration(self):
        """Test integration with units validation system."""
        # Test with valid units
        dt = DataType(units="[mV/km]/[nT]")
        assert dt.units == "[mV/km]/[nT]"
        assert "nT" in dt.units

        # Test with empty units
        dt_empty = DataType(units="")
        assert dt_empty.units == "[]"

        # Test that validator converts units properly
        dt_simple = DataType(units="mV/km/nT")
        assert isinstance(dt_simple.units, str)
        assert (
            dt_simple.units == "[mV/km]/[nT]"
        )  # Should stay the same as it's already in symbol format

    def test_field_defaults_and_requirements(self):
        """Test field defaults and requirement settings."""
        dt = DataType()

        # Check default values match expected behavior
        assert dt.name == ""
        assert dt.type == ArrayDTypeEnum.real_type
        assert dt.description == ""
        assert dt.external_url == ""
        assert dt.intention == ""
        assert dt.tag == ""
        assert dt.output == ""
        assert dt.input == ""
        assert dt.units == "[mV/km]/[nT]"

        # All fields should be accessible and modifiable
        for field_name in dt.model_fields:
            assert hasattr(dt, field_name)
            # Should be able to get field info
            field_info = dt.model_fields[field_name]
            assert field_info is not None
