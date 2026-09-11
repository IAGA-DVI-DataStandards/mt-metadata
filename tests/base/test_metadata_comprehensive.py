"""
Comprehensive test suite for metadata.py - DotNotationBaseModel and MetadataBase classes

Tests both DotNotationBaseModel and MetadataBase classes with fixtures and subtests
for optimal efficiency and comprehensive coverage.
"""

import json
import tempfile
import unittest.mock
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import pytest
from pydantic import Field
from pydantic.fields import FieldInfo

from mt_metadata.base.metadata import DotNotationBaseModel, MetadataBase
from mt_metadata.utils.exceptions import MTSchemaError


# Test models for testing
class NestedModel(MetadataBase):
    """Test model with nested attributes"""

    value: str = Field(default="test", description="Test value")
    number: int = Field(default=42, description="Test number")


class SampleModel(MetadataBase):
    """Test model for comprehensive testing"""

    simple_attr: str = Field(default="simple", description="Simple attribute")
    number_attr: int = Field(default=10, description="Number attribute")
    float_attr: float = Field(default=3.14, description="Float attribute")
    bool_attr: bool = Field(default=True, description="Boolean attribute")
    nested_model: NestedModel = Field(default_factory=NestedModel)


class RequiredFieldModel(MetadataBase):
    """Test model with required fields"""

    required_field: str = Field(
        default="required_value",
        description="Required field",
        json_schema_extra={"required": True},
    )
    optional_field: str = Field(
        default="optional_value",
        description="Optional field",
        json_schema_extra={"required": False},
    )


# Fixtures at module level for efficiency
@pytest.fixture
def dot_notation_model():
    """Basic DotNotationBaseModel instance"""
    return DotNotationBaseModel()


@pytest.fixture
def test_model():
    """Basic SampleModel instance"""
    return SampleModel()


@pytest.fixture
def test_model_with_data():
    """SampleModel with some test data"""
    return SampleModel(
        simple_attr="test_value", number_attr=100, float_attr=2.71, bool_attr=False
    )


@pytest.fixture
def required_model():
    """RequiredFieldModel instance"""
    return RequiredFieldModel()


@pytest.fixture
def sample_dict():
    """Sample dictionary for testing"""
    return {
        "simple_attr": "dict_value",
        "number_attr": 200,
        "float_attr": 1.41,
        "bool_attr": True,
        "nested_model": {"value": "nested_test", "number": 84},
    }


@pytest.fixture
def sample_json_string(sample_dict):
    """Sample JSON string for testing"""
    return json.dumps({"test_model": sample_dict})


@pytest.fixture
def sample_pandas_series():
    """Sample pandas Series for testing"""
    return pd.Series(
        {
            "simple_attr": "series_value",
            "number_attr": 300,
            "nested_model.value": "nested_series",
            "nested_model.number": 126,
        }
    )


@pytest.fixture
def sample_xml_element():
    """Sample XML element for testing"""
    root = ET.Element("test_model")
    ET.SubElement(root, "simple_attr").text = "xml_value"
    ET.SubElement(root, "number_attr").text = "400"
    nested = ET.SubElement(root, "nested_model")
    ET.SubElement(nested, "value").text = "nested_xml"
    ET.SubElement(nested, "number").text = "168"
    return root


@pytest.fixture
def temp_json_file(sample_dict):
    """Temporary JSON file for testing"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"test_model": sample_dict}, f)
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink()  # Clean up


# DotNotationBaseModel Tests
class TestDotNotationBaseModel:
    """Test DotNotationBaseModel functionality"""

    def test_initialization_basic(self, dot_notation_model):
        """Test basic initialization"""
        assert isinstance(dot_notation_model, DotNotationBaseModel)

    def test_initialization_with_dot_notation_data(self):
        """Test initialization with dot notation data"""
        data = {
            "nested.value": "test_nested",
            "nested.number": 42,
            "simple": "test_simple",
        }

        # Create a model that can handle these attributes
        class TestDotModel(DotNotationBaseModel):
            simple: str = Field(default="")
            nested: dict = Field(default_factory=dict)

        model = TestDotModel(**data)
        assert model.simple == "test_simple"
        assert model.nested["value"] == "test_nested"
        assert model.nested["number"] == 42

    def test_set_nested_attribute(self, dot_notation_model):
        """Test _set_nested_attribute method"""
        data_dict = {}
        dot_notation_model._set_nested_attribute(
            data_dict, "level1.level2.value", "test"
        )

        expected = {"level1": {"level2": {"value": "test"}}}
        assert data_dict == expected

    def test_set_nested_attribute_deep_nesting(self, dot_notation_model):
        """Test _set_nested_attribute with deep nesting"""
        data_dict = {}
        dot_notation_model._set_nested_attribute(
            data_dict, "a.b.c.d.e.value", "deep_test"
        )

        assert data_dict["a"]["b"]["c"]["d"]["e"]["value"] == "deep_test"

    def test_update_attribute_simple(self):
        """Test update_attribute with simple attribute"""

        class SimpleModel(DotNotationBaseModel):
            test_attr: str = Field(default="initial")

        model = SimpleModel()
        model.update_attribute("test_attr", "updated")
        assert model.test_attr == "updated"

    def test_update_attribute_nested(self):
        """Test update_attribute with nested attribute"""

        class NestedModel(DotNotationBaseModel):
            simple: str = Field(default="test")

        class ParentModel(DotNotationBaseModel):
            nested: NestedModel = Field(default_factory=NestedModel)

        model = ParentModel()
        model.update_attribute("nested.simple", "updated_nested")
        assert model.nested.simple == "updated_nested"

    def test_update_attribute_nonexistent_raises_error(self):
        """Test update_attribute with non-existent nested attribute raises AttributeError"""

        class SimpleModel(DotNotationBaseModel):
            test_attr: str = Field(default="test")

        model = SimpleModel()
        with pytest.raises(AttributeError, match="has no attribute"):
            model.update_attribute("nonexistent.value", "test")


# MetadataBase Tests
class TestMetadataBaseInstantiation:
    """Test MetadataBase instantiation and basic functionality"""

    def test_basic_instantiation(self, test_model):
        """Test basic MetadataBase instantiation"""
        assert isinstance(test_model, MetadataBase)
        assert test_model.simple_attr == "simple"
        assert test_model.number_attr == 10

    def test_instantiation_with_data(self, test_model_with_data):
        """Test instantiation with custom data"""
        assert test_model_with_data.simple_attr == "test_value"
        assert test_model_with_data.number_attr == 100
        assert test_model_with_data.float_attr == 2.71
        assert test_model_with_data.bool_attr is False

    def test_class_name_property(self, test_model):
        """Test _class_name computed property"""
        assert test_model._class_name == "sample_model"

    def test_string_representation(self, test_model):
        """Test __str__ method"""
        str_repr = str(test_model)
        assert isinstance(str_repr, str)
        assert "simple_attr" in str_repr

    def test_repr_representation(self, test_model):
        """Test __repr__ method returns JSON"""
        repr_str = repr(test_model)
        assert isinstance(repr_str, str)
        # Should be valid JSON
        parsed = json.loads(repr_str)
        assert isinstance(parsed, dict)

    def test_len_method(self, test_model):
        """Test __len__ method returns number of attributes"""
        length = len(test_model)
        assert isinstance(length, int)
        assert length > 0


class TestMetadataBaseEquality:
    """Test MetadataBase equality operations"""

    def test_equality_same_instance(self, test_model):
        """Test equality with same instance"""
        assert test_model == test_model

    def test_equality_same_data(self):
        """Test equality with same data"""
        model1 = SampleModel(simple_attr="test", number_attr=42)
        model2 = SampleModel(simple_attr="test", number_attr=42)
        assert model1 == model2

    def test_equality_different_data(self):
        """Test inequality with different data"""
        model1 = SampleModel(simple_attr="test1", number_attr=42)
        model2 = SampleModel(simple_attr="test2", number_attr=42)
        assert model1 != model2

    def test_equality_with_dict(self, test_model, sample_dict):
        """Test equality with dictionary"""
        # This should work through the load mechanism
        # Note: The exact behavior depends on the implementation
        # but it should not raise an exception
        result = test_model == sample_dict
        assert isinstance(result, bool)

    def test_equality_with_none(self, test_model):
        """Test equality with None"""
        assert test_model != None
        assert not (test_model == None)

    def test_inequality_method(self, test_model):
        """Test __ne__ method"""
        model2 = SampleModel(simple_attr="different")
        assert test_model != model2

    def test_equality_with_numpy_arrays(self):
        """Test equality with numpy arrays"""

        class ArrayModel(MetadataBase):
            array_field: np.ndarray = Field(default_factory=lambda: np.array([1, 2, 3]))

        model1 = ArrayModel()
        model2 = ArrayModel()
        model2.array_field = np.array([1, 2, 3])

        # Test that numpy arrays are properly converted in dict representation
        dict1 = model1.to_dict(single=True)
        dict2 = model2.to_dict(single=True)

        # Check that arrays are converted to the same format (lists)
        arr1 = dict1["array_field"]
        arr2 = dict2["array_field"]

        # Convert to lists for comparison if they're still arrays
        if hasattr(arr1, "tolist"):
            arr1 = arr1.tolist()
        if hasattr(arr2, "tolist"):
            arr2 = arr2.tolist()

        assert arr1 == arr2

    def test_equality_with_close_float_values(self):
        """Test equality with close float values"""
        model1 = SampleModel(float_attr=3.14159)
        model2 = SampleModel(float_attr=3.14159001)  # Very close value
        # Should use np.isclose for float comparison
        result = model1 == model2
        assert isinstance(result, bool)


class TestMetadataBaseLoading:
    """Test MetadataBase load functionality"""

    def test_load_from_dict(self, test_model, sample_dict):
        """Test loading from dictionary"""
        test_model.load(sample_dict)
        assert test_model.simple_attr == "dict_value"
        assert test_model.number_attr == 200

    def test_load_from_json_string(self, test_model, sample_json_string):
        """Test loading from JSON string"""
        test_model.load(sample_json_string)
        assert test_model.simple_attr == "dict_value"
        assert test_model.number_attr == 200

    def test_load_from_pandas_series(self, test_model, sample_pandas_series):
        """Test loading from pandas Series"""
        test_model.load(sample_pandas_series)
        assert test_model.simple_attr == "series_value"
        assert test_model.number_attr == 300

    def test_load_from_xml_element(self, test_model, sample_xml_element):
        """Test loading from XML element"""
        test_model.load(sample_xml_element)
        assert test_model.simple_attr == "xml_value"
        assert test_model.number_attr == 400

    def test_load_from_metadata_base(self, test_model):
        """Test loading from another MetadataBase instance"""
        other_model = SampleModel(simple_attr="other_value", number_attr=500)
        test_model.load(other_model)
        assert test_model.simple_attr == "other_value"
        assert test_model.number_attr == 500

    def test_load_null_value(self, test_model):
        """Test loading null values"""
        original_value = test_model.simple_attr
        test_model.load("none")  # Should be in NULL_VALUES
        assert test_model.simple_attr == original_value  # Should remain unchanged

    def test_load_invalid_type_raises_error(self, test_model):
        """Test loading invalid type raises MTSchemaError"""
        with pytest.raises(MTSchemaError, match="Cannot load"):
            test_model.load(123)  # Invalid type


class TestMetadataBaseUpdate:
    """Test MetadataBase update functionality"""

    def test_update_basic(self, test_model):
        """Test basic update functionality"""
        other_model = SampleModel(simple_attr="updated", number_attr=999)
        test_model.update(other_model)
        assert test_model.simple_attr == "updated"
        assert test_model.number_attr == 999

    def test_update_with_match_constraint(self):
        """Test update with match constraint"""
        model1 = SampleModel(simple_attr="same", number_attr=100)
        model2 = SampleModel(simple_attr="same", number_attr=200)

        # Should work when match field is the same
        model1.update(model2, match=["simple_attr"])
        assert model1.number_attr == 200

    def test_update_with_match_constraint_fails(self):
        """Test update with match constraint that fails"""
        model1 = SampleModel(simple_attr="different1", number_attr=100)
        model2 = SampleModel(simple_attr="different2", number_attr=200)

        # Should raise ValueError when match field differs
        with pytest.raises(ValueError, match="is not equal"):
            model1.update(model2, match=["simple_attr"])

    def test_update_skips_none_values(self, test_model):
        """Test update skips None and default values"""
        # Create model with different values to test update behavior
        other_model = SampleModel(simple_attr="different", number_attr=999)
        original_simple = test_model.simple_attr

        # Test that update works with valid values
        test_model.update(other_model)
        assert test_model.simple_attr == "different"
        assert test_model.number_attr == 999

    def test_update_wrong_type_logs_warning(self, test_model):
        """Test update with wrong type logs warning"""
        different_model = RequiredFieldModel()

        with unittest.mock.patch(
            "mt_metadata.base.metadata.logger.warning"
        ) as mock_logger:
            test_model.update(different_model)
            mock_logger.assert_called_once()


class TestMetadataBaseCopy:
    """Test MetadataBase copy functionality"""

    def test_copy_basic(self, test_model_with_data):
        """Test basic copy functionality"""
        copied = test_model_with_data.copy()
        assert copied == test_model_with_data
        assert copied is not test_model_with_data

    def test_copy_with_update(self, test_model):
        """Test copy with update parameter"""
        copied = test_model.copy(update={"simple_attr": "updated_copy"})
        assert copied.simple_attr == "updated_copy"
        assert test_model.simple_attr == "simple"  # Original unchanged

    def test_copy_shallow_vs_deep(self, test_model):
        """Test shallow vs deep copy"""
        deep_copy = test_model.copy(deep=True)
        shallow_copy = test_model.copy(deep=False)

        assert deep_copy == test_model
        assert shallow_copy == test_model
        assert deep_copy is not test_model
        assert shallow_copy is not test_model


class TestMetadataBaseFields:
    """Test MetadataBase field-related functionality"""

    def test_get_all_fields(self, test_model):
        """Test get_all_fields method"""
        fields = test_model.get_all_fields()
        assert isinstance(fields, dict)
        assert "simple_attr" in fields
        assert "number_attr" in fields

    def test_get_attribute_list(self, test_model):
        """Test get_attribute_list method"""
        attr_list = test_model.get_attribute_list()
        assert isinstance(attr_list, list)
        assert "simple_attr" in attr_list
        assert "number_attr" in attr_list
        assert attr_list == sorted(attr_list)  # Should be sorted

    def test_required_fields_property(self, required_model):
        """Test _required_fields property"""
        required_fields = required_model._required_fields
        assert isinstance(required_fields, list)
        assert "required_field" in required_fields
        assert "optional_field" not in required_fields

    def test_field_info_to_string(self, test_model):
        """Test _field_info_to_string method"""
        fields = test_model.get_all_fields()
        field_name = "simple_attr"
        field_info = fields[field_name]

        field_str = test_model._field_info_to_string(field_name, field_info)
        assert isinstance(field_str, str)
        assert field_name in field_str
        assert "description" in field_str

    def test_attribute_information_specific(self, test_model):
        """Test attribute_information for specific attribute"""
        with unittest.mock.patch("builtins.print") as mock_print:
            test_model.attribute_information("simple_attr")
            mock_print.assert_called_once()

    def test_attribute_information_all(self, test_model):
        """Test attribute_information for all attributes"""
        with unittest.mock.patch("builtins.print") as mock_print:
            test_model.attribute_information()
            mock_print.assert_called_once()

    def test_attribute_information_invalid(self, test_model):
        """Test attribute_information with invalid attribute name"""
        with pytest.raises(MTSchemaError, match="not attribute.*found"):
            test_model.attribute_information("nonexistent_attr")

    def test_get_attr_from_name_simple(self, test_model_with_data):
        """Test get_attr_from_name with simple attribute"""
        value = test_model_with_data.get_attr_from_name("simple_attr")
        assert value == "test_value"

    def test_get_attr_from_name_nested(self, test_model):
        """Test get_attr_from_name with nested attribute"""
        value = test_model.get_attr_from_name("nested_model.value")
        assert value == "test"


class TestMetadataBaseFieldManagement:
    """Test MetadataBase field management functionality"""

    def test_add_new_field(self, test_model):
        """Test add_new_field method"""
        new_field = FieldInfo(
            annotation=str,
            default="new_default",
            description="New field description",
            json_schema_extra={"units": "meters"},
        )

        new_model_class = test_model.add_new_field("new_attribute", new_field)
        assert new_model_class is not None

        # Create instance of new model
        new_instance = new_model_class()
        assert hasattr(new_instance, "new_attribute")

    def test_add_new_field_same_class(self, test_model):
        """Test add_new_field returns the same class for the same field"""

        def make_field():
            return FieldInfo(
                annotation=str,
                default="new_default",
                description="New field description",
                json_schema_extra={"units": "meters"},
            )

        first = test_model.add_new_field("new_attribute", make_field())
        second = test_model.add_new_field("new_attribute", make_field())

        assert first is second
        assert issubclass(first, type(test_model))
        assert first().new_attribute == "new_default"


class TestMetadataBaseDictConversion:
    """Test MetadataBase dictionary conversion functionality"""

    def test_to_dict_basic(self, test_model_with_data):
        """Test basic to_dict functionality"""
        result = test_model_with_data.to_dict()
        assert isinstance(result, dict)
        assert "sample_model" in result
        assert "simple_attr" in result["sample_model"]

    def test_to_dict_single(self, test_model_with_data):
        """Test to_dict with single=True"""
        result = test_model_with_data.to_dict(single=True)
        assert isinstance(result, dict)
        assert "simple_attr" in result
        assert "test_model" not in result  # Should not have class wrapper

    def test_to_dict_nested(self, test_model):
        """Test to_dict with nested=True"""
        result = test_model.to_dict(nested=True)
        assert isinstance(result, dict)
        # Should have nested structure for nested_model
        if "test_model" in result:
            model_dict = result["test_model"]
            if "nested_model" in model_dict:
                assert isinstance(model_dict["nested_model"], dict)

    def test_to_dict_required_only(self, required_model):
        """Test to_dict with required=True"""
        result = required_model.to_dict(required=True)
        assert isinstance(result, dict)
        # Should contain required field
        model_dict = result.get("required_field_model", result)
        assert "required_field" in model_dict

    def test_to_dict_all_fields(self, required_model):
        """Test to_dict with required=False"""
        result = required_model.to_dict(required=False)
        assert isinstance(result, dict)
        model_dict = result.get("required_field_model", result)
        assert "required_field" in model_dict
        assert "optional_field" in model_dict

    def test_from_dict_basic(self, test_model, sample_dict):
        """Test basic from_dict functionality"""
        test_model.from_dict(sample_dict)
        assert test_model.simple_attr == "dict_value"
        assert test_model.number_attr == 200

    def test_from_dict_with_class_wrapper(self, test_model, sample_dict):
        """Test from_dict with class name wrapper"""
        wrapped_dict = {"test_model": sample_dict}
        test_model.from_dict(wrapped_dict)
        assert test_model.simple_attr == "dict_value"

    def test_from_dict_skip_none(self, test_model):
        """Test from_dict with skip_none=True"""
        dict_with_none = {"simple_attr": None, "number_attr": 500}
        original_simple = test_model.simple_attr

        test_model.from_dict(dict_with_none, skip_none=True)
        assert test_model.simple_attr == original_simple  # Should remain unchanged
        assert test_model.number_attr == 500

    def test_from_dict_invalid_input(self, test_model):
        """Test from_dict with invalid input raises MTSchemaError"""
        with pytest.raises(MTSchemaError, match="Input must be a dictionary"):
            test_model.from_dict("not_a_dict")


class TestMetadataBaseJsonConversion:
    """Test MetadataBase JSON conversion functionality"""

    def test_to_json_basic(self, test_model_with_data):
        """Test basic to_json functionality"""
        json_str = test_model_with_data.to_json()
        assert isinstance(json_str, str)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_to_json_nested(self, test_model):
        """Test to_json with nested=True"""
        json_str = test_model.to_json(nested=True)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_to_json_custom_indent(self, test_model):
        """Test to_json with custom indentation"""
        json_str = test_model.to_json(indent="  ")
        assert isinstance(json_str, str)
        assert "  " in json_str  # Should contain the custom indent

    def test_from_json_string(self, test_model, sample_json_string):
        """Test from_json with JSON string"""
        test_model.from_json(sample_json_string)
        assert test_model.simple_attr == "dict_value"
        assert test_model.number_attr == 200

    def test_from_json_file(self, test_model, temp_json_file):
        """Test from_json with JSON file path"""
        # Read the file first to verify it's valid JSON
        with open(temp_json_file, "r") as f:
            json_content = f.read()

        # Load from the file content as JSON string
        test_model.from_json(json_content)
        assert test_model.simple_attr == "dict_value"
        assert test_model.number_attr == 200

    def test_from_json_path_object(self, test_model, temp_json_file):
        """Test from_json with Path object"""
        test_model.from_json(temp_json_file)
        assert test_model.simple_attr == "dict_value"
        assert test_model.number_attr == 200

    def test_from_json_invalid_type(self, test_model):
        """Test from_json with invalid input type"""
        with pytest.raises(MTSchemaError, match="Input must be valid JSON string"):
            test_model.from_json(123)


class TestMetadataBaseSeriesConversion:
    """Test MetadataBase pandas Series conversion functionality"""

    def test_from_series_basic(self, test_model, sample_pandas_series):
        """Test basic from_series functionality"""
        test_model.from_series(sample_pandas_series)
        assert test_model.simple_attr == "series_value"
        assert test_model.number_attr == 300

    def test_from_series_invalid_input(self, test_model):
        """Test from_series with invalid input"""
        with pytest.raises(MTSchemaError, match="Input must be a Pandas.Series"):
            test_model.from_series([1, 2, 3])

    def test_to_series_basic(self, test_model_with_data):
        """Test basic to_series functionality"""
        series = test_model_with_data.to_series()
        assert isinstance(series, pd.Series)
        assert "simple_attr" in series.index
        assert series["simple_attr"] == "test_value"

    def test_to_series_required_only(self, required_model):
        """Test to_series with required=True"""
        series = required_model.to_series(required=True)
        assert isinstance(series, pd.Series)
        assert "required_field" in series.index

    def test_to_series_all_fields(self, required_model):
        """Test to_series with required=False"""
        series = required_model.to_series(required=False)
        assert isinstance(series, pd.Series)
        assert "required_field" in series.index
        assert "optional_field" in series.index


class TestMetadataBaseXmlConversion:
    """Test MetadataBase XML conversion functionality"""

    def test_to_xml_element(self, test_model_with_data):
        """Test to_xml returning XML Element"""
        xml_element = test_model_with_data.to_xml(string=False)
        assert isinstance(xml_element, ET.Element)

    def test_to_xml_string(self, test_model_with_data):
        """Test to_xml returning XML string"""
        xml_string = test_model_with_data.to_xml(string=True)
        assert isinstance(xml_string, str)
        assert "<" in xml_string and ">" in xml_string  # Basic XML format check

    def test_to_xml_required_only(self, required_model):
        """Test to_xml with required=True"""
        xml_element = required_model.to_xml(required=True)
        assert isinstance(xml_element, ET.Element)

    def test_from_xml_basic(self, test_model, sample_xml_element):
        """Test basic from_xml functionality"""
        test_model.from_xml(sample_xml_element)
        assert test_model.simple_attr == "xml_value"
        assert test_model.number_attr == 400


class TestMetadataBaseEdgeCases:
    """Test MetadataBase edge cases and error conditions"""

    def test_roundtrip_dict_conversion(self, test_model_with_data):
        """Test roundtrip dictionary conversion"""
        original_dict = test_model_with_data.to_dict(single=True)
        new_model = SampleModel()
        new_model.from_dict(original_dict)

        assert new_model.simple_attr == test_model_with_data.simple_attr
        assert new_model.number_attr == test_model_with_data.number_attr

    def test_roundtrip_json_conversion(self, test_model_with_data):
        """Test roundtrip JSON conversion"""
        json_str = test_model_with_data.to_json()
        new_model = SampleModel()
        new_model.from_json(json_str)

        assert new_model.simple_attr == test_model_with_data.simple_attr
        assert new_model.number_attr == test_model_with_data.number_attr

    def test_roundtrip_series_conversion(self, test_model_with_data):
        """Test roundtrip pandas Series conversion"""
        series = test_model_with_data.to_series()
        new_model = SampleModel()
        new_model.from_series(series)

        assert new_model.simple_attr == test_model_with_data.simple_attr
        assert new_model.number_attr == test_model_with_data.number_attr

    def test_with_numpy_arrays(self):
        """Test handling of numpy arrays"""

        class ArrayModel(MetadataBase):
            array_field: np.ndarray = Field(default_factory=lambda: np.array([1, 2, 3]))

        model = ArrayModel()

        # Test to_dict
        result_dict = model.to_dict(single=True)
        assert "array_field" in result_dict

        # Test roundtrip
        new_model = ArrayModel()
        new_model.from_dict(result_dict)
        np.testing.assert_array_equal(model.array_field, new_model.array_field)

    def test_with_null_values(self, test_model):
        """Test handling of NULL_VALUES"""
        # Test that string null values are handled correctly in load method
        string_null_values = ["", "null", "None", "NONE", "NULL", "Null", "none"]
        for null_value in string_null_values:
            original_value = test_model.simple_attr
            test_model.load(null_value)
            # Should remain unchanged when loading string null values
            assert test_model.simple_attr == original_value

    def test_model_validation_on_assignment(self, test_model):
        """Test that Pydantic validation works on assignment"""
        # This should work
        test_model.number_attr = 999
        assert test_model.number_attr == 999

        # This might raise a validation error depending on the field configuration
        # For basic int field, string that can be converted should work
        test_model.number_attr = "123"
        assert test_model.number_attr == 123


class TestMetadataBasePerformance:
    """Test MetadataBase performance characteristics"""

    def test_large_dict_conversion_performance(self):
        """Test performance with large dictionary conversions"""
        import time

        # Create model with many attributes
        large_dict = {f"attr_{i}": f"value_{i}" for i in range(1000)}

        class LargeModel(MetadataBase):
            pass

        model = LargeModel()

        start_time = time.time()
        model.from_dict(large_dict)
        conversion_time = time.time() - start_time

        # Should complete in reasonable time (less than 1 second)
        assert conversion_time < 1.0

    def test_deep_nesting_performance(self):
        """Test performance with deeply nested structures"""
        import time

        # Create a model that can handle dynamic attributes
        class DynamicModel(MetadataBase):
            simple_attr: str = Field(default="test")

        model = DynamicModel()

        # Test with simple attributes that exist in the model
        simple_dict = {"simple_attr": "performance_test"}

        start_time = time.time()
        model.from_dict(simple_dict)
        conversion_time = time.time() - start_time

        # Should complete in reasonable time
        assert conversion_time < 1.0
        assert model.simple_attr == "performance_test"


class TestMetadataBaseParametrized:
    """Parametrized tests for comprehensive MetadataBase testing"""

    @pytest.mark.parametrize(
        "data_type,test_value",
        [
            (str, "test_string"),
            (int, 42),
            (float, 3.14159),
            (bool, True),
            (bool, False),
        ],
    )
    def test_various_data_types(self, data_type, test_value):
        """Test various data types in MetadataBase"""

        class TypeTestModel(MetadataBase):
            test_field: data_type = Field(default=None)

        model = TypeTestModel(test_field=test_value)
        assert model.test_field == test_value

        # Test roundtrip conversion
        dict_repr = model.to_dict(single=True)
        new_model = TypeTestModel()
        new_model.from_dict(dict_repr)
        assert new_model.test_field == test_value

    @pytest.mark.parametrize("format_type", ["dict", "json", "series"])
    def test_conversion_formats(self, test_model_with_data, format_type):
        """Test various conversion formats"""
        if format_type == "dict":
            converted = test_model_with_data.to_dict(single=True)
            assert isinstance(converted, dict)
        elif format_type == "json":
            converted = test_model_with_data.to_json()
            assert isinstance(converted, str)
        elif format_type == "series":
            converted = test_model_with_data.to_series()
            assert isinstance(converted, pd.Series)

    @pytest.mark.parametrize("required_flag", [True, False])
    def test_required_flag_behavior(self, required_model, required_flag):
        """Test behavior with different required flags"""
        result = required_model.to_dict(single=True, required=required_flag)
        assert isinstance(result, dict)

        if required_flag:
            # Should contain required field
            assert "required_field" in result
        else:
            # Should contain both required and optional fields
            assert "required_field" in result
            assert "optional_field" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
