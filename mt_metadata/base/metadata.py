# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 20:41:16 2020

:copyright:
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""

from __future__ import annotations

import json
from collections import OrderedDict
from enum import Enum

# =============================================================================
# Imports
# =============================================================================
from operator import itemgetter
from pathlib import Path
from typing import Any, Mapping
from xml.etree import cElementTree as et

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import (
    BaseModel,
    computed_field,
    ConfigDict,
    create_model,
    field_validator,
    model_validator,
)
from pydantic.fields import FieldInfo, PrivateAttr
from typing_extensions import deprecated

from mt_metadata import NULL_VALUES
from mt_metadata.utils.exceptions import MTSchemaError
from mt_metadata.utils.validators import validate_attribute, validate_name

from . import helpers, pydantic_helpers


# =============================================================================
#  Base class that everything else will inherit
# =============================================================================


@deprecated("Base is deprecated, use MetadataBase instead")
class Base:
    pass


class DotNotationBaseModel(BaseModel):
    """
    Base model that supports dot notation for setting nested attributes.

    This model extends Pydantic's BaseModel to allow setting nested attributes
    using dot notation (e.g., 'location.latitude' or 'time_period.start').
    It automatically handles both flat and nested dictionary structures.

    Parameters
    ----------
    **data : Any
        Keyword arguments representing field values. Supports both flat keys
        and dot-notation keys for nested attributes.

    Examples
    --------
    >>> model = DotNotationBaseModel(**{"location.latitude": 45.0})
    >>> model = DotNotationBaseModel(**{"location": {"latitude": 45.0}})
    """

    def __init__(self, **data: Any) -> None:
        # Process dot notation fields first
        flat_data = {}
        nested_data = {}

        for key, value in data.items():
            if "." in key:
                # This is a dotted field, handle specially
                self._set_nested_attribute(nested_data, key, value)
            else:
                # Regular field, pass to Pydantic as-is
                if key == validate_name(self.__class__.__name__):
                    if isinstance(value, dict):
                        # If the value is a dict, we need to flatten it
                        for nested_key, nested_value in value.items():
                            if isinstance(nested_value, dict):
                                # Flatten nested dicts
                                self._set_nested_attribute(
                                    nested_data, nested_key, nested_value
                                )
                            else:
                                flat_data[nested_key] = nested_value
                    else:
                        # Non-dict value for class name key should be treated as regular field
                        flat_data[key] = value
                else:
                    flat_data[key] = value

        # Merge the nested dict into flat dict (nested takes precedence)
        flat_data.update(nested_data)

        # Call parent constructor with processed data
        super().__init__(**flat_data)

    def _set_nested_attribute(
        self, data_dict: dict[str, Any], dotted_key: str, value: Any
    ) -> None:
        """
        Set a nested attribute in data_dict based on dotted key notation.

        Parameters
        ----------
        data_dict : dict[str, Any]
            Dictionary to populate with nested structure
        dotted_key : str
            Dot-separated key path (e.g., 'time_period.start')
        value : Any
            Value to set at the nested location

        Examples
        --------
        >>> data = {}
        >>> model._set_nested_attribute(data, "time_period.start", "2020-01-01")
        >>> # Results in: {"time_period": {"start": "2020-01-01"}}
        """
        parts = dotted_key.split(".")
        current = data_dict

        # Navigate to the deepest level, creating dicts along the way
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                # Convert to dict if it's not already
                current[part] = {}
            current = current[part]

        # Set the final value
        current[parts[-1]] = value

    def update_attribute(self, attr_name: str, attr_value: Any) -> None:
        """
        Update a nested attribute using dot notation.

        Parameters
        ----------
        attr_name : str
            Name of the attribute to update, supports dot notation for
            nested attributes (e.g., 'time_period.start')
        attr_value : Any
            New value for the attribute

        Raises
        ------
        AttributeError
            If the attribute path does not exist

        Examples
        --------
        >>> model.update_attribute("time_period.start", "2020-01-01")
        >>> model.update_attribute("latitude", 45.0)
        """
        if "." not in attr_name:
            # Directly set the attribute
            setattr(self, attr_name, attr_value)
            return

        # For nested attributes, we need to navigate the object graph
        parts = attr_name.split(".")
        current = self

        # Navigate to the deepest level
        for part in parts[:-1]:
            if not hasattr(current, part):
                raise AttributeError(
                    f"'{type(current).__name__}' has no attribute '{part}'"
                )
            current = getattr(current, part)

        # Set the final attribute
        setattr(current, parts[-1], attr_value)
        setattr(current, parts[-1], attr_value)


class MetadataBase(DotNotationBaseModel):
    """
    Base class for all metadata objects with Pydantic validation.

    MetadataBase extends DotNotationBaseModel (which inherits from Pydantic's
    BaseModel) to provide automatic validation according to metadata standards.
    It adds functionality beyond dictionaries, supporting JSON, XML, pandas
    Series, and other formats for metadata interchange.

    Attributes
    ----------
    _skip_equals : list[str]
        Private attribute listing fields to skip in equality comparisons
    _fields : dict[str, Any]
        Private attribute caching field information

    Notes
    -----
    - All field assignments are validated automatically via Pydantic
    - None values are converted to appropriate defaults (empty string or 0.0)
    - Supports nested attribute access via dot notation
    - Thread-safe for read operations after initialization
    """

    model_config = ConfigDict(
        validate_assignment=True,
        use_attribute_docstrings=True,
        extra="allow",
        arbitrary_types_allowed=True,  # need this for numpy and pd types
        use_enum_values=True,
        coerce_numbers_to_str=True,
    )

    _skip_equals: list[str] = PrivateAttr(["processed_date", "creation_time"])
    _fields: dict[str, Any] = PrivateAttr(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def convert_none_to_empty(cls, values):
        """Convert None values to empty strings or 0.0 for numeric fields, except for fields that explicitly default to None."""
        # Ensure values is a dictionary before processing
        if not isinstance(values, dict):
            return values

        for field, field_info in cls.model_fields.items():
            # Skip conversion if the field's default is explicitly None
            if field_info.default is None:
                continue

            # Only process fields that are in the input values and are None
            if field in values and values[field] is None:
                try:
                    annotation = field_info.annotation
                    # Convert None to empty string for str fields
                    if annotation is str:
                        values[field] = ""
                    # Convert None to 0.0 for float/int fields
                    elif annotation in (float, int):
                        values[field] = 0.0
                except (AttributeError, TypeError):
                    # If there's any issue checking the annotation, skip conversion
                    pass
        return values

    @field_validator("*", mode="before")
    @classmethod
    def validate_none_on_assignment(cls, value: Any, info: Any) -> Any:
        """
        Convert None values to appropriate defaults when attributes are set.

        This validator runs for all fields due to 'validate_assignment=True' in
        model config. It works generically for string and numeric fields without
        requiring subclass-specific validators.

        Parameters
        ----------
        value : Any
            The value being assigned to the field
        info : Any
            Pydantic validation info containing field name and metadata

        Returns
        -------
        Any
            Converted value (empty string for str, 0.0 for numeric) or original value

        Notes
        -----
        - For complex types, skips conversion and lets Pydantic handle validation
        - Does NOT convert None if the field explicitly has None as its default
        - Conversion rules: str -> '', float/int -> 0.0
        """
        if value is None:
            field_name = info.field_name
            # Get field info from the class model fields
            if field_name in cls.model_fields:
                field_info = cls.model_fields[field_name]

                # Skip conversion if the field's default is explicitly None
                if field_info.default is None:
                    return value

                # Only attempt conversion for primitive types
                try:
                    # Check the annotation, handling both direct types and Annotated types
                    annotation = field_info.annotation

                    # Convert None to empty string for str fields
                    if annotation is str:
                        return ""
                    # Convert None to 0.0 for float/int fields
                    elif annotation in (float, int):
                        return 0.0
                except (AttributeError, TypeError):
                    # If there's any issue checking the annotation, let Pydantic handle it normally
                    pass
        return value

    @computed_field
    @property
    def _class_name(self) -> str:
        return validate_attribute(self.__class__.__name__)

    def __str__(self) -> str:
        """

        :return: table describing attributes
        :rtype: string

        """
        return str(self.model_dump())

    def __repr__(self) -> str:
        return self.to_json()

    def __eq__(
        self, other: "MetadataBase" | dict | str | pd.Series | et.Element
    ) -> bool:
        """
        Compare this metadata object with another for equality.

        This method supports comparison with various types by converting them
        to MetadataBase objects first. Uses Pydantic's equality comparison after
        loading the other object.

        Parameters
        ----------
        other : MetadataBase | dict | str | pd.Series | et.Element
            Object to compare with. Can be:
            - Another MetadataBase instance
            - Dictionary with metadata
            - JSON string
            - Pandas Series
            - XML Element

        Returns
        -------
        bool
            True if objects are equal, False otherwise

        Notes
        -----
        The following Pydantic equality logic is used:

           if isinstance(other, BaseModel):
            # When comparing instances of generic types for equality, as long as all field values are equal,
            # only require their generic origin types to be equal, rather than exact type equality.
            # This prevents headaches like MyGeneric(x=1) != MyGeneric[Any](x=1).
            self_type = self.__pydantic_generic_metadata__['origin'] or self.__class__
            other_type = other.__pydantic_generic_metadata__['origin'] or other.__class__

            # Perform common checks first
            if not (
                self_type == other_type
                and getattr(self, '__pydantic_private__', None) == getattr(other, '__pydantic_private__', None)
                and self.__pydantic_extra__ == other.__pydantic_extra__
            ):
                return False

            # We only want to compare pydantic fields but ignoring fields is costly.
            # We'll perform a fast check first, and fallback only when needed
            # See GH-7444 and GH-7825 for rationale and a performance benchmark

            # First, do the fast (and sometimes faulty) __dict__ comparison
            if self.__dict__ == other.__dict__:
                # If the check above passes, then pydantic fields are equal, we can return early
                return True

            # We don't want to trigger unnecessary costly filtering of __dict__ on all unequal objects, so we return
            # early if there are no keys to ignore (we would just return False later on anyway)
            model_fields = type(self).model_fields.keys()
            if self.__dict__.keys() <= model_fields and other.__dict__.keys() <= model_fields:
                return False

            # If we reach here, there are non-pydantic-fields keys, mapped to unequal values, that we need to ignore
            # Resort to costly filtering of the __dict__ objects
            # We use operator.itemgetter because it is much faster than dict comprehensions
            # NOTE: Contrary to standard python class and instances, when the Model class has a default value for an
            # attribute and the model instance doesn't have a corresponding attribute, accessing the missing attribute
            # raises an error in BaseModel.__getattr__ instead of returning the class attribute
            # So we can use operator.itemgetter() instead of operator.attrgetter()
            getter = operator.itemgetter(*model_fields) if model_fields else lambda _: _utils._SENTINEL
            try:
                return getter(self.__dict__) == getter(other.__dict__)
            except KeyError:
                # In rare cases (such as when using the deprecated BaseModel.copy() method),
                # the __dict__ may not contain all model fields, which is how we can get here.
                # getter(self.__dict__) is much faster than any 'safe' method that accounts
                # for missing keys, and wrapping it in a `try` doesn't slow things down much
                # in the common case.
                self_fields_proxy = _utils.SafeGetItemProxy(self.__dict__)
                other_fields_proxy = _utils.SafeGetItemProxy(other.__dict__)
                return getter(self_fields_proxy) == getter(other_fields_proxy)

        # other instance is not a BaseModel
        else:
            return NotImplemented  # delegate to the other item in the comparison
        """
        if other in [None]:
            return False

        elif isinstance(other, (dict, str, pd.Series, et.Element)):
            try:
                # Attempt to load the other object into a new instance of MetadataBase
                # This will ensure that the other object has the same attributes as self
                other_obj = __class__().load(other)
            except Exception as e:
                logger.error(
                    f"Failed to load other object of type {type(other)}: {other}. Error is: {e} "
                )
                return False
            if not other_obj:
                return False

            if hasattr(other_obj, "to_dict") and callable(other_obj.to_dict):
                other_dict = other_obj.to_dict(single=True, required=False)
            else:
                return False

        elif isinstance(other, MetadataBase):
            other_dict = other.to_dict(single=True, required=False)
        else:
            raise ValueError(
                f"Cannot compare {self.__class__.__name__} with {type(other)}"
            )
        home_dict = self.to_dict(single=True, required=False)
        try:
            if home_dict == other_dict:
                return True
        except ValueError:
            # Handle numpy arrays in dictionaries which cannot be directly compared
            pass

        equals = True
        for key, value in home_dict.items():
            skip_key_bool = False
            for skip_key in self._skip_equals:
                if skip_key in key:
                    skip_key_bool = True
            if skip_key_bool:
                continue
            try:
                other_value = other_dict[key]
                if isinstance(value, np.ndarray):
                    if value.size != other_value.size:
                        msg = f"Array sizes for {key} differ: {value.size} != {other_value.size}"
                        logger.info(msg)
                        equals = False
                        continue
                    if not (value == other_value).all():
                        msg = f"{key}: {value} != {other_value}"
                        logger.info(msg)
                        equals = False
                elif isinstance(value, (float, int, complex)):
                    # Handle None values in numeric comparisons
                    if other_value is None or value is None:
                        # Special case for coordinate fields: treat None and 0.0 as equal
                        coordinate_fields = ["x", "y", "z", "x2", "y2", "z2"]
                        is_coordinate_field = any(
                            key.endswith(f".{coord}") for coord in coordinate_fields
                        )

                        if is_coordinate_field and (
                            (value is None and other_value == 0.0)
                            or (value == 0.0 and other_value is None)
                        ):
                            # Coordinate fields: None and 0.0 are considered equivalent
                            continue
                        elif value != other_value:
                            msg = f"{key}: {value} != {other_value}"
                            logger.info(msg)
                            equals = False
                    elif not np.isclose(value, other_value):
                        msg = f"{key}: {value} != {other_value}"
                        logger.info(msg)
                        equals = False
                else:
                    if value in NULL_VALUES and other_value in NULL_VALUES:
                        continue
                    if value != other_value:
                        msg = f"{key}: {value} != {other_value}"
                        logger.info(msg)
                        equals = False
            except KeyError:
                msg = "Cannot find {0} in other".format(key)
                logger.info(msg)

        return equals

    def __ne__(
        self, other: "MetadataBase" | dict | str | pd.Series | et.Element
    ) -> bool:
        """
        Compare this metadata object with another for inequality.

        Parameters
        ----------
        other : MetadataBase | dict | str | pd.Series | et.Element
            Object to compare with

        Returns
        -------
        bool
            True if objects are not equal, False otherwise
        """
        return not self.__eq__(other)

    def __len__(self) -> int:
        """
        Return the number of attributes in this metadata object.

        Returns
        -------
        int
            Number of attributes (including nested attributes)
        """
        return len(self.get_attribute_list())

    def load(self, other: "MetadataBase" | dict | str | pd.Series | et.Element) -> None:
        """
        Load metadata from various formats and populate attributes.

        The other object should have the same attributes as the current object.
        If there are different attributes, validation may not be accurate.
        Consider making a new model if you need a different object structure.

        Parameters
        ----------
        other : MetadataBase | dict | str | pd.Series | et.Element
            Source object from which to fill attributes. Supported types:
            - MetadataBase: Another metadata instance
            - dict: Dictionary with metadata
            - str: JSON string representation
            - pd.Series: Pandas Series with metadata
            - et.Element: XML Element with metadata

        Raises
        ------
        MTSchemaError
            If the input type is not supported

        Examples
        --------
        >>> metadata = MetadataBase()
        >>> metadata.load({"latitude": 45.0, "longitude": -120.0})
        >>> metadata.load('{"latitude": 45.0}')
        """
        if isinstance(other, MetadataBase):
            self.update(other)
        elif isinstance(other, dict):
            self.from_dict(other)
        elif isinstance(other, str):
            if other.lower() in NULL_VALUES:
                return
            self.from_json(other)
        elif isinstance(other, pd.Series):
            self.from_series(other)
        elif isinstance(other, et.Element):
            self.from_xml(other)
        else:
            msg = f"Cannot load {type(other)} into {self.__class__.__name__}"
            logger.error(msg)
            raise MTSchemaError(msg)

    def update(self, other: "MetadataBase", match: list[str] = []) -> None:
        """
        Update attribute values from another like element, skipping None

        Parameters
        ----------
        other : MetadataBase
            other Base object from which to update attributes
        """
        if not isinstance(other, type(self)):
            # Allow updates between compatible metadata classes (e.g. enhanced vs original)
            if not (
                hasattr(other, "__class__")
                and hasattr(self, "__class__")
                and other.__class__.__name__ == self.__class__.__name__
            ):
                logger.warning(f"Cannot update {type(self)} with {type(other)}")
                return
        for k in match:
            if self.get_attr_from_name(k) != other.get_attr_from_name(k):
                msg = (
                    f"{k} is not equal {self.get_attr_from_name(k)} != "
                    f"{other.get_attr_from_name(k)}"
                )
                logger.error(msg)
                raise ValueError(msg)
        for k, v in other.to_dict(single=True).items():
            if hasattr(v, "size"):
                if v.size > 0:
                    self.update_attribute(k, v)
            else:
                if (
                    v
                    not in [None, 0.0, [], "", "1980-01-01T00:00:00+00:00"]
                    + NULL_VALUES
                ):
                    self.update_attribute(k, v)

    ## cannot override the __deepcopy__ method in pydantic.BaseModel otherwise bad
    ## things happen
    def copy(
        self, update: Mapping[str, Any] | None = None, deep: bool = True
    ) -> "MetadataBase":
        """
        Create a copy of the current metadata object.

        This is a wrapper around Pydantic's copy method with special handling
        for non-copyable objects like HDF5 references. Non-copyable objects
        are set to None in the copied object.

        Parameters
        ----------
        update : Mapping[str, Any] | None, optional
            Values to change/add in the new model. Note: the data is not
            validated before creating the new model, so ensure it's trustworthy.
            Default is None.
        deep : bool, optional
            If True, create a deep copy of the object. Default is True.

        Returns
        -------
        MetadataBase
            A copy of the current object with updates applied

        Raises
        ------
        TypeError
            If the object contains non-copyable objects and fallback fails

        Notes
        -----
        - HDF5 references cannot be deep copied and will be set to None
        - If deep copy fails, falls back to dictionary-based copying

        Examples
        --------
        >>> original = MetadataBase(latitude=45.0)
        >>> copy = original.copy(update={"latitude": 46.0})
        """

        # Handle HDF5 references and other non-copyable objects
        if update is None:
            update = {}
        else:
            update = dict(update)  # Convert to mutable dict

        # Check for HDF5 references that cannot be deep copied
        if deep and hasattr(self, "hdf5_reference"):
            hdf5_ref = getattr(self, "hdf5_reference", None)
            if hdf5_ref is not None:
                # Set to None to avoid deepcopy issues
                update["hdf5_reference"] = None

        # Also check for any other MTH5-specific fields that might not be copyable
        if hasattr(self, "mth5_type"):
            mth5_type_value = getattr(self, "mth5_type", None)
            # Only preserve mth5_type if it has a valid non-None value
            if mth5_type_value is not None:
                update["mth5_type"] = mth5_type_value

        try:
            copied_obj = self.model_copy(update=update, deep=deep)
        except (TypeError, AttributeError) as e:
            if "no default __reduce__" in str(e) or "__cinit__" in str(e):
                # Fallback: create a new instance from dictionary representation
                # This avoids any non-copyable objects entirely
                self_dict = self.to_dict()
                new_instance = type(self)()
                new_instance.from_dict(self_dict)

                # Apply any updates
                for key, value in update.items():
                    if hasattr(new_instance, key):
                        setattr(new_instance, key, value)

                return new_instance
            else:
                # Re-raise if it's a different error
                raise

        return copied_obj

    def get_all_fields(self) -> dict:
        """
        Get all field attributes in the Metadata class.  Will
        search recursively and return dotted keys.  For
        instance `{location.latitude: ...}`.

        Returns
        -------
        Dict
            A flattened dictionary of dotted keys of all attributes
            within the class.
        """

        if not self._fields:
            self._fields = pydantic_helpers.flatten_field_tree_map(
                pydantic_helpers.get_all_fields_serializable(self)
            )
        return self._fields

    def get_attribute_list(self) -> list[str]:
        """
        return a list of the attributes

        Returns
        -------
        list[str]
            A list of attribute names
        """

        return sorted(self.get_all_fields().keys())

    @property
    def _required_fields(self) -> list[str]:
        """
        Get a list of required fields according to metadata standards.

        There is a distinction between "required" in Pydantic (must be defined
        on instantiation) and "required" in metadata standards (must be present
        in the standard even if the value is None).

        Returns
        -------
        list[str]
            List of required field names in the metadata standards

        Notes
        -----
        Required fields are determined by the 'required' flag in field metadata,
        not by Pydantic's required_on_init behavior.
        """
        required_fields = []
        for name, field_dict in self.get_all_fields().items():
            required = field_dict.get("required", False)
            if required:
                required_fields.append(name)

        return required_fields

    def _field_info_to_string(self, name: str, field_dict: dict[str, Any]) -> str:
        """
        Create a formatted string from field information for pretty printing.

        Parameters
        ----------
        name : str
            Name of the field
        field_dict : dict[str, Any]
            Dictionary containing field metadata (type, description, units, etc.)

        Returns
        -------
        str
            Formatted string representation of the field information

        Examples
        --------
        >>> info = {"type": "float", "description": "Latitude", "units": "degrees"}
        >>> result = obj._field_info_to_string("latitude", info)
        """

        line = [f"{name}:"]

        for key, value in field_dict.items():
            line.append(f"\t{key}: {value}")

        return "\n".join(line)

    def attribute_information(self, name: str | None = None) -> None:
        """
        Print descriptive information about attributes.

        If name is provided, prints information for that specific attribute.
        Otherwise, prints information for all attributes.

        Parameters
        ----------
        name : str | None, optional
            Attribute name for a specific attribute. If None, prints information
            for all attributes. Default is None.

        Raises
        ------
        MTSchemaError
            If the specified attribute name is not found

        Examples
        --------
        >>> metadata.attribute_information("latitude")
        >>> metadata.attribute_information()  # Print all attributes
        """
        attr_dict = self.get_all_fields()
        lines = []
        if name:
            try:
                v_dict = attr_dict[name]
            except KeyError as error:
                msg = f"{error} not attribute {name} found."
                logger.error(msg)
                raise MTSchemaError(msg)
            lines.append(self._field_info_to_string(name, v_dict))
        else:
            lines = []
            for name, v_dict in attr_dict.items():
                lines.append(self._field_info_to_string(name, v_dict))
                lines.append("=" * 50)
        print("\n".join(lines))

    def get_attr_from_name(self, name: str) -> Any:
        """
        Access attribute from the given name, supporting dot notation.

        The name can contain nested object references separated by dots,
        e.g., 'location.latitude' or 'time_period.start'.

        Parameters
        ----------
        name : str
            Name of attribute to get, may include dots for nested attributes

        Returns
        -------
        Any
            The attribute value

        Raises
        ------
        KeyError
            If the attribute is not found
        AttributeError
            If the attribute path is invalid

        Examples
        --------
        >>> metadata = MetadataBase(**{'location.latitude': 45.0})
        >>> metadata.get_attr_from_name('location.latitude')
        45.0

        Notes
        -----
        This is a helper function for names with '.' for easier access when
        reading from dictionaries or other flat structures.
        """
        value, _ = helpers.recursive_split_getattr(self, name)
        return value

    @deprecated(
        "set_attr_from_name will be deprecated in the future. Use update_attribute."
    )
    def set_attr_from_name(self, name: str, value: Any) -> None:
        """
        Helper function to set attribute from the given name.

        The name can contain the name of an object which must be separated
        by a '.' for  e.g. {object_name}.{name} --> location.latitude

        .. note:: this is a helper function for names with '.' in the name for
         easier getting when reading from dictionary.

        :param name: name of attribute to get.
        :type name: string
        :param value: attribute value
        :type value: type is defined by the attribute name

        :Example:

        >>> b = Base(**{'category.test_attr':10})
        >>> b.set_attr_from_name('category.test_attr', '10')
        >>> print(b.category.test_attr)
        '10'
        """

    @deprecated("add_base_attribute is deprecated. Use add_new_field.")
    def add_base_attribute(
        self,
    ):
        pass

    def add_new_field(self, name: str, new_field_info: FieldInfo) -> BaseModel:
        """
        This is going to be much different from older versions of mt_metadata.

        This will return a new BaseModel with the added attribute.  Going to use
        `pydantid.create_model` from the exsiting attribute information and the
        added attribute.

        Add an attribute to _attr_dict so it will be included in the
        output dictionary

        Parameters
        ----------
        name : str
            name of attribute
        new_field_info : FieldInfo
            value of the new attribute

        Returns
        -------
        BaseModel
            A new BaseModel instance with the added attribute.

        Should include:

        * annotated --> the data type [ str | int | float | bool ]
        * required --> required in the standards [ True | False ]
        * units --> units of the attribute, must be a string
        * alias --> other possible names for the attribute
        * options --> if only a few options are accepted, separated by | or
           comma.b [ option_01 | option_02 | other ]. 'other' means other options
           available but not yet defined.
        * example --> an example of the attribute

        :Example:

        .. code-block:: python

        from pydantic.fields import FieldInfo
        new_field = FieldInfo(
            annotated=str,
            default="default_value",
            required=False,
            description="new field description",
            alias="new_field_alias",
            json_schema_extra={"units":"km"}
            )

        existing_basemodel = MetadataBase()
        new_basemodel = existing_basemodel.add_new_field("new_attribute", new_field)
        new_basemodel_object = new_basemodel()

        """
        existing_model_fields = self.__pydantic_fields__.copy()
        existing_model_fields[name] = new_field_info
        all_fields = {k: (v.annotation, v) for k, v in existing_model_fields.items()}

        return create_model(  # type: ignore
            self.__class__.__name__,  # Preserve the original class name
            __base__=self.__class__,  # Preserve the original class hierarchy
            **all_fields,
        )

    def to_dict(
        self, nested: bool = False, single: bool = False, required: bool = True
    ) -> dict[str, Any]:
        """
        Convert metadata to a dictionary representation.

        Parameters
        ----------
        nested : bool, optional
            If True, return a nested dictionary structure. If False, use
            dot-notation for nested keys. Default is False.
        single : bool, optional
            If True, return just the metadata dictionary without the class name
            wrapper (meta_dict[class_name]). Default is False.
        required : bool, optional
            If True, return only required elements and elements with non-None
            values. If False, include all fields. Default is True.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the metadata

        Notes
        -----
        - Comment objects are converted to simple strings for backward compatibility
          when they only contain a value (no author or custom timestamp)
        - Numpy arrays, Enums, and nested MetadataBase objects are handled specially
        - Required fields are always included even if None

        Examples
        --------
        >>> metadata.to_dict(nested=True, single=True)
        >>> metadata.to_dict(required=False)  # Include all fields
        """

        meta_dict = {}

        # Keep track of processed comment attributes to avoid duplication
        processed_comments = set()

        for name in self.get_attribute_list():
            # Special handling for comment attributes for backwards compatibility
            if (
                ".value" in name
                and name.replace(".value", "") not in processed_comments
            ):
                base_attr_name = name.replace(".value", "")
                # Check if this is a comment attribute
                try:
                    comment_obj = self.get_attr_from_name(base_attr_name)
                    if (
                        hasattr(comment_obj, "__class__")
                        and comment_obj.__class__.__name__ == "Comment"
                    ):
                        # Check if this is a simple comment (only value set, no author or custom timestamp)
                        default_timestamp = "1980-01-01T00:00:00+00:00"
                        is_simple_comment = (
                            hasattr(comment_obj, "value")
                            and comment_obj.value is not None
                            and isinstance(comment_obj.value, str)
                            and (
                                not hasattr(comment_obj, "author")
                                or comment_obj.author is None
                                or comment_obj.author == ""
                            )
                            and (
                                not hasattr(comment_obj, "time_stamp")
                                or comment_obj.time_stamp is None
                                or str(comment_obj.time_stamp) == default_timestamp
                            )
                        )

                        if is_simple_comment and not nested:
                            # Use simple string format for backwards compatibility
                            if required:
                                if comment_obj.value not in [
                                    None,
                                    "1980-01-01T00:00:00+00:00",
                                    "1980",
                                    [],
                                    "",
                                ]:
                                    meta_dict[base_attr_name] = str(comment_obj.value)
                            else:
                                meta_dict[base_attr_name] = str(comment_obj.value)

                            # Mark this comment as processed to skip its nested attributes
                            processed_comments.add(base_attr_name)
                            continue
                        else:
                            # Use nested format - let individual attributes be processed normally
                            pass
                except (AttributeError, KeyError):
                    # Not a comment object or attribute doesn't exist, process normally
                    pass

            # Skip nested comment attributes if we already processed the base comment
            skip_attribute = False
            for processed_comment in processed_comments:
                if name.startswith(processed_comment + "."):
                    skip_attribute = True
                    break

            if skip_attribute:
                continue

            try:
                value = self.get_attr_from_name(name)
                # Special handling for Comment objects for backwards compatibility
                if (
                    hasattr(value, "__class__")
                    and value.__class__.__name__ == "Comment"
                ):
                    # Check if this is a simple comment (only value set, no author or custom timestamp)
                    default_timestamp = "1980-01-01T00:00:00+00:00"
                    is_simple_comment = (
                        hasattr(value, "value")
                        and value.value is not None
                        and isinstance(value.value, str)
                        and (
                            not hasattr(value, "author")
                            or value.author is None
                            or value.author == ""
                        )
                        and (
                            not hasattr(value, "time_stamp")
                            or value.time_stamp is None
                            or str(value.time_stamp) == default_timestamp
                        )
                    )

                    if is_simple_comment and not nested:
                        # Return simple string for backwards compatibility
                        value = str(value.value)
                    else:
                        # Return full nested format
                        value = value.to_dict(nested=nested, required=required)
                elif hasattr(value, "to_dict"):
                    value = value.to_dict(nested=nested, required=required)
                elif isinstance(value, dict):
                    for key, obj in value.items():
                        if hasattr(obj, "to_dict"):
                            value[key] = obj.to_dict(nested=nested, required=required)
                        elif isinstance(obj, Enum):
                            value[key] = obj.value
                        else:
                            value[key] = obj
                elif isinstance(value, list):
                    v_list = []
                    for obj in value:
                        if hasattr(obj, "to_dict"):
                            v_list.append(obj.to_dict(nested=nested, required=required))
                        elif isinstance(obj, Enum):
                            v_list.append(obj.value)
                        else:
                            v_list.append(obj)
                    value = v_list
                elif isinstance(value, Enum):
                    value = value.value
                elif hasattr(value, "unicode_string"):
                    value = value.unicode_string()
                elif isinstance(value, (str, int, float, bool)):
                    value = value
            except AttributeError as error:
                logger.debug(error)
                value = None
            if required:
                if isinstance(value, (np.ndarray)):
                    if name == "zeros" or name == "poles":
                        meta_dict[name] = value
                    elif value.all() != 0:
                        meta_dict[name] = value
                elif hasattr(value, "size"):
                    if value.size > 0:
                        meta_dict[name] = value
                elif (
                    value not in [None, "1980-01-01T00:00:00+00:00", "1980", [], ""]
                    or name in self._required_fields
                    or helpers._should_include_coordinate_field(name)
                    or helpers._should_convert_none_to_empty_string(name)
                ):
                    # Convert None coordinate fields to 0.0 for backward compatibility
                    if helpers._should_include_coordinate_field(name) and value is None:
                        value = 0.0
                    # Convert None string fields to empty string for backward compatibility
                    elif (
                        helpers._should_convert_none_to_empty_string(name)
                        and value is None
                    ):
                        value = ""
                    meta_dict[name] = value
            else:
                meta_dict[name] = value
        if nested:
            meta_dict = helpers.structure_dict(meta_dict)
        meta_dict = {
            validate_name(self.__class__.__name__): OrderedDict(
                sorted(meta_dict.items(), key=itemgetter(0))
            )
        }

        if single:
            meta_dict = meta_dict[list(meta_dict.keys())[0]]
        return meta_dict

    def from_dict(self, meta_dict: dict, skip_none: bool = False) -> None:
        """
        Fill attributes from a dictionary.

        The dictionary can be nested or flat with dot-notation keys. If the
        dictionary has a single key matching the class name, it will be
        unwrapped automatically.

        Parameters
        ----------
        meta_dict : dict
            Dictionary with keys equal to metadata attribute names. Supports
            both nested dictionaries and flat dictionaries with dot-notation keys.
        skip_none : bool, optional
            If True, skip attributes with None values. Default is False.

        Raises
        ------
        MTSchemaError
            If the input is not a valid dictionary

        Examples
        --------
        >>> metadata.from_dict({"latitude": 45.0, "longitude": -120.0})
        >>> metadata.from_dict({"location": {"latitude": 45.0}})
        """
        if not isinstance(meta_dict, (dict, OrderedDict)):
            msg = f"Input must be a dictionary not {type(meta_dict)}"
            logger.error(msg)
            raise MTSchemaError(msg)
        keys = list(meta_dict.keys())
        if len(keys) == 1:
            if isinstance(meta_dict[keys[0]], (dict, OrderedDict)):
                class_name = keys[0]
                if class_name.lower() != validate_name(self.__class__.__name__):
                    msg = (
                        "name of input dictionary is not the same as class type "
                        f"input = {class_name}, class type = {self.__class__.__name__}"
                    )
                    logger.debug(msg, class_name, self.__class__.__name__)
                meta_dict = helpers.flatten_dict(meta_dict[class_name])
            else:
                meta_dict = helpers.flatten_dict(meta_dict)

        else:
            logger.debug(
                f"Assuming input dictionary is of type {self.__class__.__name__}",
            )
            meta_dict = helpers.flatten_dict(meta_dict)
        # set attributes by key.
        for name, value in meta_dict.items():
            if skip_none:
                if value in NULL_VALUES:
                    continue
            if value is None:
                try:
                    self.update_attribute(name, value)
                except Exception:
                    logger.debug(
                        f"Skipping attribute '{name}' because None is not valid "
                        f"for this field; keeping existing value."
                    )
                    continue
            else:
                self.update_attribute(name, value)

    def to_json(
        self, nested: bool = False, indent: str = " " * 4, required: bool = True
    ) -> str:
        """
        Write a json string from a given object, taking into account other
        class objects contained within the given object.

        Parameters
        ----------
        indent : str
            indentation for the json string, default is 4 spaces

        nested : bool
            make the returned json nested
        required : bool
            return just the required elements and any elements with non-None values

        Returns
        -------
        str
            json string representation of the object

        """

        return json.dumps(
            self.to_dict(nested=nested, required=required),
            cls=helpers.NumpyEncoder,
            indent=indent,
        )

    def from_json(self, json_str: str | Path) -> None:
        """
        read in a json string and update attributes of an object

        Parameters
        ----------
        json_str : str | Path
            json string or file path to json file

        """
        if isinstance(json_str, str):
            try:
                json_path = Path(json_str)
                if json_path.exists():
                    with open(json_path, "r") as fid:
                        json_dict = json.load(fid)
            except OSError:
                pass
            json_dict = json.loads(json_str)
        elif isinstance(json_str, Path):
            if json_str.exists():
                with open(json_str, "r") as fid:
                    json_dict = json.load(fid)
        elif not isinstance(json_str, (str, Path)):
            msg = f"Input must be valid JSON string not {type(json_str)}"
            logger.error(msg)
            raise MTSchemaError(msg)
        self.from_dict(json_dict)

    def from_series(self, pd_series: pd.Series) -> None:
        """
        Fill attributes from a Pandas Series.

        Parameters
        ----------
        pd_series : pd.Series
            Series containing metadata information. The series must be single
            layered with key names separated by dots for nested attributes
            (e.g., 'location.latitude').

        Raises
        ------
        MTSchemaError
            If the input is not a Pandas Series

        Examples
        --------
        >>> series = pd.Series({"latitude": 45.0, "longitude": -120.0})
        >>> metadata.from_series(series)

        Notes
        -----
        Types are not currently enforced from the series - validation occurs
        via Pydantic after assignment.
        """
        if not isinstance(pd_series, pd.Series):
            msg = f"Input must be a Pandas.Series not type {type(pd_series)}"
            logger.error(msg)
            raise MTSchemaError(msg)
        for key, value in pd_series.items():
            key = str(key)
            self.update_attribute(key, value)

    def to_series(self, required: bool = True) -> pd.Series:
        """
        Convert attribute list to a pandas.Series

        .. note:: this is a flattened version of the metadata

        Parameters
        ----------
        required : bool
            return just the required elements and any elements with non-None values

        Returns
        -------
        pandas.Series
            Series containing the metadata information

        """

        return pd.Series(self.to_dict(single=True, required=required))

    def to_xml(self, string: bool = False, required: bool = True) -> str | et.Element:
        """
        Convert metadata to an XML representation.

        Creates an XML element with type and unit information for each attribute.

        Parameters
        ----------
        string : bool, optional
            If True, return XML as a string. If False, return an XML Element.
            Default is False.
        required : bool, optional
            If True, include only required elements and elements with non-None
            values. If False, include all elements. Default is True.

        Returns
        -------
        str | et.Element
            XML Element object if string=False, otherwise XML string

        Examples
        --------
        >>> xml_elem = metadata.to_xml()
        >>> xml_str = metadata.to_xml(string=True)
        """
        attr_dict = self.get_all_fields()
        element = helpers.dict_to_xml(
            self.to_dict(nested=True, required=required), attr_dict
        )
        if not string:
            return element
        else:
            return helpers.element_to_string(element)

    def from_xml(self, xml_element: et.Element) -> None:
        """
        Fill attributes from an XML element.

        Parameters
        ----------
        xml_element : et.Element
            XML element from which to fill attributes. The element structure
            should match the metadata schema.

        Examples
        --------
        >>> import xml.etree.ElementTree as et
        >>> xml_str = '<metadata><latitude>45.0</latitude></metadata>'
        >>> elem = et.fromstring(xml_str)
        >>> metadata.from_xml(elem)

        Notes
        -----
        The XML element is converted to a dictionary first, then loaded
        via the from_dict method.
        """

        self.from_dict(helpers.element_to_dict(xml_element))
