# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 20:37:52 2020

:copyright:
    Jared Peacock (jpeacock@usgs.gov)

:license: MIT

"""

import copy
import json
import logging
import re

# =============================================================================
# Imports
# =============================================================================
import textwrap
from collections import defaultdict, OrderedDict
from collections.abc import MutableMapping
from operator import itemgetter
from pathlib import Path
from xml.dom import minidom
from xml.etree import cElementTree as et

import numpy as np
from loguru import logger
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

filter_descriptions = {
    "zpk": "poles and zeros filter",
    "coefficient": "coefficient filter",
    "time delay": "time delay filter",
    "fir": "finite impaulse response filter",
    "fap": "frequency amplitude phase lookup table",
    "frequency response table": "frequency amplitude phase lookup table",
    "base": "base filter",
}

# =============================================================================
# write doc strings
# =============================================================================


def wrap_description(description, column_width):
    """
    split a description into separate lines
    """
    if isinstance(description, list):
        description = " ".join([str(d) for d in description])
        description = description.strip()
    elif not isinstance(description, str):
        description = str(description)
        description = description.strip()
    d_lines = textwrap.wrap(description, column_width)
    if len(d_lines) < 11:
        d_lines += [""] * (11 - len(d_lines))
    return d_lines


def validate_c1(attr_dict, c1):
    """
    Validate column 1 width based on attribute dictionary

    Parameters
    ----------
    attr_dict : dict
        DESCRIPTION
    c1 : int
        DESCRIPTION

    Returns
    -------
    int
        DESCRIPTION
    """
    try:
        max_c1 = max([len(key) for key in attr_dict.keys()])

        if max_c1 > (c1 - 4):
            c1 = max_c1 + 6
    except ValueError:
        pass

    return c1


def write_lines(field_dict, c1=45, c2=45, c3=15):
    """
    Takes a dictionary of field names to FieldInfo objects and parses it into a table
    Returns a string representation of this table.  This overwrites the doc.

    Parameters
    ----------
    field_dict : dict
        dictionary mapping field names to FieldInfo objects
    c1 : int, optional
        column 1 width, by default 45
    c2 : int, optional
        column 2 width, by default 45
    c3 : int, optional
        column 3 width, by default 15

    Returns
    -------
    str
        doc string
    """
    c1 = validate_c1(field_dict, c1)

    line = "       | {0:<{1}}| {2:<{3}} | {4:<{5}}|"
    hline = "       +{0}+{1}+{2}+".format(
        "-" * (c1 + 1), "-" * (c2 + 2), "-" * (c3 + 1)
    )
    mline = "       +{0}+{1}+{2}+".format(
        "=" * (c1 + 1), "=" * (c2 + 2), "=" * (c3 + 1)
    )

    lines = [
        hline,
        line.format("**Metadata Key**", c1, "**Description**", c2, "**Example**", c3),
        mline,
    ]

    for key, field_info in field_dict.items():
        if isinstance(field_info, logging.Logger):
            continue

        # Extract description from FieldInfo
        description = field_info.description or ""
        d_lines = wrap_description(description, c2)

        # Extract examples from json_schema_extra
        examples = ""
        if field_info.json_schema_extra and isinstance(
            field_info.json_schema_extra, dict
        ):
            examples = field_info.json_schema_extra.get("examples", "")
        e_lines = wrap_description(examples, c3)

        # Get required status
        required = "False"
        if field_info.json_schema_extra and isinstance(
            field_info.json_schema_extra, dict
        ):
            required = str(field_info.json_schema_extra.get("required", False))

        # Get units
        units = ""
        if field_info.json_schema_extra and isinstance(
            field_info.json_schema_extra, dict
        ):
            units = str(field_info.json_schema_extra.get("units", ""))

        # Get type from annotation
        field_type = str(field_info.annotation) if field_info.annotation else "string"

        # Get style
        style = "free form"
        if field_info.json_schema_extra and isinstance(
            field_info.json_schema_extra, dict
        ):
            style = field_info.json_schema_extra.get("style", "free form")

        # line 1 is with the entry
        lines.append(line.format(f"**{key}**", c1, d_lines[0], c2, e_lines[0], c3))
        # line 2 skip an entry in the
        lines.append(line.format("", c1, d_lines[1], c2, e_lines[1], c3))
        # line 3 required
        lines.append(
            line.format(
                f"Required: {required}",
                c1,
                d_lines[2],
                c2,
                e_lines[2],
                c3,
            )
        )
        # line 4 blank
        lines.append(line.format("", c1, d_lines[3], c2, e_lines[3], c3))

        # line 5 units
        lines.append(line.format(f"Units: {units}", c1, d_lines[4], c2, e_lines[4], c3))

        # line 6 blank
        lines.append(line.format("", c1, d_lines[5], c2, e_lines[5], c3))

        # line 7 type
        lines.append(
            line.format(f"Type: {field_type}", c1, d_lines[6], c2, e_lines[6], c3)
        )

        # line 8 blank
        lines.append(line.format("", c1, d_lines[7], c2, e_lines[7], c3))

        # line 9 style
        lines.append(line.format(f"Style: {style}", c1, d_lines[8], c2, e_lines[8], c3))

        # line 10 blank
        lines.append(line.format("", c1, d_lines[9], c2, e_lines[9], c3))

        # Handle default value - similar to write_block
        default_value = field_info.default
        if default_value is PydanticUndefined:
            if (
                field_info.default_factory is not None
                and field_info.default_factory is not PydanticUndefined
            ):
                try:
                    # Some default factories may require arguments, handle both cases
                    if callable(field_info.default_factory):
                        try:
                            default_value = field_info.default_factory()
                        except TypeError:
                            # If it needs arguments, we can't call it
                            default_value = f"<{field_info.default_factory.__name__}>"
                    else:
                        default_value = field_info.default_factory
                    # If it's a complex object, just show the type name
                    if not isinstance(
                        default_value, (str, int, float, bool, type(None))
                    ):
                        default_value = type(default_value).__name__
                except Exception:
                    default_value = None
            else:
                default_value = None

        default = [str(default_value)] + [""] * 5
        if len(str(default_value)) > c1 - 15:
            default = [""] + wrap_description(str(default_value), c1)

        # line 9 type
        lines.append(
            line.format(
                f"**Default**: {default[0]}",
                c1,
                d_lines[8],
                c2,
                e_lines[8],
                c3,
            )
        )

        # line 10 blank
        lines.append(line.format(default[1], c1, d_lines[9], c2, e_lines[9], c3))

        # line 9 type
        lines.append(line.format(default[2], c1, d_lines[10], c2, e_lines[10], c3))

        # line 10 blank
        if len(d_lines) > 11:
            lines.append(line.format(default[3], c1, d_lines[11], c2, "", c3))
            for index, d_line in enumerate(d_lines[12:], 4):
                try:
                    lines.append(line.format(default[index], c1, d_line, c2, "", c3))
                except IndexError:
                    lines.append(line.format("", c1, d_line, c2, "", c3))

        # long default value
        if len(default) > 7:
            lines.append(line.format(default[3], c1, "", c2, "", c3))
            for index, d_line in enumerate(default[4:], 12):
                try:
                    lines.append(line.format(d_line, c1, d_lines[index], c2, "", c3))
                except IndexError:
                    lines.append(line.format(d_line, c1, "", c2, "", c3))
        lines.append(hline)
    return "\n".join(lines)


def write_block(key, field_info: FieldInfo, c1=45, c2=45, c3=15):
    """

    :param key: key to write from attr dict
    :type key: string
    :param field_info: field information dictionary
    :type field_info: dict
    :param c1: column 1 width, defaults to 45
    :type c1: int, optional
    :param c2: column 2 width, defaults to 45
    :type c2: int, optional
    :param c3: column 3 width, defaults to 15
    :type c3: int, optional
    :return: list of lines
    :rtype: list

    """
    if len(key) > c1 - 4:
        c1 = len(key) + 6

    line = "       | {0:<{1}}| {2:<{3}} | {4:<{5}}|"
    hline = "       +{0}+{1}+{2}+".format(
        "-" * (c1 + 1), "-" * (c2 + 2), "-" * (c3 + 1)
    )
    mline = "       +{0}+{1}+{2}+".format(
        "=" * (c1 + 1), "=" * (c2 + 2), "=" * (c3 + 1)
    )
    section = f":navy:`{key}`"

    lines = [
        section,
        "~" * len(section),
        "",
        ".. container::",
        "",
        "   .. table::",
        "       :class: tight-table",
        f"       :widths: {c1} {c2} {c3}",
        "",
        hline,
        line.format(f"**{key}**", c1, "**Description**", c2, "**Example**", c3),
        mline,
    ]

    t_lines = wrap_description(field_info.annotation, c1 - 10)
    d_lines = wrap_description(field_info.description, c2)

    # Safely get examples from json_schema_extra
    examples = ""
    if field_info.json_schema_extra and isinstance(field_info.json_schema_extra, dict):
        examples = field_info.json_schema_extra.get("examples", "")
    e_lines = wrap_description(examples, c3)

    # Safely get required and units
    required = "False"
    units = ""
    if field_info.json_schema_extra and isinstance(field_info.json_schema_extra, dict):
        required = str(field_info.json_schema_extra.get("required", False))
        units = str(field_info.json_schema_extra.get("units", ""))

    # line 1 is with the entry
    lines.append(
        line.format(
            f"**Required**: {required}",
            c1,
            d_lines[0],
            c2,
            e_lines[0],
            c3,
        )
    )
    # line 2 skip an entry in the
    lines.append(line.format("", c1, d_lines[1], c2, e_lines[1], c3))
    # line 3 required
    lines.append(
        line.format(
            f"**Units**: {units}",
            c1,
            d_lines[2],
            c2,
            e_lines[2],
            c3,
        )
    )
    # line 4 blank
    lines.append(line.format("", c1, d_lines[3], c2, e_lines[3], c3))

    # line 5 units
    lines.append(
        line.format(
            f"**Type**: {t_lines[0]}",
            c1,
            d_lines[4],
            c2,
            e_lines[4],
            c3,
        )
    )

    # line 6 blank
    lines.append(line.format(t_lines[1], c1, d_lines[5], c2, e_lines[5], c3))

    # line 7 - continuation of type if needed
    lines.append(line.format(t_lines[2], c1, d_lines[6], c2, e_lines[6], c3))

    # Add additional lines if type annotation is very long (more than 2 lines)
    if len(t_lines) > 3:
        for i in range(3, min(len(t_lines), 6)):  # Add up to 3 more lines for type
            desc_index = 7 + (i - 3)
            example_index = 7 + (i - 3)
            lines.append(
                line.format(
                    t_lines[i],
                    c1,
                    d_lines[desc_index] if desc_index < len(d_lines) else "",
                    c2,
                    e_lines[example_index] if example_index < len(e_lines) else "",
                    c3,
                )
            )

    # line 8+ blank (adjust based on how many type lines we added)
    type_lines_used = min(len(t_lines), 6)
    desc_start_index = 5 + type_lines_used - 3  # Adjust description index
    example_start_index = 5 + type_lines_used - 3  # Adjust example index

    lines.append(
        line.format(
            "",
            c1,
            d_lines[desc_start_index] if desc_start_index < len(d_lines) else "",
            c2,
            e_lines[example_start_index] if example_start_index < len(e_lines) else "",
            c3,
        )
    )

    # Handle default value - always convert to string
    default_value = field_info.default
    if default_value is PydanticUndefined:
        if (
            field_info.default_factory is not None
            and field_info.default_factory is not PydanticUndefined
        ):
            try:
                # Some default factories may require arguments, handle both cases
                if callable(field_info.default_factory):
                    try:
                        default_value = field_info.default_factory()
                    except TypeError:
                        # If it needs arguments, we can't call it
                        default_value = f"<{field_info.default_factory.__name__}>"
                else:
                    default_value = field_info.default_factory
                # If it's a complex object, just show the type name
                if not isinstance(default_value, (str, int, float, bool, type(None))):
                    default_value = type(default_value).__name__
            except Exception:
                default_value = "None"
        else:
            default_value = "None"

    # Ensure default_value is always a string
    default_value_str = str(default_value)

    # Handle special cases for display
    if default_value_str == "":
        default_value_str = '""'  # Show empty string explicitly
    elif default_value_str == "None":
        default_value_str = "None"  # Keep None as is

    # Wrap default value if it's too long
    if len(default_value_str) > c1 - 15:
        default_lines = wrap_description(default_value_str, c1)
        default = [""] + default_lines
    else:
        default = [default_value_str] + [""] * 10  # Ensure we have enough empty strings

    # Ensure we have at least 11 items in default list
    while len(default) < 11:
        default.append("")

    # Calculate the description and example line indices for the default section
    # Account for the additional type lines we may have added
    default_desc_start = 8 + max(
        0, min(len(t_lines) - 3, 3)
    )  # Start after type section
    default_example_start = 8 + max(0, min(len(t_lines) - 3, 3))

    # line N - Default value (where N depends on type length)
    lines.append(
        line.format(
            f"**Default**: {default[0]}",
            c1,
            d_lines[default_desc_start] if default_desc_start < len(d_lines) else "",
            c2,
            (
                e_lines[default_example_start]
                if default_example_start < len(e_lines)
                else ""
            ),
            c3,
        )
    )

    # line N+1 - continuation of default/description
    lines.append(
        line.format(
            default[1],
            c1,
            (
                d_lines[default_desc_start + 1]
                if (default_desc_start + 1) < len(d_lines)
                else ""
            ),
            c2,
            (
                e_lines[default_example_start + 1]
                if (default_example_start + 1) < len(e_lines)
                else ""
            ),
            c3,
        )
    )

    # line N+2 - continuation of default/description
    lines.append(
        line.format(
            default[2],
            c1,
            (
                d_lines[default_desc_start + 2]
                if (default_desc_start + 2) < len(d_lines)
                else ""
            ),
            c2,
            (
                e_lines[default_example_start + 2]
                if (default_example_start + 2) < len(e_lines)
                else ""
            ),
            c3,
        )
    )

    # Handle additional description lines if they exist
    if len(d_lines) > 11:
        lines.append(line.format(default[3], c1, d_lines[11], c2, "", c3))
        for index, d_line in enumerate(d_lines[12:], 4):
            if index < len(default):
                lines.append(line.format(default[index], c1, d_line, c2, "", c3))
            else:
                lines.append(line.format("", c1, d_line, c2, "", c3))

    # Handle long default values that span multiple lines
    if len(default) > 4:
        start_index = 4
        # Only add additional lines if we haven't already handled them above
        if len(d_lines) <= 11:
            for index, default_line in enumerate(default[start_index:], start_index):
                if default_line.strip():  # Only add non-empty default lines
                    lines.append(line.format(default_line, c1, "", c2, "", c3))

    lines.append(hline)
    lines.append("")

    return lines


# code to convert ini_dict to flattened dictionary
# default seperater '_'
def flatten_dict(meta_dict, parent_key=None, sep="."):
    """

    :param meta_dict: DESCRIPTION
    :type meta_dict: TYPE
    :param parent_key: DESCRIPTION, defaults to None
    :type parent_key: TYPE, optional
    :param sep: DESCRIPTION, defaults to '.'
    :type sep: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    """
    items = []
    for key, value in meta_dict.items():
        if parent_key:
            new_key = f"{parent_key}{sep}{key}"
        else:
            new_key = key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def flatten_list(x_list):
    """
    Flatten a nested list
    flatten = lambda l: [item for sublist in l for item in sublist]

    Returns
    -------
    None.

    """

    flat_list = [item for sublist in x_list for item in sublist]

    return flat_list


def recursive_split_dict(key, value, remainder, sep="."):
    """
    recursively split a dictionary

    :param key: DESCRIPTION
    :type key: TYPE
    :param value: DESCRIPTION
    :type value: TYPE
    :param remainder: DESCRIPTION
    :type remainder: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """

    key, *other = key.split(sep, 1)
    if other:
        recursive_split_dict(other[0], value, remainder.setdefault(key, {}))
    else:
        remainder[key] = value


def get_by_alias(model, alias_name):
    # Find the field name that corresponds to the given alias
    # Use __pydantic_fields__ instead of model_fields (which is deprecated)
    for field_name, field_info in model.__pydantic_fields__.items():
        if field_info.alias == alias_name:
            return getattr(model, field_name)
    return None


# def get_alias_key(model, key: str) -> str:
#     """
#     Try to find an alias for a field name in a Pydantic BaseModel

#     Parameters
#     ----------
#     model : BaseModel
#         The Pydantic model to search for the field
#     key : str
#         The field name to find the alias for

#     Returns
#     -------
#     str or None
#         The alias name if found, None otherwise
#     """
#     try:
#         field_info = model.__pydantic_fields__.get(key)
#         if field_info.validation_alias:

#         if field_info and field_info.alias:
#             return field_info.alias
#         return key  # Return the original key if no alias found
#     except (AttributeError, KeyError):
#         return key  # Return the original key if any errors occur


def recursive_split_getattr(base_object, name, sep="."):
    key, *other = name.split(sep, 1)

    if other:
        base_object = getattr(base_object, key)
        value, prop = recursive_split_getattr(base_object, other[0])
    else:
        # with Pydantic, if the attribute does not exist an attribute error
        # will be raised, which is desired. The only issue will be if the
        # attribute is an alias, then TODO create a get from alias method.
        try:
            value = getattr(base_object, key)
        except AttributeError:
            value = None
            prop = False
        try:
            if isinstance(getattr(type(base_object), key), property):
                prop = True
        except AttributeError:
            prop = False
    return value, prop


def recursive_split_setattr(base_object, name, value, sep=".", skip_validation=False):
    """
    Recursively split a name and set the value of the last key. Recursion splits on the separator present in the name.

    :param base_object: The object having its attribute set, or a "parent" object in the recursive/nested scenario
    :type base_object: object
    :param name: The name of the attribute to set
    :type name: str
    :param value: The value to set the attribute to
    :type value: any
    :param sep: The separator to split the name on, defaults to "."
    :type sep: str, optional
    :param skip_validation: Whether to skip validation/parse of the attribute, defaults to False
    :type skip_validation: Optional[bool]

    :return: None
    :rtype: NoneType

    """
    key, *other = name.split(sep, 1)

    if other:
        base_object = getattr(base_object, key)
        recursive_split_setattr(base_object, other[0], value)
    else:
        # if the value is a list or dict then we need to add accordingly
        if isinstance(value, list):
            if len(value) == 0:
                value = []
            elif isinstance(value[0], (dict, OrderedDict)):
                new_list = []
                for obj_dict in value:
                    obj_key = list(obj_dict.keys())[0]
                    try:
                        obj = base_object._objects_included[obj_key]()
                        obj.from_dict(obj_dict)
                        new_list.append(obj)
                    except KeyError:
                        raise KeyError(
                            f"Could not find {obj_key} in {base_object._objects_included}"
                        )
                value = new_list

        setattr(base_object, key, value)


def structure_dict(meta_dict, sep="."):
    """

    :param meta_dict: DESCRIPTION
    :type meta_dict: TYPE
    :param sep: DESCRIPTION, defaults to '.'
    :type sep: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    """
    structured_dict = {}
    for key, value in meta_dict.items():
        recursive_split_dict(key, value, structured_dict, sep=sep)
    return structured_dict


def get_units(name, attr_dict):
    """ """
    try:
        units = attr_dict["json_schema_extra"]["units"]
        if not isinstance(units, str):
            units = "{0}".format(units)
    except KeyError:
        units = None
    if units in [None, "None", "none"]:
        return None
    return units


def get_type(name, attr_dict):
    """ """
    try:
        v_type = attr_dict[name]["type"]
        if v_type in ["string", str, "str", "String"]:
            v_type = None
    except KeyError:
        v_type = None
    return v_type


def recursive_split_xml(element, item, base, name, attr_dict=None):
    """ """
    key = None
    if isinstance(item, dict):
        for key, value in item.items():
            attr_name = ".".join([base, key])

            sub_element = et.SubElement(element, key)
            recursive_split_xml(sub_element, value, attr_name, key, attr_dict)
    elif isinstance(item, (tuple, list)):
        for ii in item:
            sub_element = et.SubElement(element, "item")
            recursive_split_xml(sub_element, ii, base, name, attr_dict)
    elif isinstance(item, str):
        element.text = item
    elif item is None:
        # Leave element.text as None so XML has empty element (no text)
        pass
    elif isinstance(item, (float, int)):
        element.text = str(item)
    else:
        # if the value is an hdf5 reference make it a string
        if "reference" in str(type(item)).lower():
            element.text = str(item)
        else:
            raise ValueError("Value cannot be {0}".format(type(item)))
    if attr_dict:
        units = get_units(base, attr_dict)
        if units:
            element.set("units", str(units))
        # v_type = get_type(base, attr_dict)
        # if v_type:
        #     element.set("type", v_type)
    return element, name


def dict_to_xml(meta_dict, attr_dict=None):
    """
    Assumes dictionary is structured {class:{attribute_dict}}

    :param meta_dict: DESCRIPTION
    :type meta_dict: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """
    class_name = list(meta_dict.keys())[0]
    root = et.Element(class_name)

    for key, value in meta_dict[class_name].items():
        element = et.SubElement(root, key)
        recursive_split_xml(element, value, key, key, attr_dict)
    return root


def element_to_dict(element):
    """

    .. todo:: Add way to read in attritues like units and validate them.

    :param element: DESCRIPTION
    :type element: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """
    meta_dict = {element.tag: {} if element.attrib else None}
    children = list(element)
    if children:
        child_dict = defaultdict(list)
        for dc in map(element_to_dict, children):
            for k, v in dc.items():
                child_dict[k].append(v)
        meta_dict = {
            element.tag: {k: v[0] if len(v) == 1 else v for k, v in child_dict.items()}
        }
        if "item" in meta_dict[element.tag].keys():
            meta_dict[element.tag] = meta_dict[element.tag]["item"]
    # going to skip attributes for now, later can check them against
    # standards, neet to skip units and type
    if element.attrib:
        pop_units = False
        pop_type = False
        for k, v in element.attrib.items():
            if k in ["units"]:
                if "type" in element.attrib.keys():
                    pop_type = True
                if len(element.attrib.keys()) <= 2:
                    pop_units = True
                    continue
            if k in ["type"]:
                if len(element.attrib.keys()) <= 1:
                    if v in [
                        "float",
                        "string",
                        "integer",
                        "boolean",
                        "list",
                        "tuple",
                    ]:
                        pop_type = True
                        continue

            meta_dict[element.tag][k] = v
        if pop_units:
            element.attrib.pop("units")
        if pop_type:
            element.attrib.pop("type")
    if element.text:
        text = element.text.strip()
        if children or element.attrib:
            if text:
                if len(element.attrib.keys()) > 0:
                    meta_dict[element.tag]["value"] = text
                else:
                    meta_dict[element.tag] = text
        else:
            meta_dict[element.tag] = text
    return OrderedDict(sorted(meta_dict.items(), key=itemgetter(0)))


# comments, processing instructions, carriage returns, tabs and control
# characters serialize differently between the two paths below
_MINIDOM_ONLY = re.compile("<!--|<\\?|[\r\t\x00-\x08\x0b\x0c\x0e-\x1f]")


def element_to_string(element):
    """
    Serialize an element pretty-printed.

    Indents in place instead of round-tripping the document through minidom,
    which parses and serializes it a second time. Trees the minidom path
    treated differently are still sent through it.
    """
    raw = et.tostring(element, encoding="unicode")
    if _MINIDOM_ONLY.search(raw):
        return (
            minidom.parseString(raw)
            .toprettyxml(indent="    ", encoding="UTF-8")
            .decode()
        )
    element = copy.deepcopy(element)
    et.indent(element, space="    ")
    # ElementTree closes an empty tag as "<tag />", minidom as "<tag/>";
    # ">" is escaped in text and attributes, so only tag closes can match
    text = et.tostring(element, encoding="unicode").replace(" />", "/>")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + text + "\n"


# =============================================================================
# Helper function to be sure everything is encoded properly
# =============================================================================
class NumpyEncoder(json.JSONEncoder):
    """
    Need to encode numpy ints and floats for json to work
    """

    def default(self, obj):
        """

        :param obj:
        :type obj:
        :return:
        """
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray)):
            if obj.dtype == complex:
                return {"real": obj.real.tolist(), "imag": obj.imag.tolist()}
            else:
                return obj.tolist()
        # For now turn references into a generic string
        elif "h5" in str(type(obj)):
            return str(obj)
        elif hasattr(obj, "unicode_string"):
            return obj.unicode_string()
        elif isinstance(obj, Path):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def validate_name(name, pattern=None):
    """
    Validate name

    :param name: DESCRIPTION
    :type name: TYPE
    :param pattern: DESCRIPTION, defaults to None
    :type pattern: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    """
    if name is None:
        return "unknown"
    return name.replace(" ", "_")


def has_numbers(text):
    """
    Check if a string contains any numeric characters.

    Parameters
    ----------
    text : str
        The string to check for numeric characters.

    Returns
    -------
    bool
        True if the string contains any digits (0-9), False otherwise.

    Examples
    --------
    >>> has_numbers("abc123")
    True
    >>> has_numbers("hello")
    False
    >>> has_numbers("test1")
    True
    >>> has_numbers("")
    False
    """
    if not isinstance(text, str):
        return False
    return any(char.isdigit() for char in text)


def is_numeric_string(text):
    """
    Check if a string represents a valid number (int or float).

    Parameters
    ----------
    text : str
        The string to check if it represents a number.

    Returns
    -------
    bool
        True if the string can be converted to a number, False otherwise.

    Examples
    --------
    >>> is_numeric_string("123")
    True
    >>> is_numeric_string("12.34")
    True
    >>> is_numeric_string("-45.6")
    True
    >>> is_numeric_string("1.23e-4")
    True
    >>> is_numeric_string("abc")
    False
    >>> is_numeric_string("12abc")
    False
    """
    if not isinstance(text, str):
        return False

    # Handle empty string
    if not text.strip():
        return False

    try:
        float(text)
        return True
    except ValueError:
        return False


def extract_numbers(text):
    """
    Extract all numeric values from a string.

    Parameters
    ----------
    text : str
        The string to extract numbers from.

    Returns
    -------
    list
        List of float values found in the string.

    Examples
    --------
    >>> extract_numbers("abc123def45.6")
    [123.0, 45.6]
    >>> extract_numbers("no numbers here")
    []
    >>> extract_numbers("1.5 and -2.3e4")
    [1.5, -23000.0]
    """
    import re

    if not isinstance(text, str):
        return []

    # Pattern to match integers, floats, and scientific notation
    number_pattern = r"[-+]?(?:\d*\.?\d+(?:[eE][-+]?\d+)?)"

    matches = re.findall(number_pattern, text)

    numbers = []
    for match in matches:
        try:
            numbers.append(float(match))
        except ValueError:
            continue

    return numbers


def requires(**requirements):
    """Decorate a function with optional dependencies.

    Parameters
    ----------
    **requirements : obj
        keywords of package name and the required object for
        a function.

    Returns
    -------
    decorated_function : function
        Original function if all soft dependencies are met, otherwise
        it returns an empty function which prints why it is not running.

    Examples
    --------
    ```
    try:
        import obspy
    except ImportError:
        obspy = None

    @requires(obspy=obspy)
    def obspy_function():
        ...
        # does something using obspy

    """
    # Check the requirements, add missing package name in the list `missing`.
    missing = []
    for key, item in requirements.items():
        if not item:
            missing.append(key)

    def decorated_function(function):
        """Wrap function."""
        if not missing:
            return function
        else:

            def passer(*args, **kwargs):
                logger.warning(f"Missing dependencies: {missing}.")
                logger.warning(f"Not running `{function.__name__}`.")

            return passer

    return decorated_function


def object_to_array(value, dtype=float):
    """
    Convert a value to a numpy array.

    Parameters
    ----------
    value : any
        The value to convert.

    Returns
    -------
    np.ndarray
        The converted numpy array.

    """
    if value is None:
        return np.empty(0)
    elif isinstance(value, (list, tuple)):
        return np.array(value, dtype=dtype)
    elif isinstance(value, np.ndarray):
        return value.astype(dtype)
    elif isinstance(value, str):
        if value in ["", "none", "None"]:
            return np.empty(0)

        if not has_numbers(value) and not is_numeric_string(value):
            msg = f"String input must be a single number or a list of numbers, not '{value}'"
            raise TypeError(msg)

        elif has_numbers(value) and not is_numeric_string(value):
            value = extract_numbers(value)
            return np.array(value, dtype=dtype)

        if "j" in value and has_numbers(value):
            dtype = complex

        if "," in value:
            separator = ","
        else:
            separator = " "  # Use space as default separator for whitespace

        try:
            return np.fromstring(value, sep=separator, dtype=dtype)

        except ValueError:
            msg = (
                f"input values must be a list, tuple, or np.ndarray, not {type(value)}"
            )
            raise TypeError(msg)
    elif isinstance(value, (int, float)):
        # Handle single numeric input
        return np.array([float(value)], dtype=dtype)
    elif isinstance(value, bytes):
        # Handle bytes input (e.g., from binary files)
        try:
            return np.frombuffer(value, dtype=dtype)
        except ValueError:
            msg = (
                f"input values must be a list, tuple, or np.ndarray, not {type(value)}"
            )
            raise TypeError(msg)
    else:
        msg = f"input values must be an list, tuple, or np.ndarray, not {type(value)}"
        raise TypeError(msg)


def _should_include_coordinate_field(field_name: str) -> bool:
    """
    Helper function to determine if a coordinate field should be included
    in to_dict output even when it has None/default values.

    This ensures backward compatibility for coordinate fields that tests expect.
    """
    coordinate_fields = {
        "negative.x",
        "negative.y",
        "negative.z",
        "positive.x2",
        "positive.y2",
        "positive.z2",
        "location.x",
        "location.y",
        "location.z",
    }
    return field_name in coordinate_fields


def _should_convert_none_to_empty_string(field_name: str) -> bool:
    """
    Helper function to determine if a field should convert None to empty string
    for backward compatibility.
    """
    string_fields = {
        "data_logger.firmware.author",
        "provenance.software.author",
        "provenance.software.version",
    }
    # Convert external URL None -> "" for backward compatibility in to_dict
    string_fields.add("url")
    return field_name in string_fields
