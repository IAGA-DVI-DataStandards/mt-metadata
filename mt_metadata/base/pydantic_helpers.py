"""
Field introspection utilities for Pydantic BaseModel classes with
lazy in-memory caching and optional on-disk caching.

This module builds a JSON-serializable nested "field tree" for any
Pydantic BaseModel, avoiding instantiation and guarding against
infinite recursion.

Leaf nodes are serializable summaries and include:
    - type
    - default
    - deprecated
    - description
    - title
    - default_factory (if present)
    - enum (for Enum/Literal types)
    - enum_names (for Enum subclasses)
    - examples (from Field(..., json_schema_extra={'examples': [...]}))
    - required (from Field(..., json_schema_extra={'required': True/False}))
    - units (from Field(..., json_schema_extra={'units': '...'}))
    - has_validators (True if any field validators are present)
    - constraints:
        * ge, le, gt, lt
        * multiple_of
        * min_length, max_length, pattern
        * min_items, max_items, unique_items
        * const, format
        * nullable

Nested nodes represent BaseModel-typed fields and contain further trees.

Notes
-----
- List, Dict, and Union types are treated as simple fields (non-expanded),
  unless the Union directly contains a BaseModel, in which case the first
  BaseModel type is expanded.
- A special-case hook (`SPECIAL_CASE_MODEL_NAMES`) lets you treat certain
  BaseModel types (e.g., "MTime") as simple fields.
- Constraints are derived from Pydantic's JSON Schema via `TypeAdapter(annotation).json_schema()`.
"""

from __future__ import annotations

import enum
import hashlib
import json
import os
import sys
import weakref
from pathlib import Path
from threading import RLock
from typing import Annotated, Any, Dict, get_args, get_origin, Literal, Union

from pydantic import __version__ as _PYDANTIC_VERSION
from pydantic import TypeAdapter  # Pydantic v2
from pydantic import BaseModel

# try:
#     # Optional dependency for platform-aware cache directory
#     from platformdirs import user_cache_dir
# except Exception:  # pragma: no cover - optional
#     user_cache_dir = None  # Fallback handled below


# -------------------------------
# Configuration & Globals
# -------------------------------

APP_NAME = "mt_metadata"

# Treat these BaseModel names as simple fields (no expansion)
SPECIAL_CASE_MODEL_NAMES = {"MTime"}

# Thread-safe in-memory cache of computed field trees (per class), weakly keyed
# so a class built at run time is not kept alive by the cache
_FIELDS_TREE_CACHE: weakref.WeakKeyDictionary[type[BaseModel], Dict[str, Any]] = (
    weakref.WeakKeyDictionary()
)
_CACHE_LOCK = RLock()

# Environment flag to disable disk caching (e.g., for tests)
_DISABLE_DISK_CACHE = os.environ.get("MT_METADATA_DISABLE_DISK_CACHE", "0") in {
    "1",
    "true",
    "True",
}


# -------------------------------
# Public API
# -------------------------------


def get_all_fields_serializable(
    model_or_cls: Union[type[BaseModel], BaseModel],
) -> Dict[str, Any]:
    """
    Build a JSON-serializable nested dictionary of fields for a Pydantic BaseModel.

    This function avoids instantiating models, caches results in memory,
    and (optionally) persists/retrieves the serialized tree to/from disk.

    Parameters
    ----------
    model_or_cls : type[BaseModel] or BaseModel
        The BaseModel class (preferred) or an instance. If an instance is provided,
        its class will be used.

    Returns
    -------
    Dict[str, Any]
        A nested, JSON-serializable dictionary describing the model's fields.
        Leaf nodes are field summaries; nested nodes correspond to BaseModel-typed fields.

    Notes
    -----
    - Uses a sentinel write to the cache prior to recursion to break cycles.
    - The on-disk cache file name is derived from the class's fully-qualified name,
      Pydantic version, and a fingerprint of the field schema.
    """
    model_cls: type[BaseModel] = (
        model_or_cls if isinstance(model_or_cls, type) else type(model_or_cls)
    )

    with _CACHE_LOCK:
        # In-memory hit
        if model_cls in _FIELDS_TREE_CACHE:
            return _FIELDS_TREE_CACHE[model_cls]

        # Try disk cache
        if not _DISABLE_DISK_CACHE:
            disk = _load_fields_from_disk(model_cls)
            if disk is not None:
                _FIELDS_TREE_CACHE[model_cls] = disk
                return disk

        # Sentinel to break cycles
        _FIELDS_TREE_CACHE[model_cls] = {}

        # Compute and persist
        tree = _compute_fields_tree(model_cls)
        _FIELDS_TREE_CACHE[model_cls] = tree

        if not _DISABLE_DISK_CACHE:
            _save_fields_to_disk(model_cls, tree)

        return tree


def flatten_field_tree_map(
    tree: Dict[str, Any], prefix: str = ""
) -> Dict[str, Dict[str, Any]]:
    """
    Flatten a nested field tree (as returned by `get_all_fields_serializable`) into
    a dictionary keyed by dotted field paths, where each value is the leaf field's
    serializable summary.

    Parameters
    ----------
    tree : Dict[str, Any]
        The nested field tree. Leaf nodes are dicts that contain `"__field__": True`;
        nested nodes are dictionaries whose values are more field trees.
    prefix : str, optional
        A prefix to prepend to each key (useful when flattening under a known root),
        by default "".

    Returns
    -------
    Dict[str, Dict[str, Any]]
        A mapping from dotted paths (e.g., "inner.a") to the corresponding leaf summary
        dictionaries (e.g., {"__field__": True, "type": "<class 'int'>", ...}).

    Notes
    -----
    - Only leaf nodes marked with `"__field__": True` are included in the output.
    - Nested BaseModel nodes (i.e., dictionaries without `"__field__": True") are traversed.
    - Keys are constructed using dot notation to reflect the hierarchy.
    """
    out: Dict[str, Dict[str, Any]] = {}

    for name, node in tree.items():
        path = f"{prefix}.{name}" if prefix else name

        # Leaf: field summary dicts have "__field__": True
        if isinstance(node, dict) and node.get("__field__") is True:
            out[path] = node
            continue

        # Nested: recurse into sub-dicts that are not leaf summaries
        if isinstance(node, dict):
            out.update(flatten_field_tree_map(node, path))

    return out


def clear_field_caches() -> None:
    """
    Clear the in-memory field tree cache.

    This does not remove any on-disk cache files.
    """
    with _CACHE_LOCK:
        _FIELDS_TREE_CACHE.clear()


# -------------------------------
# Internal helpers
# -------------------------------


def _compute_fields_tree(model_cls: type[BaseModel]) -> Dict[str, Any]:
    """
    Compute the nested, serializable field tree for a BaseModel class.

    Parameters
    ----------
    model_cls : type[BaseModel]
        The Pydantic BaseModel subclass to introspect.

    Returns
    -------
    Dict[str, Any]
        Nested dict of fields; leaf nodes are serializable summaries.

    Notes
    -----
    - Uses the public `model_fields` API where available; falls back to `__pydantic_fields__`.
    - Skips fields marked as deprecated (if `FieldInfo.deprecated` is present).
    - Computes `has_validators` flags by inspecting model-level decorators.
    """
    validators_map = _collect_field_validator_map(model_cls)
    field_map = getattr(model_cls, "model_fields", None) or getattr(
        model_cls, "__pydantic_fields__", {}
    )
    out: Dict[str, Any] = {}

    for field_name, field_info in field_map.items():
        deprecated = getattr(field_info, "deprecated", None)
        if deprecated is not None:
            continue

        annotation = getattr(field_info, "annotation", None)
        base_type = _extract_base_type(annotation)

        if (
            base_type
            and _is_basemodel_subclass(base_type)
            and base_type.__name__ not in SPECIAL_CASE_MODEL_NAMES
        ):
            out[field_name] = get_all_fields_serializable(base_type)
        else:
            out[field_name] = _to_serializable_field(
                field_info, model_cls, field_name, validators_map
            )

    return out


def _extract_base_type(annotation: Any) -> Any:
    """
    Extract a primary base type from complex type annotations (Optional/Union, Annotated, List, Dict).

    Parameters
    ----------
    annotation : Any
        The annotation to inspect.

    Returns
    -------
    Any or None
        The extracted base type if a direct class can be resolved, otherwise None.

    Notes
    -----
    - Annotated[T, ...] unwraps to T.
    - List[T] and Dict[K, V] return None (treated as simple fields).
    - Union[...] returns the first BaseModel subtype if present; otherwise the first non-None type.
    """
    if annotation is None:
        return None

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Annotated:
        return _extract_base_type(args[0]) if args else None
    if origin in (list,) or (hasattr(origin, "__name__") and origin.__name__ == "list"):
        return None
    if origin in (dict,) or (
        hasattr(origin, "__name__") and origin.__name__ in {"dict", "Dict"}
    ):
        return None
    if origin and (
        origin is Union or getattr(origin, "__name__", "") in {"Union", "UnionType"}
    ):
        for arg in args:
            if _is_basemodel_subclass(arg):
                return arg
        for arg in args:
            if arg is not type(None):
                return _extract_base_type(arg)
        return None

    if isinstance(annotation, type) and annotation is not type(None):
        return annotation

    return None


def _is_basemodel_subclass(cls: Any) -> bool:
    """
    Check whether a class is a subclass of Pydantic BaseModel with field metadata.

    Parameters
    ----------
    cls : Any
        The candidate class.

    Returns
    -------
    bool
        True if `cls` is a BaseModel subclass with field metadata, else False.
    """
    try:
        return (
            isinstance(cls, type)
            and issubclass(cls, BaseModel)
            and (hasattr(cls, "model_fields") or hasattr(cls, "__pydantic_fields__"))
        )
    except Exception:
        return False


def _to_serializable_field(
    field_info: Any,
    model_cls: type[BaseModel],
    field_name: str,
    validators_map: Dict[str, bool],
) -> Dict[str, Any]:
    """
    Convert a Pydantic FieldInfo into a JSON-serializable summary dict, enriched with
    enum values/names, examples, required, units, validators presence, and constraints.

    Parameters
    ----------
    field_info : Any
        The Pydantic FieldInfo-like object.
    model_cls : type[BaseModel]
        The BaseModel class owning the field, used to resolve validators presence.
    field_name : str
        The name of the field on the model.
    validators_map : Dict[str, bool]
        A mapping from field name to a boolean indicating if any validators target that field.

    Returns
    -------
    Dict[str, Any]
        A serializable summary including type, default, doc metadata, enum info,
        examples, `required`, `units`, `has_validators`, and `constraints`.
    """
    ann = getattr(field_info, "annotation", None)

    enum_values, enum_names = _extract_enum_info(ann)
    extras = _extract_json_schema_extras(field_info)  # examples, required, units
    constraints = _extract_constraints(ann)

    summary = {
        "__field__": True,
        "type": repr(ann),
        "default": _safe_repr(getattr(field_info, "default", None)),
        "deprecated": _safe_repr(getattr(field_info, "deprecated", None)),
        "description": getattr(field_info, "description", None),
        "enum": enum_values,
        "enum_names": enum_names,
        "examples": extras.get("examples"),
        "required": extras.get("required"),
        "units": extras.get("units"),
        "has_validators": bool(validators_map.get(field_name, False)),
        "constraints": constraints or {},
    }

    default_factory = getattr(field_info, "default_factory", None)
    if default_factory is not None:
        summary["default_factory"] = repr(default_factory)

    return summary


def _extract_json_schema_extras(field_info: Any) -> Dict[str, Any]:
    """
    Extract selected keys from a FieldInfo's `json_schema_extra`.

    Parameters
    ----------
    field_info : Any
        The Pydantic FieldInfo-like object.

    Returns
    -------
    Dict[str, Any]
        A dictionary possibly containing:
            - "examples": list or None
            - "required": bool or None
            - "units": str or None

    Notes
    -----
    - Ensures `examples` are JSON-serializable; falls back to `repr(...)` for complex items.
    - Passes through `required` and `units` if present (no type coercion beyond JSON compatibility).
    """
    out: Dict[str, Any] = {"examples": None, "required": None, "units": None}
    extra = getattr(field_info, "json_schema_extra", None)
    if not isinstance(extra, dict):
        return out

    # examples
    ex = extra.get("examples")
    if ex is not None:
        try:
            json.dumps(ex)
            out["examples"] = ex
        except Exception:
            out["examples"] = (
                [repr(item) for item in ex]
                if isinstance(ex, (list, tuple, set))
                else repr(ex)
            )

    # required
    # Note: in Pydantic, "required" is typically controlled at the model level,
    # but if your project uses json_schema_extra to signal requiredness, we surface it.
    req = extra.get("required")
    if isinstance(req, bool):
        out["required"] = req
    elif req is not None:
        # Allow strings like "true"/"false" to be normalized
        if str(req).lower() in {"true", "1"}:
            out["required"] = True
        elif str(req).lower() in {"false", "0"}:
            out["required"] = False
        else:
            out["required"] = repr(req)  # preserve value, but keep serializable

    # units
    units = extra.get("units")
    if units is not None:
        try:
            json.dumps(units)
            out["units"] = units
        except Exception:
            out["units"] = repr(units)

    return out


def _extract_enum_info(annotation: Any) -> tuple[list[Any] | None, list[str] | None]:
    """
    Extract enum values and names from annotations that are Enum subclasses or Literal[...] types.

    Parameters
    ----------
    annotation : Any
        The type annotation to inspect.

    Returns
    -------
    tuple
        (enum_values, enum_names)
        - enum_values : list or None
            The list of allowed values for the field (primitive values preferred).
        - enum_names : list of str or None
            Enum member names if the annotation is an Enum subclass; otherwise None.
    """
    if annotation is None:
        return None, None

    try:
        if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
            values = [m.value for m in annotation]
            names = [m.name for m in annotation]
            return values, names
    except Exception:
        pass

    origin = get_origin(annotation)
    if origin is Literal:
        args = list(get_args(annotation))
        values: list[Any] = []
        for v in args:
            try:
                json.dumps(v)
                values.append(v)
            except Exception:
                values.append(repr(v))
        return values, None

    return None, None


def _extract_constraints(annotation: Any) -> Dict[str, Any] | None:
    """
    Extract constraints from the type annotation using Pydantic's JSON Schema.

    Parameters
    ----------
    annotation : Any
        The type annotation to inspect.

    Returns
    -------
    Dict[str, Any] or None
        A dictionary of constraints (ge, le, gt, lt, multiple_of, min_length, max_length,
        pattern, min_items, max_items, unique_items, const, format, nullable).
        Returns None if no constraints can be extracted.

    Notes
    -----
    - Uses `TypeAdapter(annotation).json_schema()` to derive constraints.
    """
    if annotation is None:
        return None

    try:
        schema = TypeAdapter(annotation).json_schema()
    except Exception:
        return None

    def _is_nullable(s: Dict[str, Any]) -> bool:
        if s.get("nullable") is True:
            return True
        t = s.get("type")
        if isinstance(t, list) and "null" in t:
            return True
        for key in ("anyOf", "oneOf", "allOf"):
            for sub in s.get(key, []) or []:
                if isinstance(sub, dict) and sub.get("type") == "null":
                    return True
        return False

    constraints: Dict[str, Any] = {
        "ge": schema.get("minimum"),
        "le": schema.get("maximum"),
        "gt": schema.get("exclusiveMinimum"),
        "lt": schema.get("exclusiveMaximum"),
        "multiple_of": schema.get("multipleOf"),
        "min_length": schema.get("minLength"),
        "max_length": schema.get("maxLength"),
        "pattern": schema.get("pattern"),
        "min_items": schema.get("minItems"),
        "max_items": schema.get("maxItems"),
        "unique_items": schema.get("uniqueItems"),
        "const": schema.get("const"),
        "format": schema.get("format"),
        "nullable": _is_nullable(schema),
    }

    return {k: v for k, v in constraints.items() if v is not None}


def _safe_repr(obj: Any) -> Any:
    """
    Safely repr() an object for serialization, returning None if repr fails.

    Parameters
    ----------
    obj : Any
        The object to represent.

    Returns
    -------
    Any
        The repr string of the object, or None if not representable.
    """
    try:
        return repr(obj) if obj is not None else None
    except Exception:
        return None


def _collect_field_validator_map(model_cls: type[BaseModel]) -> Dict[str, bool]:
    """
    Collect a mapping of field names to a boolean indicating presence of field validators.

    Parameters
    ----------
    model_cls : type[BaseModel]
        The BaseModel class to inspect.

    Returns
    -------
    Dict[str, bool]
        Mapping from field name to True/False, where True means at least one
        field validator is declared for that field on the model.

    Notes
    -----
    - Best-effort for Pydantic v2 by introspecting `__pydantic_decorators__`.
    """
    result: Dict[str, bool] = {}
    decs = getattr(model_cls, "__pydantic_decorators__", None)
    if decs is None:
        return result

    fv = getattr(decs, "field_validators", None)
    if isinstance(fv, dict):
        for fname, validators in fv.items():
            result[fname] = bool(validators)

    vals = getattr(decs, "validators", None)
    if vals:
        for v in vals:
            fields = getattr(v, "fields", None) or getattr(v, "field", None)
            if fields is None:
                continue
            if isinstance(fields, (list, tuple, set)):
                for fname in fields:
                    result[fname] = True
            elif isinstance(fields, str):
                result[fields] = True

    return result


# -------------------------------
# Disk cache utilities
# -------------------------------


def _cache_dir() -> str:
    """
    Resolve a user-specific cache directory for the application using only the stdlib.

    Priority
    --------
    1. Environment variable override: MT_METADATA_CACHE_DIR
    2. OS-specific conventional cache directories:
       - Linux: $XDG_CACHE_HOME or ~/.cache/<APP_NAME>
       - macOS: ~/Library/Caches/<APP_NAME>
       - Windows: %LOCALAPPDATA%\\<APP_NAME> or ~/AppData/Local/<APP_NAME>

    Returns
    -------
    str
        Absolute path to the cache directory. The directory is created if it does not exist.

    Notes
    -----
    - Uses only Python's standard library (no external dependencies).
    - Provides a portable behavior that aligns with common platform conventions.
    """
    # 1) Explicit override
    override = os.environ.get("MT_METADATA_CACHE_DIR")
    if override:
        path = Path(override).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    # 2) Platform-specific default
    plat = sys.platform
    home = Path.home()

    if plat.startswith("linux"):
        base = Path(os.environ.get("XDG_CACHE_HOME", home / ".cache"))
        path = base / APP_NAME

    elif plat == "darwin":
        # macOS: ~/Library/Caches/<APP_NAME>
        path = home / "Library" / "Caches" / APP_NAME

    elif plat.startswith("win"):
        # Windows: %LOCALAPPDATA% preferred, else fallback
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            path = Path(local_appdata) / APP_NAME
        else:
            path = home / "AppData" / "Local" / APP_NAME

    else:
        # Fallback for unknown platforms
        path = home / ".cache" / APP_NAME

    path.mkdir(parents=True, exist_ok=True)
    return str(path)


# def _cache_dir() -> str:
#     """
#     Resolve a user-specific cache directory for the application.

#     Returns
#     -------
#     str
#         The path to the cache directory.

#     Notes
#     -----
#     - Uses `platformdirs.user_cache_dir(APP_NAME)` if available; otherwise
#       falls back to `~/.cache/<APP_NAME>`.
#     """
#     if user_cache_dir is not None:
#         path = user_cache_dir(APP_NAME)
#     else:
#         path = os.path.join(os.path.expanduser("~"), ".cache", APP_NAME)
#     os.makedirs(path, exist_ok=True)
#     return path


def _model_fingerprint(model_cls: type[BaseModel]) -> str:
    """
    Compute a stable fingerprint for a model class's fields.

    Parameters
    ----------
    model_cls : type[BaseModel]
        The model class to fingerprint.

    Returns
    -------
    str
        A SHA-256 hex digest representing the schema shape.

    Notes
    -----
    - Uses a sorted JSON of tuples:
        (name, deprecated, annotation repr, default repr, json_schema_extra snapshot)
    - Change in any of these will produce a different fingerprint, refreshing the disk cache.
    """
    field_map = getattr(model_cls, "model_fields", None) or getattr(
        model_cls, "__pydantic_fields__", {}
    )
    parts = []
    for name, info in field_map.items():
        extra = getattr(info, "json_schema_extra", None)
        if isinstance(extra, dict):
            try:
                extra_snapshot = json.dumps(
                    extra, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                )
            except Exception:
                extra_snapshot = []
                print("Non-serializable json_schema_extra for field:", name)
                for key, value in sorted(extra.items()):
                    extra_snapshot.append((key, _safe_repr(value)))
        else:
            extra_snapshot = _safe_repr(extra)
        parts.append(
            (
                name,
                _safe_repr(getattr(info, "deprecated", None)),
                repr(getattr(info, "annotation", None)),
                _safe_repr(getattr(info, "default", None)),
                extra_snapshot,
            )
        )
    raw = json.dumps(sorted(parts), separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _disk_cache_path(model_cls: type[BaseModel]) -> str:
    """
    Construct the on-disk cache path for a given model class.

    Parameters
    ----------
    model_cls : type[BaseModel]
        The model class.

    Returns
    -------
    str
        Absolute path to the cache JSON file.
    """
    fqname = f"{model_cls.__module__}.{model_cls.__qualname__}"
    fp = _model_fingerprint(model_cls)
    fname = f"{fqname}__pyd{_PYDANTIC_VERSION}__{fp}.json"
    return os.path.join(_cache_dir(), fname)


def _load_fields_from_disk(model_cls: type[BaseModel]) -> Dict[str, Any] | None:
    """
    Load a serialized field tree from disk cache if present.

    Parameters
    ----------
    model_cls : type[BaseModel]
        The model class.

    Returns
    -------
    Dict[str, Any] or None
        The field tree if found, otherwise None.

    Notes
    -----
    - Returns None on any read/parse error.
    """
    path = _disk_cache_path(model_cls)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_fields_to_disk(model_cls: type[BaseModel], tree: Dict[str, Any]) -> None:
    """
    Persist a serialized field tree to disk cache.

    Parameters
    ----------
    model_cls : type[BaseModel]
        The model class.
    tree : Dict[str, Any]
        The serialized field tree.

    Returns
    -------
    None

    Notes
    -----
    - Overwrites any existing file for the same model fingerprint.
    """
    path = _disk_cache_path(model_cls)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tree, f, indent=2, ensure_ascii=False)
    except Exception:
        # Best-effort caching; ignore write errors
        pass
