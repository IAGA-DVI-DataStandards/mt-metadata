import gc
import weakref

from pydantic import create_model

from mt_metadata.base import pydantic_helpers
from mt_metadata.base.metadata import MetadataBase


def test_fields_tree_cache_hit():
    """
    Test the field tree cache returns the same tree for the same class.
    """
    first = pydantic_helpers.get_all_fields_serializable(MetadataBase)
    second = pydantic_helpers.get_all_fields_serializable(MetadataBase)

    assert first is second


def test_fields_tree_cache_releases_transient_class():
    """
    Test the field tree cache lets go of a class nothing else references.
    """
    transient = create_model("Transient", __base__=MetadataBase)
    pydantic_helpers.get_all_fields_serializable(transient)

    assert transient in pydantic_helpers._FIELDS_TREE_CACHE

    reference = weakref.ref(transient)
    del transient
    gc.collect()

    assert reference() is None
