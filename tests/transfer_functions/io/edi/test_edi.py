# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 11:42:55 2021

@author: jpeacock
"""

import numpy as np

# =============================================================================
#
# =============================================================================
import pytest
from pydantic import HttpUrl

from mt_metadata.transfer_functions.io.edi import EDI
from mt_metadata.transfer_functions.io.edi.metadata import EMeasurement, HMeasurement
from mt_metadata.utils.validators import validate_doi


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture(scope="module")
def e_measurement_data():
    """Fixture to provide EMeasurement test data."""
    return {
        "id": 14.001,
        "azm": 0,
        "chtype": "EX",
        "x": -50.0,
        "y": 0.0,
        "x2": 50.0,
        "y2": 0.0,
    }


@pytest.fixture(scope="module")
def e_measurement_azm_data():
    """Fixture to provide EMeasurement azimuth test data."""
    return {
        "id": 14.001,
        "azm": 0,
        "chtype": "EX",
        "x": -50.0,
        "y": 0.0,
        "x2": 50.0,
        "y2": 10.0,
    }


@pytest.fixture(scope="module")
def h_measurement_data():
    """Fixture to provide HMeasurement test data."""
    return {
        "id": 12.001,
        "chtype": "HY",
        "x": 0.0,
        "y": 0.0,
        "azm": 90,
    }


@pytest.fixture(scope="module")
def edi_after_frequency_assert():
    """Fixture to create EDI object after asserting descending frequency."""
    edi = EDI()
    nf = 7
    edi.frequency = np.logspace(-3, 3, nf)
    z = np.arange(nf * 4).reshape((nf, 2, 2))
    t = np.arange(nf * 2).reshape((nf, 1, 2))
    edi.z = z.copy()
    edi.z_err = z.copy()
    edi.t = t.copy()
    edi.t_err = t.copy()

    # Store original arrays before transformation
    original_z = z.copy()
    original_t = t.copy()

    # Apply the frequency assertion
    edi._assert_descending_frequency()

    return {"edi": edi, "original_z": original_z, "original_t": original_t}


# =============================================================================
# EMeasurement Tests
# =============================================================================
class TestEMeasurement:
    """Test EMeasurement class functionality."""

    def test_attributes(self, e_measurement_data):
        """Test EMeasurement attributes."""
        ex = EMeasurement(**e_measurement_data)

        for k, v in e_measurement_data.items():
            actual = getattr(ex, k)
            if isinstance(v, (float, int)):
                assert (
                    abs(actual - v) < 1e-10
                ), f"Attribute {k} mismatch: expected {v}, got {actual}"
            else:
                assert (
                    actual == v
                ), f"Attribute {k} mismatch: expected {v}, got {actual}"


class TestEMeasurementAZM:
    """Test EMeasurement azimuth calculation."""

    def test_azimuth_calculation(self, e_measurement_azm_data):
        """Test EMeasurement azimuth calculation."""
        ex = EMeasurement(**e_measurement_azm_data)

        for k, v in e_measurement_azm_data.items():
            actual = getattr(ex, k)
            if k != "azm":
                assert (
                    actual == v
                ), f"Attribute {k} mismatch: expected {v}, got {actual}"
            else:
                expected = 5.7105931374996
                assert (
                    abs(actual - expected) < 1e-10
                ), f"Azimuth mismatch: expected {expected}, got {actual}"


# =============================================================================
# HMeasurement Tests
# =============================================================================
class TestHMeasurement:
    """Test HMeasurement class functionality."""

    def test_attributes(self, h_measurement_data):
        """Test HMeasurement attributes."""
        hy = HMeasurement(**h_measurement_data)

        for k, v in h_measurement_data.items():
            actual = getattr(hy, k)
            assert actual == v, f"Attribute {k} mismatch: expected {v}, got {actual}"


# =============================================================================
# Descending Frequency Tests
# =============================================================================
class TestAssertDescendingFrequency:
    """Test assert descending frequency functionality."""

    @pytest.mark.parametrize("array_type", ["z", "z_err", "t", "t_err"])
    def test_array_reversal(self, edi_after_frequency_assert, array_type):
        """Test that arrays are properly reversed when asserting descending frequency."""
        data = edi_after_frequency_assert
        edi = data["edi"]
        actual_array = getattr(edi, array_type)

        if array_type.startswith("z"):
            original_array = data["original_z"]
        else:
            original_array = data["original_t"]

        expected_array = original_array[::-1]

        assert np.allclose(
            expected_array, actual_array
        ), f"{array_type} array not properly reversed"


class TestSurveyMetadataDoiHandling:
    """Test mapping survey DOI keys to DOI or URL fields."""

    @pytest.mark.parametrize(
        "value, expected_field",
        [
            ("https://doi.org/10.1234/test", "citation_dataset.doi"),
            ("10.1234/test", "citation_dataset.doi"),
            ("doi:10.1234/test", "citation_dataset.doi"),
            ("https://example.com/dataset", "citation_dataset.url"),
            (HttpUrl("https://doi.org/10.1234/test"), "citation_dataset.doi"),
            (HttpUrl("https://example.com/dataset"), "citation_dataset.url"),
        ],
    )
    def test_survey_doi_values_map_to_expected_field(self, value, expected_field):
        edi = EDI()
        edi.Info.info_dict["survey.citation_dataset.doi"] = value

        survey_metadata = edi.survey_metadata
        actual_value = survey_metadata.get_attr_from_name(expected_field)

        if expected_field.endswith("doi"):
            assert actual_value.unicode_string() == validate_doi(value).unicode_string()
        elif isinstance(value, HttpUrl):
            assert actual_value.unicode_string() == value.unicode_string()
        else:
            assert actual_value == value

        other_field = (
            "citation_dataset.url"
            if expected_field.endswith("doi")
            else "citation_dataset.doi"
        )
        assert survey_metadata.get_attr_from_name(other_field) in [None, "None"]


# =============================================================================
# run
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__])
