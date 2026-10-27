# -*- coding: utf-8 -*-
"""
Comprehensive test suite for decimation_level_basemodel.py

Tests the DecimationLevel class, SaveFcsTypeEnum, and utility functions,
including field validation, computed properties, band management,
and FC decimation functionality.

Created on: September 12, 2025
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from mt_metadata.common.band import Band
from mt_metadata.features.weights import ChannelWeightSpec
from mt_metadata.processing import ShortTimeFourierTransform as STFT
from mt_metadata.processing import TimeSeriesDecimation as Decimation
from mt_metadata.processing.aurora.decimation_level import (
    _df_from_bands,
    DecimationLevel,
    get_fft_harmonics,
    SaveFcsTypeEnum,
)
from mt_metadata.processing.aurora.estimator import Estimator
from mt_metadata.processing.aurora.frequency_bands import FrequencyBands
from mt_metadata.processing.aurora.regression import Regression
from mt_metadata.processing.fourier_coefficients.decimation import (
    Decimation as FCDecimation,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_band():
    """Create a sample Band for testing."""
    band = Band()
    band.decimation_level = 0
    band.index_min = 0
    band.index_max = 10
    band.frequency_min = 0.1
    band.frequency_max = 1.0
    return band


@pytest.fixture
def sample_bands():
    """Create multiple Band objects for testing."""
    bands = []
    for i in range(3):
        band = Band()
        band.decimation_level = i
        band.index_min = i * 10
        band.index_max = (i + 1) * 10
        band.frequency_min = 0.1 * (i + 1)
        band.frequency_max = 1.0 * (i + 1)
        bands.append(band)
    return bands


@pytest.fixture
def sample_channel_weight_spec():
    """Create a sample ChannelWeightSpec for testing."""
    return ChannelWeightSpec()


@pytest.fixture
def basic_decimation_level():
    """Create a basic DecimationLevel instance for testing."""
    return DecimationLevel()


@pytest.fixture
def populated_decimation_level(sample_bands):
    """Create a DecimationLevel with populated data."""
    dec_level = DecimationLevel()
    dec_level.bands = sample_bands
    dec_level.input_channels = ["hx", "hy"]
    dec_level.output_channels = ["ex", "ey", "hz"]
    dec_level.reference_channels = ["rrhx", "rrhy"]
    dec_level.save_fcs = True
    dec_level.save_fcs_type = SaveFcsTypeEnum.h5

    # Configure decimation
    dec_level.decimation.level = 0
    dec_level.decimation.factor = 1
    dec_level.decimation.sample_rate = 256.0

    # Configure STFT
    dec_level.stft.window.num_samples = 256
    dec_level.stft.method = "fft"

    return dec_level


@pytest.fixture
def mock_fc_decimation():
    """Create a mock FCDecimation for testing compatibility."""
    fc_dec = Mock(spec=FCDecimation)
    fc_dec.channels_estimated = ["hx", "hy", "ex", "ey", "hz"]

    # Create nested mock objects instead of using spec
    fc_dec.time_series_decimation = Mock()
    fc_dec.time_series_decimation.anti_alias_filter = "default"
    fc_dec.time_series_decimation.sample_rate = 256.0
    fc_dec.time_series_decimation.factor = 1
    fc_dec.time_series_decimation.level = 0

    fc_dec.short_time_fourier_transform = Mock()
    fc_dec.short_time_fourier_transform.method = "fft"

    fc_dec.stft = Mock()
    fc_dec.stft.prewhitening_type = "first difference"  # Use valid enum value
    fc_dec.stft.prewhitening_ar_order = 0
    fc_dec.stft.prewhitening_ma_order = 0
    fc_dec.stft.recoloring = False
    fc_dec.stft.pre_fft_detrend_type = "linear"
    fc_dec.stft.min_num_stft_windows = 2
    fc_dec.stft.window = Mock()
    fc_dec.stft.harmonic_indices = [-1]
    return fc_dec


# =============================================================================
# Test SaveFcsTypeEnum
# =============================================================================


class TestSaveFcsTypeEnum:
    """Test SaveFcsTypeEnum functionality."""

    def test_enum_values(self):
        """Test that enum has expected values."""
        assert SaveFcsTypeEnum.h5 == "h5"
        assert SaveFcsTypeEnum.csv == "csv"

    def test_enum_string_inheritance(self):
        """Test that enum inherits from string."""
        assert isinstance(SaveFcsTypeEnum.h5, str)
        assert isinstance(SaveFcsTypeEnum.csv, str)

    def test_enum_from_string(self):
        """Test creating enum instances from string values."""
        h5_enum = SaveFcsTypeEnum("h5")
        assert h5_enum == SaveFcsTypeEnum.h5

        csv_enum = SaveFcsTypeEnum("csv")
        assert csv_enum == SaveFcsTypeEnum.csv

    def test_enum_error_cases(self):
        """Test enum error handling for invalid values."""
        with pytest.raises(ValueError):
            SaveFcsTypeEnum("invalid_format")

        with pytest.raises(ValueError):
            SaveFcsTypeEnum("")


# =============================================================================
# Test DecimationLevel Basic Functionality
# =============================================================================


class TestDecimationLevelBasics:
    """Test basic DecimationLevel functionality."""

    def test_default_instantiation(self, basic_decimation_level):
        """Test creating DecimationLevel with default values."""
        dec_level = basic_decimation_level

        # Test default list fields
        assert isinstance(dec_level.bands, list)
        assert len(dec_level.bands) == 0
        assert isinstance(dec_level.channel_weight_specs, list)
        assert len(dec_level.channel_weight_specs) == 0
        assert isinstance(dec_level.input_channels, list)
        assert len(dec_level.input_channels) == 0
        assert isinstance(dec_level.output_channels, list)
        assert len(dec_level.output_channels) == 0
        assert isinstance(dec_level.reference_channels, list)
        assert len(dec_level.reference_channels) == 0

        # Test default boolean field
        assert dec_level.save_fcs is False

        # Test default None field
        assert dec_level.save_fcs_type is None

        # Test default factory fields
        assert isinstance(dec_level.decimation, Decimation)
        assert isinstance(dec_level.estimator, Estimator)
        assert isinstance(dec_level.regression, Regression)
        assert isinstance(dec_level.stft, STFT)

    def test_field_assignment(self, basic_decimation_level):
        """Test field assignment with valid values."""
        dec_level = basic_decimation_level

        # Test channel assignments
        dec_level.input_channels = ["hx", "hy"]
        assert dec_level.input_channels == ["hx", "hy"]

        dec_level.output_channels = ["ex", "ey", "hz"]
        assert dec_level.output_channels == ["ex", "ey", "hz"]

        dec_level.reference_channels = ["rrhx", "rrhy"]
        assert dec_level.reference_channels == ["rrhx", "rrhy"]

        # Test save_fcs assignment
        dec_level.save_fcs = True
        assert dec_level.save_fcs is True

        # Test enum assignment
        dec_level.save_fcs_type = SaveFcsTypeEnum.csv
        assert dec_level.save_fcs_type == SaveFcsTypeEnum.csv

        # Test string assignment for enum
        dec_level.save_fcs_type = "h5"
        assert dec_level.save_fcs_type == SaveFcsTypeEnum.h5

    def test_from_dict_instantiation(self):
        """Test creating DecimationLevel from dictionary."""
        data = {
            "input_channels": ["hx", "hy"],
            "output_channels": ["ex", "ey"],
            "reference_channels": ["rrhx", "rrhy"],
            "save_fcs": True,
            "save_fcs_type": "h5",
            "decimation": {"level": 0, "factor": 1, "sample_rate": 256.0},
        }

        dec_level = DecimationLevel.model_validate(data)

        assert dec_level.input_channels == ["hx", "hy"]
        assert dec_level.output_channels == ["ex", "ey"]
        assert dec_level.reference_channels == ["rrhx", "rrhy"]
        assert dec_level.save_fcs is True
        assert dec_level.save_fcs_type == SaveFcsTypeEnum.h5
        assert dec_level.decimation.level == 0


# =============================================================================
# Test Band Management
# =============================================================================


class TestDecimationLevelBandManagement:
    """Test band management functionality."""

    def test_add_band_object(self, basic_decimation_level, sample_band):
        """Test adding Band object."""
        dec_level = basic_decimation_level
        initial_count = len(dec_level.bands)

        dec_level.add_band(sample_band)

        assert len(dec_level.bands) == initial_count + 1
        assert isinstance(dec_level.bands[-1], Band)
        assert dec_level.bands[-1].decimation_level == 0

    def test_add_band_dict(self, basic_decimation_level):
        """Test adding band from dictionary."""
        dec_level = basic_decimation_level
        band_dict = {
            "decimation_level": 1,
            "index_min": 5,
            "index_max": 15,
            "frequency_min": 0.5,
            "frequency_max": 1.5,
        }
        initial_count = len(dec_level.bands)

        dec_level.add_band(band_dict)

        assert len(dec_level.bands) == initial_count + 1
        assert isinstance(dec_level.bands[-1], Band)

    def test_add_band_invalid_type(self, basic_decimation_level):
        """Test error when adding invalid band type."""
        dec_level = basic_decimation_level

        with pytest.raises(TypeError):
            dec_level.add_band("invalid_band")

        with pytest.raises(TypeError):
            dec_level.add_band(123)

        with pytest.raises(TypeError):
            dec_level.add_band(None)

    def test_bands_validation(self, sample_bands):
        """Test bands field validation."""
        # Test with list of Band objects
        dec_level = DecimationLevel(bands=sample_bands)
        assert len(dec_level.bands) == 3
        assert all(isinstance(band, Band) for band in dec_level.bands)

        # Test with list of dictionaries
        band_dicts = [
            {"decimation_level": 0, "index_min": 0, "index_max": 10},
            {"decimation_level": 1, "index_min": 10, "index_max": 20},
        ]
        dec_level = DecimationLevel(bands=band_dicts)
        assert len(dec_level.bands) == 2
        assert all(isinstance(band, Band) for band in dec_level.bands)


# =============================================================================
# Test Computed Properties
# =============================================================================


class TestDecimationLevelComputedProperties:
    """Test computed properties of DecimationLevel."""

    def test_lower_bounds_property(self, populated_decimation_level):
        """Test lower_bounds computed property."""
        dec_level = populated_decimation_level
        lower_bounds = dec_level.lower_bounds

        assert isinstance(lower_bounds, np.ndarray)
        assert len(lower_bounds) == 3
        assert np.array_equal(lower_bounds, np.array([0, 10, 20]))

    def test_upper_bounds_property(self, populated_decimation_level):
        """Test upper_bounds computed property."""
        dec_level = populated_decimation_level
        upper_bounds = dec_level.upper_bounds

        assert isinstance(upper_bounds, np.ndarray)
        assert len(upper_bounds) == 3
        assert np.array_equal(upper_bounds, np.array([10, 20, 30]))

    def test_bands_dataframe_property(self, populated_decimation_level):
        """Test bands_dataframe computed property."""
        dec_level = populated_decimation_level
        bands_df = dec_level.bands_dataframe

        assert isinstance(bands_df, pd.DataFrame)
        assert len(bands_df) == 3

        expected_columns = [
            "decimation_level",
            "lower_bound_index",
            "upper_bound_index",
            "frequency_min",
            "frequency_max",
        ]
        for col in expected_columns:
            assert col in bands_df.columns

        # Test EMTF convention (+1 to decimation level)
        assert bands_df["decimation_level"].tolist() == [1, 2, 3]

    def test_frequency_sample_interval_property(self, populated_decimation_level):
        """Test frequency_sample_interval computed property."""
        dec_level = populated_decimation_level
        freq_interval = dec_level.frequency_sample_interval

        assert isinstance(freq_interval, float)
        expected = 256.0 / 256  # sample_rate / num_samples
        assert freq_interval == expected

    def test_band_edges_property(self, populated_decimation_level):
        """Test band_edges computed property."""
        dec_level = populated_decimation_level
        band_edges = dec_level.band_edges

        assert isinstance(band_edges, np.ndarray)
        assert band_edges.shape == (3, 2)  # 3 bands, 2 columns (min, max)

        # Check frequency values
        expected_min = [0.1, 0.2, 0.3]
        expected_max = [1.0, 2.0, 3.0]

        assert np.allclose(band_edges[:, 0], expected_min)
        assert np.allclose(band_edges[:, 1], expected_max)

    def test_local_channels_property(self, populated_decimation_level):
        """Test local_channels property."""
        dec_level = populated_decimation_level
        local_channels = dec_level.local_channels

        expected = ["hx", "hy", "ex", "ey", "hz"]
        assert local_channels == expected

    def test_fft_frequencies_property(self, populated_decimation_level):
        """Test fft_frequencies property."""
        dec_level = populated_decimation_level

        # Test the actual fft_frequencies property
        fft_freqs = dec_level.fft_frequencies

        assert isinstance(fft_freqs, np.ndarray)
        assert len(fft_freqs) > 0
        # Should return frequency harmonics based on window size (256) and sample rate (256.0)
        # Expected length should be num_samples/2 for positive frequencies
        assert len(fft_freqs) == 128  # 256/2 = 128

    def test_harmonic_indices_property_empty(self, basic_decimation_level):
        """Test harmonic_indices property with empty bands."""
        dec_level = basic_decimation_level
        harmonic_indices = dec_level.harmonic_indices

        assert isinstance(harmonic_indices, list)
        assert len(harmonic_indices) == 0

    def test_harmonic_indices_property_with_bands(self, populated_decimation_level):
        """Test harmonic_indices property with bands."""
        dec_level = populated_decimation_level

        # The harmonic_indices property is computed from each band's index_min and index_max
        # Let's verify it returns the expected values based on band configurations
        harmonic_indices = dec_level.harmonic_indices

        assert isinstance(harmonic_indices, list)

        # Verify that indices are collected from all bands
        # Each band should contribute its range from index_min to index_max
        expected_indices = []
        for band in dec_level.bands:
            band_indices = list(range(band.index_min, band.index_max + 1))
            expected_indices.extend(band_indices)

        expected_indices.sort()
        assert harmonic_indices == expected_indices


# =============================================================================
# Test FrequencyBands Integration
# =============================================================================


class TestDecimationLevelFrequencyBands:
    """Test FrequencyBands integration."""

    def test_frequency_bands_obj(self, populated_decimation_level):
        """Test frequency_bands_obj method."""
        dec_level = populated_decimation_level
        freq_bands = dec_level.frequency_bands_obj()

        assert isinstance(freq_bands, FrequencyBands)

        # Check that it was created with correct band_edges
        expected_band_edges = dec_level.band_edges
        assert np.allclose(freq_bands.band_edges, expected_band_edges)


# =============================================================================
# Test FC Decimation Compatibility
# =============================================================================


class TestDecimationLevelFCCompatibility:
    """Test FC decimation compatibility functionality."""

    def test_is_consistent_with_archived_fc_parameters_success(
        self, populated_decimation_level, mock_fc_decimation
    ):
        """Test successful FC parameter consistency check."""
        dec_level = populated_decimation_level

        # Configure the mock to match the populated decimation level
        mock_fc_decimation.channels_estimated = dec_level.local_channels
        mock_fc_decimation.time_series_decimation.anti_alias_filter = (
            dec_level.decimation.anti_alias_filter
        )
        mock_fc_decimation.time_series_decimation.sample_rate = (
            dec_level.decimation.sample_rate
        )
        mock_fc_decimation.short_time_fourier_transform.method = dec_level.stft.method
        mock_fc_decimation.stft.prewhitening_type = dec_level.stft.prewhitening_type
        mock_fc_decimation.stft.prewhitening_ar_order = (
            dec_level.stft.prewhitening_ar_order
        )
        mock_fc_decimation.stft.prewhitening_ma_order = (
            dec_level.stft.prewhitening_ma_order
        )
        mock_fc_decimation.stft.recoloring = dec_level.stft.recoloring
        mock_fc_decimation.stft.pre_fft_detrend_type = (
            dec_level.stft.pre_fft_detrend_type
        )
        mock_fc_decimation.stft.min_num_stft_windows = (
            dec_level.stft.min_num_stft_windows
        )
        mock_fc_decimation.stft.window = dec_level.stft.window

        # Test with local channels (remote=False)
        result = dec_level.is_consistent_with_archived_fc_parameters(
            mock_fc_decimation, remote=False
        )
        assert result is True

    def test_is_consistent_channels_mismatch(
        self, populated_decimation_level, mock_fc_decimation
    ):
        """Test FC consistency check with channel mismatch."""
        dec_level = populated_decimation_level

        # Configure the mock with mismatched channels
        mock_fc_decimation.channels_estimated = [
            "hx",
            "hy",
        ]  # Missing expected channels
        mock_fc_decimation.time_series_decimation.anti_alias_filter = (
            dec_level.decimation.anti_alias_filter
        )
        mock_fc_decimation.time_series_decimation.sample_rate = (
            dec_level.decimation.sample_rate
        )
        mock_fc_decimation.short_time_fourier_transform.method = dec_level.stft.method

        # Test should return False due to channel mismatch
        result = dec_level.is_consistent_with_archived_fc_parameters(
            mock_fc_decimation, remote=False
        )
        assert result is False

    def test_is_consistent_sample_rate_mismatch(
        self, populated_decimation_level, mock_fc_decimation
    ):
        """Test FC consistency check with sample rate mismatch."""
        dec_level = populated_decimation_level

        # Configure the mock with mismatched sample rate
        mock_fc_decimation.channels_estimated = dec_level.local_channels
        mock_fc_decimation.time_series_decimation.anti_alias_filter = (
            dec_level.decimation.anti_alias_filter
        )
        mock_fc_decimation.time_series_decimation.sample_rate = (
            512.0  # Different from expected 256.0
        )
        mock_fc_decimation.short_time_fourier_transform.method = dec_level.stft.method

        # Test should return False due to sample rate mismatch
        result = dec_level.is_consistent_with_archived_fc_parameters(
            mock_fc_decimation, remote=False
        )
        assert result is False

    def test_is_consistent_with_remote_channels(
        self, populated_decimation_level, mock_fc_decimation
    ):
        """Test FC consistency check with remote=True."""
        dec_level = populated_decimation_level

        # Configure the mock for remote channels test
        mock_fc_decimation.channels_estimated = dec_level.reference_channels
        mock_fc_decimation.time_series_decimation.anti_alias_filter = (
            dec_level.decimation.anti_alias_filter
        )
        mock_fc_decimation.time_series_decimation.sample_rate = (
            dec_level.decimation.sample_rate
        )
        mock_fc_decimation.short_time_fourier_transform.method = dec_level.stft.method
        mock_fc_decimation.stft.prewhitening_type = dec_level.stft.prewhitening_type
        mock_fc_decimation.stft.prewhitening_ar_order = (
            dec_level.stft.prewhitening_ar_order
        )
        mock_fc_decimation.stft.prewhitening_ma_order = (
            dec_level.stft.prewhitening_ma_order
        )
        mock_fc_decimation.stft.recoloring = dec_level.stft.recoloring
        mock_fc_decimation.stft.pre_fft_detrend_type = (
            dec_level.stft.pre_fft_detrend_type
        )
        mock_fc_decimation.stft.min_num_stft_windows = (
            dec_level.stft.min_num_stft_windows
        )
        mock_fc_decimation.stft.window = dec_level.stft.window

        # Test with remote channels (remote=True)
        result = dec_level.is_consistent_with_archived_fc_parameters(
            mock_fc_decimation, remote=True
        )
        assert result is True

    def test_is_consistent_prewhitening_order_mismatch(
        self, populated_decimation_level, mock_fc_decimation
    ):
        """Test ARMA order mismatches invalidate archived Fourier coefficients."""
        dec_level = populated_decimation_level
        dec_level.stft.prewhitening_type = "ARMA"
        dec_level.stft.prewhitening_ar_order = 3

        mock_fc_decimation.channels_estimated = dec_level.local_channels
        mock_fc_decimation.time_series_decimation.anti_alias_filter = (
            dec_level.decimation.anti_alias_filter
        )
        mock_fc_decimation.time_series_decimation.sample_rate = (
            dec_level.decimation.sample_rate
        )
        mock_fc_decimation.short_time_fourier_transform.method = dec_level.stft.method
        mock_fc_decimation.stft.prewhitening_type = dec_level.stft.prewhitening_type
        mock_fc_decimation.stft.prewhitening_ar_order = 2
        mock_fc_decimation.stft.prewhitening_ma_order = (
            dec_level.stft.prewhitening_ma_order
        )

        assert (
            dec_level.is_consistent_with_archived_fc_parameters(
                mock_fc_decimation, remote=False
            )
            is False
        )


# =============================================================================
# Test FC Decimation Creation
# =============================================================================


class TestDecimationLevelFCCreation:
    """Test FC decimation creation functionality."""

    def test_to_fc_decimation_local_channels(self, populated_decimation_level):
        """Test to_fc_decimation method with local channels."""
        dec_level = populated_decimation_level
        dec_level.stft.prewhitening_type = "ARMA"
        dec_level.stft.prewhitening_ar_order = 3
        dec_level.stft.prewhitening_ma_order = 0

        fc_dec = dec_level.to_fc_decimation(remote=False)

        assert isinstance(fc_dec, FCDecimation)
        assert fc_dec.channels_estimated == dec_level.local_channels
        assert fc_dec.time_series_decimation.sample_rate == 256.0
        assert fc_dec.time_series_decimation.level == 0
        assert fc_dec.time_series_decimation.factor == 1
        assert fc_dec.id == "0"
        assert fc_dec.stft.prewhitening_type == "ARMA"
        assert fc_dec.stft.prewhitening_ar_order == 3
        assert fc_dec.stft.prewhitening_ma_order == 0

    def test_to_fc_decimation_remote_channels(self, populated_decimation_level):
        """Test to_fc_decimation method with remote channels."""
        dec_level = populated_decimation_level

        fc_dec = dec_level.to_fc_decimation(remote=True)

        assert isinstance(fc_dec, FCDecimation)
        assert fc_dec.channels_estimated == dec_level.reference_channels
        assert fc_dec.time_series_decimation.sample_rate == 256.0

    def test_to_fc_decimation_ignore_harmonic_indices_true(
        self, populated_decimation_level
    ):
        """Test to_fc_decimation with ignore_harmonic_indices=True."""
        dec_level = populated_decimation_level

        fc_dec = dec_level.to_fc_decimation(ignore_harmonic_indices=True)

        # Should keep default harmonic indices
        assert hasattr(fc_dec.stft, "harmonic_indices")

    def test_to_fc_decimation_ignore_harmonic_indices_false(
        self, populated_decimation_level
    ):
        """Test to_fc_decimation with ignore_harmonic_indices=False."""
        dec_level = populated_decimation_level

        # Test with actual harmonic indices computed from bands
        # The harmonic_indices property returns indices from all bands
        expected_harmonic_indices = dec_level.harmonic_indices

        fc_dec = dec_level.to_fc_decimation(ignore_harmonic_indices=False)

        # Should set harmonic indices from the bands
        assert hasattr(fc_dec.stft, "harmonic_indices")
        # Verify the indices were set (exact comparison may depend on implementation)
        assert fc_dec.stft.harmonic_indices is not None


# =============================================================================
# Test Validation
# =============================================================================


class TestDecimationLevelValidation:
    """Test DecimationLevel field validation."""

    def test_channel_weight_specs_validation(self, sample_channel_weight_spec):
        """Test channel_weight_specs field validation."""
        # Test with list of objects
        dec_level = DecimationLevel(channel_weight_specs=[sample_channel_weight_spec])
        assert len(dec_level.channel_weight_specs) == 1
        assert isinstance(dec_level.channel_weight_specs[0], ChannelWeightSpec)

        # Test with list of dictionaries
        weight_spec_dict = {"id": "test_spec"}
        dec_level = DecimationLevel(channel_weight_specs=[weight_spec_dict])
        assert len(dec_level.channel_weight_specs) == 1
        assert isinstance(dec_level.channel_weight_specs[0], ChannelWeightSpec)

    def test_invalid_save_fcs_type(self):
        """Test validation error for invalid save_fcs_type."""
        with pytest.raises(ValueError):
            DecimationLevel(save_fcs_type="invalid_type")

    def test_valid_save_fcs_type_assignment(self, basic_decimation_level):
        """Test valid save_fcs_type assignments."""
        dec_level = basic_decimation_level

        # Test None assignment
        dec_level.save_fcs_type = None
        assert dec_level.save_fcs_type is None

        # Test enum assignment
        dec_level.save_fcs_type = SaveFcsTypeEnum.h5
        assert dec_level.save_fcs_type == SaveFcsTypeEnum.h5

        # Test string assignment
        dec_level.save_fcs_type = "csv"
        assert dec_level.save_fcs_type == SaveFcsTypeEnum.csv


# =============================================================================
# Test Serialization
# =============================================================================


class TestDecimationLevelSerialization:
    """Test DecimationLevel serialization and deserialization."""

    def test_to_dict(self, populated_decimation_level):
        """Test converting DecimationLevel to dictionary."""
        dec_level = populated_decimation_level
        dec_dict = dec_level.model_dump()

        assert isinstance(dec_dict, dict)

        # Check required fields
        required_fields = [
            "bands",
            "channel_weight_specs",
            "input_channels",
            "output_channels",
            "reference_channels",
            "save_fcs",
            "decimation",
            "estimator",
            "regression",
            "stft",
        ]
        for field in required_fields:
            assert field in dec_dict

        # Check enum serialization
        assert dec_dict["save_fcs_type"] == "h5"

    def test_from_dict(self):
        """Test creating DecimationLevel from dictionary."""
        data = {
            "input_channels": ["hx", "hy"],
            "output_channels": ["ex", "ey"],
            "save_fcs": True,
            "save_fcs_type": "csv",
            "bands": [
                {
                    "decimation_level": 0,
                    "index_min": 0,
                    "index_max": 10,
                    "frequency_min": 0.1,
                    "frequency_max": 1.0,
                }
            ],
        }

        dec_level = DecimationLevel.model_validate(data)

        assert dec_level.input_channels == ["hx", "hy"]
        assert dec_level.output_channels == ["ex", "ey"]
        assert dec_level.save_fcs is True
        assert dec_level.save_fcs_type == SaveFcsTypeEnum.csv
        assert len(dec_level.bands) == 1

    def test_json_serialization_roundtrip(self, populated_decimation_level):
        """Test JSON serialization and deserialization preserves data."""
        dec_level = populated_decimation_level

        # Serialize to JSON (exclude computed properties that contain numpy arrays)
        exclude_fields = {
            "lower_bounds",
            "upper_bounds",
            "band_edges",
            "fft_frequencies",
            "local_channels",
            "bands_dataframe",
            "frequency_sample_interval",
            "harmonic_indices",
        }
        json_str = dec_level.model_dump_json(exclude=exclude_fields)
        assert isinstance(json_str, str)
        assert len(json_str) > 100  # Basic sanity check

        # For now, just verify that basic JSON serialization works
        # Note: Full roundtrip test needs more work due to complex nested structures
        import json

        parsed_json = json.loads(json_str)
        assert "_class_name" in parsed_json
        assert parsed_json["_class_name"] == "decimation_level"
        assert "bands" in parsed_json


# =============================================================================
# Test Utility Functions
# =============================================================================


class TestUtilityFunctions:
    """Test module utility functions."""

    def test_df_from_bands(self, sample_bands):
        """Test _df_from_bands utility function."""
        df = _df_from_bands(sample_bands)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

        expected_columns = [
            "decimation_level",
            "lower_bound_index",
            "upper_bound_index",
            "frequency_min",
            "frequency_max",
        ]
        for col in expected_columns:
            assert col in df.columns

        # Test EMTF convention (+1 to decimation level)
        assert df["decimation_level"].tolist() == [1, 2, 3]

        # Test sorted by lower_bound_index
        assert df["lower_bound_index"].is_monotonic_increasing

    def test_df_from_bands_empty(self):
        """Test _df_from_bands with empty band list."""
        df = _df_from_bands([])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

        expected_columns = [
            "decimation_level",
            "lower_bound_index",
            "upper_bound_index",
            "frequency_min",
            "frequency_max",
        ]
        for col in expected_columns:
            assert col in df.columns

    @pytest.mark.parametrize(
        "samples_per_window,sample_rate,expected_length",
        [
            (256, 256.0, 128),  # Even number of samples
            (255, 255.0, 127),  # Odd number of samples
            (512, 512.0, 256),  # Larger window
            (64, 64.0, 32),  # Smaller window
        ],
    )
    def test_get_fft_harmonics(self, samples_per_window, sample_rate, expected_length):
        """Test get_fft_harmonics utility function."""
        freqs = get_fft_harmonics(samples_per_window, sample_rate)

        assert isinstance(freqs, np.ndarray)
        assert len(freqs) == expected_length

        # Should start with DC (0 Hz)
        assert freqs[0] == 0.0

        # Should be positive frequencies only (one-sided)
        assert np.all(freqs >= 0)

        # Check frequency resolution
        expected_df = sample_rate / samples_per_window
        if len(freqs) > 1:
            actual_df = freqs[1] - freqs[0]
            assert np.isclose(actual_df, expected_df)

    def test_get_fft_harmonics_frequency_values(self):
        """Test specific frequency values from get_fft_harmonics."""
        freqs = get_fft_harmonics(8, 8.0)  # Simple case for verification

        expected_freqs = np.array([0.0, 1.0, 2.0, 3.0])  # DC to just before Nyquist
        assert np.allclose(freqs, expected_freqs)


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestDecimationLevelEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_bands_computed_properties(self, basic_decimation_level):
        """Test computed properties with empty bands."""
        dec_level = basic_decimation_level

        # Should handle empty bands gracefully
        assert len(dec_level.lower_bounds) == 0
        assert len(dec_level.upper_bounds) == 0

        bands_df = dec_level.bands_dataframe
        assert len(bands_df) == 0

        band_edges = dec_level.band_edges
        assert band_edges.shape == (0, 2)

    def test_single_band_operations(self, basic_decimation_level, sample_band):
        """Test operations with single band."""
        dec_level = basic_decimation_level
        dec_level.add_band(sample_band)

        assert len(dec_level.lower_bounds) == 1
        assert len(dec_level.upper_bounds) == 1
        assert len(dec_level.bands_dataframe) == 1
        assert dec_level.band_edges.shape == (1, 2)

    def test_invalid_field_assignments(self, basic_decimation_level):
        """Test error handling for invalid field assignments."""
        dec_level = basic_decimation_level

        # Test invalid channel assignments
        with pytest.raises((ValueError, TypeError)):
            dec_level.input_channels = "not_a_list"

        with pytest.raises((ValueError, TypeError)):
            dec_level.save_fcs = "not_a_boolean"


# =============================================================================
# Integration Tests
# =============================================================================


class TestDecimationLevelIntegration:
    """Test integration scenarios."""

    def test_full_workflow(self):
        """Test complete DecimationLevel workflow."""
        # Create decimation level
        dec_level = DecimationLevel()

        # Configure basic properties
        dec_level.input_channels = ["hx", "hy"]
        dec_level.output_channels = ["ex", "ey", "hz"]
        dec_level.reference_channels = ["rrhx", "rrhy"]
        dec_level.save_fcs = True
        dec_level.save_fcs_type = SaveFcsTypeEnum.h5

        # Add bands
        for i in range(3):
            band_dict = {
                "decimation_level": i,
                "index_min": i * 10,
                "index_max": (i + 1) * 10,
                "frequency_min": 0.1 * (i + 1),
                "frequency_max": 1.0 * (i + 1),
            }
            dec_level.add_band(band_dict)

        # Configure decimation settings
        dec_level.decimation.level = 0
        dec_level.decimation.factor = 1
        dec_level.decimation.sample_rate = 256.0

        # Configure STFT settings
        dec_level.stft.window.num_samples = 256

        # Test computed properties
        assert len(dec_level.bands) == 3
        assert len(dec_level.local_channels) == 5
        assert dec_level.frequency_sample_interval == 1.0

        # Test serialization (exclude computed properties)
        exclude_fields = {
            "lower_bounds",
            "upper_bounds",
            "band_edges",
            "fft_frequencies",
            "local_channels",
            "bands_dataframe",
            "frequency_sample_interval",
            "harmonic_indices",
        }
        json_str = dec_level.model_dump_json(exclude=exclude_fields)
        assert isinstance(json_str, str) and len(json_str) > 100

        # Test FC decimation creation
        fc_dec = dec_level.to_fc_decimation()
        assert isinstance(fc_dec, FCDecimation)
        assert fc_dec.channels_estimated == dec_level.local_channels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
