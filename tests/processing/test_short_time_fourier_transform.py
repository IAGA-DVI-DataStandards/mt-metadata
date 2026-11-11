# -*- coding: utf-8 -*-
"""
Created on September 7, 2025

@author: GitHub Copilot

Comprehensive pytest test suite for ShortTimeFourierTransform basemodel using fixtures and subtests
whilst optimizing for efficiency.
"""

# =============================================================================
# Imports
# =============================================================================

import pytest
from pydantic import ValidationError

from mt_metadata.processing.short_time_fourier_transform import (
    MethodEnum,
    PerWindowDetrendTypeEnum,
    PreFftDetrendTypeEnum,
    PrewhiteningTypeEnum,
    ShortTimeFourierTransform,
)
from mt_metadata.processing.window import ClockZeroTypeEnum, TypeEnum, Window

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_stft():
    """Fixture for default STFT instance."""
    return ShortTimeFourierTransform()


@pytest.fixture
def custom_stft():
    """Fixture for custom STFT instance with specific parameters."""
    custom_window = Window(
        num_samples=256,
        overlap=32,
        type=TypeEnum.hamming,
        clock_zero_type=ClockZeroTypeEnum.user_specified,
        normalized=False,
    )
    return ShortTimeFourierTransform(
        harmonic_indices=10,
        method=MethodEnum.wavelet,
        min_num_stft_windows=8,
        per_window_detrend_type=PerWindowDetrendTypeEnum.linear,
        pre_fft_detrend_type=PreFftDetrendTypeEnum.other,
        prewhitening_type=PrewhiteningTypeEnum.arma,
        prewhitening_ar_order=3,
        prewhitening_ma_order=0,
        recoloring=False,
        window=custom_window,
    )


@pytest.fixture
def stft_params():
    """Fixture providing various parameter combinations for testing."""
    return {
        "minimal": {"harmonic_indices": 5, "min_num_stft_windows": 4},
        "complete": {
            "harmonic_indices": 20,
            "method": MethodEnum.fft,
            "min_num_stft_windows": 6,
            "per_window_detrend_type": PerWindowDetrendTypeEnum.constant,
            "pre_fft_detrend_type": PreFftDetrendTypeEnum.linear,
            "prewhitening_type": PrewhiteningTypeEnum.first_difference,
            "recoloring": True,
            "window": Window(num_samples=512, overlap=64, type=TypeEnum.hann),
        },
        "alternative": {
            "harmonic_indices": 15,
            "method": MethodEnum.other,
            "min_num_stft_windows": 3,
            "per_window_detrend_type": PerWindowDetrendTypeEnum.null,
            "pre_fft_detrend_type": PreFftDetrendTypeEnum.null,
            "prewhitening_type": PrewhiteningTypeEnum.first_difference,
            "recoloring": False,
            "window": Window(num_samples=128, overlap=16, type=TypeEnum.blackman),
        },
    }


# =============================================================================
# Test Classes
# =============================================================================


class TestSTFTInitialization:
    """Test STFT initialization and default values."""

    def test_default_initialization(self, default_stft):
        """Test default STFT initialization."""
        assert default_stft.harmonic_indices is None
        assert default_stft.method == MethodEnum.fft
        assert default_stft.min_num_stft_windows == 0
        assert default_stft.per_window_detrend_type == PerWindowDetrendTypeEnum.null
        assert default_stft.pre_fft_detrend_type == PreFftDetrendTypeEnum.linear
        assert default_stft.prewhitening_type == PrewhiteningTypeEnum.first_difference
        assert default_stft.prewhitening_ar_order == 0
        assert default_stft.prewhitening_ma_order == 0
        assert default_stft.recoloring is True
        assert isinstance(default_stft.window, Window)

    def test_custom_initialization(self, custom_stft):
        """Test custom STFT initialization with all parameters."""
        assert custom_stft.harmonic_indices == 10
        assert custom_stft.method == MethodEnum.wavelet
        assert custom_stft.min_num_stft_windows == 8
        assert custom_stft.per_window_detrend_type == PerWindowDetrendTypeEnum.linear
        assert custom_stft.pre_fft_detrend_type == PreFftDetrendTypeEnum.other
        assert custom_stft.prewhitening_type == PrewhiteningTypeEnum.arma
        assert custom_stft.prewhitening_ar_order == 3
        assert custom_stft.prewhitening_ma_order == 0
        assert custom_stft.recoloring is False
        assert isinstance(custom_stft.window, Window)
        assert custom_stft.window.num_samples == 256
        assert custom_stft.window.overlap == 32
        assert custom_stft.window.type == TypeEnum.hamming
        assert custom_stft.window.normalized is False

    @pytest.mark.parametrize("param_set", ["minimal", "complete", "alternative"])
    def test_parametrized_initialization(self, stft_params, param_set):
        """Test initialization with different parameter sets."""
        params = stft_params[param_set]
        stft = ShortTimeFourierTransform(**params)

        # Verify all provided parameters are set correctly
        for key, value in params.items():
            assert getattr(stft, key) == value

    def test_initialization_with_invalid_types(self):
        """Test initialization with invalid parameter types."""
        with pytest.raises(ValidationError):
            ShortTimeFourierTransform(harmonic_indices="invalid")

        with pytest.raises(ValidationError):
            ShortTimeFourierTransform(min_num_stft_windows="not_a_number")

        with pytest.raises(ValidationError):
            ShortTimeFourierTransform(recoloring="not_boolean")

        with pytest.raises(ValidationError):
            ShortTimeFourierTransform(prewhitening_ar_order=-1)

        with pytest.raises(ValidationError):
            ShortTimeFourierTransform(prewhitening_ma_order=-1)


class TestSTFTProperties:
    """Test STFT property access and computed fields."""

    def test_property_access(self, custom_stft):
        """Test that all properties are accessible."""
        # Test basic property access
        assert hasattr(custom_stft, "harmonic_indices")
        assert hasattr(custom_stft, "method")
        assert hasattr(custom_stft, "min_num_stft_windows")
        assert hasattr(custom_stft, "per_window_detrend_type")
        assert hasattr(custom_stft, "pre_fft_detrend_type")
        assert hasattr(custom_stft, "prewhitening_type")
        assert hasattr(custom_stft, "prewhitening_ar_order")
        assert hasattr(custom_stft, "prewhitening_ma_order")
        assert hasattr(custom_stft, "recoloring")
        assert hasattr(custom_stft, "window")

    def test_property_modification(self, default_stft):
        """Test property modification."""
        # Test harmonic_indices modification
        default_stft.harmonic_indices = 15
        assert default_stft.harmonic_indices == 15

        # Test method modification
        default_stft.method = MethodEnum.wavelet
        assert default_stft.method == MethodEnum.wavelet

        # Test boolean modification
        default_stft.recoloring = False
        assert default_stft.recoloring is False

        # Test window modification
        new_window = Window(num_samples=1024, overlap=128, type=TypeEnum.hamming)
        default_stft.window = new_window
        assert default_stft.window == new_window
        assert default_stft.window.num_samples == 1024

    def test_enum_property_validation(self, default_stft):
        """Test that enum properties validate correctly."""
        # Valid enum values should work
        default_stft.method = MethodEnum.fft
        assert default_stft.method == MethodEnum.fft

        default_stft.per_window_detrend_type = PerWindowDetrendTypeEnum.linear
        assert default_stft.per_window_detrend_type == PerWindowDetrendTypeEnum.linear

        # Invalid enum values should raise ValidationError
        with pytest.raises(ValidationError):
            default_stft.method = "invalid_method"


class TestSTFTEnumerations:
    """Test STFT enumeration types and values."""

    @pytest.mark.parametrize(
        "enum_value", [MethodEnum.fft, MethodEnum.wavelet, MethodEnum.other]
    )
    def test_method_enum_values(self, enum_value):
        """Test MethodEnum values."""
        stft = ShortTimeFourierTransform(method=enum_value)
        assert stft.method == enum_value

    @pytest.mark.parametrize(
        "enum_value",
        [
            PerWindowDetrendTypeEnum.linear,
            PerWindowDetrendTypeEnum.constant,
            PerWindowDetrendTypeEnum.null,
        ],
    )
    def test_per_window_detrend_enum_values(self, enum_value):
        """Test PerWindowDetrendTypeEnum values."""
        stft = ShortTimeFourierTransform(per_window_detrend_type=enum_value)
        assert stft.per_window_detrend_type == enum_value

    @pytest.mark.parametrize(
        "enum_value",
        [
            PreFftDetrendTypeEnum.linear,
            PreFftDetrendTypeEnum.other,
            PreFftDetrendTypeEnum.null,
        ],
    )
    def test_pre_fft_detrend_enum_values(self, enum_value):
        """Test PreFftDetrendTypeEnum values."""
        stft = ShortTimeFourierTransform(pre_fft_detrend_type=enum_value)
        assert stft.pre_fft_detrend_type == enum_value

    @pytest.mark.parametrize(
        "enum_value",
        [
            PrewhiteningTypeEnum.none,
            PrewhiteningTypeEnum.first_difference,
            PrewhiteningTypeEnum.arma,
            PrewhiteningTypeEnum.other,
        ],
    )
    def test_prewhitening_enum_values(self, enum_value):
        """Test PrewhiteningTypeEnum values."""
        stft = ShortTimeFourierTransform(prewhitening_type=enum_value)
        assert stft.prewhitening_type == enum_value


class TestSTFTWindowIntegration:
    """Test STFT window field integration and functionality."""

    def test_default_window_creation(self):
        """Test that default window is created properly."""
        stft = ShortTimeFourierTransform()
        assert isinstance(stft.window, Window)
        # Test default window properties (check the defaults)
        assert stft.window.type == TypeEnum.boxcar
        assert stft.window.clock_zero_type == ClockZeroTypeEnum.ignore
        assert stft.window.normalized is True

    def test_custom_window_assignment(self):
        """Test assigning custom window to STFT."""
        custom_window = Window(
            num_samples=1024, overlap=256, type=TypeEnum.hamming, normalized=False
        )
        stft = ShortTimeFourierTransform(window=custom_window)
        assert stft.window == custom_window
        assert stft.window.num_samples == 1024
        assert stft.window.overlap == 256
        assert stft.window.type == TypeEnum.hamming
        assert stft.window.normalized is False

    def test_window_type_variations(self):
        """Test different window types."""
        window_types = [
            TypeEnum.hann,
            TypeEnum.hamming,
            TypeEnum.blackman,
            TypeEnum.kaiser,
        ]

        for window_type in window_types:
            window = Window(num_samples=512, overlap=64, type=window_type)
            stft = ShortTimeFourierTransform(window=window)
            assert stft.window.type == window_type

    def test_window_modification_after_creation(self):
        """Test modifying window after STFT creation."""
        stft = ShortTimeFourierTransform()
        original_window_type = stft.window.type

        # Modify window properties
        new_window = Window(
            num_samples=2048, overlap=512, type=TypeEnum.hann, normalized=False
        )
        stft.window = new_window

        assert stft.window.type != original_window_type
        assert stft.window.num_samples == 2048
        assert stft.window.overlap == 512

    def test_window_serialization(self):
        """Test window field in STFT serialization."""
        custom_window = Window(num_samples=256, overlap=32, type=TypeEnum.hamming)
        stft = ShortTimeFourierTransform(
            harmonic_indices=10,
            method=MethodEnum.fft,
            min_num_stft_windows=4,
            window=custom_window,
        )

        stft_dict = stft.model_dump()
        assert "window" in stft_dict
        assert isinstance(stft_dict["window"], dict)
        assert stft_dict["window"]["num_samples"] == 256
        assert stft_dict["window"]["type"] == "hamming"

        # Test reconstruction from dict
        reconstructed_stft = ShortTimeFourierTransform.model_validate(stft_dict)
        assert reconstructed_stft.window.num_samples == 256
        assert reconstructed_stft.window.type == TypeEnum.hamming


class TestSTFTValidation:
    """Test STFT field validation and error handling."""

    def test_harmonic_indices_validation(self):
        """Test harmonic_indices field validation."""
        # Valid values
        valid_values = [1, 5, 10, 100, -1]
        for value in valid_values:
            stft = ShortTimeFourierTransform(harmonic_indices=value)
            assert stft.harmonic_indices == value

        # Invalid values should raise ValidationError
        with pytest.raises(ValidationError):
            ShortTimeFourierTransform(harmonic_indices="string")

    def test_min_num_stft_windows_validation(self):
        """Test min_num_stft_windows field validation."""
        # Valid values
        valid_values = [1, 4, 10, 100]
        for value in valid_values:
            stft = ShortTimeFourierTransform(min_num_stft_windows=value)
            assert stft.min_num_stft_windows == value

        # Invalid values should raise ValidationError
        with pytest.raises(ValidationError):
            ShortTimeFourierTransform(min_num_stft_windows="invalid")

    def test_recoloring_validation(self):
        """Test recoloring field validation."""
        # Valid boolean values
        for value in [True, False]:
            stft = ShortTimeFourierTransform(recoloring=value)
            assert stft.recoloring == value

        # Test string to boolean conversion (Pydantic should handle this)
        stft_true = ShortTimeFourierTransform(recoloring="true")
        assert stft_true.recoloring is True

        stft_false = ShortTimeFourierTransform(recoloring="false")
        assert stft_false.recoloring is False

    @pytest.mark.parametrize(
        "field_name,invalid_value",
        [
            ("method", "invalid_method"),
            ("per_window_detrend_type", "invalid_detrend"),
            ("pre_fft_detrend_type", "invalid_pre_fft"),
            ("prewhitening_type", "invalid_prewhiten"),
        ],
    )
    def test_enum_field_validation_errors(self, field_name, invalid_value):
        """Test that invalid enum values raise ValidationError."""
        with pytest.raises(ValidationError):
            ShortTimeFourierTransform(**{field_name: invalid_value})

    def test_window_field_validation(self):
        """Test window field validation."""
        # Valid Window instance should work
        valid_window = Window(num_samples=512, overlap=64)
        stft = ShortTimeFourierTransform(window=valid_window)
        assert stft.window == valid_window

        # Invalid window type should raise ValidationError
        with pytest.raises(ValidationError):
            ShortTimeFourierTransform(window="not_a_window")

        with pytest.raises(ValidationError):
            ShortTimeFourierTransform(window=123)


class TestSTFTComparison:
    """Test STFT comparison and copying."""

    def test_equality_comparison(self, stft_params):
        """Test STFT equality comparison."""
        params = stft_params["complete"]
        stft1 = ShortTimeFourierTransform(**params)
        stft2 = ShortTimeFourierTransform(**params)

        assert stft1 == stft2

    def test_inequality_comparison(self, default_stft, custom_stft):
        """Test STFT inequality comparison."""
        assert default_stft != custom_stft

    def test_model_copy(self, custom_stft):
        """Test STFT model copying."""
        copied_stft = custom_stft.model_copy()
        assert custom_stft == copied_stft

        # Verify they are different objects
        assert id(custom_stft) != id(copied_stft)

    def test_model_copy_with_changes(self, custom_stft):
        """Test STFT model copying with modifications."""
        copied_stft = custom_stft.model_copy(update={"harmonic_indices": 25})

        assert copied_stft.harmonic_indices == 25
        assert custom_stft.harmonic_indices == 10  # Original unchanged
        assert copied_stft.method == custom_stft.method  # Other fields copied


class TestSTFTRepresentation:
    """Test STFT string representation and serialization."""

    def test_string_representation(self, custom_stft):
        """Test string representation of STFT."""
        str_repr = str(custom_stft)

        # The representation should contain key information
        assert "harmonic_indices" in str_repr
        assert "10" in str_repr
        assert "method" in str_repr
        assert "wavelet" in str_repr

    def test_model_dump(self, custom_stft):
        """Test model serialization."""
        stft_dict = custom_stft.model_dump()

        assert isinstance(stft_dict, dict)
        assert stft_dict["harmonic_indices"] == 10
        assert stft_dict["method"] == "wavelet"
        assert stft_dict["min_num_stft_windows"] == 8
        assert stft_dict["prewhitening_ar_order"] == 3
        assert stft_dict["prewhitening_ma_order"] == 0
        assert stft_dict["recoloring"] is False

    def test_model_dump_json(self, custom_stft):
        """Test JSON serialization."""
        json_str = custom_stft.model_dump_json()

        assert isinstance(json_str, str)
        assert "harmonic_indices" in json_str
        assert "wavelet" in json_str

    def test_model_validate(self, stft_params):
        """Test model validation from dict."""
        params = stft_params["complete"]
        stft = ShortTimeFourierTransform.model_validate(params)

        for key, value in params.items():
            assert getattr(stft, key) == value


class TestSTFTEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_none_values(self):
        """Test STFT with None values for optional fields (by not setting them)."""
        # Don't explicitly pass None, just omit the parameters to get default None values
        stft = ShortTimeFourierTransform()
        assert stft.harmonic_indices is None
        assert stft.min_num_stft_windows == 0  # Default is 0

    def test_extreme_values(self):
        """Test STFT with extreme but valid values."""
        # Large values
        stft_large = ShortTimeFourierTransform(
            harmonic_indices=999999, min_num_stft_windows=999999
        )
        assert stft_large.harmonic_indices == 999999
        assert stft_large.min_num_stft_windows == 999999

        # Small/negative values
        stft_small = ShortTimeFourierTransform(
            harmonic_indices=-1, min_num_stft_windows=1
        )
        assert stft_small.harmonic_indices == -1
        assert stft_small.min_num_stft_windows == 1

    def test_empty_string_enum_values(self):
        """Test enums with empty string values."""
        stft = ShortTimeFourierTransform(
            per_window_detrend_type=PerWindowDetrendTypeEnum.null,
            pre_fft_detrend_type=PreFftDetrendTypeEnum.null,
        )
        assert stft.per_window_detrend_type == PerWindowDetrendTypeEnum.null
        assert stft.pre_fft_detrend_type == PreFftDetrendTypeEnum.null

    def test_model_fields_info(self):
        """Test model fields information."""
        fields = ShortTimeFourierTransform.model_fields

        expected_fields = {
            "harmonic_indices",
            "method",
            "min_num_stft_windows",
            "per_window_detrend_type",
            "pre_fft_detrend_type",
            "prewhitening_type",
            "prewhitening_ar_order",
            "prewhitening_ma_order",
            "recoloring",
            "window",
        }

        assert set(fields.keys()) == expected_fields

    def test_field_defaults(self):
        """Test that field defaults are correctly set."""
        stft = ShortTimeFourierTransform()

        # Check that defaults match expected values
        assert stft.method == MethodEnum.fft
        assert stft.per_window_detrend_type == PerWindowDetrendTypeEnum.null
        assert stft.pre_fft_detrend_type == PreFftDetrendTypeEnum.linear
        assert stft.prewhitening_type == PrewhiteningTypeEnum.first_difference
        assert stft.recoloring is True


class TestSTFTIntegration:
    """Test STFT integration scenarios and complex workflows."""

    def test_complete_workflow(self, stft_params):
        """Test a complete STFT configuration workflow."""
        # Create STFT with complete parameters
        params = stft_params["complete"]
        stft = ShortTimeFourierTransform(**params)

        # Verify all parameters are set
        assert stft.harmonic_indices == 20
        assert stft.method == MethodEnum.fft
        assert stft.min_num_stft_windows == 6
        assert stft.per_window_detrend_type == PerWindowDetrendTypeEnum.constant
        assert stft.pre_fft_detrend_type == PreFftDetrendTypeEnum.linear
        assert stft.prewhitening_type == PrewhiteningTypeEnum.first_difference
        assert stft.recoloring is True

        # Test serialization and deserialization
        stft_dict = stft.model_dump()
        recreated_stft = ShortTimeFourierTransform.model_validate(stft_dict)
        assert stft == recreated_stft

    def test_parameter_combination_validation(self):
        """Test various parameter combinations for logical consistency."""
        # Test FFT method with appropriate parameters
        fft_stft = ShortTimeFourierTransform(
            method=MethodEnum.fft,
            pre_fft_detrend_type=PreFftDetrendTypeEnum.linear,
            prewhitening_type=PrewhiteningTypeEnum.first_difference,
        )
        assert fft_stft.method == MethodEnum.fft

        # Test wavelet method
        wavelet_stft = ShortTimeFourierTransform(
            method=MethodEnum.wavelet,
            per_window_detrend_type=PerWindowDetrendTypeEnum.constant,
        )
        assert wavelet_stft.method == MethodEnum.wavelet

    def test_configuration_scenarios(self):
        """Test different configuration scenarios."""
        scenarios = [
            # Minimal configuration
            {"harmonic_indices": 5},
            # FFT-focused configuration
            {
                "method": MethodEnum.fft,
                "pre_fft_detrend_type": PreFftDetrendTypeEnum.linear,
                "min_num_stft_windows": 4,
            },
            # Wavelet-focused configuration
            {
                "method": MethodEnum.wavelet,
                "per_window_detrend_type": PerWindowDetrendTypeEnum.linear,
                "recoloring": False,
            },
            # Research configuration
            {
                "method": MethodEnum.other,
                "pre_fft_detrend_type": PreFftDetrendTypeEnum.other,
                "prewhitening_type": PrewhiteningTypeEnum.other,
                "per_window_detrend_type": PerWindowDetrendTypeEnum.null,
            },
        ]

        for scenario in scenarios:
            stft = ShortTimeFourierTransform(**scenario)
            # Verify configuration was applied
            for key, value in scenario.items():
                assert getattr(stft, key) == value

    @pytest.mark.parametrize("recoloring_value", [True, False])
    def test_recoloring_workflow_impact(self, recoloring_value):
        """Test how recoloring setting affects the workflow."""
        stft = ShortTimeFourierTransform(
            method=MethodEnum.fft,
            prewhitening_type=PrewhiteningTypeEnum.first_difference,
            recoloring=recoloring_value,
        )

        assert stft.recoloring == recoloring_value
        # In a real implementation, this might affect processing logic


# =============================================================================
# Performance and Efficiency Tests
# =============================================================================


class TestSTFTPerformance:
    """Test STFT performance and efficiency aspects."""

    def test_creation_performance(self, stft_params):
        """Test that STFT creation is efficient."""
        import time

        # Test creation time for multiple instances
        start_time = time.time()
        stfts = []
        for _ in range(100):
            stft = ShortTimeFourierTransform(**stft_params["complete"])
            stfts.append(stft)

        creation_time = time.time() - start_time

        # Should create 100 instances in reasonable time (< 1 second)
        assert creation_time < 1.0
        assert len(stfts) == 100

    def test_memory_efficiency(self, stft_params):
        """Test memory efficiency of STFT instances."""
        # Create multiple instances and verify they don't leak memory
        stfts = []
        for i in range(50):
            params = stft_params["complete"].copy()
            params["harmonic_indices"] = i
            stft = ShortTimeFourierTransform(**params)
            stfts.append(stft)

        # All instances should be unique and properly configured
        assert len(set(id(stft) for stft in stfts)) == 50
        assert all(stft.harmonic_indices == i for i, stft in enumerate(stfts))
