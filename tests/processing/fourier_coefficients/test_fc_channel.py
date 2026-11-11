# -*- coding: utf-8 -*-
"""
Created on September 7, 2025

@author: GitHub Copilot

Comprehensive pytest test suite for FCChannel basemodel using fixtures and subtests
whilst optimizing for efficiency.
"""

# =============================================================================
# Imports
# =============================================================================

import pytest
from pydantic import ValidationError

from mt_metadata.common import TimePeriod
from mt_metadata.processing.fourier_coefficients.fc_channel import FCChannel

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_fc_channel():
    """Fixture for default FCChannel instance."""
    return FCChannel()


@pytest.fixture
def custom_fc_channel():
    """Fixture for custom FCChannel instance with specific parameters."""
    return FCChannel(
        component="ex",
        frequency_max=100.0,
        frequency_min=0.1,
        sample_rate_decimation_level=256.0,
        sample_rate_window_step=8.0,
        units="mV",
        time_period=TimePeriod(
            start="2020-01-01T00:00:00+00:00", end="2020-01-02T00:00:00+00:00"
        ),
    )


@pytest.fixture
def fc_channel_params():
    """Fixture providing various parameter combinations for testing."""
    return {
        "minimal": {
            "component": "hx",
            "frequency_max": 50.0,
            "frequency_min": 0.01,
        },
        "complete": {
            "component": "ey",
            "frequency_max": 1000.0,
            "frequency_min": 0.001,
            "sample_rate_decimation_level": 1024.0,
            "sample_rate_window_step": 16.0,
            "units": "nT",
            "time_period": TimePeriod(
                start="2021-03-15T12:30:00+00:00", end="2021-03-16T12:30:00+00:00"
            ),
        },
        "electric": {
            "component": "ex",
            "frequency_max": 512.0,
            "frequency_min": 0.1,
            "sample_rate_decimation_level": 512.0,
            "sample_rate_window_step": 4.0,
            "units": "mV/km",
            "time_period": TimePeriod(
                start="2022-06-01T08:00:00+00:00", end="2022-06-02T08:00:00+00:00"
            ),
        },
        "magnetic": {
            "component": "hy",
            "frequency_max": 256.0,
            "frequency_min": 0.01,
            "sample_rate_decimation_level": 256.0,
            "sample_rate_window_step": 2.0,
            "units": "nT",
            "time_period": TimePeriod(
                start="2023-09-10T14:15:30+00:00", end="2023-09-11T14:15:30+00:00"
            ),
        },
    }


# =============================================================================
# Test Classes
# =============================================================================


class TestFCChannelInitialization:
    """Test FCChannel initialization and default values."""

    def test_default_initialization(self, default_fc_channel):
        """Test default FCChannel initialization."""
        assert default_fc_channel.component == ""
        assert default_fc_channel.frequency_max == 0.0
        assert default_fc_channel.frequency_min == 0.0
        assert default_fc_channel.sample_rate_decimation_level == 1.0
        assert default_fc_channel.sample_rate_window_step == 1.0
        assert default_fc_channel.units == "counts"
        assert isinstance(default_fc_channel.time_period, TimePeriod)

    def test_custom_initialization(self, custom_fc_channel):
        """Test custom FCChannel initialization with all parameters."""
        assert custom_fc_channel.component == "ex"
        assert custom_fc_channel.frequency_max == 100.0
        assert custom_fc_channel.frequency_min == 0.1
        assert custom_fc_channel.sample_rate_decimation_level == 256.0
        assert custom_fc_channel.sample_rate_window_step == 8.0
        assert custom_fc_channel.units == "milliVolt"
        assert isinstance(custom_fc_channel.time_period, TimePeriod)

    @pytest.mark.parametrize(
        "param_set", ["minimal", "complete", "electric", "magnetic"]
    )
    def test_parametrized_initialization(self, fc_channel_params, param_set):
        """Test initialization with different parameter sets."""
        params = fc_channel_params[param_set]
        fc_channel = FCChannel(**params)

        # Verify all provided parameters are set correctly (except units which get transformed)
        for key, value in params.items():
            if key == "units":
                # Units get transformed by the validator, so just check it's not empty
                assert fc_channel.units != ""
                assert isinstance(fc_channel.units, str)
            else:
                assert getattr(fc_channel, key) == value

    def test_initialization_with_invalid_types(self):
        """Test initialization with invalid parameter types."""
        with pytest.raises(ValidationError):
            FCChannel(frequency_max="invalid")

        with pytest.raises(ValidationError):
            FCChannel(frequency_min="not_a_number")

        with pytest.raises(ValidationError):
            FCChannel(sample_rate_decimation_level="invalid")


class TestFCChannelProperties:
    """Test FCChannel property access and computed fields."""

    def test_property_access(self, custom_fc_channel):
        """Test that all properties are accessible."""
        # Test basic property access
        assert hasattr(custom_fc_channel, "component")
        assert hasattr(custom_fc_channel, "frequency_max")
        assert hasattr(custom_fc_channel, "frequency_min")
        assert hasattr(custom_fc_channel, "sample_rate_decimation_level")
        assert hasattr(custom_fc_channel, "sample_rate_window_step")
        assert hasattr(custom_fc_channel, "units")
        assert hasattr(custom_fc_channel, "time_period")

    def test_property_modification(self, default_fc_channel):
        """Test property modification."""
        # Test component modification
        default_fc_channel.component = "hz"
        assert default_fc_channel.component == "hz"

        # Test frequency modification
        default_fc_channel.frequency_max = 200.0
        assert default_fc_channel.frequency_max == 200.0

        default_fc_channel.frequency_min = 0.5
        assert default_fc_channel.frequency_min == 0.5

        # Test sample rate modification
        default_fc_channel.sample_rate_decimation_level = 512.0
        assert default_fc_channel.sample_rate_decimation_level == 512.0

        # Test units modification
        default_fc_channel.units = "nT"
        assert default_fc_channel.units == "nanoTesla"

    def test_time_period_modification(self, default_fc_channel):
        """Test time period modification."""
        new_time_period = TimePeriod(
            start="2023-01-01T00:00:00+00:00", end="2023-01-02T00:00:00+00:00"
        )
        default_fc_channel.time_period = new_time_period
        assert default_fc_channel.time_period == new_time_period


class TestFCChannelValidation:
    """Test FCChannel field validation and error handling."""

    def test_component_validation(self):
        """Test component field validation."""
        # Valid component names
        valid_components = ["ex", "ey", "hx", "hy", "hz", "ex1", "hy2"]
        test_time_period = TimePeriod(
            start="2020-01-01T00:00:00+00:00", end="2020-01-02T00:00:00+00:00"
        )
        for component in valid_components:
            fc_channel = FCChannel(
                component=component,
                frequency_max=10.0,
                frequency_min=1.0,
                sample_rate_decimation_level=4.0,
                sample_rate_window_step=1.0,
                units="nT",
                time_period=test_time_period,
            )
            assert fc_channel.component == component

        # Component should accept any string
        fc_channel = FCChannel(
            component="custom_component",
            frequency_max=10.0,
            frequency_min=1.0,
            sample_rate_decimation_level=4.0,
            sample_rate_window_step=1.0,
            units="nT",
            time_period=test_time_period,
        )
        assert fc_channel.component == "custom_component"

    def test_frequency_validation(self):
        """Test frequency field validation."""
        test_time_period = TimePeriod(
            start="2020-01-01T00:00:00+00:00", end="2020-01-02T00:00:00+00:00"
        )
        # Valid frequency values
        valid_freqs = [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
        for freq in valid_freqs:
            fc_channel = FCChannel(
                component="hx",
                frequency_max=freq,
                frequency_min=freq / 10,
                sample_rate_decimation_level=4.0,
                sample_rate_window_step=1.0,
                units="nT",
                time_period=test_time_period,
            )
            assert fc_channel.frequency_max == freq
            assert fc_channel.frequency_min == freq / 10

        # Invalid frequency values should raise ValidationError
        with pytest.raises(ValidationError):
            FCChannel(
                component="hx",
                frequency_max="invalid",
                frequency_min=1.0,
                sample_rate_decimation_level=4.0,
                sample_rate_window_step=1.0,
                units="nT",
                time_period=test_time_period,
            )

        with pytest.raises(ValidationError):
            FCChannel(
                component="hx",
                frequency_max=10.0,
                frequency_min="not_a_number",
                sample_rate_decimation_level=4.0,
                sample_rate_window_step=1.0,
                units="nT",
                time_period=test_time_period,
            )

    def test_sample_rate_validation(self):
        """Test sample rate field validation."""
        test_time_period = TimePeriod(
            start="2020-01-01T00:00:00+00:00", end="2020-01-02T00:00:00+00:00"
        )
        # Valid sample rate values
        valid_rates = [0.1, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0]
        for rate in valid_rates:
            fc_channel = FCChannel(
                component="hx",
                frequency_max=10.0,
                frequency_min=1.0,
                sample_rate_decimation_level=rate,
                sample_rate_window_step=rate / 4,
                units="nT",
                time_period=test_time_period,
            )
            assert fc_channel.sample_rate_decimation_level == rate
            assert fc_channel.sample_rate_window_step == rate / 4

        # Invalid sample rate values should raise ValidationError
        with pytest.raises(ValidationError):
            FCChannel(
                component="hx",
                frequency_max=10.0,
                frequency_min=1.0,
                sample_rate_decimation_level="invalid",
                sample_rate_window_step=1.0,
                units="nT",
                time_period=test_time_period,
            )

        with pytest.raises(ValidationError):
            FCChannel(
                component="hx",
                frequency_max=10.0,
                frequency_min=1.0,
                sample_rate_decimation_level=4.0,
                sample_rate_window_step="not_a_number",
                units="nT",
                time_period=test_time_period,
            )

    def test_units_validation(self):
        """Test units field validation and unit conversion."""
        test_time_period = TimePeriod(
            start="2020-01-01T00:00:00+00:00", end="2020-01-02T00:00:00+00:00"
        )
        # Test standard units
        standard_units = ["mV", "nT", "counts"]
        expected_units = ["milliVolt", "nanoTesla", "counts"]
        for unit, expected in zip(standard_units, expected_units):
            fc_channel = FCChannel(
                component="hx",
                frequency_max=10.0,
                frequency_min=1.0,
                sample_rate_decimation_level=4.0,
                sample_rate_window_step=1.0,
                units=unit,
                time_period=test_time_period,
            )
            assert expected in fc_channel.units or fc_channel.units == expected

        # Test unit validation with invalid units - should set to 'unknown'
        fc_channel = FCChannel(
            component="hx",
            frequency_max=10.0,
            frequency_min=1.0,
            sample_rate_decimation_level=4.0,
            sample_rate_window_step=1.0,
            units="invalid_unit_xyz",
            time_period=test_time_period,
        )
        assert fc_channel.units == "unknown"

    def test_time_period_validation(self):
        """Test time period field validation."""
        # Valid time period
        valid_time_period = TimePeriod(
            start="2020-01-01T00:00:00+00:00", end="2020-01-02T00:00:00+00:00"
        )
        fc_channel = FCChannel(
            component="hx",
            frequency_max=10.0,
            frequency_min=1.0,
            sample_rate_decimation_level=4.0,
            sample_rate_window_step=1.0,
            units="nT",
            time_period=valid_time_period,
        )
        assert fc_channel.time_period == valid_time_period

        # Invalid time period type should raise ValidationError
        with pytest.raises(ValidationError):
            FCChannel(
                component="hx",
                frequency_max=10.0,
                frequency_min=1.0,
                sample_rate_decimation_level=4.0,
                sample_rate_window_step=1.0,
                units="nT",
                time_period="not_a_time_period",
            )


class TestFCChannelComparison:
    """Test FCChannel comparison and copying."""

    def test_equality_comparison(self, fc_channel_params):
        """Test FCChannel equality comparison."""
        params = fc_channel_params["complete"]
        fc_channel1 = FCChannel(**params)
        fc_channel2 = FCChannel(**params)

        assert fc_channel1 == fc_channel2

    def test_inequality_comparison(self, default_fc_channel, custom_fc_channel):
        """Test FCChannel inequality comparison."""
        assert default_fc_channel != custom_fc_channel

    def test_model_copy(self, custom_fc_channel):
        """Test FCChannel model copying."""
        copied_fc_channel = custom_fc_channel.model_copy()
        assert custom_fc_channel == copied_fc_channel

        # Verify they are different objects
        assert id(custom_fc_channel) != id(copied_fc_channel)

    def test_model_copy_with_changes(self, custom_fc_channel):
        """Test FCChannel model copying with modifications."""
        copied_fc_channel = custom_fc_channel.model_copy(
            update={"component": "hy", "frequency_max": 200.0}
        )

        assert copied_fc_channel.component == "hy"
        assert copied_fc_channel.frequency_max == 200.0
        assert custom_fc_channel.component == "ex"  # Original unchanged
        assert custom_fc_channel.frequency_max == 100.0  # Original unchanged
        assert copied_fc_channel.units == custom_fc_channel.units  # Other fields copied


class TestFCChannelRepresentation:
    """Test FCChannel string representation and serialization."""

    def test_string_representation(self, custom_fc_channel):
        """Test string representation of FCChannel."""
        str_repr = str(custom_fc_channel)

        # The representation should contain key information
        assert "component" in str_repr
        assert "ex" in str_repr
        assert "frequency_max" in str_repr
        assert "100.0" in str_repr

    def test_model_dump(self, custom_fc_channel):
        """Test model serialization."""
        fc_channel_dict = custom_fc_channel.model_dump()

        assert isinstance(fc_channel_dict, dict)
        assert fc_channel_dict["component"] == "ex"
        assert fc_channel_dict["frequency_max"] == 100.0
        assert fc_channel_dict["frequency_min"] == 0.1
        assert fc_channel_dict["units"] == "milliVolt"

    def test_model_dump_json(self, custom_fc_channel):
        """Test JSON serialization."""
        json_str = custom_fc_channel.model_dump_json()

        assert isinstance(json_str, str)
        assert "component" in json_str
        assert "ex" in json_str
        assert "frequency_max" in json_str

    def test_model_validate(self, fc_channel_params):
        """Test model validation from dict."""
        params = fc_channel_params["complete"]
        fc_channel = FCChannel.model_validate(params)

        for key, value in params.items():
            if key == "units":
                # Units may be transformed by validator (e.g., "nT" -> "nanoTesla")
                continue
            assert getattr(fc_channel, key) == value

    def test_prewhitening_result_serialization(self):
        """Test fitted prewhitening coefficients survive model serialization."""
        fc_channel = FCChannel(
            prewhitening_type="ARMA",
            prewhitening_fit_domain="full run",
            prewhitening_response_convention="A(z) = 1 - sum(phi_k z^-k)",
            prewhitening_ar_coefficients=[0.8, -0.2],
            prewhitening_ma_coefficients=[],
        )

        restored = FCChannel.model_validate(fc_channel.model_dump())

        assert restored.prewhitening_type == "ARMA"
        assert restored.prewhitening_fit_domain == "full run"
        assert restored.prewhitening_ar_coefficients == [0.8, -0.2]
        assert restored.prewhitening_ma_coefficients == []


class TestFCChannelEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_frequencies(self):
        """Test FCChannel with zero frequency values."""
        fc_channel = FCChannel(frequency_max=0.0, frequency_min=0.0)
        assert fc_channel.frequency_max == 0.0
        assert fc_channel.frequency_min == 0.0

    def test_extreme_frequency_values(self):
        """Test FCChannel with extreme frequency values."""
        # Very small frequencies
        fc_channel_small = FCChannel(frequency_max=0.000001, frequency_min=0.0000001)
        assert fc_channel_small.frequency_max == 0.000001
        assert fc_channel_small.frequency_min == 0.0000001

        # Very large frequencies
        fc_channel_large = FCChannel(frequency_max=1000000.0, frequency_min=100000.0)
        assert fc_channel_large.frequency_max == 1000000.0
        assert fc_channel_large.frequency_min == 100000.0

    def test_extreme_sample_rate_values(self):
        """Test FCChannel with extreme sample rate values."""
        # Very small sample rates
        fc_channel_small = FCChannel(
            sample_rate_decimation_level=0.001, sample_rate_window_step=0.0001
        )
        assert fc_channel_small.sample_rate_decimation_level == 0.001
        assert fc_channel_small.sample_rate_window_step == 0.0001

        # Very large sample rates
        fc_channel_large = FCChannel(
            sample_rate_decimation_level=100000.0, sample_rate_window_step=10000.0
        )
        assert fc_channel_large.sample_rate_decimation_level == 100000.0
        assert fc_channel_large.sample_rate_window_step == 10000.0

    def test_model_fields_info(self):
        """Test model fields information."""
        fields = FCChannel.model_fields

        expected_fields = {
            "component",
            "frequency_max",
            "frequency_min",
            "sample_rate_decimation_level",
            "sample_rate_window_step",
            "units",
            "time_period",
            "prewhitening_type",
            "prewhitening_fit_domain",
            "prewhitening_response_convention",
            "prewhitening_ar_coefficients",
            "prewhitening_ma_coefficients",
        }

        assert set(fields.keys()) == expected_fields

    def test_field_defaults(self):
        """Test that field defaults are correctly set."""
        fc_channel = FCChannel()

        # Check that defaults match expected values
        assert fc_channel.component == ""
        assert fc_channel.frequency_max == 0.0
        assert fc_channel.frequency_min == 0.0
        assert fc_channel.sample_rate_decimation_level == 1.0
        assert fc_channel.sample_rate_window_step == 1.0
        assert fc_channel.units == "counts"
        assert isinstance(fc_channel.time_period, TimePeriod)


class TestFCChannelIntegration:
    """Test FCChannel integration scenarios and complex workflows."""

    def test_complete_workflow(self, fc_channel_params):
        """Test a complete FCChannel configuration workflow."""
        # Create FCChannel with complete parameters
        params = fc_channel_params["complete"]
        fc_channel = FCChannel(**params)

        # Verify all parameters are set
        assert fc_channel.component == "ey"
        assert fc_channel.frequency_max == 1000.0
        assert fc_channel.frequency_min == 0.001
        assert fc_channel.sample_rate_decimation_level == 1024.0
        assert fc_channel.sample_rate_window_step == 16.0
        assert fc_channel.units == "nanoTesla"
        assert isinstance(fc_channel.time_period, TimePeriod)

        # Test serialization and deserialization
        fc_channel_dict = fc_channel.model_dump()
        recreated_fc_channel = FCChannel.model_validate(fc_channel_dict)
        assert fc_channel == recreated_fc_channel

    def test_electric_field_configuration(self, fc_channel_params):
        """Test electric field channel configuration."""
        params = fc_channel_params["electric"]
        fc_channel = FCChannel(**params)

        assert fc_channel.component == "ex"
        assert "volt" in fc_channel.units.lower()
        assert fc_channel.frequency_max > fc_channel.frequency_min

    def test_magnetic_field_configuration(self, fc_channel_params):
        """Test magnetic field channel configuration."""
        params = fc_channel_params["magnetic"]
        fc_channel = FCChannel(**params)

        assert fc_channel.component == "hy"
        assert "tesla" in fc_channel.units.lower()
        assert fc_channel.frequency_max > fc_channel.frequency_min

    def test_frequency_range_consistency(self):
        """Test frequency range logical consistency."""
        # Normal case where max > min
        fc_channel = FCChannel(frequency_max=100.0, frequency_min=1.0)
        assert fc_channel.frequency_max > fc_channel.frequency_min

        # Edge case where max = min (single frequency)
        fc_channel_single = FCChannel(frequency_max=50.0, frequency_min=50.0)
        assert fc_channel_single.frequency_max == fc_channel_single.frequency_min

    def test_sample_rate_relationships(self):
        """Test sample rate relationship scenarios."""
        # Typical case where decimation rate > window step rate
        fc_channel = FCChannel(
            sample_rate_decimation_level=256.0, sample_rate_window_step=4.0
        )
        assert (
            fc_channel.sample_rate_decimation_level > fc_channel.sample_rate_window_step
        )

        # Edge case where rates are equal
        fc_channel_equal = FCChannel(
            sample_rate_decimation_level=16.0, sample_rate_window_step=16.0
        )
        assert (
            fc_channel_equal.sample_rate_decimation_level
            == fc_channel_equal.sample_rate_window_step
        )

    @pytest.mark.parametrize("component_type", ["ex", "ey", "hx", "hy", "hz"])
    def test_component_type_scenarios(self, component_type):
        """Test different component type scenarios."""
        test_time_period = TimePeriod(
            start="2020-01-01T00:00:00+00:00", end="2020-01-02T00:00:00+00:00"
        )

        # Test component-specific units
        if component_type.startswith("e"):  # Electric field
            fc_channel = FCChannel(
                component=component_type,
                frequency_max=10.0,
                frequency_min=1.0,
                sample_rate_decimation_level=4.0,
                sample_rate_window_step=1.0,
                units="mV/km",
                time_period=test_time_period,
            )
            assert "volt" in fc_channel.units.lower() or "milliVolt" in fc_channel.units
        elif component_type.startswith("h"):  # Magnetic field
            fc_channel = FCChannel(
                component=component_type,
                frequency_max=10.0,
                frequency_min=1.0,
                sample_rate_decimation_level=4.0,
                sample_rate_window_step=1.0,
                units="nT",
                time_period=test_time_period,
            )
            assert (
                "tesla" in fc_channel.units.lower() or "nanoTesla" in fc_channel.units
            )

        assert fc_channel.component == component_type


class TestFCChannelTimePeriodIntegration:
    """Test FCChannel time period integration and functionality."""

    def test_default_time_period_creation(self):
        """Test that default time period is created properly."""
        fc_channel = FCChannel()
        assert isinstance(fc_channel.time_period, TimePeriod)
        # Default time period should have start and end times
        assert hasattr(fc_channel.time_period, "start")
        assert hasattr(fc_channel.time_period, "end")

    def test_custom_time_period_assignment(self):
        """Test assigning custom time period to FCChannel."""
        custom_time_period = TimePeriod(
            start="2023-05-01T12:00:00+00:00", end="2023-05-02T12:00:00+00:00"
        )
        fc_channel = FCChannel(time_period=custom_time_period)
        assert fc_channel.time_period == custom_time_period

    def test_time_period_modification_after_creation(self):
        """Test modifying time period after FCChannel creation."""
        fc_channel = FCChannel()
        original_time_period = fc_channel.time_period

        # Modify time period
        new_time_period = TimePeriod(
            start="2024-01-01T00:00:00+00:00", end="2024-01-02T00:00:00+00:00"
        )
        fc_channel.time_period = new_time_period

        assert fc_channel.time_period != original_time_period
        assert fc_channel.time_period == new_time_period

    def test_time_period_serialization(self):
        """Test time period field in FCChannel serialization."""
        custom_time_period = TimePeriod(
            start="2023-07-15T09:30:00+00:00", end="2023-07-16T09:30:00+00:00"
        )
        fc_channel = FCChannel(
            component="ex", frequency_max=100.0, time_period=custom_time_period
        )

        fc_channel_dict = fc_channel.model_dump()
        assert "time_period" in fc_channel_dict
        assert isinstance(fc_channel_dict["time_period"], dict)

        # Test reconstruction from dict
        reconstructed_fc_channel = FCChannel.model_validate(fc_channel_dict)
        assert isinstance(reconstructed_fc_channel.time_period, TimePeriod)


# =============================================================================
# Performance and Efficiency Tests
# =============================================================================


class TestFCChannelPerformance:
    """Test FCChannel performance and efficiency aspects."""

    def test_creation_performance(self, fc_channel_params):
        """Test that FCChannel creation is efficient."""
        import time

        # Test creation time for multiple instances
        start_time = time.time()
        fc_channels = []
        for _ in range(100):
            fc_channel = FCChannel(**fc_channel_params["complete"])
            fc_channels.append(fc_channel)

        creation_time = time.time() - start_time

        # Should create 100 instances in reasonable time (< 1 second)
        assert creation_time < 1.0
        assert len(fc_channels) == 100

    def test_memory_efficiency(self, fc_channel_params):
        """Test memory efficiency of FCChannel instances."""
        # Create multiple instances and verify they don't leak memory
        fc_channels = []
        for i in range(50):
            params = fc_channel_params["complete"].copy()
            params["component"] = f"ex{i}"
            params["frequency_max"] = 100.0 + i
            fc_channel = FCChannel(**params)
            fc_channels.append(fc_channel)

        # All instances should be unique and properly configured
        assert len(set(id(fc_channel) for fc_channel in fc_channels)) == 50
        assert all(
            fc_channel.component == f"ex{i}" for i, fc_channel in enumerate(fc_channels)
        )

    def test_serialization_performance(self, custom_fc_channel):
        """Test serialization/deserialization performance."""
        import time

        # Test serialization performance
        start_time = time.time()
        for _ in range(100):
            serialized = custom_fc_channel.model_dump_json()
            deserialized = FCChannel.model_validate_json(serialized)

        serialization_time = time.time() - start_time

        # Should complete 100 serialization cycles in reasonable time
        assert serialization_time < 1.0
        assert isinstance(deserialized, FCChannel)

    def test_validation_performance(self):
        """Test validation performance with various inputs."""
        import time

        # Test validation performance
        start_time = time.time()

        test_data = [
            {
                "component": f"ex{i}",
                "frequency_max": 100.0 + i,
                "frequency_min": 0.1 + i / 1000,
            }
            for i in range(100)
        ]

        for data in test_data:
            fc_channel = FCChannel(**data)
            assert fc_channel.component.startswith("ex")

        validation_time = time.time() - start_time

        # Should validate 100 instances in reasonable time
        assert validation_time < 1.0
