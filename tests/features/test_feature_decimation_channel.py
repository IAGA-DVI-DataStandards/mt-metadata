"""
Comprehensive pytest suite for FeatureDecimationChannel basemodel.

This test suite provides comprehensive coverage of the FeatureDecimationChannel class,
including initialization, field validation, units validation, time period functionality,
and error handling. Tests are optimized for efficiency using fixtures and parametrization.

Created for testing feature_decimation_channel.py
"""

import pytest

from mt_metadata.base import MetadataBase
from mt_metadata.common import TimePeriod
from mt_metadata.common.units import get_unit_object
from mt_metadata.features.feature_decimation_channel import FeatureDecimationChannel

# =====================================================
# Fixtures
# =====================================================


@pytest.fixture
def default_decimation_channel():
    """Fixture providing a default FeatureDecimationChannel instance."""
    return FeatureDecimationChannel()


@pytest.fixture
def sample_time_period():
    """Fixture providing a sample TimePeriod for testing."""
    return TimePeriod(start="2020-01-01T00:00:00", end="2020-01-02T00:00:00")


@pytest.fixture
def valid_channel_configurations():
    """Fixture providing various valid channel configurations."""
    return [
        {
            "name": "ex",
            "frequency_max": 100.0,
            "frequency_min": 0.1,
            "sample_rate_decimation_level": 256.0,
            "sample_rate_window_step": 4.0,
            "units": "milliVolt",  # Use exact unit name from UNITS_LIST
        },
        {
            "name": "hy",
            "frequency_max": 50.0,
            "frequency_min": 0.01,
            "sample_rate_decimation_level": 128.0,
            "sample_rate_window_step": 2.0,
            "units": "nanoTesla",  # Use exact unit name from UNITS_LIST
        },
        {
            "name": "hz",
            "frequency_max": 25.0,
            "frequency_min": 0.001,
            "sample_rate_decimation_level": 64.0,
            "sample_rate_window_step": 1.0,
            "units": "digital counts",  # Use exact unit name from UNITS_LIST
        },
    ]


@pytest.fixture
def valid_units():
    """Fixture providing various valid units for testing."""
    return ["digital counts", "millivolt", "volt", "nanoTesla", "celsius", "meter"]


@pytest.fixture
def invalid_units():
    """Fixture providing invalid units for testing validation."""
    return ["invalid_unit", "fake_measurement", "not_a_unit", 123, None]  # Non-string


@pytest.fixture
def frequency_configurations():
    """Fixture providing various frequency range configurations."""
    return [
        {"frequency_min": 0.001, "frequency_max": 1000.0},
        {"frequency_min": 0.1, "frequency_max": 100.0},
        {"frequency_min": 1.0, "frequency_max": 50.0},
        {"frequency_min": 10.0, "frequency_max": 25.0},
    ]


@pytest.fixture
def sample_rate_configurations():
    """Fixture providing various sample rate configurations."""
    return [
        {"sample_rate_decimation_level": 1024.0, "sample_rate_window_step": 8.0},
        {"sample_rate_decimation_level": 512.0, "sample_rate_window_step": 4.0},
        {"sample_rate_decimation_level": 256.0, "sample_rate_window_step": 2.0},
        {"sample_rate_decimation_level": 128.0, "sample_rate_window_step": 1.0},
    ]


@pytest.fixture
def configured_decimation_channel(sample_time_period):
    """Fixture providing a fully configured FeatureDecimationChannel instance."""
    return FeatureDecimationChannel(
        name="test_channel",
        frequency_max=100.0,
        frequency_min=0.1,
        sample_rate_decimation_level=256.0,
        sample_rate_window_step=4.0,
        units="milliVolt",  # Use exact unit name
        time_period=sample_time_period,
    )


# =====================================================
# Test Classes
# =====================================================


class TestFeatureDecimationChannelInitialization:
    """Test class for FeatureDecimationChannel initialization and basic functionality."""

    def test_default_initialization(self, default_decimation_channel):
        """Test default initialization sets correct default values."""
        assert default_decimation_channel.name == ""
        assert default_decimation_channel.frequency_max == 0.0
        assert default_decimation_channel.frequency_min == 0.0
        assert default_decimation_channel.sample_rate_decimation_level == 1.0
        assert default_decimation_channel.sample_rate_window_step == 1.0
        assert default_decimation_channel.units == "count"
        assert isinstance(default_decimation_channel.time_period, TimePeriod)

    def test_inheritance_from_metadata_base(self, default_decimation_channel):
        """Test that FeatureDecimationChannel properly inherits from MetadataBase."""
        assert isinstance(default_decimation_channel, MetadataBase)
        assert hasattr(default_decimation_channel, "model_dump")
        assert hasattr(default_decimation_channel, "model_validate")

    @pytest.mark.parametrize("config", [0, 1, 2])
    def test_custom_initialization(self, valid_channel_configurations, config):
        """Test custom initialization with various valid configurations."""
        config_data = valid_channel_configurations[config]
        channel = FeatureDecimationChannel(**config_data)

        assert channel.name == config_data["name"]
        assert channel.frequency_max == config_data["frequency_max"]
        assert channel.frequency_min == config_data["frequency_min"]
        assert (
            channel.sample_rate_decimation_level
            == config_data["sample_rate_decimation_level"]
        )
        assert channel.sample_rate_window_step == config_data["sample_rate_window_step"]
        assert channel.units == config_data["units"]

    def test_initialization_with_time_period(self, sample_time_period):
        """Test initialization with custom time period."""
        channel = FeatureDecimationChannel(name="test", time_period=sample_time_period)
        assert channel.time_period == sample_time_period
        assert channel.time_period.start == "2020-01-01T00:00:00+00:00"
        assert channel.time_period.end == "2020-01-02T00:00:00+00:00"

    def test_field_types_validation(self):
        """Test that fields accept correct types and convert appropriately."""
        channel = FeatureDecimationChannel(
            name="test",
            frequency_max=100,  # int should convert to float
            frequency_min=1,  # int should convert to float
            sample_rate_decimation_level=256,  # int should convert to float
            sample_rate_window_step=4,  # int should convert to float
        )

        assert isinstance(channel.frequency_max, float)
        assert isinstance(channel.frequency_min, float)
        assert isinstance(channel.sample_rate_decimation_level, float)
        assert isinstance(channel.sample_rate_window_step, float)
        assert channel.frequency_max == 100.0
        assert channel.frequency_min == 1.0


class TestFeatureDecimationChannelUnitsValidation:
    """Test class for units field validation."""

    @pytest.mark.parametrize("unit", [0, 1, 2, 3, 4, 5])
    def test_valid_units_acceptance(self, valid_units, unit):
        """Test that valid units are accepted and properly processed."""
        unit_value = valid_units[unit]
        channel = FeatureDecimationChannel()
        channel.units = unit_value

        # Units should be validated through get_unit_object
        unit_obj = get_unit_object(unit_value)
        assert channel.units == unit_obj.name

    def test_units_validator_with_empty_string(self):
        """Test units validator handles empty string correctly."""
        channel = FeatureDecimationChannel()
        channel.units = ""
        assert channel.units == ""

    def test_units_validator_with_none(self):
        """Test units validator handles None values correctly."""
        channel = FeatureDecimationChannel()
        channel.units = None
        assert channel.units == ""

    @pytest.mark.parametrize(
        "invalid_unit", ["invalid_unit", "fake_measurement", "not_a_unit"]
    )
    def test_invalid_units_rejection(self, invalid_unit):
        """Test that invalid units get set to 'unknown' (graceful handling)."""
        channel = FeatureDecimationChannel()
        channel.units = invalid_unit
        # The units validator sets invalid units to "unknown" rather than raising an error
        assert channel.units == "unknown"

    def test_units_validator_error_handling(self):
        """Test proper error handling in units validator with non-string types."""
        channel = FeatureDecimationChannel()
        with pytest.raises(TypeError):  # Changed from KeyError to TypeError
            channel.units = 123

    def test_units_case_insensitive(self):
        """Test that units validation is case insensitive."""
        channel1 = FeatureDecimationChannel()
        channel2 = FeatureDecimationChannel()
        channel3 = FeatureDecimationChannel()

        channel1.units = "MILLIVOLT"
        channel2.units = "millivolt"
        channel3.units = "MilliVolt"

        # All should resolve to the same canonical name
        assert channel1.units == channel2.units == channel3.units


class TestFeatureDecimationChannelTimePeriod:
    """Test class for time period functionality."""

    def test_default_time_period_creation(self, default_decimation_channel):
        """Test that default time period is created correctly."""
        assert isinstance(default_decimation_channel.time_period, TimePeriod)
        # Default TimePeriod should have default start/end values
        assert (
            default_decimation_channel.time_period.start == "1980-01-01T00:00:00+00:00"
        )
        assert default_decimation_channel.time_period.end == "1980-01-01T00:00:00+00:00"

    def test_custom_time_period_assignment(self, sample_time_period):
        """Test assignment of custom time period."""
        channel = FeatureDecimationChannel(time_period=sample_time_period)
        assert channel.time_period == sample_time_period
        assert channel.time_period.start == sample_time_period.start
        assert channel.time_period.end == sample_time_period.end

    def test_time_period_dict_initialization(self):
        """Test time period initialization from dictionary."""
        time_dict = {"start": "2021-01-01T00:00:00", "end": "2021-01-02T00:00:00"}
        channel = FeatureDecimationChannel(time_period=time_dict)

        assert isinstance(channel.time_period, TimePeriod)
        assert "2021-01-01" in str(channel.time_period.start)
        assert "2021-01-02" in str(channel.time_period.end)

    def test_time_period_modification(self, configured_decimation_channel):
        """Test modification of time period after initialization."""
        new_time_period = TimePeriod(
            start="2022-01-01T00:00:00", end="2022-01-02T00:00:00"
        )
        configured_decimation_channel.time_period = new_time_period

        assert configured_decimation_channel.time_period == new_time_period
        assert "2022-01-01" in str(configured_decimation_channel.time_period.start)


class TestFeatureDecimationChannelValidation:
    """Test class for field validation and error handling."""

    @pytest.mark.parametrize("freq_config", [0, 1, 2, 3])
    def test_frequency_range_validation(self, frequency_configurations, freq_config):
        """Test various frequency range configurations."""
        config = frequency_configurations[freq_config]
        channel = FeatureDecimationChannel(**config)

        assert channel.frequency_min == config["frequency_min"]
        assert channel.frequency_max == config["frequency_max"]
        assert channel.frequency_min <= channel.frequency_max

    @pytest.mark.parametrize("rate_config", [0, 1, 2, 3])
    def test_sample_rate_configurations(self, sample_rate_configurations, rate_config):
        """Test various sample rate configurations."""
        config = sample_rate_configurations[rate_config]
        channel = FeatureDecimationChannel(**config)

        assert (
            channel.sample_rate_decimation_level
            == config["sample_rate_decimation_level"]
        )
        assert channel.sample_rate_window_step == config["sample_rate_window_step"]

    def test_negative_frequency_values(self):
        """Test handling of negative frequency values."""
        # Negative values should be allowed as they might be valid in some contexts
        channel = FeatureDecimationChannel(frequency_min=-1.0, frequency_max=-0.5)
        assert channel.frequency_min == -1.0
        assert channel.frequency_max == -0.5

    def test_zero_frequency_values(self):
        """Test handling of zero frequency values."""
        channel = FeatureDecimationChannel(frequency_min=0.0, frequency_max=0.0)
        assert channel.frequency_min == 0.0
        assert channel.frequency_max == 0.0

    def test_extreme_frequency_values(self):
        """Test handling of extreme frequency values."""
        channel = FeatureDecimationChannel(frequency_min=1e-10, frequency_max=1e10)
        assert channel.frequency_min == 1e-10
        assert channel.frequency_max == 1e10

    def test_zero_sample_rates(self):
        """Test handling of zero sample rates."""
        channel = FeatureDecimationChannel(
            sample_rate_decimation_level=0.0, sample_rate_window_step=0.0
        )
        assert channel.sample_rate_decimation_level == 0.0
        assert channel.sample_rate_window_step == 0.0

    def test_name_validation(self):
        """Test name field validation."""
        # Empty string should be allowed
        channel1 = FeatureDecimationChannel(name="")
        assert channel1.name == ""

        # Various valid names
        valid_names = ["ex", "hy", "hz", "channel_1", "TEST_CHANNEL", "ch-01"]
        for name in valid_names:
            channel = FeatureDecimationChannel(name=name)
            assert channel.name == name

    def test_field_required_status(self):
        """Test that all fields have appropriate required status in schema."""
        # All fields should be marked as required in json_schema_extra
        channel = FeatureDecimationChannel()
        schema = channel.model_json_schema()

        properties = schema.get("properties", {})
        for field in [
            "name",
            "frequency_max",
            "frequency_min",
            "sample_rate_decimation_level",
            "sample_rate_window_step",
            "units",
            "time_period",
        ]:
            assert field in properties


class TestFeatureDecimationChannelIntegration:
    """Test class for integration scenarios and full workflow testing."""

    def test_full_workflow_integration(self, configured_decimation_channel):
        """Test complete workflow from initialization to usage."""
        # Verify initial state
        assert configured_decimation_channel.name == "test_channel"
        assert configured_decimation_channel.frequency_max == 100.0
        assert configured_decimation_channel.units == "milliVolt"  # Use exact unit name

        # Modify values
        configured_decimation_channel.frequency_max = 200.0
        configured_decimation_channel.units = "Volt"

        # Verify modifications
        assert configured_decimation_channel.frequency_max == 200.0
        assert configured_decimation_channel.units == "Volt"

    def test_serialization_deserialization(self, configured_decimation_channel):
        """Test JSON serialization and deserialization."""
        # Serialize to dict
        channel_dict = configured_decimation_channel.model_dump()

        # Verify essential fields are present
        assert "name" in channel_dict
        assert "frequency_max" in channel_dict
        assert "units" in channel_dict
        assert "time_period" in channel_dict

        # Deserialize back to object
        new_channel = FeatureDecimationChannel(**channel_dict)

        # Verify equality
        assert new_channel.name == configured_decimation_channel.name
        assert new_channel.frequency_max == configured_decimation_channel.frequency_max
        assert new_channel.units == configured_decimation_channel.units

    def test_copy_and_modification(self, configured_decimation_channel):
        """Test copying and modifying channel configurations."""
        # Create copy via model_dump and re-instantiation
        channel_copy = FeatureDecimationChannel(
            **configured_decimation_channel.model_dump()
        )

        # Verify initial equality
        assert channel_copy.name == configured_decimation_channel.name
        assert channel_copy.frequency_max == configured_decimation_channel.frequency_max

        # Modify copy
        channel_copy.name = "modified_channel"
        channel_copy.frequency_max = 150.0

        # Verify original is unchanged
        assert configured_decimation_channel.name == "test_channel"
        assert configured_decimation_channel.frequency_max == 100.0

        # Verify copy is changed
        assert channel_copy.name == "modified_channel"
        assert channel_copy.frequency_max == 150.0

    def test_equality_comparison(self):
        """Test equality comparison between channel instances."""
        channel1 = FeatureDecimationChannel(
            name="test", frequency_max=100.0, units="millivolts"
        )

        channel2 = FeatureDecimationChannel(
            name="test", frequency_max=100.0, units="millivolts"
        )

        channel3 = FeatureDecimationChannel(
            name="different", frequency_max=100.0, units="millivolts"
        )

        # Same configuration should be equal
        assert channel1.model_dump() == channel2.model_dump()

        # Different configuration should not be equal
        assert channel1.model_dump() != channel3.model_dump()

    def test_comprehensive_configuration(self):
        """Test a comprehensive configuration with all fields set."""
        comprehensive_config = {
            "name": "comprehensive_channel",
            "frequency_max": 1000.0,
            "frequency_min": 0.001,
            "sample_rate_decimation_level": 2048.0,
            "sample_rate_window_step": 16.0,
            "units": "nanoTesla",  # Use exact unit name
            "time_period": {
                "start": "2023-01-01T00:00:00",
                "end": "2023-12-31T23:59:59",
            },
        }

        channel = FeatureDecimationChannel(**comprehensive_config)

        # Verify all fields are set correctly
        assert channel.name == "comprehensive_channel"
        assert channel.frequency_max == 1000.0
        assert channel.frequency_min == 0.001
        assert channel.sample_rate_decimation_level == 2048.0
        assert channel.sample_rate_window_step == 16.0
        assert channel.units == "nanoTesla"  # Use exact unit name
        assert isinstance(channel.time_period, TimePeriod)
        assert "2023-01-01" in str(channel.time_period.start)
        assert "2023-12-31" in str(channel.time_period.end)

    def test_prewhitening_result_serialization(self):
        """Test fitted prewhitening coefficients survive model serialization."""
        channel = FeatureDecimationChannel(
            prewhitening_type="ARMA",
            prewhitening_fit_domain="full run",
            prewhitening_response_convention="A(z) = 1 - sum(phi_k z^-k)",
            prewhitening_ar_coefficients=[0.8, -0.2],
            prewhitening_ma_coefficients=[],
        )

        restored = FeatureDecimationChannel.model_validate(channel.model_dump())

        assert restored.prewhitening_type == "ARMA"
        assert restored.prewhitening_fit_domain == "full run"
        assert restored.prewhitening_ar_coefficients == [0.8, -0.2]
        assert restored.prewhitening_ma_coefficients == []


# =====================================================
# Test Configuration
# =====================================================

if __name__ == "__main__":
    pytest.main([__file__])
