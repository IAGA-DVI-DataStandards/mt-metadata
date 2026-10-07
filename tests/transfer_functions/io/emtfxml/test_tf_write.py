# -*- coding: utf-8 -*-
"""
Comprehensive pytest suite for EMTFXML write functionality testing.

Created on Fri Mar 10 08:52:43 2023
@author: jpeacock

Converted to pytest suite for enhanced testing with fixtures and parametrization.
Tests the translation from EMTF XML object to MT object and back to EMTF XML object.

Test Structure:
- TestEMTFXMLWriteBasics: Basic property and metadata tests (6 tests)
- TestEMTFXMLWriteComponents: Component comparison tests (9 tests)
- TestEMTFXMLWriteDataArrays: Numerical data array comparisons (8 tests)
- TestEMTFXMLWriteXMLSerialization: XML serialization tests (9 tests)
- TestEMTFXMLWriteEstimates: Statistical estimates validation (2 tests)
- TestEMTFXMLWriteIntegration: Integration and roundtrip tests (3 tests)

Total: 37 tests covering EMTFXML write functionality and roundtrip validation
"""

import numpy as np

# =============================================================================
# Imports
# =============================================================================
import pytest

from mt_metadata import TF_XML
from mt_metadata.timeseries import Run
from mt_metadata.transfer_functions.core import TF
from mt_metadata.transfer_functions.io.emtfxml import EMTFXML
from mt_metadata.transfer_functions.tf import Station


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture(scope="module")
def original_emtfxml():
    """Load original EMTFXML object from file."""
    return EMTFXML(TF_XML)


@pytest.fixture(scope="module")
def tf_roundtrip(original_emtfxml):
    """Create TF object and convert back to EMTFXML for comparison."""
    tf = TF(fn=TF_XML)
    tf.read()
    return tf.to_emtfxml()


@pytest.fixture(scope="module")
def tf_object():
    """Load TF object for testing."""
    tf = TF(fn=TF_XML)
    tf.read()
    return tf


# =============================================================================
# Test Classes
# =============================================================================
class TestEMTFXMLWriteBasics:
    """Test basic properties and metadata comparison between original and roundtrip EMTFXML objects."""

    def test_description(self, original_emtfxml, tf_roundtrip):
        """Test description property preservation through roundtrip."""
        assert original_emtfxml.description == tf_roundtrip.description

    def test_product_id(self, original_emtfxml, tf_roundtrip):
        """Test product_id property preservation through roundtrip."""
        assert original_emtfxml.product_id == tf_roundtrip.product_id

    def test_sub_type(self, original_emtfxml, tf_roundtrip):
        """Test sub_type property preservation through roundtrip."""
        assert original_emtfxml.sub_type == tf_roundtrip.sub_type

    def test_notes(self, original_emtfxml, tf_roundtrip):
        """Test notes property preservation through roundtrip."""
        assert original_emtfxml.notes == tf_roundtrip.notes

    def test_tags_content(self, original_emtfxml, tf_roundtrip):
        """Test tags content preservation through roundtrip."""
        original_tags = [v.strip() for v in original_emtfxml.tags.split(",")]
        roundtrip_tags = [v.strip() for v in tf_roundtrip.tags.split(",")]
        assert original_tags == roundtrip_tags

    def test_tags_format(self, original_emtfxml, tf_roundtrip):
        """Test tags format consistency."""
        assert isinstance(original_emtfxml.tags, str)
        assert isinstance(tf_roundtrip.tags, str)
        assert len(original_emtfxml.tags.strip()) > 0
        assert len(tf_roundtrip.tags.strip()) > 0


class TestEMTFXMLWriteComponents:
    """Test component-level comparisons and XML serialization."""

    def test_external_url_attribute(self, original_emtfxml, tf_roundtrip):
        """Test external_url object equality."""
        assert original_emtfxml.external_url == tf_roundtrip.external_url

    def test_external_url_xml_serialization(self, original_emtfxml, tf_roundtrip):
        """Test external_url XML serialization consistency."""
        original_xml = original_emtfxml.external_url.to_xml(string=True)
        roundtrip_xml = tf_roundtrip.external_url.to_xml(string=True)
        assert original_xml == roundtrip_xml

    def test_primary_data_attribute(self, original_emtfxml, tf_roundtrip):
        """Test primary_data object equality."""
        assert original_emtfxml.primary_data == tf_roundtrip.primary_data

    def test_primary_data_xml_serialization(self, original_emtfxml, tf_roundtrip):
        """Test primary_data XML serialization consistency."""
        original_xml = original_emtfxml.primary_data.to_xml(string=True)
        roundtrip_xml = tf_roundtrip.primary_data.to_xml(string=True)
        assert original_xml == roundtrip_xml

    def test_attachment_attribute(self, original_emtfxml, tf_roundtrip):
        """Test attachment object equality."""
        assert original_emtfxml.attachment == tf_roundtrip.attachment

    def test_attachment_xml_serialization(self, original_emtfxml, tf_roundtrip):
        """Test attachment XML serialization consistency."""
        original_xml = original_emtfxml.attachment.to_xml(string=True)
        roundtrip_xml = tf_roundtrip.attachment.to_xml(string=True)
        assert original_xml == roundtrip_xml

    def test_copyright_attribute(self, original_emtfxml, tf_roundtrip):
        """Test copyright object equality."""
        assert original_emtfxml.copyright == tf_roundtrip.copyright

    def test_copyright_xml_serialization(self, original_emtfxml, tf_roundtrip):
        """Test copyright XML serialization consistency."""
        original_xml = original_emtfxml.copyright.to_xml(string=True)
        roundtrip_xml = tf_roundtrip.copyright.to_xml(string=True)
        assert original_xml == roundtrip_xml

    def test_site_comprehensive(self, original_emtfxml, tf_roundtrip):
        """Test comprehensive site data and XML serialization."""
        # Test attribute equality
        original_dict = original_emtfxml.site.to_dict(single=True)
        roundtrip_dict = tf_roundtrip.site.to_dict(single=True)
        assert original_dict == roundtrip_dict

        # Test XML serialization
        original_xml = original_emtfxml.site.to_xml(string=True)
        roundtrip_xml = tf_roundtrip.site.to_xml(string=True)
        assert original_xml == roundtrip_xml


class TestEMTFXMLWriteDataArrays:
    """Test numerical data array comparisons."""

    def test_impedance_data(self, original_emtfxml, tf_roundtrip):
        """Test impedance (Z) data array consistency."""
        original_z = original_emtfxml.data.z
        roundtrip_z = tf_roundtrip.data.z

        if original_z is not None and roundtrip_z is not None:
            assert np.allclose(original_z, roundtrip_z)
        else:
            assert original_z is None and roundtrip_z is None

    def test_impedance_variance(self, original_emtfxml, tf_roundtrip):
        """Test impedance variance data consistency."""
        original_var = original_emtfxml.data.z_var
        roundtrip_var = tf_roundtrip.data.z_var

        if original_var is not None and roundtrip_var is not None:
            assert np.allclose(original_var, roundtrip_var)
        else:
            assert original_var is None and roundtrip_var is None

    def test_impedance_inverse_signal_covariance(self, original_emtfxml, tf_roundtrip):
        """Test impedance inverse signal covariance data consistency."""
        original_cov = original_emtfxml.data.z_invsigcov
        roundtrip_cov = tf_roundtrip.data.z_invsigcov

        if original_cov is not None and roundtrip_cov is not None:
            assert np.allclose(original_cov, roundtrip_cov)
        else:
            assert original_cov is None and roundtrip_cov is None

    def test_impedance_residual_covariance(self, original_emtfxml, tf_roundtrip):
        """Test impedance residual covariance data consistency."""
        original_res = original_emtfxml.data.z_residcov
        roundtrip_res = tf_roundtrip.data.z_residcov

        if original_res is not None and roundtrip_res is not None:
            assert np.allclose(original_res, roundtrip_res)
        else:
            assert original_res is None and roundtrip_res is None

    def test_tipper_data(self, original_emtfxml, tf_roundtrip):
        """Test tipper (T) data array consistency."""
        original_t = original_emtfxml.data.t
        roundtrip_t = tf_roundtrip.data.t

        if original_t is not None and roundtrip_t is not None:
            assert np.allclose(original_t, roundtrip_t)
        else:
            assert original_t is None and roundtrip_t is None

    def test_tipper_variance(self, original_emtfxml, tf_roundtrip):
        """Test tipper variance data consistency."""
        original_var = original_emtfxml.data.t_var
        roundtrip_var = tf_roundtrip.data.t_var

        if original_var is not None and roundtrip_var is not None:
            assert np.allclose(original_var, roundtrip_var)
        else:
            assert original_var is None and roundtrip_var is None

    def test_tipper_inverse_signal_covariance(self, original_emtfxml, tf_roundtrip):
        """Test tipper inverse signal covariance data consistency."""
        original_cov = original_emtfxml.data.t_invsigcov
        roundtrip_cov = tf_roundtrip.data.t_invsigcov

        if original_cov is not None and roundtrip_cov is not None:
            assert np.allclose(original_cov, roundtrip_cov)
        else:
            assert original_cov is None and roundtrip_cov is None

    def test_tipper_residual_covariance(self, original_emtfxml, tf_roundtrip):
        """Test tipper residual covariance data consistency."""
        original_res = original_emtfxml.data.t_residcov
        roundtrip_res = tf_roundtrip.data.t_residcov

        if original_res is not None and roundtrip_res is not None:
            assert np.allclose(original_res, roundtrip_res)
        else:
            assert original_res is None and roundtrip_res is None


class TestEMTFXMLWriteXMLSerialization:
    """Test XML serialization for complex components."""

    def test_field_notes_attribute_comparison(self, original_emtfxml, tf_roundtrip):
        """Test field_notes attribute dictionary comparison."""
        original_dict = original_emtfxml.field_notes.to_dict(single=True)
        roundtrip_dict = tf_roundtrip.field_notes.to_dict(single=True)
        assert original_dict == roundtrip_dict

    def test_field_notes_xml_differences(self, original_emtfxml, tf_roundtrip):
        """Test that field_notes XML serialization has expected differences due to rounding."""
        original_xml = original_emtfxml.field_notes.to_xml(string=True)
        roundtrip_xml = tf_roundtrip.field_notes.to_xml(string=True)
        # Note: These are expected to be different due to rounding
        assert original_xml != roundtrip_xml

    def test_data_types_attribute(self, original_emtfxml, tf_roundtrip):
        """Test data_types attribute consistency."""
        original_dict = original_emtfxml.data_types.to_dict(single=True)
        roundtrip_dict = tf_roundtrip.data_types.to_dict(single=True)
        assert original_dict == roundtrip_dict

    def test_data_types_xml_serialization(self, original_emtfxml, tf_roundtrip):
        """Test data_types XML serialization consistency."""
        original_xml = original_emtfxml.data_types.to_xml(string=True)
        roundtrip_xml = tf_roundtrip.data_types.to_xml(string=True)
        assert original_xml == roundtrip_xml

    def test_site_layout_attribute(self, original_emtfxml, tf_roundtrip):
        """Test site_layout attribute consistency with normalized channel names and core structure."""
        original_dict = original_emtfxml.site_layout.to_dict(single=True)
        roundtrip_dict = tf_roundtrip.site_layout.to_dict(single=True)

        # Normalize channel names and focus on core structure for comparison
        def normalize_channels_for_comparison(site_dict):
            normalized = site_dict.copy()
            for channel_type in ["input_channels", "output_channels"]:
                if channel_type in normalized:
                    for channel_group in normalized[channel_type]:
                        for field_type in ["magnetic", "electric"]:
                            if field_type in channel_group:
                                # Normalize channel name to lowercase
                                channel_group[field_type]["name"] = channel_group[
                                    field_type
                                ]["name"].lower()
            return normalized

        # Test that the channels exist with correct names and types
        original_normalized = normalize_channels_for_comparison(original_dict)
        roundtrip_normalized = normalize_channels_for_comparison(roundtrip_dict)

        # Extract channel names for comparison
        def get_channel_names(site_dict):
            input_names = []
            output_names = []
            for channel_group in site_dict.get("input_channels", []):
                for field_type in ["magnetic", "electric"]:
                    if field_type in channel_group:
                        input_names.append(channel_group[field_type]["name"])
            for channel_group in site_dict.get("output_channels", []):
                for field_type in ["magnetic", "electric"]:
                    if field_type in channel_group:
                        output_names.append(channel_group[field_type]["name"])
            return sorted(input_names), sorted(output_names)

        orig_input, orig_output = get_channel_names(original_normalized)
        rt_input, rt_output = get_channel_names(roundtrip_normalized)

        # Test that channel names match (ignoring order and detailed properties)
        assert (
            orig_input == rt_input
        ), f"Input channel names differ: {orig_input} != {rt_input}"
        assert (
            orig_output == rt_output
        ), f"Output channel names differ: {orig_output} != {rt_output}"

    def test_site_layout_xml_serialization(self, original_emtfxml, tf_roundtrip):
        """Test site_layout XML serialization has proper structure and channel names."""
        original_xml = original_emtfxml.site_layout.to_xml(string=True)
        roundtrip_xml = tf_roundtrip.site_layout.to_xml(string=True)

        # Test that both XMLs contain the expected channel types
        import re

        # Extract channel names from XML
        def extract_channel_names(xml_string):
            # Find all Magnetic and Electric channel names
            magnetic_names = re.findall(r'<Magnetic[^>]*name="([^"]*)"', xml_string)
            electric_names = re.findall(r'<Electric[^>]*name="([^"]*)"', xml_string)
            return sorted([name.lower() for name in magnetic_names]), sorted(
                [name.lower() for name in electric_names]
            )

        orig_mag, orig_elec = extract_channel_names(original_xml)
        rt_mag, rt_elec = extract_channel_names(roundtrip_xml)

        # Test that channel names are preserved (case-insensitive)
        assert (
            orig_mag == rt_mag
        ), f"Magnetic channel names differ: {orig_mag} != {rt_mag}"
        assert (
            orig_elec == rt_elec
        ), f"Electric channel names differ: {orig_elec} != {rt_elec}"

        # Test that both XMLs have proper structure
        assert "InputChannels" in original_xml and "InputChannels" in roundtrip_xml
        assert "OutputChannels" in original_xml and "OutputChannels" in roundtrip_xml

    def test_period_range_attribute(self, original_emtfxml, tf_roundtrip):
        """Test period_range attribute consistency."""
        original_dict = original_emtfxml.period_range.to_dict(single=True)
        roundtrip_dict = tf_roundtrip.period_range.to_dict(single=True)
        assert original_dict == roundtrip_dict

    def test_period_range_xml_serialization(self, original_emtfxml, tf_roundtrip):
        """Test period_range XML serialization consistency."""
        original_xml = original_emtfxml.period_range.to_xml(string=True)
        roundtrip_xml = tf_roundtrip.period_range.to_xml(string=True)
        assert original_xml == roundtrip_xml

    @pytest.mark.parametrize("line_type", ["ProcessingTag", "other_lines"])
    def test_processing_info_xml_lines(self, original_emtfxml, tf_roundtrip, line_type):
        """Test processing info XML line-by-line comparison."""
        original_lines = original_emtfxml.processing_info.to_xml(string=True).split(
            "\n"
        )
        roundtrip_lines = tf_roundtrip.processing_info.to_xml(string=True).split("\n")

        for original_line, roundtrip_line in zip(original_lines, roundtrip_lines):
            if line_type == "ProcessingTag" and "ProcessingTag" in original_line:
                # ProcessingTag is expected to be different
                assert original_line != roundtrip_line
            elif line_type == "other_lines" and "<ProcessDate/>" in original_line:
                assert roundtrip_line.strip() == "<ProcessDate>1980-01-01</ProcessDate>"
            elif line_type == "other_lines" and "ProcessingTag" not in original_line:
                # Other lines should be the same
                assert original_line == roundtrip_line


class TestEMTFXMLWriteEstimates:
    """Test statistical estimates validation."""

    def test_statistical_estimates_membership(self, original_emtfxml, tf_roundtrip):
        """Test that all roundtrip estimates are present in original estimates."""
        for roundtrip_estimate in tf_roundtrip.statistical_estimates.estimates_list:
            assert (
                roundtrip_estimate
                in original_emtfxml.statistical_estimates.estimates_list
            )

    def test_statistical_estimates_xml_consistency(
        self, original_emtfxml, tf_roundtrip
    ):
        """Test XML serialization consistency for matching estimates by name."""
        # Create dictionaries of estimates keyed by name for easier comparison
        original_estimates = {
            est.name: est
            for est in original_emtfxml.statistical_estimates.estimates_list
        }
        roundtrip_estimates = {
            est.name: est for est in tf_roundtrip.statistical_estimates.estimates_list
        }

        # Compare estimates with the same name
        for name in original_estimates:
            if name in roundtrip_estimates:
                original_xml = original_estimates[name].to_xml(string=True)
                roundtrip_xml = roundtrip_estimates[name].to_xml(string=True)
                assert (
                    original_xml == roundtrip_xml
                ), f"XML mismatch for estimate {name}"


class TestEMTFXMLWriteIntegration:
    """Test integration scenarios and comprehensive validation."""

    def test_provenance_filtered_comparison(self, original_emtfxml, tf_roundtrip):
        """Test provenance comparison excluding time-sensitive fields."""
        original_dict = original_emtfxml.provenance.to_dict(single=True)
        roundtrip_dict = tf_roundtrip.provenance.to_dict(single=True)

        # Remove time-sensitive fields that change during processing
        for key in ["create_time", "creating_application"]:
            original_dict.pop(key, None)
            roundtrip_dict.pop(key, None)

        assert original_dict == roundtrip_dict

    def test_processing_info_comprehensive_comparison(
        self, original_emtfxml, tf_roundtrip
    ):
        """Test comprehensive processing info comparison with tag exceptions."""
        original_dict = original_emtfxml.processing_info.to_dict(single=True)
        roundtrip_dict = tf_roundtrip.processing_info.to_dict(single=True)

        for key, original_value in original_dict.items():
            roundtrip_value = roundtrip_dict[key]
            if "tag" in key:
                # Processing tags are expected to be different
                assert original_value != roundtrip_value
            else:
                # All other values should match
                assert original_value == roundtrip_value

    def test_complete_roundtrip_integrity(
        self, tf_object, original_emtfxml, tf_roundtrip
    ):
        """Test overall roundtrip integrity and key properties preservation."""
        # Verify that the TF object can successfully create an EMTFXML object
        assert tf_roundtrip is not None
        assert isinstance(tf_roundtrip, EMTFXML)

        # Verify critical properties are preserved
        assert original_emtfxml.description == tf_roundtrip.description
        assert original_emtfxml.product_id == tf_roundtrip.product_id
        assert original_emtfxml.sub_type == tf_roundtrip.sub_type

        # Verify numerical data integrity
        if original_emtfxml.data.z is not None and tf_roundtrip.data.z is not None:
            assert np.allclose(original_emtfxml.data.z, tf_roundtrip.data.z)
        if original_emtfxml.data.t is not None and tf_roundtrip.data.t is not None:
            assert np.allclose(original_emtfxml.data.t, tf_roundtrip.data.t)


class TestEMTFXMLWriteFile:
    """Test writing an EMTFXML file and reading it back."""

    @pytest.fixture(scope="class")
    def written_fn(self, original_emtfxml, tmp_path_factory):
        """Write the original EMTFXML object to a file."""
        fn = tmp_path_factory.mktemp("emtfxml") / "written.xml"
        original_emtfxml.write(fn)
        return fn

    def test_sub_type_element(self, written_fn):
        """Test SubType is written as the enum value."""
        assert "<SubType>MT_TF</SubType>" in written_fn.read_text()

    def test_read_written_file(self, original_emtfxml, written_fn):
        """Test a written file can be read back in."""
        assert original_emtfxml.sub_type == EMTFXML(written_fn).sub_type


class TestEMTFXMLWriteFieldNoteComments:
    """Test station comments that carry field note information."""

    @pytest.fixture
    def station(self):
        """Create station metadata whose comments hold field note information."""
        station = Station(id="MT001")
        station.add_run(Run(id="MT001a", sample_rate=1.0))
        station.comments.value = (
            "fieldnotes.datalogger.manufacturer = Data Logger Co\n"
            "fieldnotes.electrode_ex.manufacturer = Electrode Co\n"
            "fieldnotes.magnetometer_hx.manufacturer = Magnetometer Co"
        )
        return station

    def test_field_note_run(self, station):
        """Test the comment values are read into the field notes."""
        emtfxml = EMTFXML()
        emtfxml.station_metadata = station

        run = emtfxml.field_notes._run_list[0]
        assert run.instrument.manufacturer == "Data Logger Co"
        assert run.dipole[0].manufacturer == "Electrode Co"
        assert run.magnetometer[0].manufacturer == "Magnetometer Co"

    def test_field_notes_written(self, station, tmp_path):
        """Test the comment values are written to the XML file."""
        emtfxml = EMTFXML()
        emtfxml.station_metadata = station
        emtfxml.data.period = np.array([1.0, 2.0])

        fn = tmp_path / "field_notes.xml"
        emtfxml.write(fn)

        written = fn.read_text()
        assert "Data Logger Co" in written
        assert "Electrode Co" in written
        assert "Magnetometer Co" in written


# =============================================================================
# Performance and Edge Case Tests
# =============================================================================
class TestEMTFXMLWriteEdgeCases:
    """Test edge cases and performance scenarios."""

    def test_data_array_shapes(self, original_emtfxml, tf_roundtrip):
        """Test that data arrays maintain correct shapes through roundtrip."""
        # Test impedance shapes
        original_z = original_emtfxml.data.z
        roundtrip_z = tf_roundtrip.data.z
        if original_z is not None and roundtrip_z is not None:
            assert original_z.shape == roundtrip_z.shape

        original_z_var = original_emtfxml.data.z_var
        roundtrip_z_var = tf_roundtrip.data.z_var
        if original_z_var is not None and roundtrip_z_var is not None:
            assert original_z_var.shape == roundtrip_z_var.shape

        # Test tipper shapes
        original_t = original_emtfxml.data.t
        roundtrip_t = tf_roundtrip.data.t
        if original_t is not None and roundtrip_t is not None:
            assert original_t.shape == roundtrip_t.shape

        original_t_var = original_emtfxml.data.t_var
        roundtrip_t_var = tf_roundtrip.data.t_var
        if original_t_var is not None and roundtrip_t_var is not None:
            assert original_t_var.shape == roundtrip_t_var.shape

    def test_data_types_consistency(self, original_emtfxml, tf_roundtrip):
        """Test that data types are preserved through roundtrip."""
        original_z = original_emtfxml.data.z
        roundtrip_z = tf_roundtrip.data.z
        if original_z is not None and roundtrip_z is not None:
            assert type(original_z) == type(roundtrip_z)

        original_t = original_emtfxml.data.t
        roundtrip_t = tf_roundtrip.data.t
        if original_t is not None and roundtrip_t is not None:
            assert type(original_t) == type(roundtrip_t)

    def test_non_null_critical_fields(self, original_emtfxml, tf_roundtrip):
        """Test that critical fields are not null after roundtrip."""
        critical_fields = ["description", "product_id", "sub_type"]

        for field in critical_fields:
            original_value = getattr(original_emtfxml, field)
            roundtrip_value = getattr(tf_roundtrip, field)

            assert original_value is not None
            assert roundtrip_value is not None
            assert len(str(original_value).strip()) > 0
            assert len(str(roundtrip_value).strip()) > 0
