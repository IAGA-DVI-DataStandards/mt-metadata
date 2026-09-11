"""
Tests for mt_metadata.transfer_functions.io.tools.get_nm_elev
============================================================

This module contains comprehensive pytest tests for the National Map elevation
retrieval functionality. Tests cover successful API calls, error handling,
edge cases, and proper response parsing.

Note: Due to import chain issues in the current module structure, we test
the function by loading it directly from the tools.py file.

"""

import json
from unittest.mock import Mock, patch
from urllib.error import URLError

import numpy as np
import pytest

from mt_metadata.transfer_functions.io.tools import (
    _request_nm_elev,
    get_nm_elev,
)


@pytest.fixture(autouse=True)
def clear_elevation_cache():
    """Each test gets an empty request cache."""
    _request_nm_elev.cache_clear()


# ==============================================================================
# Session-scoped fixtures for test data
# ==============================================================================
@pytest.fixture(scope="session")
def valid_coordinates():
    """Valid coordinates within the US that should return elevation data."""
    return [
        (40.0, -120.0),  # California
        (35.467, -115.3355),  # Nevada (from docstring example)
        (39.7392, -104.9903),  # Colorado (Denver)
        (47.6062, -122.3321),  # Washington (Seattle)
    ]


@pytest.fixture(scope="session")
def invalid_coordinates():
    """Invalid coordinates that should return 0 or error."""
    return [
        (0.0, 0.0),  # Ocean coordinates
        (90.0, 0.0),  # North Pole
        (-90.0, 0.0),  # South Pole
        (50.0, 0.0),  # UK coordinates (outside US)
    ]


@pytest.fixture(scope="session")
def known_elevations():
    """Known elevation values for specific coordinates (with tolerance for API variations)."""
    return {
        (40.0, -120.0): {
            "expected": [
                1899.16394043,
                1895.87854004,
                1895.33996582,
            ],  # All known values from issue #262
            "tolerance": 1e-5,
        },
        (35.467, -115.3355): {
            "expected": [809.12],  # From docstring example
            "tolerance": 10.0,  # Allow larger tolerance for this example
        },
    }


@pytest.fixture
def mock_successful_response():
    """Mock a successful API response."""
    mock_response = Mock()
    mock_response.read.return_value = json.dumps({"value": "1899.16394043"}).encode()
    return mock_response


@pytest.fixture
def mock_zero_response():
    """Mock an API response with zero elevation (ocean/invalid location)."""
    mock_response = Mock()
    mock_response.read.return_value = json.dumps({"value": "0"}).encode()
    return mock_response


@pytest.fixture
def mock_no_value_response():
    """Mock an API response without elevation value."""
    mock_response = Mock()
    mock_response.read.return_value = json.dumps(
        {"error": "No data available"}
    ).encode()
    return mock_response


@pytest.fixture
def mock_invalid_json_response():
    """Mock an invalid JSON response."""
    mock_response = Mock()
    mock_response.read.return_value = b"<html>Error page</html>"
    return mock_response


@pytest.fixture
def mock_non_numeric_response():
    """Mock a response with non-numeric elevation value."""
    mock_response = Mock()
    mock_response.read.return_value = json.dumps({"value": "invalid"}).encode()
    return mock_response


# ==============================================================================
# Test National Map Elevation Retrieval
# ==============================================================================
class TestGetNMElevation:
    """Test National Map elevation retrieval functionality."""

    def test_function_signature(self):
        """Test that function accepts latitude and longitude parameters."""
        # This should not raise an exception with valid numeric inputs
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = Mock()
            mock_response.read.return_value = json.dumps({"value": "100.0"}).encode()
            mock_urlopen.return_value = mock_response

            try:
                result = get_nm_elev(40.0, -120.0)
                assert isinstance(result, (int, float))
            except TypeError:
                pytest.fail(
                    "get_nm_elev should accept latitude and longitude parameters"
                )

    @pytest.mark.parametrize(
        "lat,lon,queried",
        [
            (40.0, -120.0, True),
            (35.467, -115.3355, True),
            (0.0, 0.0, False),
            (-90.0, 180.0, False),
            (90.0, -180.0, False),
        ],
    )
    def test_coordinate_range_acceptance(self, lat, lon, queried):
        """Coordinates outside the US National Map return 0 without a request."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.read.return_value = json.dumps(
                {"value": "100.0"}
            ).encode()
            result = get_nm_elev(lat, lon)
            assert isinstance(result, (int, float))
            assert mock_urlopen.called is queried
            if not queried:
                assert result == 0.0

    def test_successful_elevation_retrieval(self, mock_successful_response):
        """Test successful elevation retrieval from API."""
        with patch("urllib.request.urlopen", return_value=mock_successful_response):
            elevation = get_nm_elev(40.0, -120.0)
            assert elevation == 1899.16394043
            assert isinstance(elevation, float)

    def test_zero_elevation_retrieval(self, mock_zero_response):
        """Test retrieval of zero elevation (ocean/invalid coordinates)."""
        with patch("urllib.request.urlopen", return_value=mock_zero_response):
            elevation = get_nm_elev(36.0, -122.5)
            assert elevation == 0.0
            assert isinstance(elevation, float)

    def test_network_connection_error(self):
        """Test handling of network connection errors."""
        with patch("urllib.request.urlopen", side_effect=URLError("Connection failed")):
            elevation = get_nm_elev(40.0, -120.0)
            assert elevation == 0.0

    def test_http_error_handling(self):
        """Test handling of HTTP errors."""
        with patch("urllib.request.urlopen", side_effect=Exception("HTTP Error")):
            elevation = get_nm_elev(40.0, -120.0)
            assert elevation == 0.0

    def test_invalid_json_response(self, mock_invalid_json_response):
        """Test handling of invalid JSON responses."""
        with patch("urllib.request.urlopen", return_value=mock_invalid_json_response):
            elevation = get_nm_elev(40.0, -120.0)
            assert elevation == 0.0

    def test_missing_value_key(self, mock_no_value_response):
        """Test handling of API response without elevation value."""
        with patch("urllib.request.urlopen", return_value=mock_no_value_response):
            elevation = get_nm_elev(40.0, -120.0)
            assert elevation == 0.0

    def test_non_numeric_elevation_value(self, mock_non_numeric_response):
        """Test handling of non-numeric elevation values."""
        with patch("urllib.request.urlopen", return_value=mock_non_numeric_response):
            elevation = get_nm_elev(40.0, -120.0)
            assert elevation == 0.0

    @pytest.mark.parametrize(
        "elevation_str,expected",
        [
            ("1899.16394043", 1899.16394043),
            ("0", 0.0),
            ("123.456", 123.456),
            ("-50.0", -50.0),  # Below sea level
            ("8848.86", 8848.86),  # Mount Everest height
        ],
    )
    def test_elevation_value_parsing(self, elevation_str, expected):
        """Test parsing of various elevation value formats."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps({"value": elevation_str}).encode()

        with patch("urllib.request.urlopen", return_value=mock_response):
            elevation = get_nm_elev(40.0, -120.0)
            assert elevation == expected

    def test_api_url_construction(self):
        """Test that the correct API URL is constructed."""
        expected_base = "https://epqs.nationalmap.gov/v1/json?"

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.read.return_value = json.dumps(
                {"value": "100.0"}
            ).encode()

            get_nm_elev(40.123, -120.456)

            # Check that urlopen was called with correct URL pattern
            assert mock_urlopen.called
            called_url = mock_urlopen.call_args[0][0]
            assert expected_base in called_url
            assert "x=-120.456" in called_url
            assert "y=40.123" in called_url
            assert "units=Meters" in called_url
            assert "wkid=4326" in called_url
            assert "includeDate=False" in called_url


# ==============================================================================
# Integration Tests (can be slow, marked as such)
# ==============================================================================
class TestGetNMElevationIntegration:
    """Integration tests that actually call the API (marked as slow)."""

    @pytest.mark.slow
    @pytest.mark.internet
    def test_known_good_coordinate(self, known_elevations):
        """Test with known coordinates that should return valid elevation.

        Note: This test requires internet connection and is marked as slow.
        The elevation values may vary slightly due to different resolution data.
        """
        lat, lon = 40.0, -120.0
        elevation = get_nm_elev(lat, lon)

        # Handle the known issue from GitHub issue #262
        expected_values = known_elevations[(lat, lon)]["expected"]
        tolerance = known_elevations[(lat, lon)]["tolerance"]

        if elevation == 0.0:
            # If we get 0, it might be a connection issue - just verify it's a number
            assert isinstance(elevation, (int, float))
            pytest.skip(
                "API returned 0 - possible connection issue or coordinates outside US"
            )
        else:
            # Check if elevation matches any of the known possible values
            matches = [
                np.isclose(elevation, expected, rtol=tolerance)
                for expected in expected_values
            ]
            assert any(
                matches
            ), f"Elevation {elevation} doesn't match any expected values {expected_values}"

    @pytest.mark.slow
    @pytest.mark.internet
    @pytest.mark.parametrize(
        "lat,lon",
        [
            (0.0, 0.0),  # Ocean coordinates
            (51.5074, -0.1278),  # London (outside US)
        ],
    )
    def test_known_bad_coordinates(self, lat, lon):
        """Test with coordinates that should return 0 or error.

        Note: This test requires internet connection.
        """
        elevation = get_nm_elev(lat, lon)
        assert elevation == 0.0

    @pytest.mark.slow
    @pytest.mark.internet
    def test_multiple_requests_consistency(self):
        """Test that multiple requests to the same coordinate return consistent results.

        Note: This addresses the bug mentioned in issue #262 where the same
        coordinates can return different elevation values.
        """
        lat, lon = 40.0, -120.0
        elevations = []

        # Make multiple requests
        for _ in range(3):
            elev = get_nm_elev(lat, lon)
            if elev != 0.0:  # Skip if we get connection errors
                elevations.append(elev)

        if len(elevations) >= 2:
            # Check if we get consistent results or if we hit the known bug
            unique_elevations = list(set(elevations))
            if len(unique_elevations) > 1:
                # We hit the known bug - verify the values are among the known possibilities
                known_values = [1899.16394043, 1895.87854004, 1895.33996582]
                for elev in unique_elevations:
                    matches = [
                        np.isclose(elev, known, rtol=1e-5) for known in known_values
                    ]
                    assert any(matches), f"Unexpected elevation value: {elev}"
            else:
                # Consistent results - good!
                assert len(unique_elevations) == 1


# ==============================================================================
# Error Handling and Edge Cases
# ==============================================================================
class TestGetNMElevationEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.parametrize(
        "lat,lon",
        [
            (90.1, 0.0),  # Latitude out of range
            (-90.1, 0.0),  # Latitude out of range
            (0.0, 180.1),  # Longitude out of range
            (0.0, -180.1),  # Longitude out of range
        ],
    )
    def test_out_of_range_coordinates(self, lat, lon):
        """Test handling of coordinates outside valid ranges."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.read.return_value = json.dumps(
                {"value": "0"}
            ).encode()
            elevation = get_nm_elev(lat, lon)
            assert isinstance(elevation, (int, float))

    def test_extreme_precision_coordinates(self):
        """Test with very high precision coordinates."""
        lat, lon = 40.123456789012345, -120.987654321098765

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.read.return_value = json.dumps(
                {"value": "1500.0"}
            ).encode()
            elevation = get_nm_elev(lat, lon)
            assert elevation == 1500.0
            assert mock_urlopen.called

    def test_boundary_coordinates(self):
        """Coordinates at the valid boundaries sit outside the footprint."""
        boundary_coords = [
            (90.0, 0.0),  # North Pole
            (-90.0, 0.0),  # South Pole
            (0.0, 180.0),  # International Date Line
            (0.0, -180.0),  # International Date Line (other side)
        ]

        for lat, lon in boundary_coords:
            with patch("urllib.request.urlopen") as mock_urlopen:
                elevation = get_nm_elev(lat, lon)
                assert elevation == 0.0
                assert not mock_urlopen.called

    def test_api_timeout_handling(self):
        """Test handling of API timeouts."""
        import socket

        with patch(
            "urllib.request.urlopen", side_effect=socket.timeout("Request timed out")
        ):
            elevation = get_nm_elev(40.0, -120.0)
            assert elevation == 0.0

    def test_json_with_extra_fields(self):
        """Test API response with extra fields besides 'value'."""
        mock_response = Mock()
        response_data = {
            "value": "1500.25",
            "units": "Meters",
            "datasource": "NED 1/3 arc-second",
            "resolution": "1/3 arc-second",
            "timestamp": "2023-09-05T12:00:00Z",
        }
        mock_response.read.return_value = json.dumps(response_data).encode()

        with patch("urllib.request.urlopen", return_value=mock_response):
            elevation = get_nm_elev(40.0, -120.0)
            assert elevation == 1500.25

    def test_empty_response(self):
        """Test handling of empty API response."""
        mock_response = Mock()
        mock_response.read.return_value = b""

        with patch("urllib.request.urlopen", return_value=mock_response):
            elevation = get_nm_elev(40.0, -120.0)
            assert elevation == 0.0


# ==============================================================================
# Performance and Reliability Tests
# ==============================================================================
class TestGetNMElevationPerformance:
    """Test performance and reliability aspects."""

    def test_multiple_coordinate_batch(self, valid_coordinates):
        """Test processing multiple coordinates efficiently."""
        elevations = []

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.read.return_value = json.dumps(
                {"value": "1500.0"}
            ).encode()

            for lat, lon in valid_coordinates:
                elevation = get_nm_elev(lat, lon)
                elevations.append(elevation)

            assert len(elevations) == len(valid_coordinates)
            assert all(isinstance(elev, (int, float)) for elev in elevations)
            assert mock_urlopen.call_count == len(valid_coordinates)

    def test_response_consistency(self):
        """Test that the same coordinates return the same result (when mocked)."""
        lat, lon = 40.0, -120.0

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.read.return_value = json.dumps(
                {"value": "1500.0"}
            ).encode()

            elevation1 = get_nm_elev(lat, lon)
            elevation2 = get_nm_elev(lat, lon)

            assert elevation1 == elevation2
            assert elevation1 == 1500.0

    @pytest.mark.parametrize("iterations", [1, 5, 10])
    def test_repeated_calls_reliability(self, iterations):
        """Test reliability of repeated API calls."""
        lat, lon = 35.467, -115.3355

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.read.return_value = json.dumps(
                {"value": "809.12"}
            ).encode()

            elevations = []
            for _ in range(iterations):
                elevation = get_nm_elev(lat, lon)
                elevations.append(elevation)

            assert len(elevations) == iterations
            assert all(elev == 809.12 for elev in elevations)


if __name__ == "__main__":
    pytest.main([__file__])
