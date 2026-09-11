from unittest.mock import patch

import pytest
from pydantic import ValidationError
from pyproj import CRS

from mt_metadata.common import BasicLocation, Location, StationLocation


@pytest.fixture(scope="module")
def default_basic_location():
    """
    Fixture to provide a default BasicLocation object.
    """
    return BasicLocation()


@pytest.fixture(scope="module")
def custom_basic_location():
    """
    Fixture to provide a BasicLocation object with custom values.
    """
    return BasicLocation(latitude=45.0, longitude=-120.0, elevation=500.0)


@pytest.fixture(scope="module")
def default_location():
    """
    Fixture to provide a default Location object.
    """
    return Location()


@pytest.fixture(scope="module")
def default_station_location():
    """
    Fixture to provide a default StationLocation object.
    """
    return StationLocation()


def test_basic_location_default_values(default_basic_location, subtests):
    """
    Test the default values of the BasicLocation model.
    """
    location = default_basic_location

    with subtests.test("latitude default is 0.0"):
        assert location.latitude == 0.0

    with subtests.test("longitude default is 0.0"):
        assert location.longitude == 0.0

    with subtests.test("elevation default is 0.0"):
        assert location.elevation == 0.0


def test_basic_location_custom_values(custom_basic_location, subtests):
    """
    Test the BasicLocation model with custom values.
    """
    location = custom_basic_location

    with subtests.test("latitude is set correctly"):
        assert location.latitude == 45.0

    with subtests.test("longitude is set correctly"):
        assert location.longitude == -120.0

    with subtests.test("elevation is set correctly"):
        assert location.elevation == 500.0


def test_basic_location_from_kwargs(subtests):
    """
    Test initializing BasicLocation from kwargs.
    """
    kwargs = {"latitude": 35.5, "longitude": -110.25, "elevation": 1200.0}

    location = BasicLocation(**kwargs)

    with subtests.test("latitude from kwargs is set correctly"):
        assert location.latitude == 35.5

    with subtests.test("longitude from kwargs is set correctly"):
        assert location.longitude == -110.25

    with subtests.test("elevation from kwargs is set correctly"):
        assert location.elevation == 1200.0


def test_location_default_values(default_location, subtests):
    """
    Test the default values of the Location model.
    """
    location = default_location

    with subtests.test("latitude default is 0.0"):
        assert location.latitude == 0.0

    with subtests.test("longitude default is 0.0"):
        assert location.longitude == 0.0

    with subtests.test("elevation default is 0.0"):
        assert location.elevation == 0.0

    with subtests.test("latitude_uncertainty default is 0.0"):
        assert location.latitude_uncertainty == 0.0

    with subtests.test("longitude_uncertainty default is 0.0"):
        assert location.longitude_uncertainty == 0.0

    with subtests.test("elevation_uncertainty default is 0.0"):
        assert location.elevation_uncertainty == 0.0

    with subtests.test("datum default is WGS 84"):
        assert location.datum == "WGS 84"

    with subtests.test("x default is 0.0"):
        assert location.x == 0.0

    with subtests.test("y default is 0.0"):
        assert location.y == 0.0

    with subtests.test("z default is 0.0"):
        assert location.z == 0.0

    with subtests.test("x_uncertainty default is 0.0"):
        assert location.x_uncertainty == 0.0

    with subtests.test("y_uncertainty default is 0.0"):
        assert location.y_uncertainty == 0.0

    with subtests.test("z_uncertainty default is 0.0"):
        assert location.z_uncertainty == 0.0


def test_location_custom_values(subtests):
    """
    Test the Location model with custom values.
    """
    location = Location(
        latitude=23.134,
        longitude=14.23,
        elevation=123.4,
        latitude_uncertainty=0.01,
        longitude_uncertainty=0.02,
        elevation_uncertainty=0.03,
        datum="NAD83",
        x=10.0,
        y=20.0,
        z=30.0,
        x_uncertainty=0.1,
        y_uncertainty=0.2,
        z_uncertainty=0.3,
    )

    with subtests.test("latitude is set correctly"):
        assert location.latitude == 23.134

    with subtests.test("longitude is set correctly"):
        assert location.longitude == 14.23

    with subtests.test("elevation is set correctly"):
        assert location.elevation == 123.4

    with subtests.test("latitude_uncertainty is set correctly"):
        assert location.latitude_uncertainty == 0.01

    with subtests.test("longitude_uncertainty is set correctly"):
        assert location.longitude_uncertainty == 0.02

    with subtests.test("elevation_uncertainty is set correctly"):
        assert location.elevation_uncertainty == 0.03

    with subtests.test("datum is set correctly and normalized"):
        assert location.datum == "NAD83"

    with subtests.test("x is set correctly"):
        assert location.x == 10.0

    with subtests.test("y is set correctly"):
        assert location.y == 20.0

    with subtests.test("z is set correctly"):
        assert location.z == 30.0

    with subtests.test("x_uncertainty is set correctly"):
        assert location.x_uncertainty == 0.1

    with subtests.test("y_uncertainty is set correctly"):
        assert location.y_uncertainty == 0.2

    with subtests.test("z_uncertainty is set correctly"):
        assert location.z_uncertainty == 0.3


def test_location_from_kwargs(subtests):
    """
    Test initializing Location from kwargs.
    """
    kwargs = {
        "latitude": 42.5,
        "longitude": -113.75,
        "elevation": 1500.0,
        "latitude_uncertainty": 0.05,
        "longitude_uncertainty": 0.05,
        "elevation_uncertainty": 1.0,
        "datum": "NAD83",
        "x": 15.0,
        "y": 25.0,
        "z": 35.0,
    }

    location = Location(**kwargs)

    with subtests.test("latitude from kwargs is set correctly"):
        assert location.latitude == 42.5

    with subtests.test("longitude from kwargs is set correctly"):
        assert location.longitude == -113.75

    with subtests.test("elevation from kwargs is set correctly"):
        assert location.elevation == 1500.0

    with subtests.test("latitude_uncertainty from kwargs is set correctly"):
        assert location.latitude_uncertainty == 0.05

    with subtests.test("longitude_uncertainty from kwargs is set correctly"):
        assert location.longitude_uncertainty == 0.05

    with subtests.test("elevation_uncertainty from kwargs is set correctly"):
        assert location.elevation_uncertainty == 1.0

    with subtests.test("datum from kwargs is set correctly and normalized"):
        assert location.datum == "NAD83"

    with subtests.test("x from kwargs is set correctly"):
        assert location.x == 15.0

    with subtests.test("y from kwargs is set correctly"):
        assert location.y == 25.0

    with subtests.test("z from kwargs is set correctly"):
        assert location.z == 35.0


def test_location_invalid_latitude(subtests):
    """
    Test the Location model with an invalid latitude value.
    """
    with subtests.test("latitude over 90 raises ValidationError"):
        with pytest.raises(ValidationError):
            Location(latitude=100.0)  # Latitude must be between -90 and 90

    with subtests.test("latitude under -90 raises ValidationError"):
        with pytest.raises(ValidationError):
            Location(latitude=-100.0)


def test_location_invalid_longitude(subtests):
    """
    Test the Location model with an invalid longitude value.
    """
    with subtests.test("longitude over 180 raises ValidationError"):
        with pytest.raises(ValidationError):
            Location(longitude=200.0)  # Longitude must be between -180 and 180

    with subtests.test("longitude under -180 raises ValidationError"):
        with pytest.raises(ValidationError):
            Location(longitude=-200.0)


def test_location_invalid_datum(subtests):
    """
    Test the Location model with an invalid datum value.
    """
    with subtests.test("invalid datum raises ValidationError"):
        with pytest.raises(ValidationError):
            Location(datum="INVALID_DATUM")  # Datum must be a valid DatumEnum value


def test_location_datum_crs_reused(subtests):
    """
    Test the Location model does not rebuild a CRS for a datum it has seen.
    """
    Location(datum="NAD27")

    with patch.object(CRS, "from_user_input", wraps=CRS.from_user_input) as mock_crs:
        datums = [Location(datum="NAD27").datum for _ in range(5)]

    with subtests.test("datum is normalized on every call"):
        assert datums == ["NAD27"] * 5

    with subtests.test("CRS is not built again"):
        assert mock_crs.call_count == 0


def test_location_partial_values(subtests):
    """
    Test the Location model with partial values.
    """
    location = Location(latitude=45.0, longitude=-120.0)

    with subtests.test("latitude is set correctly"):
        assert location.latitude == 45.0

    with subtests.test("longitude is set correctly"):
        assert location.longitude == -120.0

    with subtests.test("elevation has default value"):
        assert location.elevation == 0.0

    with subtests.test("datum has default value"):
        assert location.datum == "WGS 84"

    with subtests.test("latitude_uncertainty is 0.0"):
        assert location.latitude_uncertainty == 0.0

    with subtests.test("longitude_uncertainty is 0.0"):
        assert location.longitude_uncertainty == 0.0

    with subtests.test("elevation_uncertainty is 0.0"):
        assert location.elevation_uncertainty == 0.0


def test_location_validate_position(subtests):
    """
    Test the validate_position method for latitude and longitude.
    """
    location = Location(latitude="45:30:00", longitude="-120:15:00")

    with subtests.test("latitude is converted from DMS to decimal"):
        assert location.latitude == 45.5

    with subtests.test("longitude is converted from DMS to decimal"):
        assert location.longitude == -120.25


def test_station_location_default_values(default_station_location, subtests):
    """
    Test the default values of the StationLocation model.
    """
    location = default_station_location

    with subtests.test("latitude default is 0.0"):
        assert location.latitude == 0.0

    with subtests.test("longitude default is 0.0"):
        assert location.longitude == 0.0

    with subtests.test("elevation default is 0.0"):
        assert location.elevation == 0.0

    with subtests.test("datum default is WGS 84"):
        assert location.datum == "WGS 84"

    with subtests.test("declination default is 0.0"):
        assert location.declination.value == 0.0

    with subtests.test("declination_epoch default is None"):
        assert location.declination.epoch is None


def test_station_location_custom_values(subtests):
    """
    Test the StationLocation model with custom values.
    """
    location = StationLocation(
        latitude=37.75,
        longitude=-122.42,
        elevation=15.0,
        datum="NAD83",
        **{"declination": {"value": 15.5, "epoch": 2020}},
    )

    with subtests.test("latitude is set correctly"):
        assert location.latitude == 37.75

    with subtests.test("longitude is set correctly"):
        assert location.longitude == -122.42

    with subtests.test("elevation is set correctly"):
        assert location.elevation == 15.0

    with subtests.test("datum is set correctly and normalized"):
        assert location.datum == "NAD83"

    with subtests.test("declination is set correctly"):
        assert location.declination.value == 15.5

    with subtests.test("declination_epoch is set correctly"):
        assert location.declination.epoch == "2020"


def test_station_location_from_kwargs(subtests):
    """
    Test initializing StationLocation from kwargs.
    """
    kwargs = {
        "latitude": 39.5,
        "longitude": -105.0,
        "elevation": 1700.0,
        "declination.value": 10.5,
        "declination.epoch": "2023",
        "datum": "NAD27",
    }

    location = StationLocation(**kwargs)

    with subtests.test("latitude from kwargs is set correctly"):
        assert location.latitude == 39.5

    with subtests.test("longitude from kwargs is set correctly"):
        assert location.longitude == -105.0

    with subtests.test("elevation from kwargs is set correctly"):
        assert location.elevation == 1700.0

    with subtests.test("declination from kwargs is set correctly"):
        assert location.declination.value == 10.5

    with subtests.test("declination_epoch from kwargs is set correctly"):
        assert location.declination.epoch == "2023"

    with subtests.test("datum from kwargs is set correctly and normalized"):
        assert location.datum == "NAD27"


# TODO: check to see if this is needed.
# def test_station_location_invalid_declination_epoch(subtests):
#     """
#     Test the StationLocation model with an invalid declination_epoch value.
#     """
#     with subtests.test("declination_epoch before 1500 raises ValidationError"):
#         with pytest.raises(ValidationError):
#             StationLocation(**{"declination.epoch": 1400})

#     with subtests.test("declination_epoch after 2030 raises ValidationError"):
#         with pytest.raises(ValidationError):
#             StationLocation(**{"declination.epoch": 2030})


def test_station_location_dms_conversion(subtests):
    """
    Test DMS conversion in StationLocation.
    """
    location = StationLocation(latitude="38:53:42.5", longitude="-77:02:12")

    with subtests.test("latitude is converted from DMS to decimal"):
        assert pytest.approx(location.latitude, 0.0001) == 38.8951

    with subtests.test("longitude is converted from DMS to decimal"):
        assert pytest.approx(location.longitude, 0.0001) == -77.0367
