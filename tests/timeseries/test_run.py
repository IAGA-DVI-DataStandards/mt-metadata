import json
from collections import OrderedDict
from operator import itemgetter

import pandas as pd
import pytest
from pydantic import ValidationError

from mt_metadata.timeseries import Auxiliary, Electric, Magnetic, Run


@pytest.fixture(scope="module")
def meta_dict():
    meta = {
        "run": OrderedDict(
            [
                ("acquired_by.comments", "lazy"),
                ("acquired_by.author", "MT guru"),
                ("channels_recorded_auxiliary", ["temperature"]),
                ("channels_recorded_electric", ["ex", "ey"]),
                ("channels_recorded_magnetic", ["hx", "hy", "hz"]),
                ("comments", "Cloudy solar panels failed"),
                ("data_logger.firmware.author", "MT instruments"),
                ("data_logger.firmware.name", "FSGMT"),
                ("data_logger.firmware.version", "12.120"),
                ("data_logger.id", "mt091"),
                ("data_logger.manufacturer", "T. Lurric"),
                ("data_logger.model", "Ichiban"),
                ("data_logger.power_source.comments", "rats"),
                ("data_logger.power_source.id", "12"),
                ("data_logger.power_source.type", "pb acid"),
                ("data_logger.power_source.voltage.end", 12.0),
                ("data_logger.power_source.voltage.start", 14.0),
                ("data_logger.timing_system.comments", "solid"),
                ("data_logger.timing_system.drift", 0.001),
                ("data_logger.timing_system.type", "GPS"),
                ("data_logger.timing_system.uncertainty", 1e-06),
                ("data_logger.type", "broadband"),
                ("data_type", "MT"),
                ("id", "mt01a"),
                ("metadata_by.comments", "lazy"),
                ("metadata_by.author", "x"),
                ("provenance.archive.name", ""),
                ("provenance.comments", "provenance comments"),
                ("provenance.creation_time", "1980-01-01T00:00:00+00:00"),
                ("provenance.creator.author", ""),
                ("provenance.log", "provenance log"),
                ("provenance.software.author", ""),
                ("provenance.software.name", ""),
                ("provenance.software.version", ""),
                ("provenance.submitter.author", ""),
                ("sample_rate", 256.0),
                ("time_period.end", "1980-01-01T00:00:00+00:00"),
                ("time_period.start", "1980-01-01T00:00:00+00:00"),
            ]
        )
    }
    return {"run": OrderedDict(sorted(meta["run"].items(), key=itemgetter(0)))}


@pytest.fixture
def run_object():
    return Run()


def test_in_out_dict(run_object, meta_dict):
    run_object.from_dict(meta_dict)
    assert meta_dict == run_object.to_dict()


def test_in_out_series(run_object, meta_dict):
    run_series = pd.Series(meta_dict["run"])
    run_object.from_series(run_series)
    assert meta_dict == run_object.to_dict()


def test_in_out_json(run_object, meta_dict):
    run_json = json.dumps(meta_dict)
    run_object.from_json(run_json)
    assert meta_dict == run_object.to_dict()


def test_start(run_object, subtests):
    run_object.time_period.start = "2020/01/02T12:20:40.4560Z"
    with subtests.test("from Z time"):
        assert run_object.time_period.start == "2020-01-02T12:20:40.456000+00:00"

    run_object.time_period.start = "01/02/20T12:20:40.4560"
    with subtests.test("from mixed"):
        assert run_object.time_period.start == "2020-01-02T12:20:40.456000+00:00"


def test_end_date(run_object, subtests):
    run_object.time_period.end = "2020/01/02T12:20:40.4560Z"
    with subtests.test("from Z time"):
        assert run_object.time_period.end == "2020-01-02T12:20:40.456000+00:00"

    run_object.time_period.end = "01/02/20T12:20:40.4560"
    with subtests.test("from mixed"):
        assert run_object.time_period.end == "2020-01-02T12:20:40.456000+00:00"


def test_n_channels(run_object, meta_dict, subtests):
    run_object.from_dict(meta_dict)
    with subtests.test("n_channels"):
        assert run_object.n_channels == 6
    with subtests.test("length"):
        assert run_object.n_channels == 6


def test_set_channels_recorded_electric(run_object, subtests):
    run_object.channels_recorded_electric = ["ex", "ey"]
    with subtests.test("length"):
        assert run_object.n_channels == 2
    with subtests.test("in list"):
        assert run_object.channels_recorded_electric == ["ex", "ey"]


def test_set_channels(run_object, subtests):
    run_object.channels = [Electric(component="ez")]
    with subtests.test("length"):
        assert run_object.n_channels == 1
    with subtests.test("in list"):
        assert run_object.channels_recorded_all == ["ez"]


def test_set_channels_fail(run_object, subtests):
    def set_channels(value):
        run_object.channels = value

    with subtests.test("fail from input int"):
        with pytest.raises(TypeError):
            set_channels(10)
    with subtests.test("fail from mixed input"):
        with pytest.raises(ValidationError):
            set_channels([Run(), Electric(component="ex")])


def test_add_channels(run_object, subtests):
    run_02 = Run()
    run_02.add_channel(Electric(component="ex"))
    run_02.add_channel(Magnetic(component="hx"))
    run_02.add_channel(Auxiliary(component="temperature"))
    run_object.add_channel(Electric(component="ey"))
    run_object.merge(run_02)
    with subtests.test("length"):
        assert run_object.n_channels == 4
    with subtests.test("In list all"):
        assert sorted(run_object.channels_recorded_all) == sorted(
            ["ex", "ey", "hx", "temperature"]
        )
    with subtests.test("in list electric"):
        assert run_object.channels_recorded_electric == sorted(["ex", "ey"])
    with subtests.test("in list magnetic"):
        assert run_object.channels_recorded_magnetic == ["hx"]
    with subtests.test("in list auxiliary"):
        assert run_object.channels_recorded_auxiliary == ["temperature"]


def test_remove_channel(run_object):
    run_object.add_channel(Electric(component="ex"))
    run_object.remove_channel("ex")
    assert run_object.channels_recorded_all == []


def test_update_time_period(run_object, subtests):
    ch = Electric(component="ex")
    ch.time_period.start = "2020-01-01T00:00:00"
    ch.time_period.end = "2020-12-01T12:12:12"
    run_object.add_channel(ch)

    with subtests.test("Test new start"):
        assert run_object.time_period.start == "2020-01-01T00:00:00+00:00"

    with subtests.test("Test new end"):
        assert run_object.time_period.end == "2020-12-01T12:12:12+00:00"


@pytest.fixture
def populated_run():
    """Fixture to create a Run object with pre-populated channels."""
    run = Run()
    run.add_channel(Electric(component="ex"))
    run.add_channel(Electric(component="ey"))
    run.add_channel(Magnetic(component="hx"))
    run.add_channel(Auxiliary(component="temperature"))
    return run


def test_add_channel(run_object, subtests):
    """Test adding channels to the Run object."""
    with subtests.test("Add electric channel"):
        run_object.add_channel(Electric(component="ex"))
        assert "ex" in run_object.channels_recorded_electric

    with subtests.test("Add magnetic channel"):
        run_object.add_channel(Magnetic(component="hx"))
        assert "hx" in run_object.channels_recorded_magnetic

    with subtests.test("Add auxiliary channel"):
        run_object.add_channel(Auxiliary(component="temperature"))
        assert "temperature" in run_object.channels_recorded_auxiliary

    with subtests.test("Add channel from string electric"):
        run_object.add_channel("ex")
        assert "ex" in run_object.channels_recorded_electric
        assert "ex" in run_object.channels_recorded_all
        assert "ex" in run_object.channels.keys()

    with subtests.test("Add channel from string magnetic"):
        run_object.add_channel("hx")
        assert "hx" in run_object.channels_recorded_magnetic
        assert "hx" in run_object.channels_recorded_all
        assert "hx" in run_object.channels.keys()

    with subtests.test("Add channel from string auxiliary"):
        run_object.add_channel("temperature")
        assert "temperature" in run_object.channels_recorded_auxiliary
        assert "temperature" in run_object.channels_recorded_all
        assert "temperature" in run_object.channels.keys()

    with subtests.test("Add channel from dict magnetic"):
        run_object.add_channel({"type": None, "component": "hy"})
        assert isinstance(run_object.channels["hy"], Magnetic)
        assert "hy" in run_object.channels_recorded_magnetic


def test_remove_channel(populated_run, subtests):
    """Test removing channels from the Run object."""
    with subtests.test("Remove electric channel"):
        populated_run.remove_channel("ex")
        assert "ex" not in populated_run.channels_recorded_electric

    with subtests.test("Remove magnetic channel"):
        populated_run.remove_channel("hx")
        assert "hx" not in populated_run.channels_recorded_magnetic

    with subtests.test("Remove auxiliary channel"):
        populated_run.remove_channel("temperature")
        assert "temperature" not in populated_run.channels_recorded_auxiliary


def test_update_time_period(run_object, subtests):
    """Test updating the time period based on channel data."""
    ch1 = Electric(component="ex")
    ch1.time_period.start = "2020-01-01T00:00:00"
    ch1.time_period.end = "2020-01-01T12:00:00"

    ch2 = Magnetic(component="hx")
    ch2.time_period.start = "2020-01-01T06:00:00"
    ch2.time_period.end = "2020-01-01T18:00:00"

    run_object.add_channel(ch1)
    run_object.add_channel(ch2)
    run_object.update_time_period()

    with subtests.test("Start time"):
        assert run_object.time_period.start == "2020-01-01T00:00:00+00:00"

    with subtests.test("End time"):
        assert run_object.time_period.end == "2020-01-01T18:00:00+00:00"


def test_channels_recorded_all(populated_run):
    """Test the combined list of all recorded channels."""
    assert sorted(populated_run.channels_recorded_all) == sorted(
        ["ex", "ey", "hx", "temperature"]
    )


def test_merge_runs(run_object, populated_run, subtests):
    """Test merging two Run objects."""
    run_object.add_channel(Electric(component="ez"))
    run_object.merge(populated_run)

    with subtests.test("Number of channels"):
        assert run_object.n_channels == 5

    with subtests.test("All channels recorded"):
        assert sorted(run_object.channels_recorded_all) == sorted(
            ["ex", "ey", "hx", "temperature", "ez"]
        )


def test_invalid_channel_addition(run_object, subtests):
    """Test adding invalid channels to the Run object."""
    with subtests.test("Invalid channel type"):
        with pytest.raises(ValueError):
            run_object.add_channel([])

    with subtests.test("Missing component for non-auxiliary channel"):
        with pytest.raises(ValidationError):
            run_object.add_channel(Electric(component=None))
