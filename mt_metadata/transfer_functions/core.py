# -*- coding: utf-8 -*-
"""
.. module:: mt_metadata.transfer_functions.core
   :synopsis: The main container for transfer functions

.. moduleauthor:: Jared Peacock <jpeacock@usgs.gov>
"""

from collections import OrderedDict
from copy import deepcopy

# ==============================================================================
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xarray as xr
from loguru import logger
from typing_extensions import Self

from mt_metadata import DEFAULT_CHANNEL_NOMENCLATURE
from mt_metadata.base.helpers import validate_name
from mt_metadata.common.list_dict import ListDict
from mt_metadata.timeseries import Electric, Magnetic, Run
from mt_metadata.timeseries import Station as TSStation
from mt_metadata.timeseries import Survey
from mt_metadata.transfer_functions.io import EDI, EMTFXML, JFile, ZMM, ZongeMTAvg
from mt_metadata.transfer_functions.io.zfiles.metadata import Channel as ZChannel
from mt_metadata.transfer_functions.tf import Station

# =============================================================================


class TF:
    """
    Generic container to hold information about an electromagnetic
    transfer funtion

    The thought here is to have a central container TF.dataset which is an
    xarray.Dataset that contains the impedance, tipper, errors and covariance
    values.  There are helper functions to get and set these from the
    TF.dataset.  Cause most of the time the user will want just the impedance
    or the tipper and associated errors.  We are accommodating EMTF style
    covariances to accurately rotated data errors.

    When reading and writing edi files this information will be lost.

    """

    # Class-level template cache
    _template_cache = {}

    def __init__(self, fn: str | Path | None = None, **kwargs):
        # set metadata for the station
        self._survey_metadata = self._initialize_metadata()
        self.channel_nomenclature = DEFAULT_CHANNEL_NOMENCLATURE
        self._inverse_channel_nomenclature = {}

        self._rotation_angle = 0
        self.save_dir = Path.cwd()

        self._dataset_attr_dict = {
            "survey": "survey_metadata.id",
            "project": "survey_metadata.project",
            "id": "station_metadata.id",
            "name": "station_metadata.geographic_name",
            "latitude": "station_metadata.location.latitude",
            "longitude": "station_metadata.location.longitude",
            "elevation": "station_metadata.location.elevation",
            "declination": "station_metadata.location.declination.value",
            "datum": "station_metadata.location.datum",
            "acquired_by": "station_metadata.acquired_by.author",
            "start": "station_metadata.time_period.start",
            "end": "station_metadata.time_period.end",
            "runs_processed": "station_metadata.run_list",
            "coordinate_system": "station_metadata.orientation.reference_frame",
        }

        self._read_write_dict = {
            "edi": {"write": self.to_edi, "read": self.from_edi},
            "xml": {"write": self.to_emtfxml, "read": self.from_emtfxml},
            "emtfxml": {"write": self.to_emtfxml, "read": self.from_emtfxml},
            "j": {"write": self.to_jfile, "read": self.from_jfile},
            "zmm": {"write": self.to_zmm, "read": self.from_zmm},
            "zrr": {"write": self.to_zrr, "read": self.from_zrr},
            "zss": {"write": self.to_zss, "read": self.from_zss},
            "avg": {"write": self.to_avg, "read": self.from_avg},
        }

        tf_set = False
        try:
            period = kwargs.pop("period")
            self._transfer_function = self._initialize_transfer_function(periods=period)
            tf_set = True
        except KeyError:
            try:
                period = 1.0 / kwargs.pop("frequency")
                self._transfer_function = self._initialize_transfer_function(
                    periods=period
                )
                tf_set = True
            except KeyError:
                pass

        for key, value in kwargs.items():
            setattr(self, key, value)

        if not tf_set:
            self._transfer_function = self._initialize_transfer_function()

        self.fn = fn

    @property
    def inverse_channel_nomenclature(self) -> dict[str, str]:
        if not self._inverse_channel_nomenclature:
            self._inverse_channel_nomenclature = {
                v: k for k, v in self.channel_nomenclature.items()
            }
        return self._inverse_channel_nomenclature

    def __str__(self) -> str:
        lines = [f"Station: {self.station}", "-" * 50]
        lines.append(f"\tSurvey:            {self.survey_metadata.id}")
        lines.append(f"\tProject:           {self.survey_metadata.project}")
        lines.append(f"\tAcquired by:       {self.station_metadata.acquired_by.author}")
        lines.append(f"\tAcquired date:     {self.station_metadata.time_period.start}")
        lines.append(f"\tLatitude:          {self.latitude:.3f}")
        lines.append(f"\tLongitude:         {self.longitude:.3f}")
        lines.append(f"\tElevation:         {self.elevation:.3f}")
        lines.append("\tDeclination:   ")
        lines.append(
            f"\t\tValue:     {self.station_metadata.location.declination.value}"
        )
        lines.append(
            f"\t\tModel:     {self.station_metadata.location.declination.model}"
        )
        lines.append(
            f"\tCoordinate System: {self.station_metadata.orientation.reference_frame}"
        )

        lines.append(f"\tImpedance:         {self.has_impedance()}")
        lines.append(f"\tTipper:            {self.has_tipper()}")

        if self.period is not None:
            lines.append(f"\tN Periods:     {len(self.period)}")

            lines.append("\tPeriod Range:")
            lines.append(f"\t\tMin:   {self.period.min():.5E} s")
            lines.append(f"\t\tMax:   {self.period.max():.5E} s")

            lines.append("\tFrequency Range:")
            lines.append(f"\t\tMin:   {1./self.period.max():.5E} Hz")
            lines.append(f"\t\tMax:   {1./self.period.min():.5E} Hz")
        return "\n".join(lines)

    def __repr__(self) -> str:
        lines = []
        lines.append(f"survey='{self.survey}'")
        lines.append(f"station='{self.station}'")
        lines.append(f"latitude={self.latitude:.2f}")
        lines.append(f"longitude={self.longitude:.2f}")
        lines.append(f"elevation={self.elevation:.2f}")

        return f"TF( {(', ').join(lines)} )"

    def __eq__(self, other: object) -> bool:
        """
        Check if two TF objects are equal.

        Parameters
        ----------
        other: object
            Another object to compare with

        Returns
        -------
        bool
            True if equal, False otherwise
        """
        is_equal = True
        if not isinstance(other, TF):
            logger.info(f"Comparing object is not TF, type {type(other)}")
            is_equal = False
        if self.station_metadata != other.station_metadata:
            logger.info("Station metadata is not equal")
            is_equal = False
        if self.survey_metadata != other.survey_metadata:
            logger.info("Survey Metadata is not equal")
            is_equal = False
        if self.has_transfer_function() and other.has_transfer_function():
            if not self.transfer_function.equals(other.transfer_function):
                logger.info("TF is not equal")
                is_equal = False
        elif not self.has_transfer_function() and not other.has_transfer_function():
            pass
        else:
            logger.info("TF is not equal")
            is_equal = False

        return is_equal

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ["logger"]:
                continue

            setattr(result, k, deepcopy(v, memo))
        return result

    def copy(self) -> Self:
        """
        Create a deep copy of the current object.

        Returns
        -------
        Self
            A deep copy of the current object.
        """
        return deepcopy(self)

    def _add_channels(
        self, run_metadata: Run, default: list[str] = ["ex", "ey", "hx", "hy", "hz"]
    ) -> Run:
        """
        Add channels to a run.

        Parameters
        ----------
        run_metadata: Run
            The run metadata to add channels to.
        default: list[str], optional
            The default list of channels to add.

        Returns
        -------
        Run
            The updated run metadata.
        """
        for ch in [cc for cc in default if cc.startswith("e")]:
            run_metadata.add_channel(Electric(component=ch))
        for ch in [cc for cc in default if cc.startswith("h")]:
            run_metadata.add_channel(Magnetic(component=ch))

        return run_metadata

    def _initialize_metadata(self) -> Survey:
        """
        Create a single `Survey` object to store all metadata

        This will include all stations and runs.

        """

        survey_metadata = Survey(id="0")
        survey_metadata.stations.append(Station(id="0"))
        survey_metadata.stations[0].runs.append(Run(id="0"))

        self._add_channels(survey_metadata.stations[0].runs[0])

        return survey_metadata

    def _validate_run_metadata(self, run_metadata: Run) -> Run:
        """
        Validate run metadata.

        Parameters
        ----------
        run_metadata: Run
            The run metadata to validate.

        Returns
        -------
        Run
            The validated run metadata.

        """

        if not isinstance(run_metadata, Run):
            if isinstance(run_metadata, dict):
                if "run" not in [cc.lower() for cc in run_metadata.keys()]:
                    run_metadata = {"Run": run_metadata}
                r_metadata = Run()
                r_metadata.from_dict(run_metadata)
                logger.debug("Loading from metadata dict")
                return r_metadata
            else:
                msg = (
                    f"input metadata must be type {type(self.run_metadata)} "
                    f"or dict, not {type(run_metadata)}"
                )
                logger.error(msg)
                raise TypeError(msg)
        return run_metadata

    def _validate_station_metadata(self, station_metadata: Station) -> Station:
        """
        Validate station metadata.

        Parameters
        ----------
        station_metadata: Station
            The station metadata to validate.

        Returns
        -------
        Station
            The validated station metadata.

        """

        if not isinstance(station_metadata, Station):
            if isinstance(station_metadata, dict):
                if "station" not in [cc.lower() for cc in station_metadata.keys()]:
                    station_metadata = {"Station": station_metadata}
                st_metadata = Station()
                st_metadata.from_dict(station_metadata)
                logger.debug("Loading from metadata dict")
                return st_metadata
            else:
                msg = (
                    f"input metadata must be type {type(self.station_metadata)}"
                    f" or dict, not {type(station_metadata)}"
                )
                logger.error(msg)
                raise TypeError(msg)
        return station_metadata

    def _validate_survey_metadata(self, survey_metadata: Survey) -> Survey:
        """
        Validate survey metadata.

        Parameters
        ----------
        survey_metadata: Survey
            The survey metadata to validate.

        Returns
        -------
        Survey
            The validated survey metadata.
        """

        if not isinstance(survey_metadata, Survey):
            if isinstance(survey_metadata, dict):
                if "survey" not in [cc.lower() for cc in survey_metadata.keys()]:
                    survey_metadata = {"Survey": survey_metadata}
                sv_metadata = Survey()
                sv_metadata.from_dict(survey_metadata)
                logger.debug("Loading from metadata dict")
                return sv_metadata
            else:
                msg = (
                    f"input metadata must be type {type(self.survey_metadata)}"
                    f" or dict, not {type(survey_metadata)}"
                )
                logger.error(msg)
                raise TypeError(msg)
        return survey_metadata

    ### Properties ------------------------------------------------------------
    @property
    def survey_metadata(self) -> Survey:
        """
        Survey metadata.
        """
        return self._survey_metadata

    @survey_metadata.setter
    def survey_metadata(self, survey_metadata: Survey) -> None:
        """
        Set survey metadata.

        Parameters
        ----------
        survey_metadata: Survey
            The survey metadata object or dictionary to set.

        """

        if survey_metadata is not None:
            survey_metadata = self._validate_survey_metadata(survey_metadata)
            self._survey_metadata.update(survey_metadata)
            for station in survey_metadata.stations:
                station.update_time_period()
                self._survey_metadata.add_station(station)

            if len(self._survey_metadata.stations.keys()) > 1:
                if "0" in self._survey_metadata.stations.keys():
                    self._survey_metadata.stations.remove("0")

            self._survey_metadata.update_time_period()

    @property
    def station_metadata(self) -> Station:
        """
        Station metadata from survey_metadata.stations[0]
        """

        return self.survey_metadata.stations[0]

    @station_metadata.setter
    def station_metadata(self, station_metadata: Station | None = None) -> None:
        """
        Set station metadata from a valid input.

        Parameters
        ----------
        station_metadata: Station | None
            The station metadata object or dictionary to set.
        """

        if station_metadata is not None:
            station_metadata = self._validate_station_metadata(station_metadata)

            runs = ListDict()
            if self.run_metadata.id not in ["0", 0, None]:
                runs.append(self.run_metadata.copy())
            runs.extend(station_metadata.runs)
            if len(runs) == 0:
                runs[0] = Run(id="0")
            # be sure there is a level below
            if len(runs[0].channels) == 0:
                self._add_channels(runs[0])
            stations = ListDict()
            stations.append(station_metadata)
            stations[0].runs = runs
            stations[0].update_time_period()

            self.survey_metadata.stations = stations
            self._survey_metadata.update_time_period()

    @property
    def run_metadata(self) -> Run:
        """
        Run metadata from survey_metadata.stations[0].runs[0]
        """

        return self.survey_metadata.stations[0].runs[0]

    @run_metadata.setter
    def run_metadata(self, run_metadata: Run | None = None) -> None:
        """
        Set run metadata from a valid input.

        Parameters
        ----------
        run_metadata: Run | None
            The run metadata object or dictionary to set.
        """

        # need to make sure the first index is the desired channel
        if run_metadata is not None:
            run_metadata = self._validate_run_metadata(run_metadata)

            runs = ListDict()
            runs.append(run_metadata)
            channels = ListDict()
            if self.component is not None:
                key = str(self.component)

                channels.append(self.station_metadata.runs[0].channels[key])
                # add existing channels
                channels.extend(self.run_metadata.channels, skip_keys=[key, "0"])
            # add channels from input metadata
            channels.extend(run_metadata.channels)

            runs[0].channels = channels
            runs.extend(self.station_metadata.runs, skip_keys=[run_metadata.id, "0"])

            self._survey_metadata.stations[0].runs = runs

    def _get_template_key(self):
        """Generate a cache key based on channel nomenclature"""
        return tuple(sorted(self.channel_nomenclature.items()))

    def _initialize_transfer_function(self, periods=[1]):
        """
        Create transfer function dataset efficiently using a cached template.
        """
        template_key = self._get_template_key()

        # Create template if not cached
        if template_key not in self._template_cache:
            tf = xr.DataArray(
                data=0.0 + 0j,
                dims=["period", "output", "input"],
                coords={
                    "period": [1],  # Single period for template
                    "output": self._ch_output_dict["all"],
                    "input": self._ch_input_dict["all"],
                },
                name="transfer_function",
            )

            tf_err = xr.DataArray(
                data=0.0,
                dims=["period", "output", "input"],
                coords={
                    "period": [1],
                    "output": self._ch_output_dict["all"],
                    "input": self._ch_input_dict["all"],
                },
                name="transfer_function_error",
            )

            tf_model_err = xr.DataArray(
                data=0.0,
                dims=["period", "output", "input"],
                coords={
                    "period": [1],
                    "output": self._ch_output_dict["all"],
                    "input": self._ch_input_dict["all"],
                },
                name="transfer_function_model_error",
            )

            inv_signal_power = xr.DataArray(
                data=0.0 + 0j,
                dims=["period", "output", "input"],
                coords={
                    "period": [1],
                    "output": self._ch_output_dict["all"],
                    "input": self._ch_input_dict["all"],
                },
                name="inverse_signal_power",
            )

            residual_covariance = xr.DataArray(
                data=0.0 + 0j,
                dims=["period", "output", "input"],
                coords={
                    "period": [1],
                    "output": self._ch_output_dict["all"],
                    "input": self._ch_input_dict["all"],
                },
                name="residual_covariance",
            )

            # will need to add in covariance in some fashion
            template = xr.Dataset(
                {
                    tf.name: tf,
                    tf_err.name: tf_err,
                    tf_model_err.name: tf_model_err,
                    inv_signal_power.name: inv_signal_power,
                    residual_covariance.name: residual_covariance,
                },
                coords={
                    "period": [1],
                    "output": self._ch_output_dict["all"],
                    "input": self._ch_input_dict["all"],
                },
            )
            self._template_cache[template_key] = template

        # Copy template and adjust periods
        dataset = self._template_cache[template_key].copy(deep=True)

        if len(periods) != 1 or periods[0] != 1:
            # Expand/adjust to match requested periods
            dataset = dataset.reindex(period=periods, fill_value=0.0)

        return dataset

    # ==========================================================================
    # Properties
    # ==========================================================================
    @property
    def channel_nomenclature(self) -> dict:
        """Channel nomenclature dictionary keyed by channel names.

        For example:

        {'ex': 'ex', 'ey': 'ey', 'hx': 'hx', 'hy': 'hy', 'hz': 'hz'}
        """
        return self._channel_nomenclature

    @channel_nomenclature.setter
    def channel_nomenclature(self, ch_dict: dict) -> None:
        """
        Set the channel nomenclature dictionary.

        Parameters
        ----------
        ch_dict : dict
            A dictionary containing channel names and their corresponding labels.
        """

        if not isinstance(ch_dict, dict):
            raise TypeError(
                "Channel_nomenclature must be a dictionary with keys "
                "['ex', 'ey', 'hx', 'hy', 'hz']."
            )

        self._channel_nomenclature = ch_dict
        # unpack channel nomenclature dict
        self.ex = self._channel_nomenclature["ex"]
        self.ey = self._channel_nomenclature["ey"]
        self.hx = self._channel_nomenclature["hx"]
        self.hy = self._channel_nomenclature["hy"]
        self.hz = self._channel_nomenclature["hz"]
        self.ex_ey = [self.ex, self.ey]
        self.hx_hy = [self.hx, self.hy]
        self.ex_ey_hz = [self.ex, self.ey, self.hz]

    @property
    def _ch_input_dict(self) -> dict:
        return {
            "impedance": self.hx_hy,
            "tipper": self.hx_hy,
            "impedance_error": self.hx_hy,
            "impedance_model_error": self.hx_hy,
            "tipper_error": self.hx_hy,
            "tipper_model_error": self.hx_hy,
            "isp": self.hx_hy,
            "res": self.ex_ey_hz,
            "tf": self.hx_hy,
            "tf_error": self.hx_hy,
            "tf_model_error": self.hx_hy,
            "all": [self.ex, self.ey, self.hz, self.hx, self.hy],
        }

    @property
    def _ch_output_dict(self) -> dict:
        return {
            "impedance": self.ex_ey,
            "tipper": [self.hz],
            "impedance_error": self.ex_ey,
            "impedance_model_error": self.ex_ey,
            "tipper_error": [self.hz],
            "tipper_model_error": [self.hz],
            "isp": self.hx_hy,
            "res": self.ex_ey_hz,
            "tf": self.ex_ey_hz,
            "tf_error": self.ex_ey_hz,
            "tf_model_error": self.ex_ey_hz,
            "all": [self.ex, self.ey, self.hz, self.hx, self.hy],
        }

    @property
    def index_zxx(self) -> dict:
        return {"input": self.hx, "output": self.ex}

    @property
    def index_zxy(self) -> dict:
        return {"input": self.hy, "output": self.ex}

    @property
    def index_zyx(self) -> dict:
        return {"input": self.hx, "output": self.ey}

    @property
    def index_zyy(self) -> dict:
        return {"input": self.hy, "output": self.ey}

    @property
    def index_tzx(self) -> dict:
        return {"input": self.hx, "output": self.hz}

    @property
    def index_tzy(self) -> dict:
        return {"input": self.hy, "output": self.hz}

    @property
    def fn(self) -> Path:
        """reference to original data file"""
        return self._fn

    @fn.setter
    def fn(self, value: Path | str | None) -> None:
        """set file name

        Parameters
        ----------
        value : Path | str | None
            The file name to set.
        """
        if value is None:
            self._fn = None
            return
        self._fn = Path(value)
        self.save_dir = self._fn.parent

    @property
    def latitude(self) -> float:
        """Latitude"""
        return self.station_metadata.location.latitude

    @latitude.setter
    def latitude(self, latitude: float) -> None:
        """
        set latitude making sure the input is in decimal degrees

        upon setting utm coordinates are recalculated
        """
        self.station_metadata.location.latitude = latitude

    @property
    def longitude(self) -> float:
        """Longitude"""
        return self.station_metadata.location.longitude

    @longitude.setter
    def longitude(self, longitude: float) -> None:
        """
        set longitude making sure the input is in decimal degrees

        upon setting utm coordinates are recalculated
        """
        self.station_metadata.location.longitude = longitude

    @property
    def elevation(self) -> float:
        """Elevation"""
        return self.station_metadata.location.elevation

    @elevation.setter
    def elevation(self, elevation: float) -> None:
        """
        set elevation, should be input as meters
        """

        self.station_metadata.location.elevation = elevation

    @property
    def dataset(self) -> xr.Dataset:
        """
        This will return an xarray dataset with proper metadata

        Returns
        -------
        xr.Dataset
            The xarray dataset with metadata.
        """

        for key, mkey in self._dataset_attr_dict.items():
            obj, attr = mkey.split(".", 1)
            value = getattr(self, obj).get_attr_from_name(attr)

            self._transfer_function.attrs[key] = value
        return self._transfer_function

    def _validate_input_ndarray(
        self, ndarray: np.ndarray, atype: str = "impedance"
    ) -> None:
        """
        Validate the input based on array type and component

        Parameters
        ----------
        ndarray : np.ndarray
            The input array to validate.
        atype : str
            The type of the array (e.g. "impedance", "tipper").

        """
        shape_dict = {
            "impedance": (2, 2),
            "tipper": (1, 2),
            "impedance_error": (2, 2),
            "impedance_model_error": (2, 2),
            "tipper_error": (1, 2),
            "tipper_model_error": (1, 2),
            "isp": (2, 2),
            "res": (3, 3),
            "transfer_function": (3, 2),
            "transfer_function_error": (3, 2),
            "transfer_function_model_error": (3, 2),
            "tf": (3, 2),
            "tf_error": (3, 2),
            "tf_model_error": (3, 2),
        }

        shape = shape_dict[atype]
        if ndarray.shape[1:] != shape:
            msg = (
                f"{atype} must be have shape (n_periods, {shape[0]}, "
                f"{shape[1]}), not {ndarray.shape}"
            )
            logger.error(msg)
            raise TFError(msg)
        if ndarray.shape[0] != self.period.size:
            msg = (
                f"New {atype} shape {ndarray.shape} not same as old {shape}, "
                "suggest creating a new instance."
            )
            logger.error(msg)
            raise TFError(msg)

    def _validate_input_dataarray(
        self, da: xr.DataArray, atype: str = "impedance"
    ) -> xr.DataArray:
        """
        Validate an input data array

        Parameters
        ----------
        da : xr.DataArray
            The input data array to validate.
        atype : str
            The type of the array (e.g. "impedance", "tipper").

        """

        ch_in = self._ch_input_dict[atype]
        ch_out = self._ch_output_dict[atype]

        # should test for shape
        if "period" not in da.coords.keys() or "input" not in da.coords.keys():
            msg = f"Coordinates must be period, output, input, not {list(da.coords.keys())}"
            logger.error(
                msg,
            )
            raise TFError(msg)
        if sorted(ch_out) != sorted(da.coords["output"].data.tolist()):
            msg = (
                f"Output dimensions must be {ch_out} not "
                f"{da.coords['output'].data.tolist()}"
            )
            logger.error(msg)
            raise TFError(msg)
        if sorted(ch_in) != sorted(da.coords["input"].data.tolist()):
            msg = (
                f"Input dimensions must be {ch_in} not "
                f"{da.coords['input'].data.tolist()}"
            )
            logger.error(msg)
            raise TFError(msg)
        # need to reorder the data array to the expected coordinates
        da = da.reindex(output=ch_out, input=ch_in)
        # if this is the first instantiation then just resize the
        # transfer function to fit the input
        if (
            self._transfer_function.transfer_function.data.shape[0] == 1
            and not self.has_tipper()
            and not self.has_impedance()
        ):
            self._transfer_function = self._initialize_transfer_function(da.period)
            return da
        elif (
            self._transfer_function.transfer_function.data.shape[0] == da.data.shape[0]
        ):
            return da
        else:
            msg = "Reassigning with a different shape is dangerous.  Should re-initialize transfer_function or make a new instance of TF"
            logger.error(msg)
            raise TFError(msg)

    def _set_data_array(
        self, value: xr.DataArray | np.ndarray | list | tuple | None, atype: str
    ) -> None:
        """

        Parameters
        ----------
        value : xr.DataArray | np.ndarray | list | tuple | None
            The data array to set.
        atype : str
            The type of the array (e.g. "impedance", "tipper").

        """
        if value is None:
            return
        key_dict = {
            "tf": "transfer_function",
            "impedance": "transfer_function",
            "tipper": "transfer_function",
            "isp": "inverse_signal_power",
            "res": "residual_covariance",
            "transfer_function": "transfer_function",
            "impedance_error": "transfer_function_error",
            "impedance_model_error": "transfer_function_model_error",
            "tipper_error": "transfer_function_error",
            "tipper_model_error": "transfer_function_model_error",
            "tf_error": "transfer_function_error",
            "tf_model_error": "transfer_function_model_error",
            "transfer_function_error": "transfer_function_error",
            "transfer_function_model_error": "transfer_function_model_error",
        }
        key = key_dict[atype]
        ch_in = self._ch_input_dict[atype]
        ch_out = self._ch_output_dict[atype]
        comps = dict(input=ch_in, output=ch_out)

        if isinstance(value, (list, tuple, np.ndarray)):
            value = np.array(value)
            self._validate_input_ndarray(value, atype=atype)

            self._transfer_function[key].loc[comps] = value
        elif isinstance(value, xr.DataArray):
            nda = self._validate_input_dataarray(value, atype=atype)

            self._transfer_function[key].loc[comps] = nda
        else:
            msg = (
                f"Data type {type(value)} not supported use a numpy "
                "array or xarray.DataArray"
            )
            logger.error(msg)
            raise TFError(msg)

    def has_transfer_function(self) -> bool:
        """
        Check to see if the transfer function is not 0 and has
        transfer function components

        Returns
        -------
        bool
            True if the transfer function is not 0 and has components, False otherwise.

        """
        outputs = self._transfer_function.transfer_function.coords[
            "output"
        ].data.tolist()
        if self.ex in outputs or self.ey in outputs or self.hz in outputs:
            if np.all(
                self._transfer_function.transfer_function.loc[
                    dict(
                        input=self._ch_input_dict["tf"],
                        output=self._ch_output_dict["tf"],
                    )
                ].data
                == 0
            ):
                return False
            return True
        return False

    @property
    def transfer_function(self) -> xr.DataArray | None:
        """

        Returns
        -------
        xr.DataArray | None
            The transfer function data array or None if not set.

        """
        if self.has_transfer_function():
            ds = self.dataset.transfer_function.loc[
                dict(input=self.hx_hy, output=self.ex_ey_hz)
            ]
            for key, mkey in self._dataset_attr_dict.items():
                obj, attr = mkey.split(".", 1)
                value = getattr(self, obj).get_attr_from_name(attr)

                ds.attrs[key] = value
            return ds

    @transfer_function.setter
    def transfer_function(self, value: xr.DataArray | np.ndarray | list | tuple | None):
        """
        Set the impedance from values

        Parameters
        ----------
        value : xr.DataArray | np.ndarray | list | tuple | None
            The data array to set.
        atype : str
            The type of the array (e.g. "impedance", "tipper").

        """
        self._set_data_array(value, "tf")

    @property
    def transfer_function_error(self) -> xr.DataArray | None:
        """

        Returns
        -------
        xr.DataArray | None
            The transfer function error data array or None if not set.

        """
        if self.has_transfer_function():
            ds = self.dataset.transfer_function_error.loc[
                dict(input=self.hx_hy, output=self.ex_ey_hz)
            ]
            for key, mkey in self._dataset_attr_dict.items():
                obj, attr = mkey.split(".", 1)
                value = getattr(self, obj).get_attr_from_name(attr)

                ds.attrs[key] = value
            return ds

    @transfer_function_error.setter
    def transfer_function_error(
        self, value: xr.DataArray | np.ndarray | list | tuple | None
    ):
        """
        Set the impedance from values

        Parameters
        ----------
        value : xr.DataArray | np.ndarray | list | tuple | None
            The data array to set.
        atype : str
            The type of the array (e.g. "impedance", "tipper").
        """
        self._set_data_array(value, "tf_error")

    @property
    def transfer_function_model_error(self) -> xr.DataArray | None:
        """

        Returns
        -------
        xr.DataArray | None
            The transfer function model error data array or None if not set.

        """
        if self.has_transfer_function():
            ds = self.dataset.transfer_function_model_error.loc[
                dict(input=self.hx_hy, output=self.ex_ey_hz)
            ]
            for key, mkey in self._dataset_attr_dict.items():
                obj, attr = mkey.split(".", 1)
                value = getattr(self, obj).get_attr_from_name(attr)

                ds.attrs[key] = value
            return ds

    @transfer_function_model_error.setter
    def transfer_function_model_error(
        self, value: xr.DataArray | np.ndarray | list | tuple | None
    ):
        """
        Set the impedance from values

        Parameters
        ----------
        value : xr.DataArray | np.ndarray | list | tuple | None
            The data array to set.
        atype : str
            The type of the array (e.g. "impedance", "tipper").
        """
        self._set_data_array(value, "tf_model_error")

    def has_impedance(self) -> bool:
        """
        Check to see if the transfer function is not 0 and has
        transfer function components

        Returns
        -------
        bool
            True if the transfer function has impedance components, False otherwise.

        """
        outputs = self._transfer_function.transfer_function.coords[
            "output"
        ].data.tolist()
        if self.ex in outputs or self.ey in outputs:
            if np.all(
                self._transfer_function.transfer_function.loc[
                    dict(
                        input=self._ch_input_dict["impedance"],
                        output=self._ch_output_dict["impedance"],
                    )
                ].data
                == 0
            ):
                return False
            return True
        return False

    @property
    def impedance(self) -> xr.DataArray | None:
        """

        Returns
        -------
        xr.DataArray | None
            The impedance data array or None if not set.
        """
        if self.has_impedance():
            z = self.dataset.transfer_function.loc[
                dict(
                    input=self._ch_input_dict["impedance"],
                    output=self._ch_output_dict["impedance"],
                )
            ]
            z.name = "impedance"
            for key, mkey in self._dataset_attr_dict.items():
                obj, attr = mkey.split(".", 1)
                value = getattr(self, obj).get_attr_from_name(attr)

                z.attrs[key] = value
            return z

    @impedance.setter
    def impedance(self, value: xr.DataArray | np.ndarray | list | tuple | None):
        """
        Set the impedance from values

        Parameters
        ----------
        value : xr.DataArray | np.ndarray | list | tuple | None
            The data array to set.
        """
        self._set_data_array(value, "impedance")

    @property
    def impedance_error(self) -> xr.DataArray | None:
        """

        Returns
        -------
        xr.DataArray | None
            The impedance error data array or None if not set.

        """
        if self.has_impedance():
            z_err = self.dataset.transfer_function_error.loc[
                dict(
                    input=self._ch_input_dict["impedance"],
                    output=self._ch_output_dict["impedance"],
                )
            ]
            z_err.name = "impedance_error"

            for key, mkey in self._dataset_attr_dict.items():
                obj, attr = mkey.split(".", 1)
                value = getattr(self, obj).get_attr_from_name(attr)

                z_err.attrs[key] = value
            return z_err

    @impedance_error.setter
    def impedance_error(self, value: xr.DataArray | np.ndarray | list | tuple | None):
        """
        Set the impedance from values

        Parameters
        ----------
        value : xr.DataArray | np.ndarray | list | tuple | None
            The data array to set.
        """
        self._set_data_array(value, "impedance_error")

    @property
    def impedance_model_error(self) -> xr.DataArray | None:
        """

        Returns
        -------
        xr.DataArray | None
            The impedance model error data array or None if not set.

        """
        if self.has_impedance():
            z_err = self.dataset.transfer_function_model_error.loc[
                dict(
                    input=self._ch_input_dict["impedance"],
                    output=self._ch_output_dict["impedance"],
                )
            ]
            z_err.name = "impedance_model_error"

            for key, mkey in self._dataset_attr_dict.items():
                obj, attr = mkey.split(".", 1)
                value = getattr(self, obj).get_attr_from_name(attr)

                z_err.attrs[key] = value
            return z_err

    @impedance_model_error.setter
    def impedance_model_error(
        self, value: xr.DataArray | np.ndarray | list | tuple | None
    ):
        """
        Set the impedance model errors from values

        Parameters
        ----------
        value : xr.DataArray | np.ndarray | list | tuple | None
            The data array to set.
        """
        self._set_data_array(value, "impedance_model_error")

    def has_tipper(self) -> bool:
        """
        Check to see if the transfer function is not 0 and has
        transfer function components

        Returns
        -------
        bool
            True if the transfer function has tipper components, False otherwise.
        """
        outputs = self._transfer_function.transfer_function.coords[
            "output"
        ].data.tolist()
        if self.hz in outputs:
            if np.all(
                np.nan_to_num(
                    self._transfer_function.transfer_function.loc[
                        dict(
                            input=self._ch_input_dict["tipper"],
                            output=self._ch_output_dict["tipper"],
                        )
                    ].data
                )
                == 0
            ):
                return False
            return True
        return False

    @property
    def tipper(self) -> xr.DataArray | None:
        """

        Returns
        -------
        xr.DataArray | None
            The tipper data array or None if not set.

        """
        if self.has_tipper():
            t = self.dataset.transfer_function.loc[
                dict(
                    input=self._ch_input_dict["tipper"],
                    output=self._ch_output_dict["tipper"],
                )
            ]
            t.name = "tipper"

            for key, mkey in self._dataset_attr_dict.items():
                obj, attr = mkey.split(".", 1)
                value = getattr(self, obj).get_attr_from_name(attr)

                t.attrs[key] = value
            return t

    @tipper.setter
    def tipper(self, value: xr.DataArray | np.ndarray | list | tuple | None):
        """

        Parameters
        ----------
        value : xr.DataArray | np.ndarray | list | tuple | None
            The data array to set.
        """

        self._set_data_array(value, "tipper")

    @property
    def tipper_error(self) -> xr.DataArray | None:
        """

        Returns
        -------
        xr.DataArray | None
            The tipper error data array or None if not set.

        """

        if self.has_tipper():
            t = self.dataset.transfer_function_error.loc[
                dict(
                    input=self._ch_input_dict["tipper"],
                    output=self._ch_output_dict["tipper"],
                )
            ]
            t.name = "tipper_error"
            for key, mkey in self._dataset_attr_dict.items():
                obj, attr = mkey.split(".", 1)
                value = getattr(self, obj).get_attr_from_name(attr)

                t.attrs[key] = value
            return t

    @tipper_error.setter
    def tipper_error(self, value: xr.DataArray | np.ndarray | list | tuple | None):
        """

        Parameters
        ----------
        value : xr.DataArray | np.ndarray | list | tuple | None
            The data array to set.

        """
        self._set_data_array(value, "tipper_error")

    @property
    def tipper_model_error(self) -> xr.DataArray | None:
        """

        Returns
        -------
        xr.DataArray | None
            The tipper model error data array or None if not set.

        """
        if self.has_tipper():
            t = self.dataset.transfer_function_model_error.loc[
                dict(
                    input=self._ch_input_dict["tipper"],
                    output=self._ch_output_dict["tipper"],
                )
            ]
            t.name = "tipper_model_error"
            for key, mkey in self._dataset_attr_dict.items():
                obj, attr = mkey.split(".", 1)
                value = getattr(self, obj).get_attr_from_name(attr)

                t.attrs[key] = value
            return t

    @tipper_model_error.setter
    def tipper_model_error(
        self, value: xr.DataArray | np.ndarray | list | tuple | None
    ):
        """

        Parameters
        ----------
        value : xr.DataArray | np.ndarray | list | tuple | None
            The data array to set.

        """
        self._set_data_array(value, "tipper_model_error")

    def has_inverse_signal_power(self) -> bool:
        """
        Check to see if the transfer function is not 0 and has
        transfer function components

        Returns
        -------
        bool
            True if the inverse signal power is set and not zero, False otherwise.

        """

        if np.all(
            self._transfer_function.inverse_signal_power.loc[
                dict(
                    input=self._ch_input_dict["isp"],
                    output=self._ch_output_dict["isp"],
                )
            ].data
            == 0
        ):
            return False
        return True

    @property
    def inverse_signal_power(self) -> xr.DataArray | None:
        """
        Get the inverse signal power data array.

        Returns
        -------
        xr.DataArray | None
            The inverse signal power data array or None if not set.
        """
        if self.has_inverse_signal_power():
            ds = self.dataset.inverse_signal_power.loc[
                dict(
                    input=self._ch_input_dict["isp"],
                    output=self._ch_output_dict["isp"],
                )
            ]
            for key, mkey in self._dataset_attr_dict.items():
                obj, attr = mkey.split(".", 1)
                value = getattr(self, obj).get_attr_from_name(attr)

                ds.attrs[key] = value
            return ds
        return None

    @inverse_signal_power.setter
    def inverse_signal_power(
        self, value: xr.DataArray | np.ndarray | list | tuple | None
    ):
        """
        Set the inverse signal power

        Parameters
        ----------
        value : xr.DataArray | np.ndarray | list | tuple | None
            The data array to set.

        """
        self._set_data_array(value, "isp")
        if self.has_residual_covariance():
            self._compute_error_from_covariance()

    def has_residual_covariance(self) -> bool:
        """
        Check to see if the transfer function is not 0 and has
        transfer function components

        Returns
        -------
        bool
            True if the residual covariance is set and not zero, False otherwise.

        """

        if np.all(
            self._transfer_function.residual_covariance.loc[
                dict(
                    input=self._ch_input_dict["res"],
                    output=self._ch_output_dict["res"],
                )
            ].data
            == 0
        ):
            return False
        return True

    @property
    def residual_covariance(self) -> xr.DataArray | None:
        """
        Get the residual covariance data array.

        Returns
        -------
        xr.DataArray | None
            The residual covariance data array or None if not set.
        """
        if self.has_residual_covariance():
            ds = self.dataset.residual_covariance.loc[
                dict(
                    input=self._ch_input_dict["res"],
                    output=self._ch_output_dict["res"],
                )
            ]
            for key, mkey in self._dataset_attr_dict.items():
                obj, attr = mkey.split(".", 1)
                value = getattr(self, obj).get_attr_from_name(attr)

                ds.attrs[key] = value
            return ds
        return None

    @residual_covariance.setter
    def residual_covariance(
        self, value: xr.DataArray | np.ndarray | list | tuple | None
    ):
        """
        Set the residual covariance

        Parameters
        ----------
        value : xr.DataArray | np.ndarray | list | tuple | None
            The data array to set.

        """
        self._set_data_array(value, "res")
        if self.has_inverse_signal_power():
            self._compute_error_from_covariance()

    def _compute_impedance_error_from_covariance(self) -> None:
        """
        Compute transfer function errors from covariance matrices

        This will become important when writing edi files.

        Translated from code written by Ben Murphy.

        """
        sigma_e = self.residual_covariance.loc[
            dict(input=self.ex_ey, output=self.ex_ey)
        ]
        sigma_s = self.inverse_signal_power.loc[
            dict(input=self.hx_hy, output=self.hx_hy)
        ]

        z_err = np.zeros((self.period.size, 2, 2), dtype=float)
        z_err[:, 0, 0] = np.abs(
            sigma_e.loc[dict(input=[self.ex], output=[self.ex])].data.flatten()
            * sigma_s.loc[dict(input=[self.hx], output=[self.hx])].data.flatten()
        )
        z_err[:, 0, 1] = np.abs(
            sigma_e.loc[dict(input=[self.ex], output=[self.ex])].data.flatten()
            * sigma_s.loc[dict(input=[self.hy], output=[self.hy])].data.flatten()
        )
        z_err[:, 1, 0] = np.abs(
            sigma_e.loc[dict(input=[self.ey], output=[self.ey])].data.flatten()
            * sigma_s.loc[dict(input=[self.hx], output=[self.hx])].data.flatten()
        )
        z_err[:, 1, 1] = np.abs(
            sigma_e.loc[dict(input=[self.ey], output=[self.ey])].data.flatten()
            * sigma_s.loc[dict(input=[self.hy], output=[self.hy])].data.flatten()
        )

        z_err = np.sqrt(np.abs(z_err))

        self.dataset.transfer_function_error.loc[
            dict(input=self.hx_hy, output=self.ex_ey)
        ] = z_err

    def _compute_tipper_error_from_covariance(self) -> None:
        """
        Compute transfer function errors from covariance matrices

        This will become important when writing edi files.

        Translated from code written by Ben Murphy.

        """
        sigma_e = self.residual_covariance.loc[dict(input=[self.hz], output=[self.hz])]
        sigma_s = self.inverse_signal_power.loc[
            dict(input=self.hx_hy, output=self.hx_hy)
        ]

        t_err = np.zeros((self.period.size, 1, 2), dtype=float)
        t_err[:, 0, 0] = np.abs(
            sigma_e.loc[dict(input=[self.hz], output=[self.hz])].data.flatten()
            * sigma_s.loc[dict(input=[self.hx], output=[self.hx])].data.flatten()
        )
        t_err[:, 0, 1] = np.abs(
            sigma_e.loc[dict(input=[self.hz], output=[self.hz])].data.flatten()
            * sigma_s.loc[dict(input=[self.hy], output=[self.hy])].data.flatten()
        )

        t_err = np.sqrt(np.abs(t_err))

        self.dataset.transfer_function_error.loc[
            dict(input=self.hx_hy, output=[self.hz])
        ] = t_err

    def _compute_error_from_covariance(self) -> None:
        """
        convenience method to compute errors from covariance

        """
        self._compute_impedance_error_from_covariance()
        self._compute_tipper_error_from_covariance()

    @property
    def period(self) -> np.ndarray | None:
        """Periods of the transfer function"""
        return self.dataset.period.data

    @period.setter
    def period(self, value: np.ndarray | None):
        """
        Set the periods of the transfer function.

        Parameters
        ----------
        value : np.ndarray | None
            The new periods for the transfer function.

        Raises
        ------
        TFError
            If the new periods are not compatible with the existing ones.
        """
        if self.period is not None:
            if len(self.period) == 1 and (self.period == np.array([1])).all():
                self._transfer_function = self._initialize_transfer_function(
                    periods=value
                )
            elif len(value) != len(self.period):
                msg = (
                    f"New period size {value.size} is not the same size as "
                    f"old ones {self.period.size}, suggest creating a new "
                    "instance of TF."
                )
                logger.error(msg)
                raise TFError(msg)
            elif not (self.period == value).all():
                self.dataset["period"] = value
        else:
            self._transfer_function = self._initialize_transfer_function(periods=value)
        return

    @property
    def frequency(self) -> np.ndarray | None:
        if self.period is not None:
            return 1.0 / self.period
        return None

    @frequency.setter
    def frequency(self, value: np.ndarray | None):
        if value is not None:
            self.period = 1.0 / value

    @property
    def station(self) -> str:
        """station name"""
        return self.station_metadata.id

    @station.setter
    def station(self, station_name: str):
        """
        set station name
        """
        self.station_metadata.id = validate_name(station_name)
        if self.station_metadata.runs[0].id is None:
            r = self.station_metadata.runs.pop(None)
            r.id = f"{self.station_metadata.id}a"
            self.station_metadata.runs.append(r)

    @property
    def survey(self) -> str:
        """
        Survey ID
        """
        return self.survey_metadata.id

    @survey.setter
    def survey(self, survey_id: str):
        """
        set survey id
        """
        if survey_id is None:
            survey_id = "unkown_survey"
        self.survey_metadata.id = validate_name(survey_id)

    @property
    def tf_id(self) -> str:
        """transfer function id"""
        return self.station_metadata.transfer_function.id

    @tf_id.setter
    def tf_id(self, value: str):
        """set transfer function id"""
        self.station_metadata.transfer_function.id = validate_name(value)

    def to_ts_station_metadata(self) -> TSStation:
        """
        need a convinience function to translate to ts station metadata
        for MTH5

        """

        ts_station_metadata = TSStation()  # type: ignore
        for key, value in self.station_metadata.to_dict(single=True).items():
            if "transfer_function" in key:
                continue
            try:
                ts_station_metadata.update_attribute(key, value)
            except AttributeError:
                logger.debug(f"Attribute {key} could not be set.")
        return ts_station_metadata

    def from_ts_station_metadata(self, ts_station_metadata: TSStation):
        """
        need a convinience function to translate to ts station metadata
        for MTH5

        """

        for key, value in ts_station_metadata.to_dict(single=True).items():
            try:
                self.station_metadata.update_attribute(key, value)
            except AttributeError:
                continue

    def merge(
        self,
        other: "TF",
        period_min: float | None = None,
        period_max: float | None = None,
        inplace: bool = False,
    ) -> "TF | None":
        """
        metadata will be assumed to be from self.

        Merge transfer functions together. `other` can be another `TF` object
        or a tuple of `TF` objects

        to set bounds should be of the format

        [{"tf": tf_01, "period_min": .01, "period_max": 100},
         {"tf": tf_02, "period_min": 100, "period_max": 1000}]

        or to just use whats in the transfer function
        [tf_01, tf_02, ...]

        The bounds are inclusive, so if you want to merge at say 1 s choose
        the best one and set the other to a value lower or higher depending
        on the periods for that transfer function, for example

        [{"tf": tf_01, "period_min": .01, "period_max": 100},
         {"tf": tf_02, "period_min": 100.1, "period_max": 1000}]

        Parameters
        ----------
        other: TF, list of dicts, list of TF objects, dict
            other transfer functions to merge with
        period_min: float
            minimum period for the original TF
        period_max: float
            maximum period for the original TF
        inplace: bool
            whether to modify the original TF or return a new one

        Returns
        -------
        TF | None
            merged transfer function or None if inplace=True


        """

        def get_slice_dict(period_min: float, period_max: float) -> dict[str, slice]:
            """
            Get an the correct dictionary for slicing an xarray.

            Parameters
            ----------
            period_min: float
                minimum period
            period_max: float
                maximum period

            Returns
            -------
            dict[str, slice]
                variable to slice an xarray

            """
            return {"period": slice(period_min, period_max)}

        def sort_by_period(tf: xr.Dataset) -> xr.Dataset:
            """
            period needs to be monotonically increasing for slice to work.
            """
            return tf.sortby("period")

        def is_tf(item: xr.Dataset) -> xr.Dataset:
            """
            If the item is a transfer function return it sorted by period

            Parameters
            ----------
            item: transfer function
            type item: :class:`mt_metadata.transfer_function.core.TF`

            Returns
            -------
            sorted by period transfer function
            rtype: xarray.Dataset

            """
            return sort_by_period(item._transfer_function)

        def validate_dict(item: dict[str, Any]) -> dict[str, Any]:
            """
            Make sure input dictionary has proper keys.

            - **tf** :class:`mt_metadata.transfer_function.core.TF`
            - **period_min** minumum period (s)
            - **period_max** maximum period (s)

            Parameters
            ----------
            item: dict
                dictionary to slice a transfer function

            Returns
            -------
            validated dictionary
            rtype: dict

            Raises
            -------
            KeyError
                If keys are not what they should be

            """
            accepted_keys = sorted(["tf", "period_min", "period_max"])

            if accepted_keys != sorted(list(item.keys())):
                msg = f"Input dictionary must have keys of {accepted_keys}"
                logger.error(msg)
                raise KeyError(msg)
            return item

        def is_dict(item: dict) -> xr.Dataset:
            """
            If the item is a dictionary then be sure to sort the transfer
            function and then apply the slice.

            Parameters
            ----------
            item: dict
                dictionary with keys 'tf', 'period_min', 'period_max'

            Returns
            -------
            sliced transfer function
            rtype: xarray.Dataset

            Raises
            ------
            KeyError
                If keys are not what they should be

            """
            item = validate_dict(item)
            period_slice = get_slice_dict(item["period_min"], item["period_max"])
            item["tf"]._transfer_function = sort_by_period(
                item["tf"]._transfer_function
            )
            return get_slice(item["tf"], period_slice)

        def get_slice(tf, period_slice: dict[str, slice]) -> xr.Dataset | None:
            """
            Get slice of a transfer function most of the time we can use .loc
            but sometimes a key error occurs if the period index is not
            monotonic (which is should be now after using .sortby('period')),
            but leaving in place just in case.  If .loc does not work, then
            we can use .where(conditions) to slice the transfer function.

            Parameters
            ----------
            tf: xarray.Dataset
                The transfer function to slice.
            period_slice: dict[str, slice]
                The slice to apply to the period dimension.

            Returns
            -------
            xarray.Dataset
                The sliced transfer function.
            """
            try:
                return tf._transfer_function.loc[period_slice]

            except KeyError:
                if (
                    period_slice["period"].start is not None
                    and period_slice["period"].stop is not None
                ):
                    return tf._transfer_function.where(
                        (tf._transfer_function.period >= period_slice["period"].start)
                        & (tf._transfer_function.period <= period_slice["period"].stop),
                        drop=True,
                    )
                elif (
                    period_slice["period"].start is None
                    and period_slice["period"].stop is not None
                ):
                    return tf._transfer_function.where(
                        (tf._transfer_function.period <= period_slice["period"].stop),
                        drop=True,
                    )
                elif (
                    period_slice["period"].start is not None
                    and period_slice["period"].stop is None
                ):
                    return tf._transfer_function.where(
                        (tf._transfer_function.period >= period_slice["period"].start),
                        drop=True,
                    )

        period_slice_self = get_slice_dict(period_min, period_max)
        tf_list = [get_slice(self, period_slice_self)]
        if not isinstance(other, list):
            other = [other]

        for item in other:
            if isinstance(item, TF):
                tf_list.append(is_tf(item))
            elif isinstance(item, dict):
                tf_list.append(is_dict(item))
            else:
                msg = f"Type {type(item)} not supported"
                logger.error(msg)
                raise TypeError(msg)

        new_tf = xr.combine_by_coords(tf_list, combine_attrs="override")

        if inplace:
            self._transfer_function = new_tf
        else:
            return_tf = self.copy()
            return_tf._transfer_function = new_tf
            return return_tf

    def write(
        self,
        fn: str | Path | None = None,
        save_dir: str | Path | None = None,
        fn_basename: str | None = None,
        file_type: Literal["edi", "xml", "zmm", "avg", "j"] = "edi",
        **kwargs,
    ):
        """
        Write an mt file, the supported file types are EDI and XML.

        .. todo:: j-files

        Parameters
        ----------
        fn: str | Path | None
            Full path to file to save to.
        save_dir: str | Path | None
            Full path save directory.
        fn_basename: str | None
            Name of file with or without extension.
        file_type: Literal["edi", "xml", "zmm", "avg", "j"]
            Type of file to write.

        Optional Keyword Arguments
        ---------------------------
        longitude_format:  str
            whether to write longitude as longitude or LONG.
            options are 'longitude' or 'LONG', default 'longitude'

        longitude_format:  string
        latlon_format:  format of latitude and longitude in output edi,
                       degrees minutes seconds ('dms') or decimal
                       degrees ('dd')

        Returns
        -------
        str
            Full path to the written file.

        :Example: ::

            >>> tf_obj.write(file_type='xml')

        """

        if fn is not None:
            new_fn = Path(fn)
            self.save_dir = new_fn.parent
            fn_basename = new_fn.name
            file_type = new_fn.suffix.lower()[1:]
        if save_dir is not None:
            self.save_dir = Path(save_dir)
        if fn_basename is not None:
            fn_basename = Path(fn_basename)
            if fn_basename.suffix in ["", None]:
                fn_basename = fn_basename.with_name(f"{fn_basename.name}.{file_type}")
        if fn_basename is None:
            fn_basename = Path(f"{self.station}.{file_type}")
        if file_type is None:
            file_type = fn_basename.suffix.lower()[1:]
        if file_type not in self._read_write_dict.keys():
            msg = f"File type {file_type} not supported yet."
            logger.error(msg)
            raise TFError(msg)
        fn = self.save_dir.joinpath(fn_basename)

        if not isinstance(self._rotation_angle, (float, int)):
            if self._rotation_angle.size != self.period.size:
                self._rotation_angle = np.repeat(
                    self._rotation_angle.mean(), self.period.size
                )

        obj = self._read_write_dict[file_type]["write"]()
        obj._fn = fn
        obj.write(fn, **kwargs)

        return obj

    def write_tf_file(self, **kwargs):
        logger.error("'write_tf_file' has been deprecated use 'write()'")

    def read_tf_file(self, **kwargs):
        logger.error("'read_tf_file' has been deprecated use 'read()'")

    def read(
        self,
        fn: str | Path | None = None,
        file_type: str | None = None,
        get_elevation: bool = False,
        **kwargs,
    ):
        """

        Read an TF response file.

        .. note:: Currently only .edi, .xml, .j, .zmm/rr/ss, .avg
           files are supported

        Parameters
        ----------
        fn: str | Path | None
            Full path to input file.
        file_type: str | None
            Type of file to read. If None, automatically detects file type by
            the extension. Options are [edi | j | xml | avg | zmm | zrr | zss | ...]
        get_elevation: bool
            Whether to get elevation from US National Map DEM

        :Example: ::

            >>> import mt_metadata.transfer_functions import TF
            >>> tf_obj = TF()
            >>> tf_obj.read(fn=r"/home/mt/mt01.xml")

        .. note:: If your internet is slow try setting 'get_elevation' = False,
         It can get hooked in a slow loop and slow down reading.

        """
        if fn is not None:
            self.fn = fn
        self.save_dir = self.fn.parent
        if file_type is None:
            file_type = self.fn.suffix.lower()[1:]
        self._read_write_dict[file_type]["read"](
            self.fn, get_elevation=get_elevation, **kwargs
        )

        self.station_metadata.update_time_period()
        self.survey_metadata.update_bounding_box()
        self.survey_metadata.update_time_period()

    def to_edi(self) -> EDI:
        """

        Convert the TF object to a
        :class:`mt_metadata.transfer_functions.io.edi.EDI` object.  From there
        attributes of an EDI object can be manipulated previous to writing
        to a file.

        Returns
        -------
            EDI object

        >>> from mt_metadata.transfer_functions import TF
        >>> from mt_metadata import TF_XML
        >>> t = TF(TF_XML)
        >>> t.read()
        >>> edi_object = t.to_edi()
        >>> edi_object.Header.acqby = "me"
        >>> edi_object.write()

        """

        edi_obj = EDI()
        if self.has_impedance():
            edi_obj.z = self.impedance.data
            edi_obj.z_err = self.impedance_error.data
        if self.has_tipper():
            edi_obj.t = self.tipper.data
            edi_obj.t_err = self.tipper_error.data
        edi_obj.frequency = 1.0 / self.period

        if isinstance(self._rotation_angle, (int, float)):
            edi_obj.rotation_angle = np.repeat(self._rotation_angle, self.period.size)
        else:
            edi_obj.rotation_angle = self._rotation_angle

        # fill from survey metadata
        edi_obj.survey_metadata = self.survey_metadata

        # fill from station metadata
        edi_obj.station_metadata = self.station_metadata

        # input data section
        edi_obj.Data.data_type = self.station_metadata.data_type
        edi_obj.Data.nfreq = self.period.size
        edi_obj.Data.sectid = self.station
        edi_obj.Data.nchan = len(edi_obj.Measurement.channel_ids.keys())
        edi_obj.Data.maxblks = 999

        for comp in ["ex", "ey", "hx", "hy", "hz", "rrhx", "rrhy"]:
            if hasattr(edi_obj.Measurement, f"meas_{comp}"):
                setattr(
                    edi_obj.Data,
                    comp,
                    getattr(edi_obj.Measurement, f"meas_{comp}").id,
                )
        edi_obj.Data.read_data(edi_obj.Data.write_data())

        edi_obj.Measurement.read_measurement(edi_obj.Measurement.write_measurement())

        return edi_obj

    def from_edi(
        self, edi_obj: str | Path | EDI, get_elevation: bool = False, **kwargs
    ) -> None:
        """
        Read in an EDI file or a
        :class:`mt_metadata.transfer_functions.io.edi.EDI` object

        Parameters
        ----------

        edi_obj: str | Path | EDI
           Path to EDI file or EDI object
           If a path is provided, the file will be read from disk.
           If an EDI object is provided, it will be used directly.
        get_elevation: bool
           Try to get elevation from US National Map,
           defaults to False

        Raises
        ------
        TypeError
            If input is incorrect

        """

        if isinstance(edi_obj, (str, Path)):
            self._fn = Path(edi_obj)
            edi_obj = EDI(**kwargs)
            edi_obj.read(self._fn, get_elevation=get_elevation)
        if not isinstance(edi_obj, EDI):
            raise TypeError(f"Input must be a EDI object not {type(edi_obj)}")
        if edi_obj.tf is not None and edi_obj.tf.shape[1:] == (3, 2):
            k_dict = OrderedDict(
                {
                    "period": "period",
                    "transfer_function": "tf",
                    "inverse_signal_power": "signal_inverse_power",
                    "residual_covariance": "residual_covariance",
                    "transfer_function_error": "tf_err",
                    "survey_metadata": "survey_metadata",
                    # "station_metadata": "station_metadata",
                    "_rotation_angle": "rotation_angle",
                }
            )
        else:
            k_dict = OrderedDict(
                {
                    "period": "period",
                    "impedance": "z",
                    "impedance_error": "z_err",
                    "tipper": "t",
                    "tipper_error": "t_err",
                    "survey_metadata": "survey_metadata",
                    # "station_metadata": "station_metadata",
                    "_rotation_angle": "rotation_angle",
                }
            )
        for tf_key, edi_key in k_dict.items():
            setattr(self, tf_key, getattr(edi_obj, edi_key))

    def to_emtfxml(self) -> EMTFXML:
        """
        Convert TF to a :class:`mt_metadata.transfer_function.io.emtfxml.EMTFXML`
        object.

        Returns
        -------
        :return: EMTFXML object
        :rtype: :class:`mt_metadata.transfer_function.io.emtfxml.EMTFXML`

        >>> from mt_metadata.transfer_functions import TF
        >>> from mt_metadata import TF_XML
        >>> t = TF(TF_XML)
        >>> t.read()
        >>> xml_object = t.to_emtfxml()
        >>> xml_object.site.country = "Here"
        >>> xml_object.write()

        """

        emtf = EMTFXML()
        emtf.survey_metadata = self.survey_metadata
        emtf.station_metadata = self.station_metadata

        if emtf.description is None:
            emtf.description = "Magnetotelluric Transfer Functions"
        if emtf.product_id is None:
            emtf.product_id = (
                f"{emtf.survey_metadata.project}."
                f"{emtf.station_metadata.id}."
                f"{emtf.station_metadata.time_period.start.year}"
            )
        tags = []

        emtf.data.period = self.period

        if self.has_impedance():
            tags += ["impedance"]
            emtf.data.z = self.impedance.data
            emtf.data.z_var = self.impedance_error.data**2
        if self.has_residual_covariance() and self.has_inverse_signal_power():
            emtf.data.z_invsigcov = self.inverse_signal_power.loc[
                dict(input=self.hx_hy, output=self.hx_hy)
            ].data
            emtf.data.z_residcov = self.residual_covariance.loc[
                dict(input=self.ex_ey, output=self.ex_ey)
            ].data
        if self.has_tipper():
            tags += ["tipper"]
            emtf.data.t = self.tipper.data
            emtf.data.t_var = self.tipper_error.data**2
        if self.has_residual_covariance() and self.has_inverse_signal_power():
            emtf.data.t_invsigcov = self.inverse_signal_power.loc[
                dict(input=self.hx_hy, output=self.hx_hy)
            ].data
            emtf.data.t_residcov = self.residual_covariance.loc[
                dict(
                    input=[self.channel_nomenclature["hz"]],
                    output=[self.channel_nomenclature["hz"]],
                )
            ].data
        emtf.tags = ", ".join(tags)
        emtf.period_range.min = emtf.data.period.min()
        emtf.period_range.max = emtf.data.period.max()

        emtf._get_data_types()
        emtf._get_statistical_estimates()
        # Update site layout after data is set to populate channels correctly
        emtf._update_site_layout()

        return emtf

    def from_emtfxml(
        self, emtfxml_obj: str | Path | EMTFXML, get_elevation: bool = False, **kwargs
    ) -> None:
        """

        Parameters
        ----------
        emtfxml_obj: str | Path | EMTFXML
            The input object to convert from.
        get_elevation: bool
            Try to get elevation from US National Map, defaults to True.

        Returns
        -------
        None

        """

        if isinstance(emtfxml_obj, (str, Path)):
            self._fn = Path(emtfxml_obj)
            emtfxml_obj = EMTFXML(**kwargs)
            emtfxml_obj.read(self._fn, get_elevation=get_elevation)
        if not isinstance(emtfxml_obj, EMTFXML):
            raise TypeError(f"Input must be a EMTFXML object not {type(emtfxml_obj)}")
        self.survey_metadata = emtfxml_obj.survey_metadata
        self.station_metadata = self.survey_metadata.stations[0]

        self.period = emtfxml_obj.data.period
        self.impedance = emtfxml_obj.data.z
        # Handle negative or invalid values in z_var before taking sqrt
        z_var = emtfxml_obj.data.z_var
        with np.errstate(invalid="ignore"):
            self.impedance_error = np.sqrt(np.where(z_var >= 0, z_var, np.nan))
        self._transfer_function.inverse_signal_power.loc[
            dict(input=["hx", "hy"], output=["hx", "hy"])
        ] = emtfxml_obj.data.z_invsigcov
        self._transfer_function.residual_covariance.loc[
            dict(input=["ex", "ey"], output=["ex", "ey"])
        ] = emtfxml_obj.data.z_residcov

        self.tipper = emtfxml_obj.data.t
        self.tipper_error = np.sqrt(emtfxml_obj.data.t_var)
        self._transfer_function.inverse_signal_power.loc[
            dict(input=["hx", "hy"], output=["hx", "hy"])
        ] = emtfxml_obj.data.t_invsigcov
        self._transfer_function.residual_covariance.loc[
            dict(input=["hz"], output=["hz"])
        ] = emtfxml_obj.data.t_residcov

    def to_jfile(self) -> None:
        """

        Translate TF object ot JFile object.

        .. note:: Not Implemented yet

        :return: JFile object
        :rtype: :class:`mt_metadata.transfer_functions.io.jfile.JFile`

        """

        raise NotImplementedError("to_jfile not implemented yet.")

    def from_jfile(
        self, j_obj: str | Path | JFile, get_elevation: bool = False, **kwargs
    ) -> None:
        """

        Parameters
        ----------
        jfile_obj: str | Path | JFile
            The input object to convert from.
        get_elevation: bool
            Try to get elevation from US National Map, defaults to True.

        Returns
        -------
        None

        """
        if isinstance(j_obj, (str, Path)):
            self._fn = Path(j_obj)
            j_obj = JFile(**kwargs)
            j_obj.read(self._fn, get_elevation=get_elevation)
        if not isinstance(j_obj, JFile):
            raise TypeError(f"Input must be a JFile object not {type(j_obj)}")
        k_dict = OrderedDict(
            {
                "period": "periods",
                "impedance": "z",
                "impedance_error": "z_err",
                "tipper": "t",
                "tipper_error": "t_err",
                "survey_metadata": "survey_metadata",
                # "station_metadata": "station_metadata",
            }
        )

        for tf_key, j_key in k_dict.items():
            setattr(self, tf_key, getattr(j_obj, j_key))

    def make_zmm_run(self, zmm_obj: ZMM, number_dict: dict) -> Run:
        """
        Helper function to provide a run for a zmm object to aid writing z-file

        Parameters
        ----------
        zmm_obj: ZMM
            A ZMM that will be written to file, that needs a run associated.

        number_dict: dict
            Mapping between hexy keys and integers, needed for emtf z-files,
            e.g. {"hx": 1, "hy": 2, "hz": 3, "ex": 4, "ey": 5}
        :type number_dict: dictionary

        :return: run
        :rtype: :class:` mt_metadata.timeseries.run.Run`
        """
        run = Run()
        for ch, ch_num in number_dict.items():
            c = ZChannel()
            c.channel = ch
            c.number = ch_num
            setattr(zmm_obj, c.channel, c)
            if ch in ["ex", "ey"]:
                rc = Electric(component=ch, channel_number=ch_num)
                run.add_channel(rc)
            elif ch in ["hx", "hy", "hz"]:
                rc = Magnetic(component=ch, channel_number=ch_num)
                run.add_channel(rc)
        return run

    def to_zmm(self) -> ZMM:
        """

        Translate TF object to ZMM object.

        :return: ZMM object
        :rtype: :class:`mt_metadata.transfer_function.io.zfiles.ZMM`

        >>> from mt_metadata.transfer_functions import TF
        >>> from mt_metadata import TF_XML
        >>> t = TF(TF_XML)
        >>> t.read()
        >>> zmm_object = t.to_zmm()
        >>> zmm_object.processing_type = "new and fancy"
        >>> zmm_object.write()

        """
        zmm_kwargs = {}
        zmm_kwargs["channel_nomenclature"] = self.channel_nomenclature
        zmm_kwargs["inverse_channel_nomenclature"] = self.inverse_channel_nomenclature
        if hasattr(self, "decimation_dict"):
            zmm_kwargs["decimation_dict"] = self.decimation_dict
        zmm_obj = ZMM(**zmm_kwargs)

        zmm_obj.dataset = self.dataset
        zmm_obj.station_metadata = self.station_metadata

        # need to set the channel numbers according to the z-file format
        # with input channels (h's) and output channels (hz, e's).
        if self.has_tipper():
            if self.has_impedance():
                zmm_obj.num_channels = 5
                number_dict = {"hx": 1, "hy": 2, "hz": 3, "ex": 4, "ey": 5}
            else:
                zmm_obj.num_channels = 3
                number_dict = {"hx": 1, "hy": 2, "hz": 3}
        else:
            if self.has_impedance():
                zmm_obj.num_channels = 4
                number_dict = {"hx": 1, "hy": 2, "ex": 3, "ey": 4}
        if len(self.station_metadata.runs) == 0:
            run = self.make_zmm_run(zmm_obj, number_dict)
            self.station_metadata.add_run(run)
        elif len(self.station_metadata.runs[0].channels_recorded_all) == 0:
            # avoid the default metadata getting interpretted as a real metadata object
            # Overwrite this "spoof" run with a run that has recorded channels
            if len(self.station_metadata.runs[0].channels_recorded_all) == 0:
                run = self.make_zmm_run(zmm_obj, number_dict)
                self.station_metadata.runs[0] = run
        else:
            for comp in self.station_metadata.runs[0].channels_recorded_all:
                if "rr" in comp:
                    continue
                ch = self.station_metadata.runs[0].get_channel(comp)
                ch.component = self.inverse_channel_nomenclature[comp]
                c = ZChannel()
                c.from_dict(ch.to_dict(single=True))
                ch.component = comp
                try:
                    c.number = number_dict[c.channel]
                    setattr(zmm_obj, c.channel, c)
                except KeyError:
                    logger.debug(f"Could not find channel {c.channel}")
        zmm_obj.survey_metadata.update(self.survey_metadata)
        zmm_obj.num_freq = self.period.size

        return zmm_obj

    def from_zmm(
        self, zmm_obj: str | Path | ZMM, get_elevation: bool = False, **kwargs
    ) -> None:
        """

        Parameters
        ----------
        zmm_obj: str | Path | ZMM
            Path to .zmm file or ZMM object
        get_elevation: bool
            Try to get elevation from US National Map, defaults to True
        kwargs: dict
            Keyword arguments for ZMM object
            Can include channel_nomenclature, inverse_channel_nomenclature
            rotate_to_measurement_coordinates : bool, optional
                If True, rotate impedance to the provided reference frame of the
                channel metadata, by default True
            use_declination : bool, optional
                If True, rotate impedance to true north using declination value in metadata,
                by default False

        """

        if isinstance(zmm_obj, (str, Path)):
            self._fn = Path(zmm_obj)
            zmm_obj = ZMM(**kwargs)
            zmm_obj.read(
                self._fn,
                get_elevation=get_elevation,
                rotate_to_measurement_coordinates=kwargs.get(
                    "rotate_to_measurement_coordinates", True
                ),
                use_declination=kwargs.get("use_declination", False),
            )
        if not isinstance(zmm_obj, ZMM):
            raise TypeError(f"Input must be a ZMM object not {type(zmm_obj)}")
        self.decimation_dict = zmm_obj.decimation_dict
        k_dict = OrderedDict(
            {
                "survey_metadata": "survey_metadata",
                "station_metadata": "station_metadata",
                "period": "periods",
            }
        )

        for tf_key, j_key in k_dict.items():
            setattr(self, tf_key, getattr(zmm_obj, j_key))

        self._transfer_function["transfer_function"].loc[
            dict(input=zmm_obj.input_channels, output=zmm_obj.output_channels)
        ] = zmm_obj.dataset.transfer_function.loc[
            dict(input=zmm_obj.input_channels, output=zmm_obj.output_channels)
        ]

        self._transfer_function["transfer_function_error"].loc[
            dict(input=zmm_obj.input_channels, output=zmm_obj.output_channels)
        ] = zmm_obj.dataset.transfer_function_error.loc[
            dict(input=zmm_obj.input_channels, output=zmm_obj.output_channels)
        ]

        self._transfer_function["inverse_signal_power"].loc[
            dict(input=zmm_obj.input_channels, output=zmm_obj.input_channels)
        ] = zmm_obj.dataset.inverse_signal_power.loc[
            dict(input=zmm_obj.input_channels, output=zmm_obj.input_channels)
        ]

        self._transfer_function["residual_covariance"].loc[
            dict(input=zmm_obj.output_channels, output=zmm_obj.output_channels)
        ] = zmm_obj.dataset.residual_covariance.loc[
            dict(input=zmm_obj.output_channels, output=zmm_obj.output_channels)
        ]

        self._compute_error_from_covariance()
        if kwargs.get("use_declination", False):
            self._rotation_angle = -1 * zmm_obj.declination

    def to_zrr(self) -> ZMM:
        """

        Translate TF object to ZMM object.

        :return: ZMM object
        :rtype: :class:`mt_metadata.transfer_function.io.zfiles.ZMM`

        >>> from mt_metadata.transfer_functions import TF
        >>> from mt_metadata import TF_XML
        >>> t = TF(TF_XML)
        >>> t.read()
        >>> zmm_object = t.to_zmm()
        >>> zmm_object.processing_type = "new and fancy"
        >>> zmm_object.write()

        """
        return self.to_zmm()

    def from_zrr(
        self, zrr_obj: str | Path | ZMM, get_elevation: bool = False, **kwargs
    ) -> None:
        """
        Parameters
        ----------
        zmm_obj: str | Path | ZMM
            Path to .zmm file or ZMM object
        get_elevation: bool
            Try to get elevation from US National Map, defaults to True
        kwargs: dict
            Keyword arguments for ZMM object

        """

        self.from_zmm(zrr_obj, get_elevation=get_elevation, **kwargs)

    def to_zss(self) -> ZMM:
        """

        Translate TF object to ZMM object.

        :return: ZMM object
        :rtype: :class:`mt_metadata.transfer_function.io.zfiles.ZMM`

        >>> from mt_metadata.transfer_functions import TF
        >>> from mt_metadata import TF_XML
        >>> t = TF(TF_XML)
        >>> t.read()
        >>> zmm_object = t.to_zmm()
        >>> zmm_object.processing_type = "new and fancy"
        >>> zmm_object.write()

        """
        return self.to_zmm()

    def from_zss(
        self, zss_obj: str | Path | ZMM, get_elevation: bool = False, **kwargs
    ) -> None:
        """
        Parameters
        ----------
        zss_obj: str | Path | ZMM
            Path to .zss file or ZMM object
        get_elevation: bool
            Try to get elevation from US National Map, defaults to True

        """

        self.from_zmm(zss_obj, get_elevation=get_elevation, **kwargs)

    def to_avg(self) -> ZongeMTAvg:
        """

        Translate TF object to ZongeMTAvg object.

        .. note:: Not Implemented yet

        :return: ZongeMTAvg object
        :rtype: :class:`mt_metadata.transfer_function.io.zonge.ZongeMTAvg`


        """

        avg_obj = ZongeMTAvg()
        avg_obj.frequency = self.frequency
        avg_obj.z = self.impedance
        avg_obj.z_err = self.impedance_error
        avg_obj.t = self.tipper
        avg_obj.t_err = self.tipper_error

        logger.warning("Metadata is not properly set for a AVG file yet.")
        return avg_obj

    def from_avg(
        self, avg_obj: str | Path | ZongeMTAvg, get_elevation: bool = False, **kwargs
    ) -> None:
        """

        Parameters
        ----------
        avg_obj: str | Path | ZongeMTAvg
            Path to .avg file or ZongeMTAvg object
        get_elevation: bool
            Try to get elevation from US National Map,   defaults to True

        """
        if isinstance(avg_obj, (str, Path)):
            self._fn = Path(avg_obj)
            avg_obj = ZongeMTAvg(**kwargs)
            avg_obj.read(self._fn, get_elevation=get_elevation)
        if not isinstance(avg_obj, ZongeMTAvg):
            raise TypeError(f"Input must be a ZMM object not {type(avg_obj)}")
        self.survey_metadata = avg_obj.survey_metadata

        self.period = 1.0 / avg_obj.frequency
        self.impedance = avg_obj.z
        self.impedance_error = avg_obj.z_err

        if avg_obj.t is not None:
            self.tipper = avg_obj.t
            self.tipper_error = avg_obj.t_err


# ==============================================================================
#             Error
# ==============================================================================


class TFError(Exception):
    pass
