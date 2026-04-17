# -*- coding: utf-8 -*-
"""
.. module:: EDI
   :synopsis: Deal with EDI files.  The Edi class can read and write an .edi
             file, the 'standard format' of magnetotellurics.  Each section
             of the .edi file is given its own class, so the elements of each
             section are attributes for easy access.

.. moduleauthor:: Jared Peacock <jpeacock@usgs.gov>

Updated 2021 to used mt_metadata type metadata and how spectra are read.

"""

# ==============================================================================
#  Imports
# ==============================================================================
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger
from pydantic import HttpUrl

from mt_metadata import __version__, NULL_VALUES
from mt_metadata.timeseries import Electric, Magnetic, Run, Survey
from mt_metadata.transfer_functions import tf
from mt_metadata.transfer_functions.io.edi.metadata import (
    DataSection,
    DefineMeasurement,
    Header,
    Information,
)
from mt_metadata.transfer_functions.io.tools import (
    _validate_edi_lines,
    _validate_str_with_equals,
    get_nm_elev,
    index_locator,
)
from mt_metadata.utils.validators import validate_doi, validate_station_name


# ==============================================================================
# EDI Class
# ==============================================================================
class EDI:
    """
    This class is for .edi files, mainly reading and writing.  Has been tested
    on Winglink and Phoenix output .edi's, which are meant to follow the
    archaic EDI format put forward by SEG. Can read impedance, Tipper and/or
    spectra data.

    The Edi class contains a class for each major section of the .edi file.

    Frequency and components are ordered from highest to lowest frequency.

    :param fn: full path to .edi file to be read in.
                  *default* is None. If an .edi file is input, it is
                  automatically read in and attributes of Edi are filled
    :type fn: string or :class:`pathlib.Path`

    :Change Latitude: ::

        >>> from mt_metadata.transfer_functions.io.edi import EDI
        >>> edi_obj = EDI(fn=r"/home/mt/mt01.edi")
        >>> # change the latitude
        >>> edi_obj.lat = 45.7869
        >>> new_edi_fn = edi_obj.write()
    """

    def __init__(self, fn=None, **kwargs):
        self.logger = logger
        self._fn = None
        self._edi_lines = []

        self.Header = Header()
        self.Info = Information()
        self.Measurement = DefineMeasurement()
        self.Data = DataSection()

        self.z = None
        self.z_err = None
        self.t = None
        self.t_err = None
        self.frequency = None
        self.rotation_angle = None
        self.residual_covariance = None
        self.signal_inverse_power = None
        self.tf = None
        self.tf_err = None

        self._z_labels = [
            ["zxxr", "zxxi", "zxx.var"],
            ["zxyr", "zxyi", "zxy.var"],
            ["zyxr", "zyxi", "zyx.var"],
            ["zyyr", "zyyi", "zyy.var"],
        ]

        self._t_labels = [
            ["txr.exp", "txi.exp", "txvar.exp"],
            ["tyr.exp", "tyi.exp", "tyvar.exp"],
        ]

        self._accepted_keys = [
            "freq",
            "zxxr",
            "zxxi",
            "zxyr",
            "zxyi",
            "zyxr",
            "zyxi",
            "zyyr",
            "zyyi",
            "zxx.var",
            "zxy.var",
            "zyx.var",
            "zyy.var",
            "txr.exp",
            "txi.exp",
            "tyr.exp",
            "tyi.exp",
            "txvar.exp",
            "tyvar.exp",
            "rhoxx",
            "rhoxy",
            "rhoyx",
            "rhoyy",
            "rhoxx.err",
            "rhoxy.err",
            "rhoyx.err",
            "rhoyy.err",
            "phsxx",
            "phsxy",
            "phsyx",
            "phsyy",
            "phsxx.err",
            "phsxy.err",
            "phsyx.err",
            "phsyy.err",
            "zrot",
            "rhorot",
            "trot",
        ]

        self._channel_skip_list = [
            "filter.name",
            "filter.applied",
            "time_period.start",
            "time_period.end",
            "location.elevation",
            "location.latitude",
            "location.longitude",
            "location.x",
            "location.y",
            "location.z",
            "positive.latitude",
            "positive.longitude",
            "positive.elevation",
            "positive.x",
            "positive.x2",
            "positive.y",
            "positive.y2",
            "positive.z",
            "positive.z2",
            "negative.latitude",
            "negative.longitude",
            "negative.elevation",
            "negative.x",
            "negative.x2",
            "negative.y",
            "negative.y2",
            "negative.z",
            "negative.z2",
            "sample_rate",
            "data_quality.rating.value",
            "data_quality.flag",
        ]

        self._index_dict = {
            "zxx": {"ii": 0, "jj": 0, "obj": "z", "err_obj": "z_err"},
            "zxy": {"ii": 0, "jj": 1, "obj": "z", "err_obj": "z_err"},
            "zyx": {"ii": 1, "jj": 0, "obj": "z", "err_obj": "z_err"},
            "zyy": {"ii": 1, "jj": 1, "obj": "z", "err_obj": "z_err"},
            "tx": {"ii": 0, "jj": 0, "obj": "t", "err_obj": "t_err"},
            "ty": {"ii": 0, "jj": 1, "obj": "t", "err_obj": "t_err"},
            "rhoxx": {"ii": 0, "jj": 0, "obj": "z", "err_obj": "z_err"},
            "rhoxy": {"ii": 0, "jj": 1, "obj": "z", "err_obj": "z_err"},
            "rhoyx": {"ii": 1, "jj": 0, "obj": "z", "err_obj": "z_err"},
            "rhoyy": {"ii": 1, "jj": 1, "obj": "z", "err_obj": "z_err"},
        }

        self._data_header_str = ">!****{0}****!\n"

        self._num_format = " 15.6e"
        self._block_len = 6

        self.fn = fn

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self) -> str:
        lines = [f"Station: {self.station}", "-" * 50]
        lines.append(f"\tSurvey:        {self.survey_metadata.id}")
        lines.append(f"\tProject:       {self.survey_metadata.project}")
        lines.append(f"\tAcquired by:   {self.station_metadata.acquired_by.author}")
        lines.append(f"\tAcquired date: {self.station_metadata.time_period.start}")
        lines.append(f"\tLatitude:      {self.station_metadata.location.latitude:.3f}")
        lines.append(f"\tLongitude:     {self.station_metadata.location.longitude:.3f}")
        lines.append(f"\tElevation:     {self.station_metadata.location.elevation:.3f}")
        if self.z is not None:
            lines.append("\tImpedance:     True")
        else:
            lines.append("\tImpedance:     False")
        if self.t is not None:
            lines.append("\tTipper:        True")
        else:
            lines.append("\tTipper:        False")
        if self.frequency is not None:
            lines.append(f"\tNumber of periods: {self.frequency.size}")
            lines.append(
                f"\t\tPeriod Range:   {1./self.frequency.max():.5E} -- {1./self.frequency.min():.5E} s"
            )
            lines.append(
                f"\t\tFrequency Range {self.frequency.min():.5E} -- {self.frequency.max():.5E} s"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def fn(self):
        return self._fn

    @fn.setter
    def fn(self, fn: str | Path | None):
        if fn is not None:
            self._fn = Path(fn)
            if self._fn.exists():
                self.read()

    @property
    def period(self) -> np.typing.NDArray | None:
        if self.frequency is not None:
            return 1.0 / self.frequency
        return None

    def _assert_descending_frequency(self) -> None:
        """
        Assert that the transfer function is ordered from high frequency to low
        frequency.

        """
        if self.frequency is not None:
            if self.frequency[0] < self.frequency[1]:
                self.logger.debug(
                    "Ordered arrays to be arranged from high to low frequency"
                )
                self.frequency = self.frequency[::-1]
                if self.z is not None:
                    self.z = self.z[::-1]
                if self.z_err is not None:
                    self.z_err = self.z_err[::-1]
                if self.t is not None:
                    self.t = self.t[::-1]
                if self.t_err is not None:
                    self.t_err = self.t_err[::-1]

    def read(self, fn: str | Path | None = None, get_elevation: bool = False) -> None:
        """
        Read in an edi file and fill attributes of each section's classes.
        Including:

            * Header
            * Info
            * Measurement
            * Data
            * z, z_err
            * t, t_err

            .. note:: Automatically detects if data is in spectra format.  All
                  data read in is converted to impedance and Tipper.


        :param fn: full path to .edi file to be read in
                       *default* is None
        :type fn: string

        :Example: ::

            >>> from mt_metadata.transfer_functions.io.edi import EDI
            >>> edi_obj = EDI
            >>> edi_obj.read(fn=r"/home/mt/mt01.edi")

        """

        if fn is not None:
            self._fn = Path(fn)
        if self.fn is None:
            msg = "Must input EDI file to read"
            self.logger.error(msg)
            raise IOError(msg)
        if not self.fn.exists():
            msg = f"Cannot find EDI file: {self.fn}"
            self.logger.error(msg)
            raise IOError(msg)
        self._edi_lines = _validate_edi_lines(
            self.fn.read_text(encoding="utf-8", errors="replace").splitlines()
        )
        self.Header.read_header(self._edi_lines)
        self.Info.read_info(self._edi_lines)
        self.Measurement.read_measurement(self._edi_lines)
        self.Data.read_data(self._edi_lines)
        self.Data.match_channels(self.Measurement.channel_ids)

        self._read_data()

        if self.Header.latitude in [None, 0.0]:
            self.Header.latitude = self.Measurement.reflat
            self.logger.debug(f"Got latitude from reflat for {self.Header.dataid}")
        if self.Header.longitude in [None, 0.0]:
            self.Header.longitude = self.Measurement.reflon
            self.logger.debug(f"Got longitude from reflon for {self.Header.dataid}")
        if self.Header.elevation in [None, 0.0]:
            self.Header.elevation = self.Measurement.refelev
            self.logger.debug(f"Got elevation from refelev for {self.Header.dataid}")

        if self.elev in [0, None] and get_elevation:
            if self.lat != 0 and self.lon != 0:
                self.elev = get_nm_elev(self.lat, self.lon)

    def _read_data(self) -> None:
        """
        Read either impedance or spectra data depending on what the type is
        in the data section.
        """

        lines = self._edi_lines[self.Data._line_num :]

        if self.Data._data_type_in == "spectra":
            self.logger.debug("Converting Spectra to Impedance and Tipper")
            self.logger.debug(
                "Check to make sure input channel list is correct if the data looks incorrect"
            )
            if self.Data.nchan == 5:
                c_list = ["hx", "hy", "hz", "ex", "ey"]
            elif self.Data.nchan == 4:
                c_list = ["hx", "hy", "ex", "ey"]
            elif self.Data.nchan == 6:
                c_list = ["hx", "hy", "ex", "ey", "rhx", "rhy"]
            elif self.Data.nchan == 7:
                c_list = ["hx", "hy", "hz", "ex", "ey", "rhx", "rhy"]
            self._read_spectra(lines, comp_list=c_list)
        elif self.Data._data_type_in == "z":
            self._read_mt(lines)

    def _read_mt(self, data_lines: list[str]) -> None:
        """
        Read in impedance and tipper data

        :param data_lines: list of data lines from the edi file
        :type data_lines: list
        """

        data_dict = {}
        data_find = False
        for line in data_lines:
            line = line.strip()
            if ">" in line and "!" not in line:
                line_list = line[1:].strip().split()
                if len(line_list) == 0:
                    continue
                key = line_list[0].lower()
                if key in self._accepted_keys:
                    data_find = True
                    data_dict[key] = []
                else:
                    data_find = False
            elif data_find and ">" not in line and "!" not in line:
                d_lines = line.strip().split()
                for ii, dd in enumerate(d_lines):
                    # check for empty values and set them to 0, check for any
                    # other characters sometimes there are ****** for a null
                    # component
                    try:
                        d_lines[ii] = float(dd)
                        if d_lines[ii] == self.Header.empty:
                            d_lines[ii] = 0.0
                    except ValueError:
                        d_lines[ii] = 0.0
                data_dict[key] += d_lines
        # put everything into arrays
        for key, k_list in data_dict.items():
            data_dict[key] = np.array(k_list)
        # fill useful arrays
        self.frequency = data_dict["freq"]
        self.z = np.zeros((self.frequency.size, 2, 2), dtype=complex)
        self.z_err = np.zeros((self.frequency.size, 2, 2), dtype=float)
        # fill tipper data if there it exists
        self.t = np.zeros((self.frequency.size, 1, 2), dtype=complex)
        self.t_err = np.zeros((self.frequency.size, 1, 2), dtype=float)
        self.data_dict = data_dict

        # fill tensors
        for key in sorted(self._index_dict.keys(), reverse=True):
            index = self._index_dict[key]
            ii = index["ii"]
            jj = index["jj"]
            obj = getattr(self, index["obj"])
            error_obj = getattr(self, index["err_obj"])
            try:
                if key.startswith("z"):
                    obj[:, ii, jj] = data_dict[f"{key}r"] + data_dict[f"{key}i"] * 1j
                    try:
                        error_key = [
                            k for k in data_dict.keys() if key in k and "var" in k
                        ][0]
                        error_obj[:, ii, jj] = np.abs(data_dict[error_key]) ** 0.5
                    except IndexError:
                        self.logger.debug(f"Could not find error information for {key}")
                elif key.startswith("t"):
                    obj[:, ii, jj] = (
                        data_dict[f"{key}r.exp"] + data_dict[f"{key}i.exp"] * 1j
                    )
                    try:
                        error_key = [
                            k for k in data_dict.keys() if key in k and "var" in k
                        ][0]
                        error_obj[:, ii, jj] = np.abs(data_dict[error_key]) ** 0.5
                    except IndexError:
                        self.logger.debug(f"Could not find error information for {key}")
                elif key.startswith("r") or key.startswith("p"):
                    self.logger.debug("Reading RHO and PHS to compute impedance")
                    if (self.z[:, ii, jj] == 0).all():
                        phase = data_dict[f"phs{key[-2:]}"]
                        z_real = np.sqrt(
                            (5 * self.frequency * data_dict[key])
                            / (np.tan(np.deg2rad(phase)) ** 2 + 1)
                        )
                        z_imag = (np.tan(np.deg2rad(phase))) * z_real
                        if ii == 1 and jj == 0:
                            if phase.mean() < 90 and phase.mean() > 0:
                                obj[:, ii, jj] = -1 * (z_real + 1j * z_imag)
                        else:
                            obj[:, ii, jj] = z_real + 1j * z_imag
                        error_obj[:, ii, jj] = np.deg2rad(
                            data_dict[f"phs{key[-2:]}.err"]
                        ) * np.sqrt(data_dict[key] * (self.frequency * 5))
            except KeyError as error:
                self.logger.debug(error)
        # check for order of frequency, we want high togit  low
        self._assert_descending_frequency()

        try:
            self.rotation_angle = np.array(data_dict["zrot"])
        except KeyError:
            try:
                self.rotation_angle = np.array(data_dict["rhorot"])
            except KeyError:
                self.rotation_angle = np.zeros_like(self.frequency)

    def _read_spectra(
        self,
        data_lines: list[str],
        comp_list: list[str] = ["hx", "hy", "hz", "ex", "ey", "rhx", "rhy"],
    ) -> None:
        """
        Read in spectra data and convert to impedance and Tipper.

        Translated from A. Kelbert's EMTF fortran module

        :param data_lines: list of lines from edi file
        :type data_lines: list

        :param comp_list: list of components that correspond to the columns
                          of the spectra data.
        :type comp_list: list
        """

        data_dict = {}
        avgt_dict = {}
        data_find = False
        for line in data_lines:
            if line.lower().find(">spectra") == 0 and line.find("!") == -1:
                line_list = _validate_str_with_equals(line)
                data_find = True

                # frequency will be the key
                try:
                    key = float(
                        [
                            ss.split("=")[1]
                            for ss in line_list
                            if ss.lower().find("freq") == 0
                        ][0]
                    )
                    data_dict[key] = []
                    avgt = float(
                        [
                            ss.split("=")[1]
                            for ss in line_list
                            if ss.lower().find("avgt") == 0
                        ][0]
                    )
                    avgt_dict[key] = avgt
                except ValueError:
                    self.logger.debug("did not find frequency key")
            elif data_find and line.find(">") == -1 and line.find("!") == -1:
                data_dict[key] += [float(ll) for ll in line.strip().split()]
            elif line.find(">spectra") == -1:
                data_find = False
        # get an object that contains the indices for each component
        cc = index_locator(comp_list)

        self.frequency = np.array(sorted(list(data_dict.keys()), reverse=True))

        self.z = np.zeros((self.frequency.size, 2, 2), dtype=complex)
        self.t = np.zeros((self.frequency.size, 1, 2), dtype=complex)

        self.z_err = np.zeros_like(self.z, dtype=float)
        self.t_err = np.zeros_like(self.t, dtype=float)

        self.residual_covariance = np.zeros(
            (self.frequency.size, cc.n_outputs, cc.n_outputs), dtype=complex
        )
        self.signal_inverse_power = np.zeros(
            (self.frequency.size, cc.n_inputs, cc.n_inputs), dtype=complex
        )

        self.tf = np.zeros(
            (self.frequency.size, cc.n_outputs, cc.n_inputs), dtype=complex
        )
        self.tf_err = np.zeros_like(self.tf, dtype=float)

        for kk, key in enumerate(self.frequency):
            # read in spectra  as an (n_channel x n_channel) array
            spectra_arr = np.reshape(
                np.array(data_dict[key]), (len(comp_list), len(comp_list))
            )

            # compute cross powers
            s_arr = np.zeros_like(spectra_arr, dtype=complex)
            for ii in range(s_arr.shape[0]):
                for jj in range(ii, s_arr.shape[0]):
                    if ii == jj:
                        s_arr[ii, jj] = spectra_arr[ii, jj]
                    else:
                        # minus sign for complex conjugation
                        # original spectra data are of form <A,B*>, but we need
                        # the order <B,A*>...
                        # this is achieved by complex conjugation of the
                        # original entries
                        s_arr[ii, jj] = complex(
                            spectra_arr[jj, ii], -spectra_arr[ii, jj]
                        )
                        # keep complex conjugated entries in the lower
                        # triangular matrix:
                        s_arr[jj, ii] = complex(
                            spectra_arr[jj, ii], spectra_arr[ii, jj]
                        )
            # check for empty values
            s_arr[s_arr == 0] = np.nan
            s_arr[s_arr == self.Header.empty] = np.nan

            # from A. Kelbert's EMTF
            # cross spectra matrices

            # Note we changed the indices to [ex, ey, hz] from [hz, ex, ey]
            # input channels
            rh = np.zeros((cc.n_inputs, cc.n_inputs), dtype=complex)
            rr = np.zeros((cc.n_inputs, cc.n_inputs), dtype=complex)
            hh = np.zeros((cc.n_inputs, cc.n_inputs), dtype=complex)

            # output channels
            re = np.zeros((cc.n_inputs, cc.n_outputs), dtype=complex)
            he = np.zeros((cc.n_inputs, cc.n_outputs), dtype=complex)
            ee = np.zeros((cc.n_outputs, cc.n_outputs), dtype=complex)

            # fill in cross powers for input channels
            rh[0, 0] = s_arr[cc.rhx, cc.hx]
            rh[0, 1] = s_arr[cc.rhx, cc.hy]
            rh[1, 0] = s_arr[cc.rhy, cc.hx]
            rh[1, 1] = s_arr[cc.rhy, cc.hy]

            rr[0, 0] = s_arr[cc.rhx, cc.rhx]
            rr[0, 1] = s_arr[cc.rhx, cc.rhy]
            rr[1, 0] = s_arr[cc.rhy, cc.rhx]
            rr[1, 1] = s_arr[cc.rhy, cc.rhy]

            hh[0, 0] = s_arr[cc.hx, cc.hx]
            hh[0, 1] = s_arr[cc.hx, cc.hy]
            hh[1, 0] = s_arr[cc.hy, cc.hx]
            hh[1, 1] = s_arr[cc.hy, cc.hy]

            # fill in cross powers for output channels
            if cc.has_tipper and cc.has_electric:
                re[0, 2] = s_arr[cc.rhx, cc.hz]
                re[0, 0] = s_arr[cc.rhx, cc.ex]
                re[0, 1] = s_arr[cc.rhx, cc.ey]
                re[1, 2] = s_arr[cc.rhy, cc.hz]
                re[1, 0] = s_arr[cc.rhy, cc.ex]
                re[1, 1] = s_arr[cc.rhy, cc.ey]

                he[0, 2] = s_arr[cc.hx, cc.hz]
                he[0, 0] = s_arr[cc.hx, cc.ex]
                he[0, 1] = s_arr[cc.hx, cc.ey]
                he[1, 2] = s_arr[cc.hy, cc.hz]
                he[1, 0] = s_arr[cc.hy, cc.ex]
                he[1, 1] = s_arr[cc.hy, cc.ey]

                ee[2, 2] = s_arr[cc.hz, cc.hz]
                ee[2, 0] = s_arr[cc.hz, cc.ex]
                ee[2, 1] = s_arr[cc.hz, cc.ey]
                ee[0, 2] = s_arr[cc.ex, cc.hz]
                ee[0, 0] = s_arr[cc.ex, cc.ex]
                ee[0, 1] = s_arr[cc.ex, cc.ey]
                ee[1, 2] = s_arr[cc.ey, cc.hz]
                ee[1, 0] = s_arr[cc.ey, cc.ex]
                ee[1, 1] = s_arr[cc.ey, cc.ey]
            elif not cc.has_tipper and cc.has_electric:
                re[0, 0] = s_arr[cc.rhx, cc.ex]
                re[0, 1] = s_arr[cc.rhx, cc.ey]
                re[1, 0] = s_arr[cc.rhy, cc.ex]
                re[1, 0] = s_arr[cc.rhy, cc.ey]

                he[0, 0] = s_arr[cc.hx, cc.ex]
                he[0, 1] = s_arr[cc.hx, cc.ey]
                he[0, 1] = s_arr[cc.hy, cc.ex]
                he[1, 1] = s_arr[cc.hy, cc.ey]

                ee[0, 0] = s_arr[cc.ex, cc.ex]
                ee[0, 1] = s_arr[cc.ex, cc.ey]
                ee[1, 0] = s_arr[cc.ey, cc.ex]
                ee[1, 1] = s_arr[cc.ey, cc.ey]
            elif cc.has_tipper and not cc.has_electric:
                re[0, 0] = s_arr[cc.rhx, cc.hz]
                re[1, 0] = s_arr[cc.rhy, cc.hz]

                he[0, 0] = s_arr[cc.hx, cc.hz]
                he[1, 0] = s_arr[cc.hy, cc.hz]

                ee[0, 0] = s_arr[cc.hz, cc.hz]
            # check to make sure the values are legit for accurate results
            if abs(np.linalg.det(rh)) < np.finfo(float).eps:
                self.logger.warning(
                    "spectral matrix determinant is too small "
                    f"{abs(np.linalg.det(rh))} for period {key}. "
                    "Results may be inaccurate"
                )
            tfh = np.matmul(np.linalg.inv(rh), re)
            tf = tfh.conj().T

            sig = np.matmul(
                np.linalg.inv(rh), np.matmul(rr, np.linalg.inv(rh.conj().T))
            )
            res = (
                ee
                - np.matmul(tf, he)
                - np.matmul(he.conj().T, tfh)
                + np.matmul(tf, np.matmul(hh, tfh))
            ) / avgt_dict[key]

            # variance = abs(np.dot(res[0 : cc.n_inputs, :].T, sig))
            variance = np.zeros((cc.n_outputs, cc.n_inputs), dtype=complex)
            for nn in range(cc.n_outputs):
                for mm in range(cc.n_inputs):
                    variance[nn, mm] = res[nn, nn] * sig[mm, mm]

            tf_err = np.sqrt(np.abs(variance))
            self.tf[kk, :, :] = tf
            self.tf_err[kk, :, :] = np.sqrt(np.abs(variance))
            self.signal_inverse_power[kk, :, :] = sig
            self.residual_covariance[kk, :, :] = res

            if cc.has_tipper and cc.has_electric:
                self.z[kk, :, :] = tf[0:2, :]
                self.z_err[kk, :, :] = tf_err[0:2, :]
                self.t[kk, :, :] = tf[2, :]
                self.t_err[kk, :, :] = tf_err[2, :]
                self.z_err[np.where(np.nan_to_num(self.z_err) == 0.0)] = 1.0
                self.t_err[np.nan_to_num(self.t_err) == 0.0] = 1.0
            elif not cc.has_tipper and cc.has_electric:
                self.z[kk, :, :] = tf[:, :]
                self.z_err[kk, :, :] = tf_err[:, :]
                self.z_err[np.where(np.nan_to_num(self.z_err) == 0.0)] = 1.0
            elif cc.has_tipper and not cc.has_electric:
                self.t[kk, :, :] = tf[:, :]
                self.t_err[kk, :, :] = tf_err[:, :]
                self.t_err[np.nan_to_num(self.t_err) == 0.0] = 1.0

    def write(
        self,
        new_edi_fn: str | Path | None = None,
        longitude_format: Literal["LON", "LONG"] = "LON",
        latlon_format: Literal["dms", "dd"] = "dms",
    ) -> Path:
        """
        Write a new edi file from either an existing .edi file or from data
        input by the user into the attributes of Edi.


        :param new_edi_fn: full path to new edi file.
                           *default* is None, which will write to the same
                           file as the input .edi with as:
                           r"/home/mt/mt01_1.edi"
        :type new_edi_fn: string
        :param longitude_format:  whether to write longitude as LON or LONG.
                                  options are 'LON' or 'LONG', default 'LON'
        :type longitude_format:  string
        :param latlon_format:  format of latitude and longitude in output edi,
                               degrees minutes seconds ('dms') or decimal
                               degrees ('dd')
        :type latlon_format:  string

        :returns: full path to new edi file
        :rtype: string

        :Example: ::

            >>> import mtpy.core.edi as mtedi
            >>> edi_obj = mtedi.Edi(fn=r"/home/mt/mt01/edi")
            >>> edi_obj.Header.dataid = 'mt01_rr'
            >>> n_edi_fn = edi_obj.write_edi_file()
        """

        if new_edi_fn is None:
            if self.fn is not None:
                new_edi_fn = Path(self.fn)
            else:
                new_edi_fn = Path().cwd().joinpath(f"{self.Header.dataid}.edi")
        else:
            new_edi_fn = Path(new_edi_fn)
        # write lines
        extra_lines = []
        if self.survey_metadata.summary != None:
            extra_lines.append(f"\tsurvey.summary = {self.survey_metadata.summary}\n")
        if self.Header.progname != "mt_metadata":
            extra_lines.append(f"\toriginal_program.name={self.Header.progname}\n")
        if self.Header.progvers != __version__:
            extra_lines.append(f"\toriginal_program.version={self.Header.progvers}\n")
        if self.Header.progdate != "1980-01-01":
            extra_lines.append(f"\toriginal_program.date={self.Header.progdate}\n")
        if self.Header.filedate != "1980-01-01":
            extra_lines.append(f"\toriginal_file.date={self.Header.filedate}\n")
        header_lines = self.Header.write_header(
            longitude_format=longitude_format, latlon_format=latlon_format
        )

        info_lines = self.Info.write_info()
        info_lines.insert(1, "".join(extra_lines))

        define_lines = self.Measurement.write_measurement(
            longitude_format=longitude_format, latlon_format=latlon_format
        )

        self.Data.nfreq = len(self.frequency)
        dsect_lines = self.Data.write_data()

        # write out frequencies
        freq_lines = [self._data_header_str.format("frequencies".upper())]
        freq_lines += self._write_data_block(self.frequency, "freq")

        # write out rotation angles
        zrot_lines = [self._data_header_str.format("impedance rotation angles".upper())]
        if self.rotation_angle is None:
            self.rotation_angle = np.zeros(self.frequency.size)
        elif isinstance(self.rotation_angle, (float, int)):
            self.rotation_angle = np.repeat(self.rotation_angle, self.frequency.size)
        elif len(self.rotation_angle) != self.frequency.size:
            raise ValueError(
                "rotation angle must be the same length and the number of "
                f"frequencies {len(self.rotation_angle)} != {self.frequency.size}"
            )
        zrot_lines += self._write_data_block(self.rotation_angle, "zrot")

        # write out data only impedance and tipper
        z_data_lines = [self._data_header_str.format("impedances".upper())]
        self.z = np.nan_to_num(self.z)
        self.z_err = np.nan_to_num(self.z_err)
        self.t = np.nan_to_num(self.t)
        self.t_err = np.nan_to_num(self.t_err)
        if self.z is not None:
            for ii in range(2):
                for jj in range(2):
                    z_lines_real = self._write_data_block(
                        self.z[:, ii, jj].real, self._z_labels[2 * ii + jj][0]
                    )
                    z_lines_imag = self._write_data_block(
                        self.z[:, ii, jj].imag, self._z_labels[2 * ii + jj][1]
                    )
                    z_lines_var = self._write_data_block(
                        self.z_err[:, ii, jj] ** 2.0,
                        self._z_labels[2 * ii + jj][2],
                    )

                    z_data_lines += z_lines_real
                    z_data_lines += z_lines_imag
                    z_data_lines += z_lines_var
        if self.t is None:
            trot_lines = [""]
            t_data_lines = [""]
        elif np.all(self.t == 0):
            trot_lines = [""]
            t_data_lines = [""]
        else:
            try:
                # write out rotation angles
                trot_lines = [
                    self._data_header_str.format("tipper rotation angles".upper())
                ]
                if isinstance(self.rotation_angle, float):
                    trot = np.repeat(self.rotation_angle, self.frequency.size)
                else:
                    trot = self.rotation_angle
                trot_lines += self._write_data_block(np.array(trot), "trot")

                # write out tipper lines
                t_data_lines = [self._data_header_str.format("tipper".upper())]
                for jj in range(2):
                    t_lines_real = self._write_data_block(
                        self.t[:, 0, jj].real, self._t_labels[jj][0]
                    )
                    t_lines_imag = self._write_data_block(
                        self.t[:, 0, jj].imag, self._t_labels[jj][1]
                    )
                    t_lines_var = self._write_data_block(
                        self.t_err[:, 0, jj] ** 2.0, self._t_labels[jj][2]
                    )

                    t_data_lines += t_lines_real
                    t_data_lines += t_lines_imag
                    t_data_lines += t_lines_var
            except AttributeError:
                trot_lines = [""]
                t_data_lines = [""]
        edi_lines = (
            header_lines
            + info_lines
            + define_lines
            + dsect_lines
            + freq_lines
            + zrot_lines
            + z_data_lines
            + trot_lines
            + t_data_lines
            + [">END"]
        )

        with open(new_edi_fn, "w") as fid:
            fid.write("".join(edi_lines))
        self.fn = new_edi_fn
        return new_edi_fn

    def _write_data_block(
        self, data_comp_arr: np.typing.NDArray, data_key: str
    ) -> list[str]:
        """
        Write a data block

        :param data_comp_arr: array of data components
        :type data_comp_arr: np.ndarray

        :param data_key: the component to write out
        :type data_key: string

        :returns: list of lines to write to edi file
        :rtype: list
        """
        if data_key.lower().find("z") >= 0 and data_key.lower() not in [
            "zrot",
            "trot",
        ]:
            block_lines = [
                ">{0} ROT=ZROT // {1:.0f}\n".format(
                    data_key.upper(), data_comp_arr.size
                )
            ]
        elif data_key.lower().find("t") >= 0 and data_key.lower() not in [
            "zrot",
            "trot",
        ]:
            block_lines = [
                ">{0} ROT=TROT // {1:.0f}\n".format(
                    data_key.upper(), data_comp_arr.size
                )
            ]
        elif data_key.lower() == "freq":
            block_lines = [
                ">{0} // {1:.0f}\n".format(data_key.upper(), data_comp_arr.size)
            ]
        elif data_key.lower() in ["zrot", "trot"]:
            block_lines = [
                ">{0} // {1:.0f}\n".format(data_key.upper(), data_comp_arr.size)
            ]
        else:
            raise ValueError("Cannot write block for {0}".format(data_key))
        for d_index, d_comp in enumerate(data_comp_arr, 1):
            if d_comp == 0.0 and data_key.lower() not in ["zrot", "trot"]:
                d_comp = self.Header.empty
            # write the string in the specified format
            num_str = "{0:{1}}".format(d_comp, self._num_format)

            # check to see if a new line is needed
            if d_index % self._block_len == 0:
                num_str += "\n"
            # at the end of the block add a return
            if d_index == data_comp_arr.size:
                num_str += "\n"
            block_lines.append(num_str)
        return block_lines

    # -----------------------------------------------------------------------
    # set a few important properties
    # --> Latitude
    @property
    def lat(self) -> float | None:
        """latitude in decimal degrees"""
        return self.Header.latitude

    @lat.setter
    def lat(self, input_lat) -> None:
        """set latitude and make sure it is converted to a float"""
        self.Header.latitude = input_lat

    # --> Longitude
    @property
    def lon(self) -> float | None:
        """longitude in decimal degrees"""
        return self.Header.longitude

    @lon.setter
    def lon(self, input_lon: float | None):
        """set latitude and make sure it is converted to a float"""
        self.Header.longitude = input_lon

    # --> Elevation
    @property
    def elev(self) -> float:
        """Elevation in elevation units"""
        return self.Header.elevation

    @elev.setter
    def elev(self, input_elev: float) -> None:
        """set elevation and make sure it is converted to a float"""
        self.Header.elevation = input_elev

    # --> station
    @property
    def station(self) -> str | None:
        """station name"""
        if self.Header.dataid is not None:
            return self.Header.dataid.replace(r"/", "_")
        elif self.Measurement.refloc is not None:
            return self.Measurement.refloc.replace('"', "")
        elif self.Data.sectid is not None:
            return self.Data.sectid

    @station.setter
    def station(self, new_station: str | int):
        """station name"""
        if not isinstance(new_station, str):
            new_station = f"{new_station}".replace(r"/", "_")
        self.Header.dataid = new_station
        self.Data.sectid = new_station

    @property
    def survey_metadata(self) -> Survey:
        sm = Survey()

        if self.Header.project is None:
            if self.Header.prospect is not None:
                sm.project = self.Header.prospect
        else:
            sm.project = self.Header.project

        if self.Header.survey is None:
            sm.id = "0"
        else:
            sm.id = (
                self.Header.survey.replace(",", "")
                .strip()
                .replace(" ", "_")
                .replace(".", "_")
            )
        if self.Header.acqby is not None:
            sm.acquired_by.author = self.Header.acqby
        if self.Header.loc is not None:
            sm.geographic_name = self.Header.loc
        sm.country = self.Header.country

        for key, value in self.Info.info_dict.items():
            if key is None:
                key = "extra"
            key = key.lower()
            if key.startswith("survey."):
                if "doi" in key and not self._is_doi_value(value):
                    key = key.replace("doi", "url")

                try:
                    sm.update_attribute(key.split("survey.")[1], value)
                except Exception as error:
                    self.logger.debug(
                        f"Could not update survey metadata with {key}={value}, caused {error}"
                    )

        sm.add_station(self.station_metadata)

        return sm

    @staticmethod
    def _is_doi_value(value: str | HttpUrl | object) -> bool:
        """Return True when the value can be validated as a DOI."""
        if not isinstance(value, (str, HttpUrl)):
            return False

        try:
            return validate_doi(value) is not None
        except (TypeError, ValueError):
            return False

    @survey_metadata.setter
    def survey_metadata(self, survey: Survey) -> None:
        """
        Update metadata from a survey metadata object

        :param value: DESCRIPTION
        :type value: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if not isinstance(survey, Survey):
            raise TypeError(
                "Input must be a mt_metadata.transfer_function.Survey object"
                f" not {type(survey)}"
            )
        self.Header.survey = survey.id
        self.Header.project = survey.project
        self.Header.loc = survey.geographic_name
        self.Header.country = survey.country
        if survey.summary not in NULL_VALUES:
            self.Info.info_dict[f"survey.summary"] = survey.summary

        for key in survey.to_dict(single=True).keys():
            if "northwest" in key or "southeast" in key or "time_period" in key:
                continue
            value = survey.get_attr_from_name(key)
            if value not in NULL_VALUES:
                self.Info.info_dict[f"survey.{key}"] = value

    @property
    def station_metadata(self) -> tf.Station:
        sm = tf.Station()
        sm.add_run(Run(id=f"{self.station}a"))
        if self.station is not None:
            sm.id = self.station
        sm.data_type = "MT"
        sm.channels_recorded = self.Measurement.channels_recorded
        # location
        sm.location.latitude = self.lat
        sm.location.longitude = self.lon
        sm.location.elevation = self.elev
        sm.location.datum = self.Header.datum
        sm.location.declination.value = self.Header.declination.value
        sm.orientation.reference_frame = self.Header.coordinate_system.split()[0]
        if self.Header.loc is not None:
            sm.geographic_name = self.Header.loc

        # provenance
        if self.Header.acqby is not None:
            sm.acquired_by.author = self.Header.acqby
        sm.provenance.creation_time = self.Header.filedate
        sm.provenance.submitter.author = self.Header.fileby
        sm.provenance.software.name = self.Header.fileby
        sm.provenance.software.version = self.Header.progvers
        sm.transfer_function.processed_date = self.Header.filedate
        sm.transfer_function.runs_processed = sm.run_list
        if self.station is not None:
            sm.transfer_function.id = self.station
        # dates
        if self.Header.acqdate is not None:
            sm.time_period.start = self.Header.acqdate
        if self.Header.enddate is not None:
            sm.time_period.end = self.Header.enddate

        # processing information
        for key, value in self.Info.info_dict.items():
            if key is None:
                continue
            if value in NULL_VALUES:
                continue
            key = key.lower()

            if "provenance" in key:
                if "email" in key:
                    if value.count(",") > 0:
                        value = value.split(",")[0]
                sm.update_attribute(key, value)
            elif "transfer_function" in key:
                key = key.split("transfer_function.")[1]
                if "processing_parameters" in key:
                    if isinstance(value, list):
                        for item in value:
                            if "=" in item:
                                param, val = item.split("=")
                                sm.transfer_function.processing_parameters.append(
                                    f"{param}={val}"
                                )
                            else:
                                sm.transfer_function.processing_parameters.append(item)
                    elif value not in NULL_VALUES:
                        param = key.split(".")[-1]
                        sm.transfer_function.processing_parameters.append(
                            f"{param}={value}"
                        )
                elif "remote_references" in key:
                    if key.count(".") > 1:
                        if key in ["remote_references.id"]:
                            sm.transfer_function.remote_references.append(value)
                        elif key in ["remote_references.geographic_name"]:
                            try:
                                sm.transfer_function.remote_references[
                                    -1
                                ] = f"{value}.{sm.transfer_function.remote_references[-1]}"
                            except IndexError:
                                sm.transfer_function.remote_references.append(value)
                        else:
                            # skip any keys that have remote_references.something_else
                            continue
                    else:
                        if isinstance(value, list):
                            sm.transfer_function.remote_references.extend(value)
                        elif isinstance(value, str):
                            sm.transfer_function.remote_references.append(value)
                        elif isinstance(value, (int, float)):
                            sm.transfer_function.remote_references.append(value)
                        else:
                            self.logger.warning(
                                f"Do not know how to handle remote_references {value}"
                            )
                            continue
                elif "runs_processed" in key:
                    sm.run_list = sm.transfer_function.runs_processed
                else:
                    sm.transfer_function.update_attribute(key, value)

            elif key.startswith("run."):
                comp = None
                key = key.split("run.")[1]
                if "." in key:
                    # split the key into component and attribute
                    # e.g. run.mt01a.hx.comments
                    try:
                        comp, key = key.split(".", 1)
                    except (IndexError, ValueError):
                        continue
                if comp is not None:
                    try:
                        ch = getattr(sm.runs[0], comp)
                    except AttributeError:
                        ch = None
                    if ch is None:
                        if comp in ["ex", "ey"]:
                            ch = Electric(component=comp)  # type: ignore
                            sm.runs[0].add_channel(ch)
                        elif comp in ["hx", "hy", "hz", "rrhx", "rrhy"]:
                            ch = Magnetic(component=comp)  # type: ignore
                            sm.runs[0].add_channel(ch)
                        else:
                            self.logger.warning(
                                f"Do not recognize channel {comp}, skipping..."
                            )
                            try:
                                sm.runs[0].update_attribute(f"{key}.{comp}", value)

                            except Exception as e:
                                self.logger.warning(
                                    f"Failed to update attribute {key}.{comp}: {e}"
                                )
                    if key in ["comments"]:
                        if isinstance(value, list):
                            value = ",".join(value)
                    try:
                        ch.update_attribute(key, value)
                    except Exception as e:
                        self.logger.warning(f"Failed to update attribute {key}: {e}")
            elif key.startswith("data_logger"):
                if key.count(".") == 0:
                    key = f"{key}.id"
                sm.runs[0].update_attribute(key, value)
            elif key.startswith("station."):
                if "id" in key:
                    value = validate_station_name(value)
                sm.update_attribute(key.split("station.")[1], value)
            elif "processing." in key:
                key = key.split("processing.")[1]
                if key in ["software"]:
                    sm.transfer_function.software.name = value
                elif key in ["tag"]:
                    if value.count(",") > 0:
                        sm.transfer_function.remote_references = value.split(",")
                    else:
                        sm.transfer_function.remote_references = value.split()
            elif key in ["processedby", "processed_by"]:
                sm.transfer_function.processed_by.name = value
            elif key in ["runlist", "run_list"]:
                if value.count(",") > 0:
                    runs = value.split(",")
                else:
                    runs = value.split()
                sm.run_list = []
                for rr in runs:
                    if rr not in sm.runs.keys():
                        sm.add_run(Run(id=rr))
                sm.transfer_function.runs_processed = runs
            elif key == "sitename":
                sm.geographic_name = value
            elif key == "signconvention":
                sm.transfer_function.sign_convention = value
            elif "mtft" in key or "emtf" in key or "mtedit" in key:
                sm.transfer_function.processing_parameters.append(f"{key}={value}")

        if self.Header.filedate is not None:
            sm.transfer_function.processed_date = self.Header.filedate
        # make any extra information in info list into a comment
        sm.comments = "\n".join(self.Info.write_info()[1:])

        # add information to runs
        for rr in sm.runs:
            if rr.time_period.start in NULL_VALUES:
                rr.time_period.start = sm.time_period.start
            if rr.time_period.end in NULL_VALUES:
                rr.time_period.end = sm.time_period.end

            for ch in self.Measurement.channels_recorded:
                try:
                    ch_obj = rr.channels[ch]
                    ch_obj.update(getattr(self, f"{ch}_metadata"))
                except AttributeError:
                    pass
        return sm

    @station_metadata.setter
    def station_metadata(self, sm: tf.Station) -> None:
        """
        Set EDI metadata from station metadata object

        :param sm: Station object to pull metadata from
        :type sm: :class:`mt_metadata.transfer_functions.tf.Station`

        """

        ### fill header information from station
        self.Header.acqby = sm.acquired_by.author
        self.Header.acqdate = sm.time_period.start
        self.Header.coordinate_system = sm.orientation.reference_frame
        self.Header.dataid = sm.id
        self.Header.declination = sm.location.declination
        self.Header.elevation = sm.location.elevation
        self.Header.fileby = sm.provenance.submitter.author
        self.Header.filedate = sm.provenance.creation_time
        self.Header.latitude = sm.location.latitude
        self.Header.longitude = sm.location.longitude
        self.Header.datum = sm.location.datum
        self.Header.units = sm.transfer_function.units
        self.Header.enddate = sm.time_period.end
        if sm.geographic_name is not None:
            self.Header.loc = sm.geographic_name

        ### write notes
        # write comments, which would be anything in the info section from an edi
        if isinstance(sm.comments, str):
            self.Info.read_info(sm.comments.split("\n"))
        # write transfer function info first
        for k, v in sm.transfer_function.to_dict(single=True).items():
            if not v in NULL_VALUES:
                if k in ["processing_parameters"]:
                    for item in v:
                        param, value = item.split("=", 1)
                        self.Info.info_dict[
                            f"transfer_function.processing_parameters.{param}"
                        ] = value

                else:
                    self.Info.info_dict[f"transfer_function.{k}"] = v
        # write provenance
        for k, v in sm.provenance.to_dict(single=True).items():
            if not v in NULL_VALUES:
                self.Info.info_dict[f"provenance.{k}"] = v
        # write field notes
        for run in sm.runs:
            r_dict = run.to_dict(single=True)
            for r_key, r_value in r_dict.items():
                if r_value in NULL_VALUES:
                    continue
                self.Info.info_dict[f"{run.id}.{r_key}"] = r_value
            for ch in run.channels:
                ch_dict = ch.to_dict(single=True)
                for ch_key, ch_value in ch_dict.items():
                    if ch_key not in self._channel_skip_list:
                        if ch_value in NULL_VALUES:
                            continue
                        self.Info.info_dict[
                            f"{run.id}.{ch.component}.{ch_key}"
                        ] = ch_value
                # write station information
                self.Measurement.from_metadata(ch)
                # add channel id to data section
                try:
                    setattr(self.Data, f"{ch.component}", ch.channel_id)
                except AttributeError:
                    self.logger.warning(
                        f"Cannot set {ch.component} channel id, skipping..."
                    )

        ### fill measurement
        self.Measurement.refelev = sm.location.elevation
        self.Measurement.reflat = sm.location.latitude
        self.Measurement.reflon = sm.location.longitude
        self.Measurement.refloc = sm.id
        self.Measurement.maxchan = len(sm.channels_recorded)

    def _get_electric_metadata(self, comp):
        """
        get electric information from the various metadata
        """
        comp = comp.lower()
        electric = Electric(component=comp)
        electric.positive.type = "electric"
        electric.negative.type = "electric"
        if comp in self.Measurement.measurements.keys():
            meas = self.Measurement.measurements[comp]
            for attr in [
                "negative.x",
                "negative.y",
                "positive.x2",
                "positive.y2",
                "measurement_azimuth",
                "translated_azimuth",
            ]:
                if electric.get_attr_from_name(attr) is None:
                    electric.update_attribute(attr, 0)
            electric.dipole_length = meas.dipole_length
            electric.channel_id = meas.id
            electric.measurement_azimuth = meas.azimuth
            electric.translated_azimuth = meas.azimuth
            electric.component = meas.chtype
            electric.channel_number = meas.channel_number
            electric.negative.x = meas.x
            electric.positive.x2 = meas.x2
            electric.negative.y = meas.y
            electric.positive.y2 = meas.y2
            for k, v in self.Info.info_dict.items():
                if k is None:
                    continue
                if f"{comp}." in k:
                    key = k.split(f"{comp}.")[1].strip()
                    if key == "manufacturer":
                        electric.negative.manufacturer = v
                        electric.positive.manufacturer = v
                    if key == "type":
                        electric.negative.type = v
                        electric.positive.type = v
            if (
                electric.positive.x2 == 0
                and electric.positive.y2 == 0.0
                and electric.negative.x == 0
                and electric.negative.y == 0.0
            ):
                electric.positive.x2 = electric.dipole_length * np.cos(
                    np.deg2rad(meas.azimuth)
                )
                electric.positive.y2 = electric.dipole_length * np.sin(
                    np.deg2rad(meas.azimuth)
                )
            for key, value in self.Info.info_dict.items():
                if key is None:
                    continue
                elif ".type" in key:
                    continue
                if f".{comp}." in key:
                    key = key.split(f".{comp}.", 1)[-1]
                    if value in NULL_VALUES:
                        continue
                    try:
                        electric.update_attribute(key, value)
                    except Exception as error:
                        self.logger.warning(
                            f"Cannot update attribute {key} with value {value}, causes {error}"
                        )
        return electric

    @property
    def ex_metadata(self):
        return self._get_electric_metadata("ex")

    @property
    def ey_metadata(self):
        return self._get_electric_metadata("ey")

    def _get_magnetic_metadata(self, comp):
        """

        get magnetic metadata from the various sources

        :param comp: DESCRIPTION
        :type comp: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        magnetic = Magnetic(component=comp)
        magnetic.sensor.type = "magnetic"
        if comp in self.Measurement.measurements.keys():
            meas = self.Measurement.measurements[comp]
            for attr in ["location.x", "location.y", "location.z"]:
                if magnetic.get_attr_from_name(attr) is None:
                    magnetic.update_attribute(attr, 0)
            magnetic.measurement_azimuth = meas.azm
            magnetic.translated_azimuth = meas.azm
            magnetic.component = meas.chtype
            magnetic.channel_number = meas.channel_number
            magnetic.channel_id = meas.id
            magnetic.location.x = meas.x
            magnetic.location.y = meas.y
            try:
                magnetic.sensor.id = meas.meas_magnetic.sensor
            except AttributeError:
                pass
            for k, v in self.Info.info_dict.items():
                if k is None:
                    continue
                if f"{comp}." in k:
                    key = k.split(f"{comp}.")[1].strip()
                    if key == "manufacturer":
                        magnetic.sensor.manufacturer = v
                    if key == "type":
                        magnetic.sensor.type = v
                    if key.startswith("sensor."):
                        magnetic.update_attribute(key, v)
        return magnetic

    @property
    def hx_metadata(self):
        return self._get_magnetic_metadata("hx")

    @property
    def hy_metadata(self):
        return self._get_magnetic_metadata("hy")

    @property
    def hz_metadata(self):
        return self._get_magnetic_metadata("hz")

    @property
    def rrhx_metadata(self):
        return self._get_magnetic_metadata("rrhx")

    @property
    def rrhy_metadata(self):
        return self._get_magnetic_metadata("rrhy")
