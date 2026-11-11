"""
This module contains the DecimationLevel class.
TODO: Factor or rename.  The decimation level class here has information about the entire processing.
"""

# =====================================================
# Imports
# =====================================================
from typing import Annotated, get_args, List, Union

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import computed_field, Field, field_validator, ValidationInfo

from mt_metadata.base import MetadataBase
from mt_metadata.common.band import Band
from mt_metadata.common.enumerations import StrEnumerationBase
from mt_metadata.features.weights import ChannelWeightSpec
from mt_metadata.helper_functions import cast_to_class_if_dict, validate_setter_input
from mt_metadata.processing import ShortTimeFourierTransform as STFT
from mt_metadata.processing import TimeSeriesDecimation as Decimation
from mt_metadata.processing.aurora.estimator import Estimator
from mt_metadata.processing.aurora.frequency_bands import FrequencyBands
from mt_metadata.processing.aurora.regression import Regression
from mt_metadata.processing.fourier_coefficients.decimation import (
    Decimation as FCDecimation,
)


# =====================================================
class SaveFcsTypeEnum(StrEnumerationBase):
    h5 = "h5"
    csv = "csv"


class DecimationLevel(MetadataBase):
    bands: Annotated[
        list[Band],
        Field(
            default_factory=list,
            description="List of bands",
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["[]"],
            },
        ),
    ]

    channel_weight_specs: Annotated[
        List[ChannelWeightSpec],
        Field(
            default_factory=list,
            description="List of weighting schemes to use for TF processing for each output channel",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["[]"],
            },
        ),
    ]

    input_channels: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="list of input channels (sources)",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["hx, hy"],
            },
        ),
    ]

    output_channels: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="list of output channels (responses)",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["ex, ey, hz"],
            },
        ),
    ]

    reference_channels: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="list of reference channels (remote sources)",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["hx, hy"],
            },
        ),
    ]

    save_fcs: Annotated[
        bool,
        Field(
            default=False,
            description="Whether the Fourier coefficients are saved [True] or not [False].",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": [True],
            },
        ),
    ]

    save_fcs_type: Annotated[
        SaveFcsTypeEnum | None,
        Field(
            default=None,
            description="Format to use for fc storage",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["h5"],
            },
        ),
    ]

    decimation: Annotated[
        Decimation,
        Field(
            default_factory=Decimation,  # type: ignore
            description="Decimation settings",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["Decimation()"],
            },
        ),
    ]

    estimator: Annotated[
        Estimator,
        Field(
            default_factory=Estimator,  # type: ignore
            description="Estimator settings",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["Estimator()"],
            },
        ),
    ]

    regression: Annotated[
        Regression,
        Field(
            default_factory=Regression,  # type: ignore
            description="Regression settings",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["Regression()"],
            },
        ),
    ]

    stft: Annotated[
        STFT,
        Field(
            default_factory=STFT,  # type: ignore
            description="Short-time Fourier transform settings",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["STFT()"],
            },
        ),
    ]

    @field_validator("channel_weight_specs", mode="before")
    @classmethod
    def validate_channel_weight_specs(cls, value, info: ValidationInfo):
        """
        Validator for channel_weight_specs field.
        """

        # Handle singleton cases
        if isinstance(value, (ChannelWeightSpec, dict)):
            value = [value]

        if not isinstance(value, list):
            raise TypeError(f"Not sure what to do with {type(value)}")

        # Convert dicts to ChannelWeightSpecs objects
        validated_specs = []
        for item in value:
            if isinstance(item, dict):
                validated_specs.append(ChannelWeightSpec(**item))
            elif isinstance(item, ChannelWeightSpec):
                validated_specs.append(item)
            else:
                raise TypeError(
                    f"List entry must be a ChannelWeightSpec object or dict, not {type(item)}"
                )

        return validated_specs

    @field_validator("bands", mode="before")
    @classmethod
    def validate_bands(cls, value, info: ValidationInfo):
        # Get the field type dynamically from the model
        field_name = info.field_name
        if field_name is None:
            raise ValueError("Field name is required for validation")

        field_info = cls.model_fields[field_name]

        # Extract the target class from List[TargetClass] annotation
        target_class = get_args(field_info.annotation)[0]

        values = validate_setter_input(value, target_class)
        return [cast_to_class_if_dict(obj, target_class) for obj in values]

    def add_band(self, band: Union[Band, dict]) -> None:
        """
        add a band
        """

        if not isinstance(band, (Band, dict)):
            raise TypeError(f"List entry must be a Band object not {type(band)}")
        if isinstance(band, dict):
            obj = Band()
            obj.from_dict(band)
        else:
            obj = band

        self.bands.append(obj)

    @computed_field
    @property
    def lower_bounds(self) -> np.ndarray:
        """
        get lower bounds index values into an array.
        """

        return np.array(sorted([band.index_min for band in self.bands]))

    @computed_field
    @property
    def upper_bounds(self) -> np.ndarray:
        """
        get upper bounds index values into an array.
        """

        return np.array(sorted([band.index_max for band in self.bands]))

    @computed_field
    @property
    def bands_dataframe(self) -> pd.DataFrame:
        """
        Utility function that transforms a list of bands into a dataframe

        See notes in `_df_from_bands`.

        Returns
        -------
        bands_df: pd.Dataframe
            Same format as that generated by EMTFBandSetupFile.get_decimation_level()
        """
        bands_df = _df_from_bands(self.bands)
        return bands_df

    @computed_field
    @property
    def frequency_sample_interval(self) -> float:
        """
        Returns the delta_f in frequency domain df = 1 / (N * dt)
        Here dt is the sample interval after decimation

        Returns
        -------
        frequency_sample_interval: float
            The frequency sample interval after decimation.
        """
        return self.decimation.sample_rate / self.stft.window.num_samples

    @computed_field
    @property
    def band_edges(self) -> np.ndarray:
        """
        Returns the band edges as a numpy array

        Returns
        -------
        band_edges: 2D numpy array, one row per frequency band and two columns
        """
        bands_df = self.bands_dataframe
        band_edges = np.vstack(
            (bands_df.frequency_min.values, bands_df.frequency_max.values)
        ).T
        return band_edges

    def frequency_bands_obj(self) -> FrequencyBands:
        """
        Gets a FrequencyBands object that is used as input to processing.

        Used by Aurora.

        TODO: consider adding .to_frequency_bands() method directly to self.bands

        Returns
        -------
        frequency_bands:  FrequencyBands
            A FrequencyBands object that can be used as an iterator for processing.

        """
        frequency_bands = FrequencyBands(band_edges=self.band_edges)
        return frequency_bands

    @property
    def fft_frequencies(self) -> np.ndarray:
        """
        Gets the harmonics of the STFT.

        Returns
        -------
        freqs: np.ndarray
            The frequencies at which the stft will be available.
        """
        freqs = self.stft.window.fft_harmonics(self.decimation.sample_rate)
        return freqs

    @property
    def harmonic_indices(self) -> List[int]:
        """
        Loops over all bands and returns a list of the harminic indices.
        TODO: Distinguish the bands which are a processing construction vs harmonic indices which are FFT info.

        Returns
        -------
        return_list: list of integers
            The indices of the harmonics that are needed for processing.
        """
        return_list = []
        for band in self.bands:
            fc_indices = band.harmonic_indices
            return_list += fc_indices.tolist()
        return_list.sort()
        return return_list

    @property
    def local_channels(self):
        return self.input_channels + self.output_channels

    def is_consistent_with_archived_fc_parameters(
        self, fc_decimation: FCDecimation, remote: bool
    ):
        """
            Usage: For an already existing spectrogram stored in an MTH5 archive, this compares the metadata
            within the archive (fc_decimation) with an aurora decimation level (self), and tells whether the
            parameters are in agreement. If True, this allows aurora to skip the calculation of FCs and instead
            read them from the archive.

            TODO: Merge all checks of TimeSeriesDecimation parameters into a single check.
            - e.g. Compress all decimation checks to: assert fc_decimation.decimation == self.decimation

            Parameters
            ----------
            decimation_level: FCDecimation
                metadata describing the parameters used to compute an archived spectrogram
            remote: bool
                If True, we are looking for reference channels, not local channels in the FCGroup.

            Iterates over FCDecimation attributes:
                "channels_estimated": to ensure all expected channels are in the group
                "decimation.anti_alias_filter": check that the expected AAF was applied
                "decimation.sample_rate,
                "decimation.method",
                "stft.prewhitening_type",
                "stft.recoloring",
                "stft.pre_fft_detrend_type",
                "stft.min_num_stft_windows",
                "stft.window",
                "stft.harmonic_indices",
        Returns
        -------

        :return:
        """
        # channels_estimated: Checks that the archived spectrogram has the required channels
        if remote:
            required_channels = self.reference_channels
        else:
            required_channels = self.local_channels
        try:
            assert set(required_channels).issubset(fc_decimation.channels_estimated)
        except AssertionError:
            msg = (
                f"required_channels for processing {required_channels} not available"
                f"-- fc channels estimated are {fc_decimation.channels_estimated}"
            )
            logger.info(msg)
            return False

        # anti_alias_filter: Check that the data were filtered the same way
        try:
            assert (
                fc_decimation.time_series_decimation.anti_alias_filter
                == self.decimation.anti_alias_filter
            )
        except AssertionError:
            cond1 = self.decimation.anti_alias_filter == "default"
            cond2 = fc_decimation.time_series_decimation.anti_alias_filter is None
            if cond1 & cond2:
                pass
            else:
                msg = (
                    "Antialias Filters Not Compatible -- need to add handling for "
                    f"FCdec {fc_decimation.time_series_decimation.anti_alias_filter} and "
                    f"processing config:{self.decimation.anti_alias_filter}"
                )
                raise NotImplementedError(msg)

        # sample_rate
        try:
            assert (
                fc_decimation.time_series_decimation.sample_rate
                == self.decimation.sample_rate
            )
        except AssertionError:
            msg = (
                f"Sample rates do not agree: fc {fc_decimation.time_series_decimation.sample_rate} differs from "
                f"processing config {self.decimation.sample_rate}"
            )
            logger.info(msg)
            return False

        # transform method (fft, wavelet, etc.)
        try:
            assert (
                fc_decimation.short_time_fourier_transform.method == self.stft.method
            )  # FFT, Wavelet, etc.
        except AssertionError:
            msg = (
                "Transform methods do not agree: "
                f"fc {fc_decimation.short_time_fourier_transform.method} != processing config {self.stft.method}"
            )
            logger.info(msg)
            return False

        # prewhitening_type
        try:
            assert fc_decimation.stft.prewhitening_type == self.stft.prewhitening_type
        except AssertionError:
            msg = (
                "prewhitening_type does not agree "
                f"fc {fc_decimation.stft.prewhitening_type} != processing config {self.stft.prewhitening_type}"
            )
            logger.info(msg)
            return False

        if self.stft.prewhitening_type == "ARMA":
            for order_name in (
                "prewhitening_ar_order",
                "prewhitening_ma_order",
            ):
                if getattr(fc_decimation.stft, order_name) != getattr(
                    self.stft, order_name
                ):
                    msg = (
                        f"{order_name} does not agree "
                        f"fc {getattr(fc_decimation.stft, order_name)} != "
                        f"processing config {getattr(self.stft, order_name)}"
                    )
                    logger.info(msg)
                    return False

        # recoloring
        try:
            assert fc_decimation.stft.recoloring == self.stft.recoloring
        except AssertionError:
            msg = (
                "recoloring does not agree "
                f"fc {fc_decimation.stft.recoloring} != processing config {self.stft.recoloring}"
            )
            logger.info(msg)
            return False

        # pre_fft_detrend_type
        try:
            assert (
                fc_decimation.stft.pre_fft_detrend_type
                == self.stft.pre_fft_detrend_type
            )
        except AssertionError:
            msg = (
                "pre_fft_detrend_type does not agree "
                f"fc {fc_decimation.stft.pre_fft_detrend_type} != processing config {self.stft.pre_fft_detrend_type}"
            )
            logger.info(msg)
            return False

        # min_num_stft_windows
        try:
            assert (
                fc_decimation.stft.min_num_stft_windows
                == self.stft.min_num_stft_windows
            )
        except AssertionError:
            msg = (
                "min_num_stft_windows do not agree "
                f"fc {fc_decimation.stft.min_num_stft_windows} != processing config {self.stft.min_num_stft_windows}"
            )
            logger.info(msg)
            return False

        # window
        try:
            assert fc_decimation.stft.window == self.stft.window
        except AssertionError:
            msg = "window does not agree: "
            msg = f"{msg} FC Group: {fc_decimation.stft.window} "
            msg = f"{msg} Processing Config  {self.stft.window}"
            logger.info(msg)
            return False

        if fc_decimation.stft.harmonic_indices is None:
            # harmonic_indices not set, skip this check
            pass
        elif -1 in fc_decimation.stft.harmonic_indices:
            # if harmonic_indices is -1, it means the archive kept all so we can skip this check.
            pass
        else:
            msg = "WIP: harmonic indices in AuroraDecimationlevel are derived from processing bands -- Not robustly tested to compare with FCDecimation"
            logger.debug(msg)
            harmonic_indices_requested = self.harmonic_indices
            fcdec_group_set = set(fc_decimation.stft.harmonic_indices)
            processing_set = set(harmonic_indices_requested)
            if processing_set.issubset(fcdec_group_set):
                pass
            else:
                msg = (
                    f"Processing FC indices {processing_set} is not contained "
                    f"in FC indices {fcdec_group_set}"
                )
                logger.info(msg)
                return False

        # Getting here means no checks were failed. The FCDecimation supports the processing config
        return True

    def to_fc_decimation(
        self,
        remote: bool = False,
        ignore_harmonic_indices: bool = True,
    ) -> FCDecimation:
        """
        Generates a FC Decimation() object for use with FC Layer in mth5.

        TODO: this is being tested only in aurora -- move a test to mt_metadata or move the method.
        Ignoring for now these properties
        "time_period.end": "1980-01-01T00:00:00+00:00",
        "time_period.start": "1980-01-01T00:00:00+00:00",

        TODO: FIXME: Assignment of TSDecimation can be done in one shot once #235 is addressed.

        Parameters
        ----------
        remote: bool
            If True, use reference channels, if False, use local_channels.  We may wish to not pass remote=True when
            _building_ FCs however, because then not all channels will get built.
        ignore_harmonic_indices: bool
            If True, leave harmonic indices at default [-1,], which means all indices.  If False, only the specific
            harmonic indices needed for processing will be stored.  Thus, when building FCs, it maybe best to leave
            this as True, that way all FCs will be stored, so if the band setup is changed, the FCs will still be there.

        Returns:
            fc_dec_obj:mt_metadata.transfer_functions.processing.fourier_coefficients.decimation.Decimation
            A decimation object configured for STFT processing

        """

        fc_dec_obj = FCDecimation()  # type: ignore
        fc_dec_obj.time_series_decimation.anti_alias_filter = (
            self.decimation.anti_alias_filter
        )
        if remote:
            fc_dec_obj.channels_estimated = self.reference_channels
        else:
            fc_dec_obj.channels_estimated = self.local_channels
        fc_dec_obj.time_series_decimation.factor = self.decimation.factor
        fc_dec_obj.time_series_decimation.level = self.decimation.level
        if ignore_harmonic_indices:
            pass
        else:
            # Now that harmonic_indices is list[int], this should work
            fc_dec_obj.stft.harmonic_indices = self.harmonic_indices
        fc_dec_obj.id = f"{self.decimation.level}"
        fc_dec_obj.stft.method = self.stft.method
        fc_dec_obj.stft.pre_fft_detrend_type = self.stft.pre_fft_detrend_type
        fc_dec_obj.stft.prewhitening_type = self.stft.prewhitening_type
        fc_dec_obj.stft.prewhitening_ar_order = self.stft.prewhitening_ar_order
        fc_dec_obj.stft.prewhitening_ma_order = self.stft.prewhitening_ma_order
        fc_dec_obj.stft.recoloring = self.stft.recoloring
        fc_dec_obj.time_series_decimation.sample_rate = self.decimation.sample_rate
        fc_dec_obj.stft.window = self.stft.window

        return fc_dec_obj


def _df_from_bands(band_list: List[Union[Band, dict, None]]) -> pd.DataFrame:
    """
    Utility function that transforms a list of bands into a dataframe

    Note: The decimation_level here is +1 to agree with EMTF convention.
        Not clear this is really necessary
    TODO: Consider making this a method of FrequencyBands() class.
    TODO: Check typehint -- should None be allowed value in the band_list?
    TODO: Consider adding columns lower_closed, upper_closed to df

    Parameters
    ----------
    band_list: list
        obtained from mt_metadata.transfer_functions.processing.aurora.decimation_level.DecimationLevel.bands

    Returns
    -------
    out_df: pd.Dataframe
        Same format as that generated by EMTFBandSetupFile.get_decimation_level()
    """
    df_columns = [
        "decimation_level",
        "lower_bound_index",
        "upper_bound_index",
        "frequency_min",
        "frequency_max",
    ]
    n_rows = len(band_list)
    df_columns_dict = {}
    for col in df_columns:
        df_columns_dict[col] = n_rows * [None]
    for i_band, band in enumerate(band_list):
        df_columns_dict["decimation_level"][i_band] = band.decimation_level + 1
        df_columns_dict["lower_bound_index"][i_band] = band.index_min
        df_columns_dict["upper_bound_index"][i_band] = band.index_max
        df_columns_dict["frequency_min"][i_band] = band.frequency_min
        df_columns_dict["frequency_max"][i_band] = band.frequency_max
    out_df = pd.DataFrame(data=df_columns_dict)
    out_df.sort_values(by="lower_bound_index", inplace=True)
    out_df.reset_index(inplace=True, drop=True)
    return out_df


def get_fft_harmonics(samples_per_window: int, sample_rate: float) -> np.ndarray:
    """
    Works for odd and even number of points.

    Development notes:
    Could be modified with kwargs to support one_sided, two_sided, ignore_dc
    ignore_nyquist, and etc.  Consider taking FrequencyBands as an argument.

    Parameters
    ----------
    samples_per_window: integer
        Number of samples in a window that will be Fourier transformed.
    sample_rate: float
            Inverse of time step between samples,
            Samples per second

    Returns
    -------
    harmonic_frequencies: numpy array
        The frequencies that the fft will be computed.
        These are one-sided (positive frequencies only)
        Does not return Nyquist
        Does return DC component
    """
    n_fft_harmonics = int(samples_per_window / 2)  # no bin at Nyquist,
    delta_t = 1.0 / sample_rate
    harmonic_frequencies = np.fft.fftfreq(samples_per_window, d=delta_t)
    harmonic_frequencies = harmonic_frequencies[0:n_fft_harmonics]
    return harmonic_frequencies
