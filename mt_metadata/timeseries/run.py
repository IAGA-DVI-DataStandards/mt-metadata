# =====================================================
# Imports
# =====================================================
from __future__ import annotations

from collections import OrderedDict
from typing import Annotated

import numpy as np
from loguru import logger
from pydantic import (
    computed_field,
    Field,
    field_validator,
    model_validator,
    ValidationInfo,
)
from typing_extensions import Self

from mt_metadata.base import MetadataBase
from mt_metadata.common import (
    AuthorPerson,
    Comment,
    DataTypeEnum,
    Fdsn,
    Provenance,
    TimePeriod,
)
from mt_metadata.common.list_dict import ListDict
from mt_metadata.timeseries import Auxiliary, DataLogger, Electric, Magnetic

# =====================================================


class Run(MetadataBase):
    channels_recorded_auxiliary: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="List of auxiliary channels recorded",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["[T]"],
            },
        ),
    ]

    channels_recorded_electric: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="List of electric channels recorded",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["[Ex , Ey]"],
            },
        ),
    ]

    channels_recorded_magnetic: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="List of magnetic channels recorded",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["[Hx , Hy , Hz]"],
            },
        ),
    ]

    @computed_field
    def channels_recorded_all(self) -> list[str]:
        """
        List of all channels recorded in the run.
        """
        return sorted(
            [ch.component for ch in self.channels.values() if ch.component is not None]
        )

    comments: Annotated[
        Comment,
        Field(
            default_factory=Comment,  # type: ignore
            description="Any comments on the run.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["cows chewed cables"],
            },
        ),
    ]

    data_type: Annotated[
        DataTypeEnum,
        Field(
            default=DataTypeEnum.BBMT,
            description="Type of data recorded for this run.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["BBMT"],
            },
        ),
    ]

    id: Annotated[
        str,
        Field(
            default="",
            description="Run ID should be station name followed by a number or character.  Characters should only be used if the run number is small, if the run number is high consider using digits with zeros.  For example if you have 100 runs the run ID could be 001 or {station}001.",
            alias=None,
            pattern="^[a-zA-Z0-9_]*$",
            json_schema_extra={
                "units": None,
                "required": True,
                "examples": ["001"],
            },
        ),
    ]

    sample_rate: Annotated[
        float,
        Field(
            default=0.0,
            description="Digital sample rate for the run",
            alias=None,
            json_schema_extra={
                "units": "samples per second",
                "required": True,
                "examples": ["100"],
            },
        ),
    ]

    acquired_by: Annotated[
        AuthorPerson,
        Field(
            default_factory=AuthorPerson,  # type: ignore
            description="Information about the group that collected the data.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["Person()"],
            },
        ),
    ]

    metadata_by: Annotated[
        AuthorPerson,
        Field(
            default_factory=AuthorPerson,  # type: ignore
            description="Information about the group that collected the metadata.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["Person()"],
            },
        ),
    ]

    provenance: Annotated[
        Provenance,
        Field(
            default_factory=Provenance,  # type: ignore
            description="Provenance information about the run.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["Provenance()"],
            },
        ),
    ]

    time_period: Annotated[
        TimePeriod,
        Field(
            default_factory=TimePeriod,  # type: ignore
            description="Time period for the run.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["TimePeriod(start='2020-01-01', end='2020-12-31')"],
            },
        ),
    ]

    data_logger: Annotated[
        DataLogger,
        Field(
            default_factory=DataLogger,  # type: ignore
            description="Data Logger information used to collect the run.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["DataLogger()"],
            },
        ),
    ]

    fdsn: Annotated[
        Fdsn,
        Field(
            default_factory=Fdsn,  # type: ignore
            description="FDSN information for the run.",
            alias=None,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["Fdsn()"],
            },
        ),
    ]

    channels: Annotated[
        ListDict | list | dict | OrderedDict,
        Field(
            default_factory=ListDict,
            description="ListDict of channel objects collected in this run.",
            alias=None,
            exclude=True,
            json_schema_extra={
                "units": None,
                "required": False,
                "examples": ["ListDict(Electric(), Magnetic(), Auxiliary())"],
            },
        ),
    ]

    @field_validator("comments", mode="before")
    @classmethod
    def validate_comments(cls, value, info: ValidationInfo) -> Comment:
        """
        Validate that the value is a valid comment.
        """
        if isinstance(value, str):
            return Comment(value=value)
        return value

    @field_validator("data_type", mode="before")
    @classmethod
    def validate_data_type(cls, value, info: ValidationInfo) -> str:
        """
        Validate that the data_type is a string.
        """
        if isinstance(value, DataTypeEnum):
            value = value.value
        elif isinstance(value, str):
            if "," in value:
                value = value.split(",")[0].strip()
        elif not isinstance(value, str):
            raise TypeError(f"data_type must be a string, not {type(value)}")
        return value

    @field_validator(
        "channels_recorded_electric",
        "channels_recorded_magnetic",
        "channels_recorded_auxiliary",
        mode="before",
    )
    @classmethod
    def validate_list_of_strings(
        cls, value: np.ndarray | list[str] | str, info: ValidationInfo
    ) -> list[str]:
        """
        Validate that the value is a list of strings.
        """
        if value in [None, "None", "none", "NONE", "null"]:
            return []

        if isinstance(value, np.ndarray):
            value = value.astype(str).tolist()

        elif isinstance(value, (list, tuple)):
            value = [str(v) for v in value]

        elif isinstance(value, (str)):
            value = [v.strip() for v in value.split(",")]

        else:
            raise TypeError(
                "'channels_recorded' must be set with a list of strings not "
                f"{type(value)}."
            )
        return value

    @model_validator(mode="after")
    def validate_channels_recorded(self) -> Self:
        """
        Validate that the value is a list of strings.
        """
        # need to make each another object list() otherwise the contents
        # get overwritten with the new channel.
        for electric in list(self.channels_recorded_electric):
            if electric not in self.channels.keys():
                self.add_channel(Electric(component=electric))  # type: ignore
        for magnetic in list(self.channels_recorded_magnetic):
            if magnetic not in self.channels.keys():
                self.add_channel(Magnetic(component=magnetic))  # type: ignore
        for auxiliary in list(self.channels_recorded_auxiliary):
            if auxiliary not in self.channels.keys():
                self.add_channel(Auxiliary(component=auxiliary))  # type: ignore
        return self

    @field_validator("channels", mode="before")
    @classmethod
    def validate_channels(cls, value, info: ValidationInfo) -> ListDict:
        if not isinstance(value, (list, tuple, dict, ListDict, OrderedDict)):
            msg = (
                "input run_list must be an iterable, should be a list or dict "
                f"not {type(value)}"
            )
            logger.error(msg)
            raise TypeError(msg)

        fails = []
        channels = ListDict()
        if isinstance(value, (dict, ListDict, OrderedDict)):
            value_list = value.values()

        elif isinstance(value, (list, tuple)):
            value_list = value

        for ii, channel in enumerate(value_list):
            try:
                channels.append(cls._get_correct_channel_type(channel))
            except KeyError as error:
                msg = f"Could not find type in {channel}"
                logger.error(msg)
                logger.exception(error)
                fails.append(msg)

        if len(fails) > 0:
            raise TypeError("\n".join(fails))

        return channels

    def _empty_channels_recorded(self) -> None:
        """
        Clear all channels recorded lists.

        Removes all entries from the auxiliary, electric, and magnetic
        channels recorded lists. This is typically called before updating
        the lists based on current channel contents.

        See Also
        --------
        _update_channels_recorded : Update channels recorded lists from current channels

        """
        self.channels_recorded_auxiliary.clear()
        self.channels_recorded_electric.clear()
        self.channels_recorded_magnetic.clear()

    def _update_channels_recorded(self) -> None:
        """
        Update channels recorded lists based on current channels.

        Scans all channels in the run and populates the appropriate
        channels_recorded lists (auxiliary, electric, magnetic) based on
        channel types and components. Excludes default/None components.

        Notes
        -----
        This method is automatically called when channels are added or removed.
        The lists are sorted alphabetically.

        Excluded components:

        - Auxiliary: None, 'auxiliary_default'
        - Electric: None, 'e_default'
        - Magnetic: None, 'h_default'

        Examples
        --------
        >>> run = Run(id='001')
        >>> run.add_channel(Electric(component='ex'))
        >>> run.add_channel(Magnetic(component='hx'))
        >>> # _update_channels_recorded is called automatically
        >>> print(run.channels_recorded_electric)
        ['ex']
        >>> print(run.channels_recorded_magnetic)
        ['hx']

        """
        self._empty_channels_recorded()
        aux_components = [
            ch.component
            for ch in self.channels
            if isinstance(ch, Auxiliary)
            and ch.component not in [None, "auxiliary_default"]
        ]
        elec_components = [
            ch.component
            for ch in self.channels
            if isinstance(ch, Electric) and ch.component not in [None, "e_default"]
        ]
        mag_components = [
            ch.component
            for ch in self.channels
            if isinstance(ch, Magnetic) and ch.component not in [None, "h_default"]
        ]
        self.channels_recorded_auxiliary = sorted(aux_components)
        self.channels_recorded_electric = sorted(elec_components)
        self.channels_recorded_magnetic = sorted(mag_components)

    # def __len__(self):
    #     return len(self.channels)

    def merge(self, other: Run, inplace: bool = True) -> Run | None:
        """
        Merge channels from another Run into this run.

        Combines channels from two runs and updates the channels recorded lists
        and time period.

        Parameters
        ----------
        other : Run
            Another Run object whose channels will be merged into this run.
        inplace : bool, optional
            If True, update this run and update time period. If False, return
            a copy of the merged run (default is True).

        Returns
        -------
        Run | None
            If inplace is False, returns a copy of the merged Run. Otherwise None.

        Raises
        ------
        TypeError
            If other is not a Run object.

        Examples
        --------
        Merge runs in place:

        >>> run1 = Run(id='001')
        >>> run1.add_channel(Electric(component='ex'))
        >>> run2 = Run(id='002')
        >>> run2.add_channel(Magnetic(component='hx'))
        >>> run1.merge(run2, inplace=True)
        >>> print(run1.channels_recorded_all)
        ['ex', 'hx']

        Merge and return new run:

        >>> merged_run = run1.merge(run2, inplace=False)
        >>> print(merged_run.channels_recorded_all)
        ['ex', 'hx']

        See Also
        --------
        update : Update metadata from another run

        """
        if isinstance(other, Run):
            self.channels.extend(other.channels)
            self._update_channels_recorded()
            if inplace:
                self.update_time_period()
            else:
                return self.copy()
        else:
            msg = f"Can only merge Run objects, not {type(other)}"
            logger.error(msg)
            raise TypeError(msg)

    def update(self, other: Run, match: list[str] | None = []) -> None:
        """
        Update attribute values from another Run object.

        Copies non-None, non-default attribute values from another Run object
        to this one. Skips empty values like None, 0.0, [], empty strings,
        and default timestamps.

        Parameters
        ----------
        other : Run
            Another Run object to copy attributes from.
        match : list[str] | None, optional
            List of attribute names that must match between runs before updating.
            If any don't match, raises ValueError. Typically used for 'id' to
            ensure runs are compatible (default is None).

        Raises
        ------
        ValueError
            If any attributes in match list don't have equal values.
        TypeError
            If other is not a compatible Run type.

        Examples
        --------
        Basic update:

        >>> run1 = Run(id='001', sample_rate=256.0)
        >>> run2 = Run(id='001', sample_rate=0.0)
        >>> run2.acquired_by.author = 'J. Doe'
        >>> run1.update(run2)
        >>> print(run1.acquired_by.author)
        'J. Doe'
        >>> print(run1.sample_rate)  # Not updated (run2 has default 0.0)
        256.0

        Update with matching check:

        >>> run1 = Run(id='001')
        >>> run2 = Run(id='002')
        >>> try:
        ...     run1.update(run2, match=['id'])
        ... except ValueError as e:
        ...     print("IDs don't match!")
        IDs don't match!

        Notes
        -----
        Channel metadata is also updated. For each channel in other, if the
        channel exists in this run, it's updated; if not, it's added.

        Skipped values:

        - None
        - 0.0
        - Empty lists []
        - Empty strings ''
        - Default timestamp '1980-01-01T00:00:00+00:00'

        See Also
        --------
        merge : Merge channels from another run

        """
        # Check if other is a compatible Run type (handles dynamically created classes)
        if not (
            isinstance(other, type(self))
            or (hasattr(other, "__class__") and other.__class__.__name__ == "Run")
        ):
            logger.warning(f"Cannot update {type(self)} with {type(other)}")
        for k in match:
            if self.get_attr_from_name(k) != other.get_attr_from_name(k):
                msg = (
                    f"{k} is not equal {self.get_attr_from_name(k)} != "
                    "{other.get_attr_from_name(k)}"
                )
                logger.error(msg)
                raise ValueError(msg)
        for k, v in other.to_dict(single=True).items():
            if hasattr(v, "size"):
                if v.size > 0:
                    self.update_attribute(k, v)
            else:
                if v not in [None, 0.0, [], "", "1980-01-01T00:00:00+00:00"]:
                    self.update_attribute(k, v)

        ## Need this because channels are set when setting channels_recorded
        ## and it initiates an empty channel, but we need to fill it with
        ## the appropriate metadata.
        for ch in other.channels:
            self.add_channel(ch)

    def has_channel(self, component: str) -> bool:
        """
        Check if a channel with the given component exists in the run.

        Parameters
        ----------
        component : str
            Channel component name to search for (e.g., 'ex', 'hy').

        Returns
        -------
        bool
            True if channel exists, False otherwise.

        Examples
        --------
        >>> run = Run(id='001')
        >>> run.add_channel(Electric(component='ex'))
        >>> print(run.has_channel('ex'))
        True
        >>> print(run.has_channel('ey'))
        False

        See Also
        --------
        get_channel : Retrieve a channel object
        channel_index : Get the index of a channel

        """

        if component in [cc for cc in self.channels_recorded_all]:
            return True
        return False

    def channel_index(self, component: str) -> int | None:
        """
        Get the index of a channel in the channels_recorded_all list.

        Parameters
        ----------
        component : str
            Channel component name to search for (e.g., 'ex', 'hy').

        Returns
        -------
        int | None
            Index of the channel if found, None otherwise.

        Examples
        --------
        >>> run = Run(id='001')
        >>> run.add_channel(Electric(component='ex'))
        >>> run.add_channel(Electric(component='ey'))
        >>> run.add_channel(Magnetic(component='hx'))
        >>> print(run.channel_index('ey'))
        1
        >>> print(run.channel_index('hz'))
        None

        Notes
        -----
        Channels are sorted alphabetically in channels_recorded_all.

        See Also
        --------
        has_channel : Check if channel exists
        get_channel : Retrieve channel object

        """
        if self.has_channel(component):
            return self.channels_recorded_all.index(component)
        return None

    def get_channel(self, component: str) -> Electric | Magnetic | Auxiliary | None:
        """
        Retrieve a channel object by component name.

        Parameters
        ----------
        component : str
            Channel component name to retrieve (e.g., 'ex', 'hy').

        Returns
        -------
        Electric | Magnetic | Auxiliary | None
            Channel object if found, None otherwise. Return type depends on
            the channel type.

        Examples
        --------
        >>> run = Run(id='001')
        >>> ex = Electric(component='ex', dipole_length=100.0)
        >>> run.add_channel(ex)
        >>> channel = run.get_channel('ex')
        >>> print(type(channel).__name__)
        'Electric'
        >>> print(channel.dipole_length)
        100.0
        >>> print(run.get_channel('ey'))
        None

        See Also
        --------
        has_channel : Check if channel exists
        add_channel : Add a channel to the run

        """

        if self.has_channel(component):
            return self.channels[component]

    def add_channel(
        self,
        channel_obj: Electric | Magnetic | Auxiliary | dict | str,
        update: bool = True,
    ) -> None:
        """
        Add or update a channel in the run.

        If the channel already exists (matched by component), its metadata
        is updated. If it doesn't exist, it's added to the channels list.
        Can accept channel objects, dictionaries, or component strings.

        Parameters
        ----------
        channel_obj : Electric | Magnetic | Auxiliary | dict | str
            Channel to add. Can be:

            - Channel object (Electric, Magnetic, or Auxiliary)
            - Dictionary with channel attributes (must include 'type' or 'component')
            - String component name (e.g., 'ex', 'hy', 'temp')

            If string, channel type is inferred:

            - Starts with 'e' → Electric
            - Starts with 'h' or 'b' or equals 'magnetic' → Magnetic
            - Otherwise → Auxiliary

        update : bool, optional
            If True, update the run's time period to include this channel's
            time period. If False, don't update time period (default is True).

        Examples
        --------
        Add channel objects:

        >>> run = Run(id='001')
        >>> ex = Electric(component='ex', dipole_length=100.0)
        >>> run.add_channel(ex)
        >>> print(run.channels_recorded_electric)
        ['ex']

        Add from string (infers type):

        >>> run.add_channel('hy')
        >>> run.add_channel('temperature')
        >>> print(run.channels_recorded_magnetic)
        ['hy']
        >>> print(run.channels_recorded_auxiliary)
        ['temperature']

        Add from dictionary:

        >>> channel_dict = {
        ...     'type': 'electric',
        ...     'component': 'ey',
        ...     'dipole_length': 95.0
        ... }
        >>> run.add_channel(channel_dict)

        Update existing channel:

        >>> ex_updated = Electric(component='ex', dipole_length=105.0)
        >>> run.add_channel(ex_updated)  # Updates existing 'ex'
        >>> print(run.get_channel('ex').dipole_length)
        105.0

        Add without updating time period:

        >>> run.add_channel('hz', update=False)

        Notes
        -----
        This method automatically:

        - Updates channels_recorded lists
        - Updates run time period (if update=True)
        - Converts string/dict inputs to proper channel objects
        - Logs when updating existing channels

        See Also
        --------
        remove_channel : Remove a channel from the run
        get_channel : Retrieve a channel object
        update_time_period : Manually update time period

        """
        channel_obj = self._get_correct_channel_type(channel_obj)

        if self.has_channel(channel_obj.component):
            self.channels[channel_obj.component].update(channel_obj)
            logger.debug(
                f"Run {channel_obj.component} already exists, updating metadata"
            )

        else:
            self.channels.append(channel_obj)

        self._update_channels_recorded()

        if update:
            self.update_time_period()

    def remove_channel(self, channel_id: str) -> None:
        """
        Remove a channel from the run.

        Parameters
        ----------
        channel_id : str
            Channel component name to remove (e.g., 'ex', 'hy').

        Examples
        --------
        >>> run = Run(id='001')
        >>> run.add_channel(Electric(component='ex'))
        >>> run.add_channel(Electric(component='ey'))
        >>> print(run.channels_recorded_electric)
        ['ex', 'ey']
        >>> run.remove_channel('ex')
        >>> print(run.channels_recorded_electric)
        ['ey']
        >>> run.remove_channel('ez')  # Doesn't exist
        # Logs warning: Could not find ez to remove.

        Notes
        -----
        Automatically updates the channels_recorded lists after removal.
        Logs a warning if the channel is not found.

        See Also
        --------
        add_channel : Add a channel to the run
        has_channel : Check if channel exists

        """

        if self.has_channel(channel_id):
            self.channels.remove(channel_id)

            self._update_channels_recorded()
        else:
            logger.warning(f"Could not find {channel_id} to remove.")

    def update_channel_keys(self) -> dict[str, str]:
        """
        Update channel dictionary keys to match current component values.

        Updates the keys in the channels ListDict to match current channel
        components. Useful when channel components have been modified after
        channels were added, ensuring channels can be accessed by their
        current component values.

        Returns
        -------
        dict[str, str]
            Mapping of old keys to new keys showing what was changed.

        Examples
        --------
        Fix keys after modifying components:

        >>> run = Run(id='001')
        >>> channel = Electric(component='')
        >>> run.add_channel(channel)
        >>> # Channel is stored with empty string key
        >>> channel.component = 'ex'
        >>> key_mapping = run.update_channel_keys()
        >>> print(key_mapping)
        {'': 'ex'}
        >>> # Now accessible as run.channels['ex']
        >>> print(run.get_channel('ex').component)
        'ex'

        Multiple key updates:

        >>> run = Run(id='001')
        >>> ch1 = Electric(component='e1')
        >>> ch2 = Magnetic(component='h1')
        >>> run.add_channel(ch1)
        >>> run.add_channel(ch2)
        >>> ch1.component = 'ex'
        >>> ch2.component = 'hx'
        >>> mapping = run.update_channel_keys()
        >>> print(mapping)
        {'e1': 'ex', 'h1': 'hx'}

        Notes
        -----
        This is typically only needed if you've directly modified channel
        component attributes after adding them to the run. Normal usage
        doesn't require calling this method.

        See Also
        --------
        add_channel : Add channels to the run
        get_channel : Access channels by component

        """
        return self.channels.update_keys()

    @property
    def n_channels(self) -> int:
        """
        Number of channels in the run.

        Returns
        -------
        int
            Count of channels currently in the run.

        Examples
        --------
        >>> run = Run(id='001')
        >>> print(run.n_channels)
        0
        >>> run.add_channel('ex')
        >>> run.add_channel('hy')
        >>> print(run.n_channels)
        2
        """
        return len(self.channels)

    def update_time_period(self) -> None:
        """
        Update run's time period to encompass all channel time periods.

        Examines all channels in the run and updates the run's start and end
        times to include the earliest start and latest end from all channels.
        Ignores default timestamp '1980-01-01T00:00:00+00:00'.

        Examples
        --------
        >>> from mt_metadata.timeseries import Run, Electric
        >>> run = Run(id='001')
        >>> ex = Electric(component='ex')
        >>> ex.time_period.start = '2020-01-01T00:00:00+00:00'
        >>> ex.time_period.end = '2020-01-01T01:00:00+00:00'
        >>> run.add_channel(ex, update=False)
        >>> print(run.time_period.start)
        1980-01-01T00:00:00+00:00
        >>> run.update_time_period()
        >>> print(run.time_period.start)
        2020-01-01T00:00:00+00:00

        Multiple channels:

        >>> ey = Electric(component='ey')
        >>> ey.time_period.start = '2020-01-01T00:30:00+00:00'
        >>> ey.time_period.end = '2020-01-01T02:00:00+00:00'
        >>> run.add_channel(ey, update=True)
        >>> print(run.time_period.start)  # Uses earliest
        2020-01-01T00:00:00+00:00
        >>> print(run.time_period.end)  # Uses latest
        2020-01-01T02:00:00+00:00

        Notes
        -----
        - Only updates if channels exist (n_channels > 0)
        - Ignores channels with default timestamp
        - Always expands time period, never shrinks it
        - Automatically called by add_channel() when update=True

        See Also
        --------
        add_channel : Add channel and optionally update time period

        """
        if self.n_channels > 0:
            start = []
            end = []
            for channel in self.channels:
                if channel.time_period.start != "1980-01-01T00:00:00+00:00":
                    start.append(channel.time_period.start)
                if channel.time_period.end != "1980-01-01T00:00:00+00:00":
                    end.append(channel.time_period.end)

            if start:
                if self.time_period.start == "1980-01-01T00:00:00+00:00":
                    self.time_period.start = min(start)
                else:
                    if self.time_period.start > min(start):
                        self.time_period.start = min(start)

            if end:
                if self.time_period.end == "1980-01-01T00:00:00+00:00":
                    self.time_period.end = max(end)
                else:
                    if self.time_period.end < max(end):
                        self.time_period.end = max(end)

    @classmethod
    def _get_correct_channel_type(
        cls, channel_obj: Electric | Magnetic | Auxiliary | dict | OrderedDict | str
    ) -> Electric | Magnetic | Auxiliary:
        """
        Convert input to the correct channel type object.

        Determines the appropriate channel type (Electric, Magnetic, or
        Auxiliary) based on the input and returns a properly typed channel
        object. Handles string, dict, and object inputs.

        Parameters
        ----------
        channel_obj : Electric | Magnetic | Auxiliary | dict | OrderedDict | str
            Input to convert to a channel object. Can be:

            - Channel object: returned as-is (after validation)
            - String: component name, type inferred from first letter
            - Dict: must contain 'type' or 'component' key

        Returns
        -------
        Electric | Magnetic | Auxiliary
            Properly typed channel object.

        Raises
        ------
        ValueError
            If channel_obj is a channel with None component (except Auxiliary),
            or if input type is not supported.
        KeyError
            If dict input doesn't contain required keys.

        Examples
        --------
        From string:

        >>> channel = Run._get_correct_channel_type('ex')
        >>> print(type(channel).__name__)
        'Electric'
        >>> print(channel.component)
        'ex'

        From dict:

        >>> ch_dict = {'type': 'magnetic', 'component': 'hx'}
        >>> channel = Run._get_correct_channel_type(ch_dict)
        >>> print(type(channel).__name__)
        'Magnetic'

        Type inference from component letter:

        >>> Run._get_correct_channel_type('ey')  # 'e' → Electric
        <Electric ...>
        >>> Run._get_correct_channel_type('hz')  # 'h' → Magnetic
        <Magnetic ...>
        >>> Run._get_correct_channel_type('temp')  # other → Auxiliary
        <Auxiliary ...>

        Notes
        -----
        Type inference rules:

        **From string component:**

        - Starts with 'e' → Electric
        - Starts with 'h', 'b', or equals 'magnetic' → Magnetic
        - Anything else → Auxiliary

        **From dict:**

        - Uses 'type' key if present, otherwise first letter of 'component'
        - Same inference rules apply

        See Also
        --------
        add_channel : Add a channel using this type detection

        """
        if isinstance(channel_obj, str):
            if channel_obj.lower().startswith("e"):
                return Electric(component=channel_obj)
            elif (
                channel_obj.lower().startswith("h")
                or channel_obj.lower().startswith("b")
                or channel_obj.lower() in ["magnetic"]
            ):
                return Magnetic(component=channel_obj)
            else:
                return Auxiliary(component=channel_obj)

        elif isinstance(channel_obj, (dict, OrderedDict)):
            try:
                ch_type = channel_obj["type"]
                if ch_type is None:
                    ch_type = channel_obj["component"][0]

                if ch_type.lower().startswith("e"):
                    return Electric(**channel_obj)
                elif (
                    ch_type.lower().startswith("h")
                    or ch_type.lower().startswith("b")
                    or ch_type.lower() in ["magnetic"]
                ):
                    return Magnetic(**channel_obj)
                else:
                    return Auxiliary(**channel_obj)
            except KeyError as error:
                msg = f"Could not find type in {channel_obj}"
                logger.error(msg)
                logger.exception(error)
                raise error

        elif isinstance(channel_obj, (Electric, Magnetic, Auxiliary)):
            if channel_obj.component is None:
                if not isinstance(channel_obj, Auxiliary):
                    msg = "component cannot be empty"
                    logger.error(msg)
                    raise ValueError(msg)
            return channel_obj
        else:
            msg = (
                f"Input must be mt_metadata.timeseries.Channel not {type(channel_obj)}"
            )
            logger.error(msg)
            raise ValueError(msg)
