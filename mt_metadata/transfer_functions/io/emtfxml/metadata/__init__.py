# # -*- coding: utf-8 -*-
# """
# ==================
# EMTF XML metadata
# ==================

# The metadata for transfer functions follows those proposed by Anna Kelbert `EMTF XML: New data interchange format and conversion tools for electromagnetic transfer functions <http://mr.crossref.org/iPage?doi=10.1190%2Fgeo2018-0679.1>`__.

# :copyright:
#     Jared Peacock (jpeacock@usgs.gov)

# :license:
#     MIT


# """

# # package file

# ### !!! DO NOT CHANGE THE ORDER !!!
from .external_url import ExternalUrl
from .primary_data import PrimaryData
from .attachment import Attachment
from .person import Person
from .citation import Citation
from .provenance import Provenance
from .copyright import Copyright
from .data_quality_notes import DataQualityNotes
from .data_quality_warnings import DataQualityWarnings
from .orientation import Orientation
from .location import Location
from .site import Site
from .instrument import Instrument
from .electrode import Electrode
from .dipole import Dipole
from .magnetometer import Magnetometer
from .run import Run
from .field_notes import FieldNotes
from .software import ProcessingSoftware
from .remote_ref import RemoteRef
from .remote_info import RemoteInfo
from .processing_info import ProcessingInfo
from .estimate import Estimate
from .statistical_estimates import StatisticalEstimates
from .data_type import DataType
from .data_types import DataTypes
from .magnetic import Magnetic
from .electric import Electric
from .channels import Channels
from .site_layout import SiteLayout
from .period_range import PeriodRange
from .data import TransferFunction
from .emtf import EMTF

__all__ = [
    "ExternalUrl",
    "PrimaryData",
    "Attachment",
    "Person",
    "Provenance",
    "Citation",
    "Copyright",
    "DataQualityNotes",
    "DataQualityWarnings",
    "Orientation",
    "Location",
    "Site",
    "Instrument",
    "Electrode",
    "Dipole",
    "Magnetometer",
    "Run",
    "FieldNotes",
    "ProcessingSoftware",
    "RemoteRef",
    "RemoteInfo",
    "ProcessingInfo",
    "Estimate",
    "StatisticalEstimates",
    "DataType",
    "DataTypes",
    "Magnetic",
    "Electric",
    "Channels",
    "SiteLayout",
    "PeriodRange",
    "TransferFunction",
    "EMTF",
]
