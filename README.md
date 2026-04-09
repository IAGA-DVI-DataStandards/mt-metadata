# mt_metadata version 1.0.7
 Standard MT metadata

[![PyPi version](https://img.shields.io/pypi/v/mt_metadata.svg)](https://pypi.python.org/pypi/mt-metadata)
[![Latest conda|conda-forge version](https://img.shields.io/conda/v/conda-forge/mt-metadata.svg)](https://anaconda.org/conda-forge/mt-metadata)
[![codecov](https://codecov.io/gh/kujaku11/mt_metadata/branch/main/graph/badge.svg?token=1WYF0G1L3D)](https://codecov.io/gh/kujaku11/mt_metadata)
![example workflow name](https://github.com/kujaku11/mt_metadata/workflows/TestingInConda/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/324097765.svg)](https://zenodo.org/badge/latestdoi/324097765)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kujaku11/mt_metadata/main)

# Description

MT Metadata is a project led by [IRIS-PASSCAL MT Software working group](https://www.iris.edu/hq/about_iris/governance/mt_soft>) and USGS to develop tools that standardize magnetotelluric metadata, well, at least create tools for standards that are generally accepted.  This include the two main types of magnetotelluric data

- **Time Series**
    - Structured as:
        - Experiment -> Survey -> Station -> Run -> Channel
    - Supports translation to/from **StationXML**

- **Transfer Functions**
    - Supports (will support) to/from:
        - **EDI** (most common format)
        - **ZMM** (Egberts EMTF output)
        - **JFILE** (BIRRP output)
        - **EMTFXML** (Kelbert's format)
        - **AVG** (Zonge output)

Most people will be using the transfer functions, but a lot of that metadata comes from the time series metadata.  This module supports both and has tried to make them more or less seamless to reduce complication.

* **Version**: 1.0.7
* **Free software**: MIT license
* **Documentation**: https://mt-metadata.readthedocs.io.
* **Examples**: Click the `Binder` badge above and Jupyter Notebook examples are in **mt_metadata/examples/notebooks** and **docs/source/notebooks**
* **Suggested Citation**: Peacock, J. R., Kappler, K., Ronan, T., Heagy, L.,  Kelbert, A., Frassetto, A. (2022) MTH5: An archive and exchangeable data format for magnetotelluric time series data, *Computers & Geoscience*, **162**, doi:10.1016/j.cageo.2022.105102
* **IPDS**: IP-138156


# Installation

## From Source

`git clone https://github.com/kujaku11/mt_metadata.git`

`pip install .`

You can add the flag `-e` if you want to install the source repository in an editable state.

## PIP
`pip install mt_metadata`

> You can install with optional packages by appending `[option_name]` to the package name during the
> `pip` install command. E.g:
>
> `pip install mt_metadata[obspy]`
>
> or `pip install .[obspy]` if building from source.

## Conda

`conda install mt_metadata`

# Standards

Each metadata keyword has an associated standard that goes with it.  These are stored internally in JSON file.  The JSON files are read in when the package is loaded to initialize the standards.  Each keyword is described by:

- **type** - How the value should be represented based on very basic types

    - *string*
    - *number* (float or integer)
    - *boolean*

- **required** -  A boolean (True or False) denoting whether the metadata key word required to represent the data.
- **style** - How the value should be represented within the type.  For instance is the value a controlled string where there are only a few options, or is the value a controlled naming convention where only a 5 character alpha-numeric string is allowed.  The styles are

    - *Alpha Numeric* a string with alphabetic and numberic characters
    - *Free Form* a free form string
    - *Controlled Vocabulary* only certain values are allowed according to **options**
    - *Date* a date and/or time string in ISO format
    - *Number* a float or integer
    - *Boolean* the value can only be True or False

- **units** - Units of the value
- **description** - Full description of what the metadata key is meant to convey.
- **options** - Any options of a **Controlled Vocabulary** style.
- **alias** - Any aliases that may represent the same metadata key.
- **example** - An example value to inform the user.

All input values are internally validated according to the definition providing a robust way to standardize metadata.

Each metadata object is based on a Base class that has methods:

- to/from_json
- to/from_xml
- to_from_dict
- attribute_information

And each object has a doc string that describes the standard:


| **Metadata Key**                             | **Description**                               | **Example**    |
|----------------------------------------------|-----------------------------------------------|----------------|
| **key**                                      | description of what the key describes         |  example value |
|                                              |                                               |                |
| Required: False                              |                                               |                |
|                                              |                                               |                |
| Units: None                                  |                                               |                |
|                                              |                                               |                |
| Type: string                                 |                                               |                |
|                                              |                                               |                |
| Style: controlled vocabulary                 |                                               |                |


The time series module is more mature than the transfer function module at the moment, and this is still a work in progress.


# Example

```
from mt_metadata import timeseries
x = timeseries.Instrument()

```
# Help
```
help(x)

+----------------------------------------------+-----------------------------------------------+----------------+
| **Metadata Key**                             | **Description**                               | **Example**    |
+==============================================+===============================================+================+
| **id**                                       | instrument ID number can be serial number or  | mt01           |
|                                              | a designated ID                               |                |
| Required: True                               |                                               |                |
|                                              |                                               |                |
| Units: None                                  |                                               |                |
|                                              |                                               |                |
| Type: string                                 |                                               |                |
|                                              |                                               |                |
| Style: free form                             |                                               |                |
+----------------------------------------------+-----------------------------------------------+----------------+
| **manufacturer**                             | who manufactured the instrument               | mt gurus       |
|                                              |                                               |                |
| Required: True                               |                                               |                |
|                                              |                                               |                |
| Units: None                                  |                                               |                |
|                                              |                                               |                |
| Type: string                                 |                                               |                |
|                                              |                                               |                |
| Style: free form                             |                                               |                |
+----------------------------------------------+-----------------------------------------------+----------------+
| **type**                                     | instrument type                               | broadband      |
|                                              |                                               | 32-bit         |
| Required: True                               |                                               |                |
|                                              |                                               |                |
| Units: None                                  |                                               |                |
|                                              |                                               |                |
| Type: string                                 |                                               |                |
|                                              |                                               |                |
| Style: free form                             |                                               |                |
+----------------------------------------------+-----------------------------------------------+----------------+
| **model**                                    | model version of the instrument               | falcon5        |
|                                              |                                               |                |
| Required: False                              |                                               |                |
|                                              |                                               |                |
| Units: None                                  |                                               |                |
|                                              |                                               |                |
| Type: string                                 |                                               |                |
|                                              |                                               |                |
| Style: free form                             |                                               |                |
+----------------------------------------------+-----------------------------------------------+----------------+
```

## Fill in metadata
```
x.model = "falcon 5"
x.type = "broadband 32-bit"
x.manufacturer = "MT Gurus"
x.id = "f176"
```

## to JSON
```
print(x.to_json())
{
    "instrument": {
        "id": "f176",
        "manufacturer": "MT Gurus",
        "model": "falcon 5",
        "type": "broadband 32-bit"
    }
}
```

## to XML
```
print(x.to_xml(string=True))
<?xml version="1.0" ?>
<instrument>
    <id>f176</id>
    <manufacturer>MT Gurus</manufacturer>
    <model>falcon 5</model>
    <type>broadband 32-bit</type>
</instrument>

```


Credits
-------

This project is in cooperation with the Incorporated Research Institutes of Seismology, the U.S. Geological Survey, and other collaborators.  Facilities of the IRIS Consortium are supported by the National Science Foundation’s Seismological Facilities for the Advancement of Geoscience (SAGE) Award under Cooperative Support Agreement EAR-1851048.  USGS is partially funded through the Community for Data Integration and IMAGe through the Minerals Resources Program.
