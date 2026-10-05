from typing import Annotated
from xml.etree import ElementTree as et

from pydantic import Field

from mt_metadata.common import BasicLocation, Declination
from mt_metadata.transfer_functions.io.emtfxml.metadata.helpers import element_to_string


class Location(BasicLocation):
    declination: Annotated[
        Declination,
        Field(
            default_factory=Declination,  # type: ignore
            description="Declination at the location in degrees",
            alias=None,
            json_schema_extra={
                "units": "degrees",
                "required": False,
                "examples": ["10.0"],
            },
        ),
    ]

    def to_xml(self, string=False, required=True):
        """
        Overwrite to XML to follow EMTF XML format

        Parameters
        -------------

        string: bool
            If True, return the XML as a string. If False, return an ElementTree Element. Defaults to False.
        required: bool
            If True, include all required fields in the XML. Defaults to True.

        Returns
        -----------
            XML representation of the BasicLocationDeclination object as a string or ElementTree Element.

        """
        if self.datum is None:
            self.datum = "WGS84"
        if self.declination.epoch is None:
            self.declination.epoch = "1995"

        root = et.Element(
            self.__class__.__name__.capitalize(), {"datum": self.datum.replace(" ", "")}
        )
        lat = et.SubElement(root, "Latitude")
        lat.text = f"{self.latitude:.6f}"
        lon = et.SubElement(root, "Longitude")
        lon.text = f"{self.longitude:.6f}"
        elev = et.SubElement(root, "Elevation", {"units": "meters"})
        elev.text = f"{self.elevation:.3f}"
        dec = et.SubElement(
            root, "Declination", {"epoch": self.declination.epoch.split(".", 1)[0]}
        )
        dec.text = f"{self.declination.value:.3f}"

        if not string:
            return root
        else:
            return element_to_string(root)
