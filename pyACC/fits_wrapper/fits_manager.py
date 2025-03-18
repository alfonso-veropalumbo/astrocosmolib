# Open what is needed to open fits files
from astropy.io import fits

# import numpy
import numpy as np

# Import the logger
from ..helpers.logger import Logger


class FitsManager:
    """
    A class to manage FITS files, using astropy.io.fits module
    """

    # Define the constructor
    def __init__(self, input_file):
        """
        Constructor of the class
        :param input_file: The input file to be opened
        """
        self.input_file = input_file
        self.hdulist = fits.open(input_file)
        
        # Create the logger
        self.logger = Logger("FitsManager")
        self.logger("FITS file opened successfully")

    def get_hdu_count(self):
        """
        Get the number of HDUs in the FITS file
        :return: The number of HDUs
        """

        return len(self.hdulist)
    
    def get_header(self, hdu_index):
        """
        Get the header of a given HDU
        :param hdu_index: The index of the HDU to get the header from
        :return: The header of the HDU
        """

        if hdu_index < 0 or hdu_index >= len(self.hdulist):
            self.logger.error("Invalid HDU index", ValueError)

        return self.hdulist[hdu_index].header
    
    def get_data(self, hdu_index):
        """
        Get the data of a given HDU
        :param hdu: The index of the HDU to get the data from
        :return: The data of the HDU
        """

        if hdu_index < 0 or hdu_index >= len(self.hdulist):
            self.logger.error("Invalid HDU index", ValueError)

        return self.hdulist[hdu_index].data
    