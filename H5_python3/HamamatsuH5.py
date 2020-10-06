"""
Module stores a variety of classes and functions useful for interacting with the hamamatsu data
stored in the results.hdf5 file
"""

from typing import Dict
import numpy as np
from numpy import s_
from recordclass import recordclass as rc
import warnings

FrameGrabberAqRegion = rc('FrameGrabberAqRegion', ('left', 'right', 'top', 'bottom'))


class HMROI:
    """
    Class to organize the ROI used during analysis. Supports rectangular ROI

    Use as an ROI to cut out a rectangular section of an image encoded as a numpy array. Use the
    width and height attributes as the width and height of the larger image. the ROI is set to be
    bounded by width and height.

    Typical usages:
        > width, height = original_image.shape
        > roi = HMROI(width, height, dic = {
        >     'top': 3,
        >     'bottom': 7,
        >     'left': 4,
        >     'right': 9
        > })
        > sliced_image = original_image[roi.slice]

        > width, height = original_image.shape
        > roi = HMROI(width, height)
        > roi["top"] = 3
        > roi["bottom"] = 7
        > roi.left = 4
        > roi.right = 9
        > sliced_image = original_image[roi.slice]

    Attributes:
        width: width of the image this ROI is meant to slice
        height: height of the image this ROI is meant to slice
        top: top edge of the roi
        bottom: bottom edge of the roi
        left: left edge of the roi
        right: right edge of the roi
        slice: numpy compatible slice to use as an index for the original image.
    """

    __slots__ = ('__width', '__height', '__top', '__vals')

    def __init__(self, width, height, dic: Dict[str, int] = None):
        """
        Args:
            width: width of the whole image to be sliced
            height: height of the whole image to be sliced
            dic: dict of top, bottom, left, and right edges of ROI.
                must be a dict with keys "top", "bottom", "left", "right"
        """
        self.__width = int(width)
        self.__height = int(height)

        self.__vals = {
            "top": 0,
            "bottom": height,
            "left": 0,
            "right": width
        }

        if dic is not None:
            try:
                self.top = dic["top"]
                self.bottom = dic["bottom"]
                self.left = dic["left"]
                self.right = dic["right"]
            except KeyError as e:
                raise ValueError(
                    f"dic must be a dictionary with keys {list(self.__vals.keys())}"
                ) from e

    # RO properties
    @property
    def width(self):
        return self.__width

    @property
    def height(self):
        return self.__height

    # RW properties
    @property
    def top(self):
        return self["top"]

    @top.setter
    def top(self, value):
        self["top"] = value

    @property
    def bottom(self):
        return self["bottom"]

    @bottom.setter
    def bottom(self, value):
        self["bottom"] = value

    @property
    def left(self):
        return self["left"]

    @left.setter
    def left(self, value):
        self["left"] = value

    @property
    def right(self):
        return self["right"]

    @right.setter
    def right(self, value):
        self["right"] = value

    def __len__(self):
        return len(self.__vals)

    def __getitem__(self, key):
        return self.__vals[key]

    def __setitem__(self, key, value):
        value = int(value)

        if key == "top":
            self.__vals[key] = value if 0 < value < self.bottom else 0
        elif key == "bottom":
            self.__vals[key] = value if self.top < value < self.height else self.height
        elif key == "left":
            self.__vals[key] = value if 0 < value < self.right else 0
        elif key == "right":
            self.__vals[key] = value if self.left < value < self.width else self.width
        else:
            raise KeyError(f"{key} is not a valid key")

        if self[key] != value:
            warnings.warn(f"ROI[{key}] value {value} is invalid. ROI[{key}] set to default")

    @property
    def __dict__(self):
        return self.__vals

    @property
    def slice(self):
        """
        Use as an index for a 2D numpy array to retrieve a sub-array at this roi
        Returns:
            a slice object to be used to slice a 2D numpy array
        """
        return s_[self.top:self.bottom, self.left:self.right]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.width},{self.height},dic = {self.__dict__})"

    def __str__(self):
        ms = f"ROI :\n\twidth = {self.width}\n\theight = {self.height}"
        for key, value in self.__vals.items():
            ms += f"\n\t{key} = {value}"
        return ms


def set_frame_grabber_region(results_file):
    """

    Args:
        results_file: h5py File object corresponding to the results file
    Returns:
        fg: the frame grabber acquisition region set for the hamamatsu in the settings
    """
    fg = FrameGrabberAqRegion(0, 0, 0, 0)
    fg_adr = 'settings/experiment/LabView/camera/frameGrabberAcquisitionRegion{}/function'
    for axis in list(fg.__dict__.keys()):
        adr_axis = axis[0].upper() + axis[1:]
        fg[axis] = int(results_file[fg_adr.format(adr_axis)][()])
    return fg


def load_data(results_file, roi: HMROI = None):
    """
    Load data from an instrument into a numpy array.

    results are indexed as follows
    > results = array[iterations,measurements,shots,horizontal_pixels, vertical_pixels]

    Args:
        results_file: h5file object corresponding to results.hdf5 file
        roi: region of interest from which to extract pixel data

    Returns:
        5D numpy array holding all of the data taken by the hamamatsu during the experiment
        indexed [iteration,measurement,shot,horizontal_pixel,vertical_pixel]
    """
    measurements = results_file['settings/experiment/measurementsPerIteration'][()]+1
    num_its = len(results_file['iterations'])
    shots_per_measurement = int(
        results_file['/settings/experiment/LabView/camera/shotsPerMeasurement/function'][()])

    hm_pix = np.zeros(
        (num_its, measurements, shots_per_measurement, roi.right - roi.left, roi.bottom - roi.top),
        dtype=int
    )

    for iteration, i_group in results_file['iterations'].items():
        # print(f"iteration : {iteration} : {type(iteration)}")
        for measurement, m_group in i_group['measurements'].items():
            # print(f"\tmeasurement : {measurement} : {type(measurement)}")
            for shot, s_group in m_group['data/Hamamatsu/shots'].items():
                try:
                    # print(f"\t\tshot : {shot} : {type(shot)}")
                    hm_pix[int(iteration), int(measurement), int(shot)] = s_group[()][roi.slice]
                except IndexError as e:
                    warnings.warn(
                        f"{e}\n iteration : {iteration} measurement : {measurement} shot {shot}"
                    )
                    continue

    return hm_pix