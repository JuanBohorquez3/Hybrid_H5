# HDF5 analysis package in python3
Package to perform common analysis on HDF5 files created by Cspy.

## How to use:
The idea is to create an Iterations object to store info on the cspy settings at each iteration, then perform analysis on experimental data using Iterations as a reference.

In this code base ivars or independent variables refers to experimental settings that were purposely varied during the experiment. Those settings which were left constant are not considered.

## Functionality (by file):
Specifics are documented in each file, what is bellow serves to outline the code base for first-time users.
### Iterations.py
Contains the Iterations class that wraps a pandas DataFrame created from the independent variables that were varied during an experiment.  

The DataFrame holds info on what values each independent variable took during each iteration.  
The iterations class sorts the DataFrame by the values taken by independent variables.  

If other operations are needed the internal DataFrame is exposed as:
* **data_frame**

Useful attributes from experiment settings:
* **data_frame** - the DataFrame this class wraps in case other operations are required. I advise against modifying this.
* **independent_variables** - An OrderedDict mapping the names of the independent variables with the values they took this experiment
* **ivars** - A list of the names of the independent_variables

Functions to perform data operations:
* **fold_to_nd** - reshapes a data array passed to it into an nd_array or a convenient shape for plotting or otherwise analyzing data.

Attributes and Functions wrapping DataFrame functionality
* \_\_getitem__()
* keys()
* \_\_len__()
* \_\_str__()
* \_\_repr__()
* items()
* iterrows()
* \_\_iter__()
* loc  

## PlottingH5
A package of plotting functions that are regularly useful
* **default_plotting()** - plots the data provided in data in the default manner given the number of independent variables in an experiment
    * 0 ivars:  Prints the values of data provided (and uncertainty if provided)
    * 1 ivar: Plots the values of data provided (and error bars) ona a line graph
    * 2 ivar: Plots the values of data provided as a colormap image
* **iterate_plot_2D()** - plots the data from a 2D scan by iterating through one of the varied independent variables and for each value that ivar takes, it plots a line graph of the data with the other ivar on the x-axis.

## HamamtsuH5.py

Classes:
* **HMROI** - class that organizes the ROI to be used in the analysis. Passing it in a dict with the limits of the ROI creates the object which can be used to easily create sub-images from larger images taken. Can be used on any image, not just images taken by the hamamatsu

Functions:
* **set_frame_grabber_region()** - reads setting data from the HDF5 file and figures out the frame grabber region used to acquire images for the hamamatsu, therefore the shape of the images taken.
* **load_data()** - reads data taken by the hamamatsu (in the ROI provided) and loads it into a numpy array
