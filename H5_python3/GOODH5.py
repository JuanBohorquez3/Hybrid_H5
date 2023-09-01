import h5py
from numpy import *
import pandas as pd
import os
import sys
from typing import List, Dict, Tuple
from datetime import datetime
import time
from configparser import ConfigParser
import logging
# Local imports
from Iterations import Iterations
# import origin
# Set up Reader and Config
origin_lib_path = "C:\\Users\\Hybrid\\Repos\\Origin\\lib_python3"
origin_cfg_path = "C:\\Users\\Hybrid\\Repos\\Origin\\config\\origin-server.cfg"

sys.path.append(origin_lib_path)
from origin.client.origin_reader import Reader


datetime_encoding = '%Y-%m-%d %H:%M:%S' # from cspy
time_str = "start_time_str"
ts_str = "start_time"


def _make_reader():
    """
    Creates a Reader object from origin.
    Returns:

    """
    origin_config = ConfigParser(inline_comment_prefixes=(";",))
    origin_config.read(origin_cfg_path)

    log = logging.Logger(__name__)  # useless placeholder logger

    return Reader(config=origin_config, logger=log)


def load_bypassed_origin_data(
        results_file: h5py.File,
        iterations: Iterations,
        data_filename: str,
        measurements: int,
        fields: List[str] = None
) -> List[pd.DataFrame]:
    """
    Loads origin data from csv file containing logged values over the course of the experiment. Organizes data into a
    list of dataframes. List is indexed by iteration, each dataframe contains data logged during iteration.
    Args:
        results_file: h5py file containing experiment data
        iterations: iterations object corresponding to the experiment
        data_filename: full path to csv file containing origin data
        measurements: how many measurements per iteration
        fields: list of fields in data stream which should be read. If unspecified all fields in the stream are read out

    Returns:
        GOOD_data: List of data frames. List is indexed by iteration number. Each data frame has a column containing the
            measurement time of the data (as a datetime object) and a column for each field in fields.
    """
    start, end = get_exp_start_stop(results_file, iterations)
    # make a list of the start times of the iterations
    iteration_times_ts = zeros(len(iterations), dtype=float)
    for iteration, i_group in results_file["iterations"].items():
        iteration = int(iteration)
        iteration_times_ts[iteration] = int(i_group.attrs[ts_str])  # encode times as timestamps

    origin_data = pd.read_csv(data_filename)
    origin_data["measurement_time_dt"] = [datetime.fromtimestamp(t/2**32) for t in origin_data["measurement_time"]]
    origin_data["measurement_time_ts"] = [t/2**32 for t in origin_data["measurement_time"]]

    new_fields = fields + ["measurement_time_dt"]

    it_data = []
    for i, row in iterations.iterrows():
        iteration = int(row["iteration"])
        if iteration < len(iterations) -1:
            sub_df = origin_data[
                origin_data["measurement_time_ts"].between(
                    iteration_times_ts[iteration],
                    iteration_times_ts[iteration+1]
                )
            ][new_fields]
        else:
            sub_df = origin_data[
                origin_data["measurement_time_ts"].between(
                    iteration_times_ts[iteration],
                    end
                )
            ][new_fields]
        it_data.append(sub_df)

    return it_data


def load_origin_data(
        results_file: h5py.File,
        iterations: Iterations,
        measurements: int,
        stream: str,
        fields: List[str] = None
) -> List[pd.DataFrame]:
    """
    Loads data logged to origin over the course of the experiment, organizes it into a list of dicts of lists.
    List is indexed [iteration][field][].

    [
        {measurements times:[measurement times as datetime objects],
        field: [field data as floats]}
        for iteration in range(iterations)
    ]

    Args:
        results_file: h5py file containing experiment data
        iterations: iterations object corresponding to the experiment
        measurements: how many measurements per iteration
        stream: name of the stream which should be read
        fields: list of fields in data stream which should be read. If unspecified all fields in the stream are read out

    Returns:
        List of data frames. List is indexed by iteration number. Each data frame has a column containing the
            measurement time of the data (as a datetime object) and a column for each field in fields.
    """
    # get start and end times of experiment

    reader = _make_reader()

    # make a list of the start times of the iterations
    iteration_times = zeros(len(iterations), dtype=datetime)
    iteration_times_ts = zeros(iteration_times.shape, dtype=float)
    for iteration, i_group in results_file["iterations"].items():
        iteration = int(iteration)
        iteration_times[iteration] = datetime.strptime(i_group.attrs[time_str], datetime_encoding)
        iteration_times_ts[iteration] = int(i_group.attrs[ts_str])  # encode times as timestamps

    # initial time to pull data
    start_time = min(iteration_times_ts)

    # get start time of last measurement
    meas_inds = zeros(measurements, dtype=int)
    for m, m_tup in enumerate(results_file[f"iterations/{len(iterations) - 1}/measurements"].items()):
        meas_inds[m] = int(m_tup[0])
    last_measurement = results_file[f"iterations/{len(iterations) - 1}/measurements/{max(meas_inds)}"]
    end_time_str = last_measurement.attrs[time_str]

    # end of data pulling window
    end_time_dt = datetime.strptime(end_time_str, datetime_encoding)
    end_time = last_measurement.attrs[ts_str]
    #end_time = datetime.strptime(end_time_str, datetime_encoding)

    # get all origin data in given fields over the course of the experiment
    origin_data = reader.get_stream_raw_data(
        stream=stream,
        fields=fields,
        start=start_time,
        stop=end_time
    )

    # add list to dict of measurement times as datetime objects
    origin_data.update({
        "measurement_time_dt": array(
            [datetime.fromtimestamp(t / 2**32) for t in origin_data["measurement_time"]]
        )
    })

    new_fields = fields + ["measurement_time_dt"]

    it_data = []
    for i in range(len(iterations)):
        if i < len(iterations) - 1:
            data_inds = where(
                (origin_data["measurement_time_dt"] > iteration_times[i]) *
                (origin_data["measurement_time_dt"] < iteration_times[i + 1])
            )[0]
        else:
            data_inds = where(
                (origin_data["measurement_time_dt"] > iteration_times[i]) *
                (origin_data["measurement_time_dt"] < end_time_dt)
            )[0]
        try:
            it_data.append(
                pd.DataFrame(
                    {field: dataset[min(data_inds):max(data_inds)] for field, dataset in origin_data.items() if field in new_fields}
                )
            )
        except ValueError as e:
            print(f"{e}\nissue in iteration {i}")
            it_data.append(
                None
            )
    return it_data


def get_exp_start_stop(results_file: h5py.File,
                       iterations: Iterations) -> Tuple[float, float]:
    """
    Gets the start time and stop time of an experiment
    Args:
        results_file: h5py file containing experiment data
        iterations: iterations object corresponding to the experiment

    Returns:
        start_time, stop_time
            start_time: timestamp corresponding to the start of the experiment
            stop_time: timestamp corresponding to the end of the experiment
    """

    start_time = results_file[f"iterations/0"].attrs[ts_str]
    last_it_no = len(iterations) - 1
    last_it = results_file[f"iterations/{last_it_no}"]

    max_meas_ind = 0
    for meas_ind, m_group in last_it["measurements"].items():
        meas_ind = int(meas_ind)
        max_meas_ind = meas_ind if meas_ind > max_meas_ind else max_meas_ind

    stop_time = last_it[f"measurements/{max_meas_ind}"].attrs[ts_str]

    return start_time, stop_time
