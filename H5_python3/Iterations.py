import pandas as pd
import h5py
from numpy import *
from collections import OrderedDict
from typing import List, Dict, Tuple, Any, Union


def seed_permute(x: Union[int, ndarray], seed: Any):
    """
    Reproducibly permute an array of values given a seed
    Args:
        x : array-like to be permuted. If x is an int, a permutation of arange(x) is returned
        seed : seed for random
    Returns:
        permuted array given the seed provided
    """
    random.seed(seed)
    return random.permutation(x)


class Iterations:
    """
    Class that wraps a dataframe containing data from an experiments iterations and independent
    variables

    Attributes:
        ivars : list of names of independent variables this experiment, sorted alphabetically
        data_frame : data frame indexed by iteration number, containing the values taken by each
            independent variable for that iteration
    """
    def __init__(self, results_file: h5py.File):
        if not isinstance(results_file, h5py.File):
            raise TypeError("Results file must be valid HDF5 file")
        self.__results = results_file
        self.__independent_variables: OrderedDict = self._get_independent_variables()
        self.ivars: List[str] = sorted(list(self.__independent_variables.keys()))
        self.data_frame: pd.DataFrame = self._load_df()

    @property
    def loc(self):
        return self.data_frame.loc

    @property
    def results(self) -> h5py.File:
        """
        Returns:
            The results file associated with this experiment
        """
        return self.__results

    @property
    def independent_variables(self) -> OrderedDict:
        """
        Ordered Dict of independent variables in this experiment and the values they took (in
        order)

        sorted alphabetically by name
        """
        return self.__independent_variables

    @property
    def _step_sizes(self) -> Tuple[float]:
        """
        Returns:
            the step size of each independent variable in this experiment, assuming even spacing
                for all independent variables.
                Tuple ordered alphabetically by variable name

        """
        sorted_values = [
            sorted(list(set(values))) for iVar, values in self.items() if iVar != 'iteration'
        ]
        return tuple([float(vals[1] - vals[0]) for vals in sorted_values])

    @staticmethod
    def __get_iteration_ivars(iteration: h5py.Group, *ivar_names: str) -> Dict[str, Any]:
        """
        Gets the values of independent variables for the iteration that was passed in

        Args:
            iteration: h5py data group corresponding to an iteration
            *ivar_names: list of names of independent variables
        Returns:
            map of ivar_names to their values
        """
        return {name: iteration[f"variables/{name}"][()] for name in ivar_names}

    def _get_independent_variables(self):
        """
        Finds the independent variables that were varied this experiment and puts them in a
        dictionary

        Returns:
            OrderedDict[str: List[Any]]
                dictionary mapping varied independent variable names to the values these variables
                took.
                Sorted alphabetically by keys
        """

        indep_vars = {}
        if len(self.results['iterations']) > 1:
            for variable in self.results['settings/experiment/independentVariables'].items():
                values = eval(variable[1]['function'][()])
                if iterable(values):
                    indep_vars.update({variable[0]: array(values)})
        return OrderedDict(sorted(indep_vars.items()))

    def _load_df(self) -> pd.DataFrame:
        """
        Loads a data frame mapping independent variable values to iteration number.
        Keys should be sorted alphabetically.
        Returns:
            Data frame containing each iteration number and the value each independent variable took
                during that iteration
        """
        df = pd.DataFrame(columns=["iteration"] + self.ivars)
        for iteration in self.results['iterations'].items():
            i = int(iteration[0])
            ivar_vals = self.__get_iteration_ivars(iteration[1], *self.ivars)
            ivar_vals.update({"iteration": i})
            df = df.append(pd.DataFrame(ivar_vals, index=[i]))

        # Sort the dataframe indeces by values if independent variables (not iteration number) so
        # operations can be performed somewhat intuitively  TODO : Document this better
        # breaks with 1 iteration
        if len(df) == 1:
            return df
        else:
            return df.sort_values(self.ivars[::-1], kind="mergesort", ignore_index=True)

    def fold_to_nd(self, data_array: ndarray = None) -> ndarray:
        """
        Folds data array into an ndarray conveniently shaped for operations
        Args:
            data_array : 1D array with data from each iteration. Should be indexed [iteration]
                if none, an array of iteration numbers is returned
        Returns:
            the folded data_array. Indexed
                [independent_variable_step1,independent_variable_step2,...]
                independent variables ordered by name (alphabetically)
        """
        if data_array is None:
            data_array = array(self.data_frame['iteration'], dtype=int)
        else:
            # Jumble the array to match the sorted independent variable values lists.
            # Redundant when the independent variables were generated in a sorted manner,
            # but important when they were not
            data_array = data_array[array(self.data_frame['iteration'], dtype=int)]
        return data_array.reshape(
            *[len(vals) for var, vals in self.__independent_variables.items().__reversed__()]
        ).T

# Wrapping utility functions
    def __getitem__(self, item):
        return self.data_frame[item]

    def keys(self):
        return self.data_frame.keys()

    def __len__(self):
        return len(self.data_frame)

    def __str__(self):
        return str(self.data_frame)

    def __repr__(self):
        return self.data_frame.__repr__()

    def items(self):
        return self.data_frame.items()

    def iterrows(self):
        return self.data_frame.iterrows()

    def __iter__(self):
        return self.data_frame.__iter__()