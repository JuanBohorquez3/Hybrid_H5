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
    def __init__(self, results_file: h5py.File = None, df=None):
        if results_file is not None and not isinstance(results_file, h5py.File):
            raise TypeError("Results file must be valid HDF5 file")
        if results_file is None and df is None:
            raise ValueError("Either results_file or df must be specified")
        if df is not None:
            self.__results = None
            self.data_frame: pd.DataFrame = df
        self.__results = results_file
        self.__constants_str = self._get_constants()
        self.__dependent_var_str = self._get_dependent_variables()
        self.__independent_variables: OrderedDict = self._get_independent_variables()
        self.ivars: List[str] = sorted(list(self.__independent_variables.keys()))
        if df is None:
            self.data_frame: pd.DataFrame = self._load_df()
        self.__dependent_variables: Dict[Dict[int, float]] = {}

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
        # quick deep copy to prevent mutable-ness of dict from causing bugs
        return OrderedDict({ky: value for ky, value in self.__independent_variables.items()})

    @property
    def dependent_variables(self) -> Dict[Dict[int, float]]:
        """
        Ordered Dict of dependent variable names and the values each of them too (in order).

        Dict structured [dependent variable name: [iteration number: dependent variable value] ]
        """
        # quick deep copy to prevent mutable-ness of dict from causing bugs
        return {ky: val for ky, val in self.__dependent_variables.items()}

    @property
    def dvars(self) -> List[str]:
        """
        Returns: List of dependent variable names
        """
        return [name for name in self.dependent_variables.keys()]

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

    def _get_constants(self):
        constants_str = self.results["settings/experiment/constantsStr"][()]

        # put all info from constants into local memory
        exec(constants_str, globals(), globals())

        return constants_str

    def _get_dependent_variables(self):
        dep_str = self.results["settings/experiment/dependentVariablesStr"][()]
        return dep_str

    def load_dependent_variables(self, d_var_names: Union[str, List[str]]):
        """
        Loads the dependent variable(s) in d_var_names and appends them to the iterations dataframe. Update
        self.__dependent_variables to reflect these changes
        Args:
            d_var_names: name or list of names of dependent variables to append to the dataframe
        """
        if type(d_var_names) is str:
            d_var_names = [d_var_names]

        dep_df = pd.DataFrame(columns=d_var_names)
        # Load values to a dataframe in the same way we do for independent variables
        for iteration in self.results["iterations"].items():
            try:
                i = int(iteration[0])
            except ValueError:
                print(f"Warning : {iteration[0]} is not a valid iteration number")
                continue
            d_var_vals = self.__get_iteration_ivars(iteration[1], *d_var_names)
            dep_df = dep_df.append(pd.DataFrame(d_var_vals, index=[i]))
        # Append the columns to the iterations dataframe as-if they're independent variables
        self.data_frame = self.data_frame.join(dep_df)
        # Update the __dependent_variables dict so that new dependent variables can be updated multiple times.
        # Initial dict is empty. Use the "dict" option in to_dict to preserve information of iteration number within the
        # dependent variables dict. Can't save them the same way as we do for independent_variables because the
        # variables we'd like to analyze are often complex functions of the simple functions we use for
        # independent_variables
        self.__dependent_variables.update(self.data_frame[d_var_names].to_dict("dict"))

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
        def ivar_uniques(ivar_vals: ndarray) -> ndarray:
            """
            Finds unique values within an array, while preserving order (in which value first
            appears).

            numpy.unique and set() calls sort unique values, but the ordering of ivars is important
                to maintain. The order of first appearance usually maps to the order
                in which they are created in the results file
            Args:
                ivar_vals: an ndarray of values taken by an ivar
            Returns:
                unique values within ivar_vals, with their order preserved
            """
            vals = []
            for val in ivar_vals:
                if val not in vals:
                    vals.append(val)
            return array(vals)

        indep_vars = {}
        if self.results is None:
            indep_vars = {
                ivar: ivar_uniques(self.data_frame[ivar]) for ivar in self.data_frame if ivar != 'iteration'
            }
        elif len(self.results['iterations']) > 1:
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
            try:
                i = int(iteration[0])
            except ValueError:
                print(f"Warning : {iteration[0]} is not a valid iteration number")
                continue
            ivar_vals = self.__get_iteration_ivars(iteration[1], *self.ivars)
            ivar_vals.update({"iteration": i})
            df = df.append(pd.DataFrame(ivar_vals, index=[i]))

        # When increasing iteration number does not map to increasing independent variable values many
        # plotting functions break or produce confusing plots. Here we sort the dataframe indices by increasing
        # independent variable values, holding on to the iteration number for each set of IV values.
        # This functionality should leave experiments unchanged when each independent variable array increases
        # monotonically
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
