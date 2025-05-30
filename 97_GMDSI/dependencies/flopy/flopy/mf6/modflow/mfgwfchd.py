# DO NOT MODIFY THIS FILE DIRECTLY.  THIS FILE MUST BE CREATED BY
# mf6/utils/createpackages.py
# FILE created on February 07, 2024 20:16:08 UTC
from .. import mfpackage
from ..data.mfdatautil import ListTemplateGenerator


class ModflowGwfchd(mfpackage.MFPackage):
    """
    ModflowGwfchd defines a chd package within a gwf6 model.

    Parameters
    ----------
    model : MFModel
        Model that this package is a part of. Package is automatically
        added to model when it is initialized.
    loading_package : bool
        Do not set this parameter. It is intended for debugging and internal
        processing purposes only.
    auxiliary : [string]
        * auxiliary (string) defines an array of one or more auxiliary variable
          names. There is no limit on the number of auxiliary variables that
          can be provided on this line; however, lists of information provided
          in subsequent blocks must have a column of data for each auxiliary
          variable name defined here. The number of auxiliary variables
          detected on this line determines the value for naux. Comments cannot
          be provided anywhere on this line as they will be interpreted as
          auxiliary variable names. Auxiliary variables may not be used by the
          package, but they will be available for use by other parts of the
          program. The program will terminate with an error if auxiliary
          variables are specified on more than one line in the options block.
    auxmultname : string
        * auxmultname (string) name of auxiliary variable to be used as
          multiplier of CHD head value.
    boundnames : boolean
        * boundnames (boolean) keyword to indicate that boundary names may be
          provided with the list of constant-head cells.
    print_input : boolean
        * print_input (boolean) keyword to indicate that the list of constant-
          head information will be written to the listing file immediately
          after it is read.
    print_flows : boolean
        * print_flows (boolean) keyword to indicate that the list of constant-
          head flow rates will be printed to the listing file for every stress
          period time step in which "BUDGET PRINT" is specified in Output
          Control. If there is no Output Control option and "PRINT_FLOWS" is
          specified, then flow rates are printed for the last time step of each
          stress period.
    save_flows : boolean
        * save_flows (boolean) keyword to indicate that constant-head flow
          terms will be written to the file specified with "BUDGET FILEOUT" in
          Output Control.
    timeseries : {varname:data} or timeseries data
        * Contains data for the ts package. Data can be stored in a dictionary
          containing data for the ts package with variable names as keys and
          package data as values. Data just for the timeseries variable is also
          acceptable. See ts package documentation for more information.
    observations : {varname:data} or continuous data
        * Contains data for the obs package. Data can be stored in a dictionary
          containing data for the obs package with variable names as keys and
          package data as values. Data just for the observations variable is
          also acceptable. See obs package documentation for more information.
    dev_no_newton : boolean
        * dev_no_newton (boolean) turn off Newton for unconfined cells
    maxbound : integer
        * maxbound (integer) integer value specifying the maximum number of
          constant-head cells that will be specified for use during any stress
          period.
    stress_period_data : [cellid, head, aux, boundname]
        * cellid ((integer, ...)) is the cell identifier, and depends on the
          type of grid that is used for the simulation. For a structured grid
          that uses the DIS input file, CELLID is the layer, row, and column.
          For a grid that uses the DISV input file, CELLID is the layer and
          CELL2D number. If the model uses the unstructured discretization
          (DISU) input file, CELLID is the node number for the cell. This
          argument is an index variable, which means that it should be treated
          as zero-based when working with FloPy and Python. Flopy will
          automatically subtract one when loading index variables and add one
          when writing index variables.
        * head (double) is the head at the boundary. If the Options block
          includes a TIMESERIESFILE entry (see the "Time-Variable Input"
          section), values can be obtained from a time series by entering the
          time-series name in place of a numeric value.
        * aux (double) represents the values of the auxiliary variables for
          each constant head. The values of auxiliary variables must be present
          for each constant head. The values must be specified in the order of
          the auxiliary variables specified in the OPTIONS block. If the
          package supports time series and the Options block includes a
          TIMESERIESFILE entry (see the "Time-Variable Input" section), values
          can be obtained from a time series by entering the time-series name
          in place of a numeric value.
        * boundname (string) name of the constant head boundary cell. BOUNDNAME
          is an ASCII character variable that can contain as many as 40
          characters. If BOUNDNAME contains spaces in it, then the entire name
          must be enclosed within single quotes.
    filename : String
        File name for this package.
    pname : String
        Package name for this package.
    parent_file : MFPackage
        Parent package file that references this package. Only needed for
        utility packages (mfutl*). For example, mfutllaktab package must have
        a mfgwflak package parent_file.

    """

    auxiliary = ListTemplateGenerator(("gwf6", "chd", "options", "auxiliary"))
    ts_filerecord = ListTemplateGenerator(
        ("gwf6", "chd", "options", "ts_filerecord")
    )
    obs_filerecord = ListTemplateGenerator(
        ("gwf6", "chd", "options", "obs_filerecord")
    )
    stress_period_data = ListTemplateGenerator(
        ("gwf6", "chd", "period", "stress_period_data")
    )
    package_abbr = "gwfchd"
    _package_type = "chd"
    dfn_file_name = "gwf-chd.dfn"

    dfn = [
        ["header", "multi-package", "package-type stress-package"],
        [
            "block options",
            "name auxiliary",
            "type string",
            "shape (naux)",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name auxmultname",
            "type string",
            "shape",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name boundnames",
            "type keyword",
            "shape",
            "reader urword",
            "optional true",
        ],
        [
            "block options",
            "name print_input",
            "type keyword",
            "reader urword",
            "optional true",
            "mf6internal iprpak",
        ],
        [
            "block options",
            "name print_flows",
            "type keyword",
            "reader urword",
            "optional true",
            "mf6internal iprflow",
        ],
        [
            "block options",
            "name save_flows",
            "type keyword",
            "reader urword",
            "optional true",
            "mf6internal ipakcb",
        ],
        [
            "block options",
            "name ts_filerecord",
            "type record ts6 filein ts6_filename",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
            "construct_package ts",
            "construct_data timeseries",
            "parameter_name timeseries",
        ],
        [
            "block options",
            "name ts6",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name filein",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name ts6_filename",
            "type string",
            "preserve_case true",
            "in_record true",
            "reader urword",
            "optional false",
            "tagged false",
        ],
        [
            "block options",
            "name obs_filerecord",
            "type record obs6 filein obs6_filename",
            "shape",
            "reader urword",
            "tagged true",
            "optional true",
            "construct_package obs",
            "construct_data continuous",
            "parameter_name observations",
        ],
        [
            "block options",
            "name obs6",
            "type keyword",
            "shape",
            "in_record true",
            "reader urword",
            "tagged true",
            "optional false",
        ],
        [
            "block options",
            "name obs6_filename",
            "type string",
            "preserve_case true",
            "in_record true",
            "tagged false",
            "reader urword",
            "optional false",
        ],
        [
            "block options",
            "name dev_no_newton",
            "type keyword",
            "reader urword",
            "optional true",
            "mf6internal inewton",
        ],
        [
            "block dimensions",
            "name maxbound",
            "type integer",
            "reader urword",
            "optional false",
        ],
        [
            "block period",
            "name iper",
            "type integer",
            "block_variable True",
            "in_record true",
            "tagged false",
            "shape",
            "valid",
            "reader urword",
            "optional false",
        ],
        [
            "block period",
            "name stress_period_data",
            "type recarray cellid head aux boundname",
            "shape (maxbound)",
            "reader urword",
            "mf6internal spd",
        ],
        [
            "block period",
            "name cellid",
            "type integer",
            "shape (ncelldim)",
            "tagged false",
            "in_record true",
            "reader urword",
        ],
        [
            "block period",
            "name head",
            "type double precision",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "time_series true",
        ],
        [
            "block period",
            "name aux",
            "type double precision",
            "in_record true",
            "tagged false",
            "shape (naux)",
            "reader urword",
            "optional true",
            "time_series true",
            "mf6internal auxvar",
        ],
        [
            "block period",
            "name boundname",
            "type string",
            "shape",
            "tagged false",
            "in_record true",
            "reader urword",
            "optional true",
        ],
    ]

    def __init__(
        self,
        model,
        loading_package=False,
        auxiliary=None,
        auxmultname=None,
        boundnames=None,
        print_input=None,
        print_flows=None,
        save_flows=None,
        timeseries=None,
        observations=None,
        dev_no_newton=None,
        maxbound=None,
        stress_period_data=None,
        filename=None,
        pname=None,
        **kwargs,
    ):
        super().__init__(
            model, "chd", filename, pname, loading_package, **kwargs
        )

        # set up variables
        self.auxiliary = self.build_mfdata("auxiliary", auxiliary)
        self.auxmultname = self.build_mfdata("auxmultname", auxmultname)
        self.boundnames = self.build_mfdata("boundnames", boundnames)
        self.print_input = self.build_mfdata("print_input", print_input)
        self.print_flows = self.build_mfdata("print_flows", print_flows)
        self.save_flows = self.build_mfdata("save_flows", save_flows)
        self._ts_filerecord = self.build_mfdata("ts_filerecord", None)
        self._ts_package = self.build_child_package(
            "ts", timeseries, "timeseries", self._ts_filerecord
        )
        self._obs_filerecord = self.build_mfdata("obs_filerecord", None)
        self._obs_package = self.build_child_package(
            "obs", observations, "continuous", self._obs_filerecord
        )
        self.dev_no_newton = self.build_mfdata("dev_no_newton", dev_no_newton)
        self.maxbound = self.build_mfdata("maxbound", maxbound)
        self.stress_period_data = self.build_mfdata(
            "stress_period_data", stress_period_data
        )
        self._init_complete = True
