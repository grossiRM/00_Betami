import pathlib as pl ; from typing import List, Tuple, Union
import flopy ; import numpy as np  ; from flopy.utils.gridintersect import GridIntersect  ; from shapely.geometry import LineString, Polygon
figwidth = 180  # 90 ; figwidth = figwidth / 10 / 2.54  ; figheight = figwidth ; figsize = (figwidth, figheight) ; Lx = 180000 ; Ly = 100000
import pathlib as pl ; from typing import List, Tuple, Union
import flopy ; import numpy as np ; from flopy.utils.gridintersect import GridIntersect ; from shapely.geometry import LineString, Polygon

geometries = yaml.safe_load(open(pl.Path("../../examples/data/groundwater2023/geometries.yml")))

def string2geom(geostring, conversion=None):
    if conversion is None: multiplier = 1.0
    else:                  multiplier = float(conversion)
    res = []
    for line in geostring.split("\n"):
        if not any(line):
            continue
        line = line.strip() ; line = line.split(" ") ; x = float(line[0]) * multiplier ; y = float(line[1]) * multiplier ; res.append((x, y))
    return res

def string2geom(geostring: str, conversion: float = None) -> List[tuple]:
    if conversion is None: multiplier = 1.0
    else:                  multiplier = float(conversion)
    res = []
    for line in geostring.split("\n"):
        line = line.split(" ")  ; x = float(line[0]) * multiplier  ; y = float(line[1]) * multiplier  ; res.append((x, y))
    return res

# ___________________________02a_defaults_geometry....py ___# Geometry data
geometry = {
    "boundary": """1.868012422360248456e+05 4.695652173913043953e+04
1.790372670807453396e+05 5.204968944099379587e+04
1.729813664596273447e+05 5.590062111801243009e+04
1.672360248447204940e+05 5.987577639751553215e+04
1.631987577639751253e+05 6.335403726708075556e+04
1.563664596273291972e+05 6.819875776397516893e+04
1.509316770186335489e+05 7.229813664596274612e+04
1.453416149068323139e+05 7.527950310559007630e+04
1.395962732919254631e+05 7.627329192546584818e+04
1.357142857142857101e+05 7.664596273291927355e+04
1.329192546583850926e+05 7.751552795031057030e+04
1.268633540372670832e+05 8.062111801242237561e+04
1.218944099378881947e+05 8.285714285714286962e+04
1.145962732919254486e+05 8.571428571428572468e+04
1.069875776397515583e+05 8.869565217391305487e+04
1.023291925465838431e+05 8.931677018633540138e+04
9.456521739130433707e+04 9.068322981366459862e+04
8.804347826086955320e+04 9.080745341614908830e+04
7.950310559006211406e+04 9.267080745341615693e+04
7.562111801242236106e+04 9.391304347826087906e+04
6.692546583850930620e+04 9.602484472049689793e+04
5.667701863354037778e+04 9.763975155279504543e+04
4.906832298136646568e+04 9.689440993788820924e+04
3.897515527950309479e+04 9.540372670807455142e+04
3.167701863354036323e+04 9.304347826086958230e+04
2.375776397515527788e+04 8.757763975155279331e+04
1.847826086956521613e+04 8.161490683229814749e+04
1.164596273291925172e+04 7.739130434782608063e+04
6.211180124223596977e+03 7.055900621118013805e+04
4.347826086956512881e+03 6.422360248447205959e+04
1.863354037267072272e+03 6.037267080745341809e+04
2.639751552795024509e+03 5.602484472049689793e+04
1.552795031055893560e+03 5.279503105590062478e+04
7.763975155279410956e+02 4.186335403726709046e+04
2.018633540372667312e+03 3.813664596273292409e+04
6.055900621118013078e+03 3.341614906832297856e+04
1.335403726708074100e+04 2.782608695652173992e+04
2.577639751552794405e+04 2.086956521739130767e+04
3.416149068322980747e+04 1.763975155279503815e+04
4.642857142857142753e+04 1.440993788819875044e+04
5.636645962732918997e+04 1.130434782608694877e+04
6.459627329192546313e+04 9.813664596273290954e+03
8.555900621118012350e+04 6.832298136645956220e+03
9.829192546583850344e+04 5.093167701863346338e+03
1.085403726708074391e+05 4.347826086956525614e+03
1.200310559006211115e+05 4.223602484472040487e+03
1.296583850931677007e+05 4.347826086956525614e+03
1.354037267080745369e+05 5.590062111801232277e+03
1.467391304347825935e+05 1.267080745341615875e+04
1.563664596273291972e+05 1.937888198757762802e+04
1.630434782608695677e+05 2.198757763975155467e+04
1.694099378881987650e+05 2.434782608695652743e+04
1.782608695652173774e+05 2.981366459627329095e+04
1.833850931677018234e+05 3.180124223602484562e+04
1.868012422360248456e+05 3.577639751552795497e+04""",
    "streamseg1": """1.868012422360248456e+05 4.086956521739130403e+04
1.824534161490683327e+05 4.086956521739130403e+04
1.770186335403726553e+05 4.124223602484472940e+04
1.737577639751552779e+05 4.186335403726709046e+04
1.703416149068323139e+05 4.310559006211180531e+04
1.670807453416148783e+05 4.397515527950310934e+04
1.636645962732919143e+05 4.484472049689441337e+04
1.590062111801242281e+05 4.559006211180124228e+04
1.555900621118012350e+05 4.559006211180124228e+04
1.510869565217391064e+05 4.546583850931677443e+04
1.479813664596273156e+05 4.534161490683229931e+04
1.453416149068323139e+05 4.496894409937888850e+04
1.377329192546583654e+05 4.447204968944099528e+04
1.326086956521739194e+05 4.447204968944099528e+04
1.285714285714285652e+05 4.434782608695652743e+04
1.245341614906832110e+05 4.472049689440993825e+04
1.215838509316770069e+05 4.509316770186335634e+04
1.161490683229813585e+05 4.509316770186335634e+04
1.125776397515527933e+05 4.459627329192547040e+04
1.074534161490683036e+05 4.385093167701864149e+04
1.018633540372670686e+05 4.347826086956522340e+04
9.798136645962731563e+04 4.360248447204969125e+04
9.223602484472049400e+04 4.310559006211180531e+04
8.602484472049689793e+04 4.198757763975155831e+04
7.981366459627327276e+04 4.173913043478261534e+04
7.468944099378881219e+04 4.248447204968944425e+04
7.034161490683228476e+04 4.385093167701864149e+04
6.785714285714285506e+04 4.621118012422360334e+04
6.583850931677018525e+04 4.919254658385094081e+04
6.319875776397513982e+04 5.192546583850932075e+04
6.009316770186335634e+04 5.677018633540373412e+04
5.605590062111800216e+04 5.950310559006211406e+04
5.279503105590060295e+04 6.124223602484472940e+04
4.751552795031056303e+04 6.211180124223603343e+04
3.990683229813664366e+04 6.335403726708075556e+04
3.276397515527949508e+04 6.409937888198757719e+04
2.934782608695651652e+04 6.509316770186336362e+04
2.546583850931676716e+04 6.832298136645962950e+04""",
    "streamseg2": """7.025161490683228476e+04 4.375093167701864149e+04
6.816770186335404287e+04 4.273291925465839449e+04
6.490683229813665093e+04 4.211180124223603343e+04
6.164596273291925900e+04 4.173913043478262261e+04
5.776397515527951327e+04 4.124223602484472940e+04
5.450310559006211406e+04 4.049689440993789322e+04
4.984472049689442065e+04 3.937888198757764621e+04
4.534161490683231386e+04 3.801242236024845624e+04
4.114906832298137306e+04 3.664596273291926627e+04
3.913043478260868869e+04 3.565217391304348712e+04
3.649068322981366509e+04 3.416149068322981475e+04
3.322981366459628043e+04 3.242236024844721760e+04
3.012422360248447148e+04 3.105590062111801672e+04
2.608695652173913550e+04 2.957521739130435890e+04""",
    "streamseg3": """1.059006211180124228e+05 4.335403726708074828e+04
1.029503105590062187e+05 4.223602484472050128e+04
1.004658385093167890e+05 4.024844720496894297e+04
9.937888198757765349e+04 3.788819875776398112e+04
9.627329192546584818e+04 3.490683229813664366e+04
9.285714285714286962e+04 3.316770186335403559e+04
8.897515527950311662e+04 3.093167701863354159e+04
8.338509316770188161e+04 2.795031055900621504e+04
7.872670807453416637e+04 2.670807453416148928e+04
7.329192546583851799e+04 2.385093167701863058e+04
6.863354037267081731e+04 2.111801242236025064e+04
6.304347826086958230e+04 1.863354037267081003e+04""",
    "streamseg4": """1.371118012422360480e+05 4.472049689440994553e+04
1.321428571428571595e+05 4.720496894409938250e+04
1.285714285714285652e+05 4.981366459627330187e+04
1.243788819875776535e+05 5.341614906832298584e+04
1.189440993788819906e+05 5.540372670807454415e+04
1.125776397515527933e+05 5.627329192546584818e+04
1.065217391304347839e+05 5.726708074534162733e+04
1.020186335403726698e+05 5.913043478260870324e+04
9.409937888198759174e+04 6.273291925465840177e+04
9.192546583850932075e+04 6.633540372670808574e+04
8.881987577639751544e+04 7.242236024844722124e+04
8.586956521739131131e+04 7.552795031055902655e+04
8.369565217391305487e+04 7.962732919254660374e+04""",
}

# beta 08a
def set_idomain(grid, boundary):
    ix = GridIntersect(grid, method="vertex", rtree=True)    ; result = ix.intersect(Polygon(boundary))
    idx = [coords for coords in result.cellids]              ;    idx = np.array(idx, dtype=int)          ; nr = idx.shape[0]
    if idx.ndim == 1:                                             idx = idx.reshape((nr, 1))
    idx = tuple([idx[:, i] for i in range(idx.shape[1])])    ; idomain = np.zeros(grid.shape[1:], dtype=int)
    idomain[idx] = 1                                         ; idomain = idomain.reshape(grid.shape)      ; grid.idomain = idomain
# beta 08a








def get_base_dir():
    """
    Returns the working directory for the base simulation data and results

    Returns
    -------
    dir: str
        the working directory
    """
    dir = pl.Path.cwd().joinpath("temp/base")
    return dir


def get_parallel_dir():
    """
    Returns the working directory for the parallel simulation data and results

    Returns
    -------
    dir: str
        the working directory
    """
    dir = pl.Path.cwd().joinpath("temp/parallel")
    return dir





def set_structured_idomain(
    modelgrid: flopy.discretization.StructuredGrid,
    boundary: List[tuple],
) -> None:
    """
    Set the idomain for a structured grid using a boundary line.

    Parameters
    ----------
    modelgrid: flopy.discretization.StructuredGrid
        flopy modelgrid object
    boundary: List(tuple)
        list of x,y tuples defining the boundary of the active model domain.

    Returns
    -------
    None

    """
    if modelgrid.grid_type != "structured":
        raise ValueError(
            f"modelgrid must be 'structured' not '{modelgrid.grid_type}'"
        )

    ix = GridIntersect(modelgrid, method="vertex", rtree=True)
    result = ix.intersect(Polygon(boundary))
    idx = [coords for coords in result.cellids]
    idx = np.array(idx, dtype=int)
    nr = idx.shape[0]
    if idx.ndim == 1:
        idx = idx.reshape((nr, 1))

    idx = tuple([idx[:, i] for i in range(idx.shape[1])])
    idomain = np.zeros(modelgrid.shape[1:], dtype=int)
    idomain[idx] = 1
    idomain = idomain.reshape(modelgrid.shape)

    # set modelgrid idomain
    modelgrid.idomain = idomain
    return


def intersect_segments(
    modelgrid: Union[
        flopy.discretization.StructuredGrid, flopy.discretization.VertexGrid
    ],
    segments: List[List[tuple]],
) -> Tuple[flopy.utils.GridIntersect, list, list]:
    """

    Parameters
    ----------
    modelgrid: flopy.discretization.StructuredGrid
        flopy modelgrid object
    segments: list of list of tuples
        List of segment x,y tuples

    Returns
    -------
    ixs: flopy.utils.GridIntersect
        flopy GridIntersect object
    cellids: list
        list of intersected cellids
    lengths: list
        list of intersected lengths

    """
    ixs = flopy.utils.GridIntersect(
        modelgrid,
        method=modelgrid.grid_type,
    )
    cellids = []
    lengths = []
    for sg in segments:
        v = ixs.intersect(LineString(sg), sort_by_cellid=True)
        cellids += v["cellids"].tolist()
        lengths += v["lengths"].tolist()
    return ixs, cellids, lengths


def cell_areas(
    modelgrid: Union[
        flopy.discretization.StructuredGrid, flopy.discretization.VertexGrid
    ],
) -> np.ndarray:
    """
    Calculate cell areas

    Parameters
    ----------
    modelgrid: flopy.discretization.StructuredGrid
        flopy modelgrid object

    Returns
    -------
    areas: numpy.ndarray
        cell areas

    """
    if modelgrid.grid_type == "structured":
        nrow, ncol = modelgrid.nrow, modelgrid.ncol
        areas = np.zeros((nrow, ncol), dtype=float)
        for r in range(nrow):
            for c in range(ncol):
                cellid = (r, c)
                vertices = np.array(modelgrid.get_cell_vertices(cellid))
                area = Polygon(vertices).area
                areas[cellid] = area
    elif modelgrid.grid_type == "vertex":
        areas = np.zeros(modelgrid.ncpl, dtype=float)
        for idx in range(modelgrid.ncpl):
            vertices = np.array(modelgrid.get_cell_vertices(idx))
            area = Polygon(vertices).area
            areas[idx] = area
    else:
        raise ValueError(
            "modelgrid must be 'structured' or 'vertex' not "
            + f"{modelgrid.grid_type}"
        )
    return areas


def build_drain_data(
    modelgrid: Union[
        flopy.discretization.StructuredGrid, flopy.discretization.VertexGrid
    ],
    cellids: list,
    lengths: list,
    leakance: float,
    elevation: np.ndarray,
) -> List[tuple]:
    """
    Build drain package data represent river segments

    Parameters
    ----------
    modelgrid: flopy.discretization.StructuredGrid
        flopy modelgrid object
    cellids: list
        list of intersected cellids
    lengths: list
        list of intersected lengths
    leakance: float
        drainage leakance value
    elevation: numpy.ndarray
        land surface elevation

    Returns
    -------
    drn_data: list of tuples
        Drain package data for stream segments

    """
    drn_data = []
    for cellid, length in zip(cellids, lengths):
        x = modelgrid.xcellcenters[cellid]
        width = 5.0 + (14.0 / Lx) * (Lx - x)
        conductance = leakance * length * width
        if not isinstance(cellid, tuple):
            cellid = (cellid,)
        drn_data.append((0, *cellid, elevation[cellid], conductance))
    return drn_data


def build_groundwater_discharge_data(
    modelgrid: Union[
        flopy.discretization.StructuredGrid, flopy.discretization.VertexGrid
    ],
    leakance: float,
    elevation: np.ndarray,
) -> List[tuple]:
    """
    Build drain package data represent river segments

    Parameters
    ----------
    modelgrid: flopy.discretization.StructuredGrid
        flopy modelgrid object
    leakance: float
        drainage leakance value
    elevation: numpy.ndarray
        land surface elevation

    Returns
    -------
    drn_data: list of tuples
        Drain package data for stream segments
    """
    areas = cell_areas(modelgrid)
    drn_data = []
    idomain = modelgrid.idomain[0]
    for idx in range(modelgrid.ncpl):
        if modelgrid.grid_type == "structured":
            r, c = modelgrid.get_lrc(idx)[0][1:]
            cellid = (r, c)
        else:
            cellid = idx
        area = areas[cellid]
        if idomain[cellid] == 1:
            conductance = leakance * area
            if not isinstance(cellid, tuple):
                cellid = (cellid,)
            drn_data.append(
                (0, *cellid, elevation[cellid] - 0.5, conductance, 1.0)
            )
    return drn_data


def get_model_cell_count(
    model: Union[
        flopy.mf6.ModflowGwf,
        flopy.mf6.ModflowGwt,
    ],
) -> Tuple[int, int]:
    """
    Get the total number of cells and number of active cells in a model.

    Parameters
    ----------
    model: flopy.mf6.ModflowGwf, flopy.mf6.ModflowGwt
        flopy mf6 model object

    Returns
    -------
    ncells: int
        Total number of cells in a model
    nactive: int
        Total number of active cells in a model
    """
    modelgrid = model.modelgrid
    if modelgrid.grid_type == "structured":
        nlay, nrow, ncol = modelgrid.nlay, modelgrid.nrow, modelgrid.ncol
        ncells = nlay * nrow * ncol
        idomain = modelgrid.idomain
        if idomain is None:
            nactive = nlay * nrow * ncol
        else:
            nactive = np.count_nonzero(idomain == 1)
    elif modelgrid.grid_type == "vertex":
        nlay, ncpl = modelgrid.nlay, modelgrid.ncpl
        ncells = nlay * ncpl
        idomain = modelgrid.idomain
        if idomain is None:
            nactive = nlay * ncpl
        else:
            nactive = np.count_nonzero(idomain == 1)
    else:
        raise ValueError(
            f"modelgrid grid type '{modelgrid.grid_type}' not supported"
        )

    return ncells, nactive


def get_simulation_cell_count(
    simulation: flopy.mf6.MFSimulation,
) -> Tuple[int, int]:
    """
    Get the total number of cells and number of active cells in a simulation.

    Parameters
    ----------
    simulation: flopy.mf6.MFSimulation
        flopy mf6 simulation object

    Returns
    -------
    ncells: int
        Total number of cells in a simulation
    nactive: int
        Total number of active cells in a simulation
    """
    ncells = 0
    nactive = 0
    for model_name in simulation.model_names:
        model = simulation.get_model(model_name)
        i, j = get_model_cell_count(model)
        ncells += i
        nactive += j

    return ncells, nactive





# Geometry data
# vertices defining basin boundary and river segments
geometry = {
    "boundary": """1.868012422360248456e+05 4.695652173913043953e+04
1.790372670807453396e+05 5.204968944099379587e+04
1.729813664596273447e+05 5.590062111801243009e+04
1.672360248447204940e+05 5.987577639751553215e+04
1.631987577639751253e+05 6.335403726708075556e+04
1.563664596273291972e+05 6.819875776397516893e+04
1.509316770186335489e+05 7.229813664596274612e+04
1.453416149068323139e+05 7.527950310559007630e+04
1.395962732919254631e+05 7.627329192546584818e+04
1.357142857142857101e+05 7.664596273291927355e+04
1.329192546583850926e+05 7.751552795031057030e+04
1.268633540372670832e+05 8.062111801242237561e+04
1.218944099378881947e+05 8.285714285714286962e+04
1.145962732919254486e+05 8.571428571428572468e+04
1.069875776397515583e+05 8.869565217391305487e+04
1.023291925465838431e+05 8.931677018633540138e+04
9.456521739130433707e+04 9.068322981366459862e+04
8.804347826086955320e+04 9.080745341614908830e+04
7.950310559006211406e+04 9.267080745341615693e+04
7.562111801242236106e+04 9.391304347826087906e+04
6.692546583850930620e+04 9.602484472049689793e+04
5.667701863354037778e+04 9.763975155279504543e+04
4.906832298136646568e+04 9.689440993788820924e+04
3.897515527950309479e+04 9.540372670807455142e+04
3.167701863354036323e+04 9.304347826086958230e+04
2.375776397515527788e+04 8.757763975155279331e+04
1.847826086956521613e+04 8.161490683229814749e+04
1.164596273291925172e+04 7.739130434782608063e+04
6.211180124223596977e+03 7.055900621118013805e+04
4.347826086956512881e+03 6.422360248447205959e+04
1.863354037267072272e+03 6.037267080745341809e+04
2.639751552795024509e+03 5.602484472049689793e+04
1.552795031055893560e+03 5.279503105590062478e+04
7.763975155279410956e+02 4.186335403726709046e+04
2.018633540372667312e+03 3.813664596273292409e+04
6.055900621118013078e+03 3.341614906832297856e+04
1.335403726708074100e+04 2.782608695652173992e+04
2.577639751552794405e+04 2.086956521739130767e+04
3.416149068322980747e+04 1.763975155279503815e+04
4.642857142857142753e+04 1.440993788819875044e+04
5.636645962732918997e+04 1.130434782608694877e+04
6.459627329192546313e+04 9.813664596273290954e+03
8.555900621118012350e+04 6.832298136645956220e+03
9.829192546583850344e+04 5.093167701863346338e+03
1.085403726708074391e+05 4.347826086956525614e+03
1.200310559006211115e+05 4.223602484472040487e+03
1.296583850931677007e+05 4.347826086956525614e+03
1.354037267080745369e+05 5.590062111801232277e+03
1.467391304347825935e+05 1.267080745341615875e+04
1.563664596273291972e+05 1.937888198757762802e+04
1.630434782608695677e+05 2.198757763975155467e+04
1.694099378881987650e+05 2.434782608695652743e+04
1.782608695652173774e+05 2.981366459627329095e+04
1.833850931677018234e+05 3.180124223602484562e+04
1.868012422360248456e+05 3.577639751552795497e+04""",
    "streamseg1": """1.868012422360248456e+05 4.086956521739130403e+04
1.824534161490683327e+05 4.086956521739130403e+04
1.770186335403726553e+05 4.124223602484472940e+04
1.737577639751552779e+05 4.186335403726709046e+04
1.703416149068323139e+05 4.310559006211180531e+04
1.670807453416148783e+05 4.397515527950310934e+04
1.636645962732919143e+05 4.484472049689441337e+04
1.590062111801242281e+05 4.559006211180124228e+04
1.555900621118012350e+05 4.559006211180124228e+04
1.510869565217391064e+05 4.546583850931677443e+04
1.479813664596273156e+05 4.534161490683229931e+04
1.453416149068323139e+05 4.496894409937888850e+04
1.377329192546583654e+05 4.447204968944099528e+04
1.326086956521739194e+05 4.447204968944099528e+04
1.285714285714285652e+05 4.434782608695652743e+04
1.245341614906832110e+05 4.472049689440993825e+04
1.215838509316770069e+05 4.509316770186335634e+04
1.161490683229813585e+05 4.509316770186335634e+04
1.125776397515527933e+05 4.459627329192547040e+04
1.074534161490683036e+05 4.385093167701864149e+04
1.018633540372670686e+05 4.347826086956522340e+04
9.798136645962731563e+04 4.360248447204969125e+04
9.223602484472049400e+04 4.310559006211180531e+04
8.602484472049689793e+04 4.198757763975155831e+04
7.981366459627327276e+04 4.173913043478261534e+04
7.468944099378881219e+04 4.248447204968944425e+04
7.034161490683228476e+04 4.385093167701864149e+04
6.785714285714285506e+04 4.621118012422360334e+04
6.583850931677018525e+04 4.919254658385094081e+04
6.319875776397513982e+04 5.192546583850932075e+04
6.009316770186335634e+04 5.677018633540373412e+04
5.605590062111800216e+04 5.950310559006211406e+04
5.279503105590060295e+04 6.124223602484472940e+04
4.751552795031056303e+04 6.211180124223603343e+04
3.990683229813664366e+04 6.335403726708075556e+04
3.276397515527949508e+04 6.409937888198757719e+04
2.934782608695651652e+04 6.509316770186336362e+04
2.546583850931676716e+04 6.832298136645962950e+04""",
    "streamseg2": """7.025161490683228476e+04 4.375093167701864149e+04
6.816770186335404287e+04 4.273291925465839449e+04
6.490683229813665093e+04 4.211180124223603343e+04
6.164596273291925900e+04 4.173913043478262261e+04
5.776397515527951327e+04 4.124223602484472940e+04
5.450310559006211406e+04 4.049689440993789322e+04
4.984472049689442065e+04 3.937888198757764621e+04
4.534161490683231386e+04 3.801242236024845624e+04
4.114906832298137306e+04 3.664596273291926627e+04
3.913043478260868869e+04 3.565217391304348712e+04
3.649068322981366509e+04 3.416149068322981475e+04
3.322981366459628043e+04 3.242236024844721760e+04
3.012422360248447148e+04 3.105590062111801672e+04
2.608695652173913550e+04 2.957521739130435890e+04""",
    "streamseg3": """1.059006211180124228e+05 4.335403726708074828e+04
1.029503105590062187e+05 4.223602484472050128e+04
1.004658385093167890e+05 4.024844720496894297e+04
9.937888198757765349e+04 3.788819875776398112e+04
9.627329192546584818e+04 3.490683229813664366e+04
9.285714285714286962e+04 3.316770186335403559e+04
8.897515527950311662e+04 3.093167701863354159e+04
8.338509316770188161e+04 2.795031055900621504e+04
7.872670807453416637e+04 2.670807453416148928e+04
7.329192546583851799e+04 2.385093167701863058e+04
6.863354037267081731e+04 2.111801242236025064e+04
6.304347826086958230e+04 1.863354037267081003e+04""",
    "streamseg4": """1.371118012422360480e+05 4.472049689440994553e+04
1.321428571428571595e+05 4.720496894409938250e+04
1.285714285714285652e+05 4.981366459627330187e+04
1.243788819875776535e+05 5.341614906832298584e+04
1.189440993788819906e+05 5.540372670807454415e+04
1.125776397515527933e+05 5.627329192546584818e+04
1.065217391304347839e+05 5.726708074534162733e+04
1.020186335403726698e+05 5.913043478260870324e+04
9.409937888198759174e+04 6.273291925465840177e+04
9.192546583850932075e+04 6.633540372670808574e+04
8.881987577639751544e+04 7.242236024844722124e+04
8.586956521739131131e+04 7.552795031055902655e+04
8.369565217391305487e+04 7.962732919254660374e+04""",
}


def get_base_dir():
    """
    Returns the working directory for the base simulation data and results

    Returns
    -------
    dir: str
        the working directory
    """
    dir = pl.Path.cwd().joinpath("temp/base")
    return dir


def get_parallel_dir():
    """
    Returns the working directory for the parallel simulation data and results

    Returns
    -------
    dir: str
        the working directory
    """
    dir = pl.Path.cwd().joinpath("temp/parallel")
    return dir


def string2geom(
    geostring: str,
    conversion: float = None,
) -> List[tuple]:
    """
    Convert a multi-line string of vertices to a list of x,y vertices

    Parameters
    ----------
    geostring: str
        multi-line string of x,y vertices
    conversion: float, options
        x,y, vertices conversion factor (Default is None)

    Returns
    -------
    res: List[tuple]
        list of x,y vertices
    """
    if conversion is None:
        multiplier = 1.0
    else:
        multiplier = float(conversion)
    res = []
    for line in geostring.split("\n"):
        line = line.split(" ")
        x = float(line[0]) * multiplier
        y = float(line[1]) * multiplier
        res.append((x, y))
    return res


def set_structured_idomain(
    modelgrid: flopy.discretization.StructuredGrid,
    boundary: List[tuple],
) -> None:
    """
    Set the idomain for a structured grid using a boundary line.

    Parameters
    ----------
    modelgrid: flopy.discretization.StructuredGrid
        flopy modelgrid object
    boundary: List(tuple)
        list of x,y tuples defining the boundary of the active model domain.

    Returns
    -------
    None

    """
    if modelgrid.grid_type != "structured":
        raise ValueError(
            f"modelgrid must be 'structured' not '{modelgrid.grid_type}'"
        )

    ix = GridIntersect(modelgrid, method="vertex", rtree=True)
    result = ix.intersect(Polygon(boundary))
    idx = [coords for coords in result.cellids]
    idx = np.array(idx, dtype=int)
    nr = idx.shape[0]
    if idx.ndim == 1:
        idx = idx.reshape((nr, 1))

    idx = tuple([idx[:, i] for i in range(idx.shape[1])])
    idomain = np.zeros(modelgrid.shape[1:], dtype=int)
    idomain[idx] = 1
    idomain = idomain.reshape(modelgrid.shape)

    # set modelgrid idomain
    modelgrid.idomain = idomain
    return


def intersect_segments(
    modelgrid: Union[
        flopy.discretization.StructuredGrid, flopy.discretization.VertexGrid
    ],
    segments: List[List[tuple]],
) -> Tuple[flopy.utils.GridIntersect, list, list]:
    """

    Parameters
    ----------
    modelgrid: flopy.discretization.StructuredGrid
        flopy modelgrid object
    segments: list of list of tuples
        List of segment x,y tuples

    Returns
    -------
    ixs: flopy.utils.GridIntersect
        flopy GridIntersect object
    cellids: list
        list of intersected cellids
    lengths: list
        list of intersected lengths

    """
    ixs = flopy.utils.GridIntersect(
        modelgrid,
        method=modelgrid.grid_type,
    )
    cellids = []
    lengths = []
    for sg in segments:
        v = ixs.intersect(LineString(sg), sort_by_cellid=True)
        cellids += v["cellids"].tolist()
        lengths += v["lengths"].tolist()
    return ixs, cellids, lengths


def cell_areas(
    modelgrid: Union[
        flopy.discretization.StructuredGrid, flopy.discretization.VertexGrid
    ],
) -> np.ndarray:
    """
    Calculate cell areas

    Parameters
    ----------
    modelgrid: flopy.discretization.StructuredGrid
        flopy modelgrid object

    Returns
    -------
    areas: numpy.ndarray
        cell areas

    """
    if modelgrid.grid_type == "structured":
        nrow, ncol = modelgrid.nrow, modelgrid.ncol
        areas = np.zeros((nrow, ncol), dtype=float)
        for r in range(nrow):
            for c in range(ncol):
                cellid = (r, c)
                vertices = np.array(modelgrid.get_cell_vertices(cellid))
                area = Polygon(vertices).area
                areas[cellid] = area
    elif modelgrid.grid_type == "vertex":
        areas = np.zeros(modelgrid.ncpl, dtype=float)
        for idx in range(modelgrid.ncpl):
            vertices = np.array(modelgrid.get_cell_vertices(idx))
            area = Polygon(vertices).area
            areas[idx] = area
    else:
        raise ValueError(
            "modelgrid must be 'structured' or 'vertex' not "
            + f"{modelgrid.grid_type}"
        )
    return areas


def build_drain_data(
    modelgrid: Union[
        flopy.discretization.StructuredGrid, flopy.discretization.VertexGrid
    ],
    cellids: list,
    lengths: list,
    leakance: float,
    elevation: np.ndarray,
) -> List[tuple]:
    """
    Build drain package data represent river segments

    Parameters
    ----------
    modelgrid: flopy.discretization.StructuredGrid
        flopy modelgrid object
    cellids: list
        list of intersected cellids
    lengths: list
        list of intersected lengths
    leakance: float
        drainage leakance value
    elevation: numpy.ndarray
        land surface elevation

    Returns
    -------
    drn_data: list of tuples
        Drain package data for stream segments

    """
    drn_data = []
    for cellid, length in zip(cellids, lengths):
        x = modelgrid.xcellcenters[cellid]
        width = 5.0 + (14.0 / Lx) * (Lx - x)
        conductance = leakance * length * width
        if not isinstance(cellid, tuple):
            cellid = (cellid,)
        drn_data.append((0, *cellid, elevation[cellid], conductance))
    return drn_data


def build_groundwater_discharge_data(
    modelgrid: Union[
        flopy.discretization.StructuredGrid, flopy.discretization.VertexGrid
    ],
    leakance: float,
    elevation: np.ndarray,
) -> List[tuple]:
    """
    Build drain package data represent river segments

    Parameters
    ----------
    modelgrid: flopy.discretization.StructuredGrid
        flopy modelgrid object
    leakance: float
        drainage leakance value
    elevation: numpy.ndarray
        land surface elevation

    Returns
    -------
    drn_data: list of tuples
        Drain package data for stream segments
    """
    areas = cell_areas(modelgrid)
    drn_data = []
    idomain = modelgrid.idomain[0]
    for idx in range(modelgrid.ncpl):
        if modelgrid.grid_type == "structured":
            r, c = modelgrid.get_lrc(idx)[0][1:]
            cellid = (r, c)
        else:
            cellid = idx
        area = areas[cellid]
        if idomain[cellid] == 1:
            conductance = leakance * area
            if not isinstance(cellid, tuple):
                cellid = (cellid,)
            drn_data.append(
                (0, *cellid, elevation[cellid] - 0.5, conductance, 1.0)
            )
    return drn_data


def get_model_cell_count(
    model: Union[
        flopy.mf6.ModflowGwf,
        flopy.mf6.ModflowGwt,
    ],
) -> Tuple[int, int]:
    """
    Get the total number of cells and number of active cells in a model.

    Parameters
    ----------
    model: flopy.mf6.ModflowGwf, flopy.mf6.ModflowGwt
        flopy mf6 model object

    Returns
    -------
    ncells: int
        Total number of cells in a model
    nactive: int
        Total number of active cells in a model
    """
    modelgrid = model.modelgrid
    if modelgrid.grid_type == "structured":
        nlay, nrow, ncol = modelgrid.nlay, modelgrid.nrow, modelgrid.ncol
        ncells = nlay * nrow * ncol
        idomain = modelgrid.idomain
        if idomain is None:
            nactive = nlay * nrow * ncol
        else:
            nactive = np.count_nonzero(idomain == 1)
    elif modelgrid.grid_type == "vertex":
        nlay, ncpl = modelgrid.nlay, modelgrid.ncpl
        ncells = nlay * ncpl
        idomain = modelgrid.idomain
        if idomain is None:
            nactive = nlay * ncpl
        else:
            nactive = np.count_nonzero(idomain == 1)
    else:
        raise ValueError(
            f"modelgrid grid type '{modelgrid.grid_type}' not supported"
        )

    return ncells, nactive


def get_simulation_cell_count(
    simulation: flopy.mf6.MFSimulation,
) -> Tuple[int, int]:
    """
    Get the total number of cells and number of active cells in a simulation.

    Parameters
    ----------
    simulation: flopy.mf6.MFSimulation
        flopy mf6 simulation object

    Returns
    -------
    ncells: int
        Total number of cells in a simulation
    nactive: int
        Total number of active cells in a simulation
    """
    ncells = 0
    nactive = 0
    for model_name in simulation.model_names:
        model = simulation.get_model(model_name)
        i, j = get_model_cell_count(model)
        ncells += i
        nactive += j

    return ncells, nactive
