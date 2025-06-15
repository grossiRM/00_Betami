import yaml ; import pathlib as pl ; import flopy ; import numpy as np
from typing import List, Tuple, Union

#__________________
geometries = yaml.safe_load(open(pl.Path("E:/15_REPOS/00_BETAMI/bin/Flopy_3099/geometries.yml")))
#__________________
def string2geom(geostring: str,conversion: float = None,) -> List[tuple]:
    if conversion is None:    multiplier = 1.0
    else:                     multiplier = float(conversion)
    res = []
    for line in geostring.split("\n"):
        line = line.split(" ") ;         x = float(line[0]) * multiplier ; y = float(line[1]) * multiplier ;         res.append((x, y))
    return res
def string2geom(geostring, conversion=None):
    if conversion is None:    multiplier = 1.0
    else:                     multiplier = float(conversion)
    res = []
    for line in geostring.split("\n"):
        if not any(line):
            continue
        line = line.strip() ; line = line.split(" ") ; x = float(line[0]) * multiplier ; y = float(line[1]) * multiplier ; res.append((x, y))
    return res
#__________________
#flopy.mf6.ModflowIms(sim,complexity="simple",print_option="SUMMARY",csv_outer_output_filerecord="outer.csv",
#                           csv_inner_output_filerecord="inner.csv",linear_acceleration="bicgstab",outer_maximum=1000,inner_maximum=100,
#                           outer_dvclose=1e-4,inner_dvclose=1e-5,preconditioner_levels=2,relaxation_factor=0.0)
#__________________
def read_solver_csv():
    fpath = sim_ws / "ims.inner.csv"
    return pd.read_csv(fpath)
#__________________
#stress_period_data.data - drn_data
#flopy.mf6.ModflowGwfdrn(gwf,maxbound=len(drn_data),stress_period_data=drn_data,pname="river",filename="drn_riv.drn")
#flopy.mf6.ModflowGwfdrn(gwf,auxiliary=["depth"],auxdepthname="depth",maxbound=len(gw_discharge_data),
#                                  stress_period_data=gw_discharge_data,pname="gwd",filename="drn_gwd.drn")
##gw_discharge_data = build_groundwater_discharge_data(working_grid,leakance,top_wg)  # gw_discharge_data[:10]
def build_groundwater_discharge_data(modelgrid: Union[flopy.discretization.StructuredGrid, flopy.discretization.VertexGrid],
                                     leakance: float,elevation: np.ndarray,) -> List[tuple]:
    areas = cell_areas(modelgrid)   ; drn_data = [] ; idomain = modelgrid.idomain[0]
    for idx in range(modelgrid.ncpl):
        if modelgrid.grid_type == "structured":  r, c = modelgrid.get_lrc(idx)[0][1:]  ; cellid = (r, c)
        else:                                    cellid = idx 
        area = areas[cellid]
        if idomain[cellid] == 1:
            conductance = leakance * area
            if not isinstance(cellid, tuple):    cellid = (cellid,)
            drn_data.append((0, *cellid, elevation[cellid] - 0.5, conductance, 1.0))
    return drn_data
#__________________
def densify_geometry(line, step, keep_internal_nodes=True):
    xy = []  ;     lines_strings = []
    if keep_internal_nodes:
        for idx in range(1, len(line)):      lines_strings.append(shapely.geometry.LineString(line[idx - 1 : idx + 1]))
    else:                                    lines_strings = [shapely.geometry.LineString(line)]
    for                       line_string in lines_strings:
        length_m = line_string.length  
        for distance in np.arange(0, length_m + step, step):
            point = line_string.interpolate(distance)       ; xy_tuple = (point.x, point.y)
            if xy_tuple not in xy:                            xy.append(xy_tuple)
        if keep_internal_nodes:
            xy_tuple = line_string.coords[-1]
            if xy_tuple not in xy:                            xy.append(xy_tuple)
    return xy
#__________________
def set_idomain(grid, boundary):
    ix = GridIntersect(grid, method="vertex", rtree=True)    ; result = ix.intersect(Polygon(boundary))
    idx = [coords for coords in result.cellids]              ;    idx = np.array(idx, dtype=int)          ; nr = idx.shape[0]
    if idx.ndim == 1:                                             idx = idx.reshape((nr, 1))
    idx = tuple([idx[:, i] for i in range(idx.shape[1])])    ; idomain = np.zeros(grid.shape[1:], dtype=int)
    idomain[idx] = 1                                         ; idomain = idomain.reshape(grid.shape)      ; grid.idomain = idomain
#__________________
#__________________
def set_structured_idomain(modelgrid: flopy.discretization.StructuredGrid,boundary: List[tuple]) -> None:
    if modelgrid.grid_type != "structured":
        raise ValueError(f"modelgrid must be 'structured' not '{modelgrid.grid_type}'")
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
    modelgrid.idomain = idomain
    return
#__________________
def intersect_segments(modelgrid: Union[flopy.discretization.StructuredGrid, flopy.discretization.VertexGrid],
                       segments: List[List[tuple]]) -> Tuple[flopy.utils.GridIntersect, list, list]:
    ixs = flopy.utils.GridIntersect(modelgrid,method=modelgrid.grid_type)  ; cellids = []  ; lengths = []
    for sg in segments:
        v = ixs.intersect(LineString(sg), sort_by_cellid=True) ; cellids += v["cellids"].tolist()  ; lengths += v["lengths"].tolist()
    return ixs, cellids, lengths
#__________________
def cell_areas(modelgrid: Union[flopy.discretization.StructuredGrid, flopy.discretization.VertexGrid]) -> np.ndarray:
    if modelgrid.grid_type == "structured":
        nrow, ncol = modelgrid.nrow, modelgrid.ncol ; areas = np.zeros((nrow, ncol), dtype=float)
        for r in range(nrow):
            for c in range(ncol): 
                cellid = (r, c) ; vertices = np.array(modelgrid.get_cell_vertices(cellid)) ; area = Polygon(vertices).area ; areas[cellid] = area
    elif modelgrid.grid_type == "vertex": 
        areas = np.zeros(modelgrid.ncpl, dtype=float)
        for idx in range(modelgrid.ncpl): 
            vertices = np.array(modelgrid.get_cell_vertices(idx))  ; area = Polygon(vertices).area ; areas[idx] = area
    else:   raise ValueError(+ f"{modelgrid.grid_type}")
    return areas
#__________________
def build_groundwater_discharge_data(modelgrid: Union[flopy.discretization.StructuredGrid, flopy.discretization.VertexGrid],
                                     leakance: float,elevation: np.ndarray,) -> List[tuple]:
    areas = cell_areas(modelgrid)   ; drn_data = [] ; idomain = modelgrid.idomain[0]
    for idx in range(modelgrid.ncpl):
        if modelgrid.grid_type == "structured":  r, c = modelgrid.get_lrc(idx)[0][1:]  ; cellid = (r, c)
        else:                                    cellid = idx 
        area = areas[cellid]
        if idomain[cellid] == 1:
            conductance = leakance * area
            if not isinstance(cellid, tuple):    cellid = (cellid,)
            drn_data.append((0, *cellid, elevation[cellid] - 0.5, conductance, 1.0))
    return drn_data
#__________________
def get_model_cell_count(model: Union[flopy.mf6.ModflowGwf,flopy.mf6.ModflowGwt]) -> Tuple[int, int]:
    modelgrid = model.modelgrid
    if modelgrid.grid_type == "structured":
        nlay, nrow, ncol = modelgrid.nlay, modelgrid.nrow, modelgrid.ncol ; ncells = nlay * nrow * ncol ; idomain = modelgrid.idomain
        if idomain is None:             nactive = nlay * nrow * ncol
        else:                           nactive = np.count_nonzero(idomain == 1)
    elif modelgrid.grid_type == "vertex":
        nlay, ncpl = modelgrid.nlay, modelgrid.ncpl ; ncells = nlay * ncpl ;         idomain = modelgrid.idomain
        if idomain is None:             nactive = nlay * ncpl
        else:                           nactive = np.count_nonzero(idomain == 1)
    else:         raise ValueError(f"modelgrid grid type '{modelgrid.grid_type}' not supported")
    return ncells, nactive
#__________________
def get_simulation_cell_count(simulation: flopy.mf6.MFSimulation,) -> Tuple[int, int]:
    ncells = 0 ;     nactive = 0
    for model_name in simulation.model_names:
        model = simulation.get_model(model_name) ; i, j = get_model_cell_count(model) ; ncells += i ; nactive += j
    return ncells, nactive
ncells, nactive = get_simulation_cell_count(sim) ;  print("nr. of cells:", ncells, ", active:", nactive)
#__________________
#__________________
#__________________
#__________________
#__________________
