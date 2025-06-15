import yaml ; import pathlib as pl ; import flopy ; import numpy as np
from typing import List, Tuple, Union

geometries = yaml.safe_load(open(pl.Path("E:/15_REPOS/00_BETAMI/bin/Flopy_3099/geometries.yml")))

 
#def string2geom(geostring: str,conversion: float = None,) -> List[tuple]:
#    if conversion is None:         multiplier = 1.0
#    else:         multiplier = float(conversion)
#    res = []
#    for line in geostring.split("\n"):
#        line = line.split(" ") ;         x = float(line[0]) * multiplier ; y = float(line[1]) * multiplier ;         res.append((x, y))
#    return res

def string2geom(geostring, conversion=None):
    if conversion is None:    multiplier = 1.0
    else:                     multiplier = float(conversion)
    res = []
    for line in geostring.split("\n"):
        if not any(line):
            continue
        line = line.strip() ; line = line.split(" ") ; x = float(line[0]) * multiplier ; y = float(line[1]) * multiplier ; res.append((x, y))
    return res


#flopy.mf6.ModflowIms(sim,complexity="simple",print_option="SUMMARY",csv_outer_output_filerecord="outer.csv",
#                           csv_inner_output_filerecord="inner.csv",linear_acceleration="bicgstab",outer_maximum=1000,inner_maximum=100,
#                           outer_dvclose=1e-4,inner_dvclose=1e-5,preconditioner_levels=2,relaxation_factor=0.0)

def read_solver_csv():
    fpath = sim_ws / "ims.inner.csv"
    return pd.read_csv(fpath)


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