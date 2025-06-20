import os
import pathlib as pl
from typing import List, Tuple, Union

import flopy
import numpy as np
import shapely
from flopy.utils.gridintersect import GridIntersect
from shapely.geometry import LineString, Polygon

# figures
figwidth = 180  # 90 # mm
figwidth = figwidth / 10 / 2.54  # inches
figheight = figwidth
figsize = (figwidth, figheight)

# domain information
Lx = 180000
Ly = 100000


def set_structured_idomain(modelgrid: flopy.discretization.StructuredGrid,boundary: List[tuple]) -> None:
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

def intersect_segments(modelgrid: Union[flopy.discretization.StructuredGrid, flopy.discretization.VertexGrid],segments: List[List[tuple]]) -> Tuple[flopy.utils.GridIntersect, list, list]:
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
    ixs = flopy.utils.GridIntersect(modelgrid,method=modelgrid.grid_type,)  ; cellids = [] ; lengths = []
    for sg in segments:
        v = ixs.intersect(LineString(sg), sort_by_cellid=True) ; cellids += v["cellids"].tolist() ; lengths += v["lengths"].tolist()
    return ixs, cellids, lengths