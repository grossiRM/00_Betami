import pathlib as pl
from typing import List, Tuple, Union

import flopy
import numpy as np
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

def get_base_dir():
    dir = pl.Path.cwd().joinpath("temp/base")
    return dir

def get_parallel_dir():
    dir = pl.Path.cwd().joinpath("temp/parallel")
    return dir
