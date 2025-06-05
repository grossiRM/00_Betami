import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import shutil
import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import shutil
import sys
import pyemu
import flopy

bin_path = os.path.join( "..", "bin", "win")

def prep_bins(dest_path):
    files = os.listdir(bin_path)
    for f in files:
        if os.path.exists(os.path.join(dest_path,f)):
            os.remove(os.path.join(dest_path,f))
        shutil.copy2(os.path.join(bin_path,f),os.path.join(dest_path,f))