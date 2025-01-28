
import os
import pyemu
import pandas as pd
import flopy
import platform
import shutil


#function count up convergence fails and add as obs
def converg_fails():
    data = open(os.path.join("freyberg6.lst")).read()
    flow_fail_count = data.count('FAILED')
    fails = pd.DataFrame(index=['flow_fails'], columns=['count'])
    fails.index.name = "type"
    fails.iloc[0, 0] = flow_fail_count
    fails.to_csv('convergence_failure_counts.csv')

#function to run converg fails fxn
def test_converg_fails(d):
    cwd = os.getcwd()
    os.chdir(d)
    converg_fails()
    os.chdir(cwd)

# list file processing for water budget stuff
def process_list_files():
    """process the gwf and gwt list files into a format that is actually useful....
        took this lil number from benchmark.py in zp1 to include transport list file components
    """

    class Mf6TListBudget(flopy.utils.mflistfile.ListBudget):
        """"""

        def set_budget_key(self):
            self.budgetkey = "MASS BUDGET FOR ENTIRE MODEL"
            return

    lst1 = flopy.utils.Mf6ListBudget("freyberg6.lst")
    fdf, cdf = lst1.get_dataframes(diff=True, start_datetime="1-1-2020")
    cdf.index.name = "time"
    fdf.index.name = "time"
    cdf.fillna(0.0, inplace=True)
    fdf.fillna(0.0, inplace=True)
    fdf.to_csv("inc_flow.csv", date_format="%Y-%m-%d")
    cdf.to_csv("cum_flow.csv", date_format="%Y-%m-%d")

def test_process_list_files(d):
    cwd = os.getcwd()
    os.chdir(d)
    process_list_files()
    os.chdir(cwd)

def prep_deps(d):
    """copy exes to a directory based on platform

    Args:

        d (str): directory to copy into

    """
    # copy in deps and exes
    if "window" in platform.platform().lower():
        bd = os.path.join("bin", "windows")
    elif "linux" in platform.platform().lower():
        bd = os.path.join("bin", "linux")
    else:
        bd = os.path.join("bin", "mac")
    dest_dirs = [dd for dd in os.listdir(d) if os.path.isdir(os.path.join(d, dd))]

    for f in os.listdir(bd):
        if "window" in platform.platform().lower():
            if not f.startswith('pestpp'):
                for dd in dest_dirs:
                    shutil.copy2(os.path.join(bd, f), os.path.join(d, dd, f))
                shutil.copy2(os.path.join(bd, f), os.path.join(d, f))
            else:
                shutil.copy2(os.path.join(bd, f), os.path.join(d, f))
        else:
            shutil.copy2(os.path.join(bd, f), os.path.join(d, f))
    for src_d in ["flopy"]:
        assert os.path.exists(src_d), src_d
        dest_d = os.path.join(d, src_d)
        if os.path.exists(dest_d):
            shutil.rmtree(dest_d)
        shutil.copytree(src_d, dest_d)