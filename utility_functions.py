import data_organisation_tools as DOT
import calibration_tools as CT

from ccdproc import ImageFileCollection
import pathlib
import numpy as np

root_dir = pathlib.Path(__file__).resolve().parent
all_fits_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS")
calib_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS"/"PROCESSED FRAMES")

def report_names(all_fits_path):
    lights=ImageFileCollection(all_fits_path,keywords='*',glob_exclude="bias_sub_*")
    lights.sort(["object","mjd-obs"])
    all_objects=np.array(lights.summary["object"])
    print(np.unique(all_objects))

report_names(all_fits_path)
