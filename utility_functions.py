import data_organisation_tools as DOT
import calibration_tools as CT

from ccdproc import ImageFileCollection
import ccdproc as ccdp
import pathlib
import os
import numpy as np

root_dir = pathlib.Path(__file__).resolve().parent
all_fits_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS")
calib_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS"/"PROCESSED FRAMES")

def report_names(all_fits_path):
    lights=ImageFileCollection(all_fits_path,keywords='*',glob_exclude="bias_sub_*")
    lights.sort(["object","mjd-obs"])
    all_objects=np.array(lights.summary["object"])
    names=np.unique(all_objects)
    return names

def clean_unsolved(calib_path):
    lights=ImageFileCollection(calib_path,keywords='*')
    criteria = {"plate_solved":False}
    file_names=lights.files_filtered(**criteria)
    print("FILES TO BE DELETED: ")
    for name in file_names:
        print(name)
    print("-"*10)
    confirm=input(("DELETE "+str(len(file_names))+" FILES???"))
    confirm=input(("ARE YOU ABSOLUTELY SURE?"))
    for name in file_names:
        os.remove(calib_path/name)


#names=report_names(all_fits_path)
#clean_unsolved(calib_path)