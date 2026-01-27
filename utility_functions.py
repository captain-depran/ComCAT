import data_organisation_tools as DOT
import calibration_tools as CT

from ccdproc import ImageFileCollection
import ccdproc as ccdp
from astropy.nddata import CCDData
import pathlib
import os
import numpy as np
import matplotlib.pyplot as plt

root_dir = pathlib.Path(__file__).resolve().parent
all_fits_path = pathlib.Path(root_dir/"Data_set_2"/"block_1"/"ALL_FITS")
#calib_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS"/"PROCESSED FRAMES")

def report_names(all_fits_path):
    lights=ImageFileCollection(all_fits_path,keywords='*',glob_exclude="bias_sub_*")
    lights.sort(["object","mjd-obs"])
    all_objects=np.array(lights.summary["object"])
    names=np.unique(all_objects)
    return names

def plot_times(all_fits_path):
    lights=ImageFileCollection(all_fits_path,keywords='*',glob_exclude="bias_sub_*")
    lights.sort(["object","mjd-obs"])
    all_times=np.array(lights.summary["mjd-obs"])
    plt.scatter((all_times-np.min(all_times))*24,np.zeros(all_times.shape))
    plt.show()

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

def report_exptime(all_fits_path):
    lights=ImageFileCollection(all_fits_path,keywords='*',glob_exclude="bias_sub_*")
    lights.sort(["object","mjd-obs"])
    all_objects=np.array(lights.summary["exptime"])
    times,counts=np.unique(all_objects,return_counts=True)
    return times,counts

def VLT_chip2_purge(all_fits_path):
    lights=ImageFileCollection(all_fits_path,keywords='*')
    criteria = {"EXTNAME":"CHIP2"}
    file_names=lights.files_filtered(**criteria)
    print("FILES TO BE DELETED: ")
    for name in file_names:
        print(name)
    print("-"*10)
    confirm=input(("DELETE "+str(len(file_names))+" FILES???"))
    confirm=input(("ARE YOU ABSOLUTELY SURE?"))
    for name in file_names:
        os.remove(all_fits_path/name)


#VLT_chip2_purge(pathlib.Path(root_dir/"Data_set_2"/"unpacked_data"))

#names=report_names(all_fits_path)
#print(names)
#clean_unsolved(calib_path)

#plot_times(all_fits_path)


#times,counts=report_exptime(all_fits_path)
#for time,count in zip(times,counts):
    #print("EXPTIME: ",time," -> ",count," IMAGES")

