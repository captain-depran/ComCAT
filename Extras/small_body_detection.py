import data_organisation_tools as DOT
import calibration_tools as CT
from astropy.io import fits
from astropy import wcs
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pathlib
from ccdproc import ImageFileCollection, Combiner, combine
import ccdproc as ccdp
from ccdproc import wcs_project
import numpy as np


root_dir = pathlib.Path(__file__).resolve().parent

px_scale=0.24
tgt_name="149P"
filter="R#642"

all_fits_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS")
calib_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS"/"PROCESSED FRAMES")

criteria={'object' : tgt_name, "ESO INS FILT1 NAME".lower():filter,'plate_solved':True}

science=ImageFileCollection(calib_path,keywords='*',glob_include=tgt_name+"_"+filter+"*")
science.sort(["object","mjd-obs"])
files=science.files_filtered(**criteria)
with fits.open(pathlib.Path(calib_path/files[0])) as img:
    tgt_wcs=wcs.WCS(img[0].header)

avg_stack=CT.allign_and_avgstack(science,tgt_wcs)
#CT.show_image(avg_stack)

with fits.open(pathlib.Path(calib_path/files[0])) as img:
    data=img[0].data


reprojected=[]
n_max=99
n=0

for img in science.ccds():
    n+=1
    new_image=wcs_project(img,tgt_wcs)
    new_image.data=new_image.data-avg_stack.data
    new_image.data[new_image.data < 0]=0
    
    reprojected.append(new_image)
    #plt.imshow(new_image.data-avg_stack.data,origin="lower",norm=LogNorm())
    #plt.show()

    if n>=n_max:
        break

c=Combiner(reprojected)
c.minmax_clipping(min_clip=10,max_clip=100)
big_stack=c.sum_combine()

plt.imshow(avg_stack.data,origin="lower",norm=LogNorm())
plt.show()

