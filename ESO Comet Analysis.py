import photometry_core as photo_core
import calibration_tools as CT
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
import numpy as np
import pathlib

import time

root_dir = pathlib.Path(__file__).resolve().parent
calib_path = pathlib.Path(root_dir/"Data_set_1"/"block_3"/"ALL_FITS"/"PROCESSED FRAMES")
all_fits_path = pathlib.Path(root_dir/"Data_set_1"/"block_3"/"ALL_FITS")

filter="R#642"

#names=["P29","2009AU16","93P","94P","50P","74P","P113"]
#jpl_names=["29P","P/2009 AU16","93P","94P","50P","74P","113P"]
#ephs=[90000394,0,90000917,90000921,90000587,90000821,0]

names=["29P","93P","94P","50P","P113","48P"]
jpl_names=["29P","93P","94P","50P","113P","48P"]
ephs=[90000394,90000917,90000921,90000588,0,90000568]

for name,jpl_name,eph in zip(names,jpl_names,ephs):
    comet = photo_core.study_comet(name,jpl_name,filter,calib_path,eph_code=eph,plot_stack=False)
    
    comet.plot_surf_brightness(max=30,
                               logx=False,
                               logy=True,
                               y_relative=False)
    

plt.legend()
plt.xlabel("Radius (Arcsecs)")
plt.ylabel("Log Surface Brightness (Counts/area)")
plt.title("Surface Brightness profiles for Multiple Comets")
plt.show()