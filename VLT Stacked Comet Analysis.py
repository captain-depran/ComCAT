import photometry_core as photo_core
import calibration_tools as CT
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
import numpy as np
import pathlib

import time

root_dir = pathlib.Path(__file__).resolve().parent
calib_path = pathlib.Path(root_dir/"Data_set_2"/"block_1"/"ALL_FITS"/"PROCESSED FRAMES")
all_fits_path = pathlib.Path(root_dir/"Data_set_2"/"block_1"/"ALL_FITS")

filter="R_SPECIAL"

#names=["P29","2009AU16","93P","94P","50P","74P","P113"]
#jpl_names=["29P","P/2009 AU16","93P","94P","50P","74P","113P"]
#ephs=[90000394,0,90000917,90000921,90000587,90000821,0]

names=["93P"]
jpl_names=["93P"]
ephs=[90000917]

for name,jpl_name,eph in zip(names,jpl_names,ephs):
    comet = photo_core.study_comet(name,
                                   jpl_name,
                                   filter,
                                   calib_path,
                                   eph_code=eph,
                                   plot_stack=True,
                                   comet_pixel_max=15000,
                                   show_frames=False,
                                   cutout_size=50,
                                   man_shift = [0,0])
    if comet.skip_this:
        print("No images :C ")
        continue
    """
    comet.plot_surf_brightness(max=30,
                               logx=False,
                               logy=False,
                               y_relative=False)
    """
    
    """
    print(comet.jpl_name)
    print(filter)
    print("Mag: ")
    print(comet.mags)
    print("-"*10)
    """
    #comet.show_full_stack()
    #plt.scatter(comet.t,comet.mags)
    #print(np.median(comet.mags))
    plt.errorbar(comet.t,comet.mags,yerr=comet.errors,fmt="k.")
    plt.show()

#plt.show()
plt.legend()
plt.xlabel("Radius (Arcsecs)")
plt.ylabel("Log Surface Brightness (Counts/area)")
plt.title("Surface Brightness profiles for Multiple Comets")
plt.show()
