import photometry_core as photo_core
import calibration_tools as CT
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
import numpy as np
import pathlib

import time

root_dir = pathlib.Path(__file__).resolve().parent
calib_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS"/"PROCESSED FRAMES")
all_fits_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS")

filter_1="V#641"
filter_2="R#642"


names=["93P"]
jpl_names=["93P"]
ephs=[90000917]

sol_col=0.45

colour_term_1 = -0.5555139559009306
colour_term_2 = -0.1659028667301966
#shift_col= 0.45 #Solar colour, starting point
#shift_col= 0.4085325  

def get_obs_mag(filter,name,jpl_name,eph):
    global calib_path
    comet = photo_core.study_comet(name,
                                   jpl_name,
                                   filter,
                                   calib_path,
                                   eph_code=eph,
                                   plot_stack=False,
                                   comet_pixel_max=5000,
                                   show_frames=False,
                                   cutout_size=200,
                                   man_shift = [0,0])
    
    m_obs=np.nanmedian(comet.mags)
    #plt.scatter(comet.t,comet.mags)
    plt.errorbar(comet.t,comet.mags,yerr=comet.errors,fmt="k.")
    plt.show()

    return m_obs,comet.skip_this

def shift_colour(mag,shift_col,colour_term):
     m_act=mag+(colour_term*shift_col)
     return m_act

def iter_colour(obs_1,obs_2,shift_col,term1,term2):
    m1=shift_colour(obs_1,shift_col,term1)
    m2=shift_colour(obs_2,shift_col,term2)

    return (m1-m2),m1,m2    

    
first=True
for name,jpl_name,eph in zip(names,jpl_names,ephs):

    m_obs_1,no_images=get_obs_mag(filter_1,name,jpl_name,eph)
    if no_images:
        print("No images :C ")
        continue
    m_obs_2,no_images=get_obs_mag(filter_2,name,jpl_name,eph)
    if no_images:
        print("No images :C ")
        continue
    for i in range(0,50):
        if first:
            colour,mag1,mag2=iter_colour(m_obs_1,
                                        m_obs_2,
                                        sol_col,
                                        colour_term_1,
                                        colour_term_2)
            first=False
        else:
            colour,mag1,mag2=iter_colour(m_obs_1,
                                        m_obs_2,
                                        colour,
                                        colour_term_1,
                                        colour_term_2)

        print("MAG 1: ",mag1)
        print("MAG 2: ",mag2)
        print("COLOUR: ",colour)
        print("---------------------")

    

    
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
    #plt.show()