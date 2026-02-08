import photometry_core as photo_core
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pathlib

root_dir = pathlib.Path(__file__).resolve().parent
calib_path = pathlib.Path(root_dir/"Data_set_3"/"block_4"/"ALL_FITS"/"PROCESSED FRAMES")
all_fits_path = pathlib.Path(root_dir/"Data_set_3"/"block_4"/"ALL_FITS")

filter_1="V#606"
filter_2="R#608"


#names=["137P_Shoemaker-Levy2"]
names=["123P"]
jpl_names=["123P"]
ephs=[90001024]

sol_col=0.45

colour_term_1 = -0.4811791650583872
colour_term_2 = -0.17286169884964958
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
                                   comet_pixel_max=6000,
                                   show_frames=False,
                                   cutout_size=30,
                                   man_shift = [0,0])
    comet.analyse_path()
    #comet.mark2_comet_track()
    m_obs=np.nanmedian(comet.mags)
    print(m_obs)
    #plt.scatter(comet.t,comet.mags)
    plt.errorbar(comet.t,comet.mags,yerr=comet.errors,fmt="k.")
    plt.show()

    
get_obs_mag(filter_2,names[0],jpl_names[0],ephs[0])

