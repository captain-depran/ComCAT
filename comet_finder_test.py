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

tgt_name="50P"
search_name="50P"
filter="R#642"

obs_code=809

all_frames=[]


all_image_names=photo_core.get_image_files(calib_path,tgt_name,filter)
print(len(all_image_names))
#image_name=all_image_names[1]
for image_name in all_image_names:
    img=photo_core.ESO_image(calib_path,image_name)
    comet_pic=photo_core.comet_frame(obs_code,search_name,img)
    comet_pic.find_comet()
    #comet_pic.show_comet()
    comet_pic.cutout_comet()
    all_frames.append(comet_pic.cutout)

#all_frames=np.array(all_frames)
#for img in all_frames:
    #plt.imshow(img,origin="lower",cmap="grey",norm=LogNorm())
    #plt.show()

sum=np.sum(all_frames,axis=0)
avg=np.median(all_frames,axis=0)
plt.imshow(avg,origin="lower",cmap="grey")
plt.axis("scaled")
plt.show()