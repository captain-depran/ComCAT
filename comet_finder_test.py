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

tgt_name="94P"
search_name="94P"
eph_code=90000920
filter="R#642"

obs_code=809

all_frames=[]
comet_pics=[]


all_image_names=photo_core.get_image_files(calib_path,tgt_name,filter)
print("Images of Comet "+tgt_name+": ",len(all_image_names))
if len(all_image_names)==0:
    exit()
#image_name=all_image_names[1]
for image_name in all_image_names:
    img=photo_core.ESO_image(calib_path,image_name)
    if img.solved==False:
        continue
    elif img.get_zero() == 0:
        continue
    comet_pic=photo_core.comet_frame(obs_code,search_name,img)
    comet_pic.find_comet(eph_code=eph_code)
    #comet_pic.show_comet()
    comet_pic.cutout_comet()
    comet_pics.append(comet_pic)
    all_frames.append(comet_pic.cutout)
    

all_frames=np.array(all_frames)
print(len(all_frames))
if len(all_frames)==0:
    print("ERROR: No Solved Images")
    exit()

cutout_stack = photo_core.composite_comet(all_frames)
comet_error = photo_core.lock_comet(cutout_stack,fwhm=6)
print (comet_error)
photo_core.mark_target(comet_error+((len(cutout_stack[0])-1)/2),cutout_stack)

all_frames=[]
mags=[]
for pic in comet_pics:
    pic.offset=comet_error
    pic.apply_correction()
    pic.cutout_comet(cutout_size=30)

    pic.refine_lock() #Refine the lock on the comet by refitting a gaussian per image now we know we are *basically* on top of the comet

    all_frames.append(pic.cutout)
    print(pic.img.image_path)
    mags.append(pic.comet_ap_phot(5,1.2,1.5)+pic.img.get_zero())
    #pic.show_comet()
    #photo_core.mark_target([pic.pad,pic.pad],pic.cutout)
pic.show_comet()

cutout_stack = photo_core.composite_comet(all_frames)
comet_error = photo_core.lock_comet(cutout_stack)
photo_core.mark_target(comet_error+((len(cutout_stack[0])-1)/2),cutout_stack)


plt.plot(mags)
plt.show()