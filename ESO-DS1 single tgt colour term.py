import photometry_core as photo_core
import calibration_tools as CT
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pathlib

import time


root_dir = pathlib.Path(__file__).resolve().parent
calib_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS"/"PROCESSED FRAMES")
all_fits_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS")

tgt_name="149P"
filter="R#642"
cat_filter="rmag"
pix_size=0.24  #size of a pixel in arcseconds
star_cell_size=5 #half width of the cell used for star detection around a PS1 entry

app_rad = 1.5 #Apperture radius is this multiplied by the average fwhm in the image
ann_in = 1.5
ann_out = 2

edge_pad=20 #introduces padding at the edge 

#reference img, for colour calib median and WCS
ref_name=photo_core.get_image_file(calib_path,tgt_name,filter)
ref_img=photo_core.ESO_image(calib_path,ref_name)

#wide field catalogue covering whole target area
wide_cat=photo_core.field_catalogue(ref_img,
                                    "10m",
                                    star_cell_size,
                                    pix_size,
                                    cat_filter)

all_image_names=photo_core.get_image_files(calib_path,tgt_name,filter)

pix_mask=CT.load_bad_pixel_mask(calib_path)
R_r=[]
gr=[]
ids=[]
grads=[]

mask=pix_mask
pixel_mask_data=np.array(pix_mask.data)

first=True

t=time.process_time()
count=0
for image_name in (all_image_names):

    plot_this=False
    pix_mask.data=pixel_mask_data
    img=photo_core.ESO_image(calib_path,image_name)
    subject_frame=photo_core.colour_calib_frame(img,
                                                pix_mask,
                                                edge_pad,
                                                wide_cat)
    
    subject_frame.star_fitter(star_cell_size,
                              fwhm_range=0.3)


    subject_frame.ap_phot(app_rad,
                        ann_in,
                        ann_out,
                        plot=plot_this)


    new_R_r,new_gr,id,grad = subject_frame.colour_compare()
    R_r.extend(new_R_r)
    gr.extend(new_gr)
    ids.extend(id)
    grads.append(grad)


term=np.mean(grads)
print(term)


print("Ellapsed Time: ",time.process_time()-t)
plt.scatter(gr,R_r)
plt.plot(gr,np.array(gr)*term,color="r")
plt.xlabel("g - r (Mag)")
plt.ylabel("R - r (Mag)")
for x,y,id in zip(gr,R_r,ids):
    if y>1:
        plt.annotate(str(id),[x,y])
plt.show()

