import photometry_core as photo_core
import calibration_tools as CT
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pathlib


root_dir = pathlib.Path(__file__).resolve().parent
calib_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS"/"PROCESSED FRAMES")
all_fits_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS")

tgt_name="149P"
filter="R#642"
cat_filter="rmag"
pix_size=0.24  #size of a pixel in arcseconds
star_cell_size=10 #half width of the cell used for star detection around a PS1 entry

app_rad = 6
ann_in = 1.5
ann_out = 2

edge_pad=2 * ann_out * app_rad #introduces padding at the edge scaled to aperture size

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
colour_median=0
R_r=[]
gr=[]

mask=pix_mask
pixel_mask_data=np.array(pix_mask.data)

first=True

for image_name in tqdm(all_image_names):
    pix_mask.data=pixel_mask_data
    img=photo_core.ESO_image(calib_path,image_name)
    subject_frame=photo_core.colour_calib_frame(img,
                                                pix_mask,
                                                edge_pad,
                                                wide_cat,
                                                colour_median=colour_median)
    
    subject_frame.star_fitter(star_cell_size)
    if first == True:
        subject_frame.ap_phot(app_rad,
                            ann_in,
                            ann_out,
                            plot=True)
        first=False
    else:
        subject_frame.ap_phot(app_rad,
                    ann_in,
                    ann_out)

    
    R_r,gr = subject_frame.colour_compare(R_r,gr)

plt.scatter(gr,R_r)
plt.xlabel("g - r (Mag)")
plt.ylabel("R - r (Mag)")
plt.show()