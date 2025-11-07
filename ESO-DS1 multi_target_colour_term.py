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

pix_mask=CT.load_bad_pixel_mask(calib_path)

tgt_names=["P2004F3","149P","P113","94P","93P","74P"]

filter="R#642"
cat_filter="rmag"
pix_size=0.24  #size of a pixel in arcseconds
star_cell_size=5 #half width of the cell used for star detection around a PS1 entry

app_rad = 1.5 #Apperture radius is this multiplied by the average fwhm in the image
ann_in = 1.5
ann_out = 2

edge_pad=20 #introduces padding at the edge 

process_time=2 #average number of seconds for system to process a file. Used for job time estimation

file_count=photo_core.check_job_size(tgt_names,[filter],calib_path)
print("Estimated Runtime: ",file_count*process_time," Seconds | ",file_count," Comet Images")

calib_frames=[]
grads=[]

for tgt_name in tgt_names:
    print("PROCESSING COMET: ",tgt_name)
    #reference img, for WCS
    ref_name=photo_core.get_image_file(calib_path,tgt_name,filter)
    if ref_name==9999:
        print("COMET ",tgt_name," NOT FOUND! SKIPPING...")
        continue
    ref_img=photo_core.ESO_image(calib_path,ref_name)

    #wide field catalogue covering whole target area
    wide_cat=photo_core.field_catalogue(ref_img,
                                        "10m",
                                        star_cell_size,
                                        pix_size,
                                        cat_filter)

    all_image_names=photo_core.get_image_files(calib_path,tgt_name,filter)

    R_r=[]
    gr=[]
    ids=[]
    #calib_frames=[]

    mask=pix_mask
    pixel_mask_data=np.array(pix_mask.data)

    first=True
    count=0
    for image_name in tqdm(all_image_names):
        plot_this=False
        pix_mask.data=pixel_mask_data
        img=photo_core.ESO_image(calib_path,image_name)
        if img.solved==False:
            continue
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


        new_R_r,new_gr,id,grad,filtered_R_r,mag_errors = subject_frame.colour_grad_fit()
        #print("Filtered Points: ",np.sum(filtered_R_r.mask))
        #R_r.extend(new_R_r)
        #gr.extend(new_gr)
        #ids.extend(id)
        grads.append(grad)
        calib_frames.append(subject_frame)



term=np.mean(grads)
print(grads)
print (term)

for frame in calib_frames:
    offset = frame.colour_zero(term)
    plt.errorbar(frame.target_table["g-r"],frame.target_table["R-r"]-offset,yerr=frame.target_table["mag_error"],fmt="k.")
    #plt.errorbar(frame.frame_catalogue["rmag"],frame.target_table["mag"]-offset,yerr=frame.target_table["mag_error"],fmt="k.")
#plt.plot(calib_frames[0].frame_catalogue["rmag"],(term*calib_frames[0].frame_catalogue["rmag"])-offset)
plt.show()


