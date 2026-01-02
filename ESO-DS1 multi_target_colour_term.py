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

tgt_names=["P2004F3","94P","93P","74P","2009AU16","P2005R2","29P","50P","P113","48P","149P"]

filter="R#642"
cat_filter="rmag"

colour_a="gmag"
colour_b="rmag"

plot=True

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
    print("-"*10)
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

    mask=pix_mask
    pixel_mask_data=np.array(pix_mask.data)

    first=True
    count=0
    for image_name in all_image_names:
        plot_this=False
        pix_mask.data=pixel_mask_data
        img=photo_core.ESO_image(calib_path,image_name)
        img.update_zero(0)
        #print(image_name)
        if img.solved==False:
            print("NOT SOLVED, SKIPPING")
            print("-"*10)
            continue
        
        subject_frame=photo_core.colour_calib_frame(img,
                                                    pix_mask,
                                                    edge_pad,
                                                    wide_cat,
                                                    cat_filter)
        
        check = subject_frame.star_fitter(star_cell_size,
                                fwhm_range=0.3)
        if check == 1:
            continue

        subject_frame.ap_phot(app_rad,
                            ann_in,
                            ann_out,
                            plot=plot_this)

        if subject_frame.no_stars==True:
            print("NO VALID STARS, SKIPPING")
            print("-"*10)
            continue
        new_R_r,new_gr,id,grad,filtered_R_r,mag_errors = subject_frame.colour_grad_fit(colour_a,colour_b)
        print("CALIBRATION STARS EXTRACTED!")
        print("-"*10)


        grads.append(grad)
        calib_frames.append(subject_frame)
    

print(len(calib_frames))

term=np.mean(grads)
#print(grads)
print (colour_a + " - " + colour_b + " Colour Term: ",term)


for frame in calib_frames:
    offset = frame.colour_zero(term)
    if plot:
        plt.errorbar(frame.target_table["cat_colour"],frame.target_table["colour_dif"]+offset,yerr=frame.target_table["mag_error"],fmt="k.")

    #plt.plot(np.sort(frame.target_table["cat_colour"]),(term*np.sort(frame.target_table["cat_colour"])),label=frame.frame.header["object"])
    #plt.errorbar(frame.frame_catalogue["rmag"],frame.target_table["mag"]-offset,yerr=frame.target_table["mag_error"],fmt="k.")
if plot:
    plt.legend()
    plt.show()

