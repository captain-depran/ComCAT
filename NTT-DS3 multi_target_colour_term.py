import photometry_core as photo_core
import calibration_tools as CT
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pathlib

from astropy.stats import sigma_clipped_stats, SigmaClip,sigma_clip
from astropy.modeling import fitting,models


import time


root_dir = pathlib.Path(__file__).resolve().parent
calib_path = pathlib.Path(root_dir/"Data_set_3"/"block_2"/"ALL_FITS"/"PROCESSED FRAMES")
all_fits_path = pathlib.Path(root_dir/"Data_set_3"/"block_2"/"ALL_FITS")

pix_mask=CT.load_bad_pixel_mask(calib_path)


#tgt_names=["171P_Spahr","163P_NEAT","15P_Finlay","169P_NEAT","69P_Taylor"]
tgt_names=["137P_Shoemaker-Levy2",
"138P_Shoemaker-Levy7",
"15P_Finlay",
"163P_NEAT",
"169P_NEAT",
"171P_Spahr",
"62P_Tsuchinshan1",
"68P_Klemola",
"69P_Taylor",
"78P_Gehrels2",
"93P_Lovas1",
"P2001R6_LINEAR-Skiff",
"P2002S1_Skiff",
"146P",
"123P",
"P2002_S1",
"118P",
"36P",
"43P",
"47P",
"62P",
"94P"]

filter="R#608"
cat_filter="rmag"

colour_a="gmag"
colour_b="rmag"

plot=True

pix_size=0.35  #size of a pixel in arcseconds
star_cell_size=10 #half width of the cell used for star detection around a PS1 entry

app_rad = 1.5 #Apperture radius is this multiplied by the average fwhm in the image
ann_in = 1.5
ann_out = 2

edge_pad=20 #introduces padding at the edge 

process_time=4 #average number of seconds for system to process a file. Used for job time estimation

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
    ref_img=photo_core.ESO_image(calib_path,ref_name,pix_limit=50000)
    

    #wide field catalogue covering whole target area
    wide_cat=photo_core.field_catalogue(ref_img,
                                        "10m",
                                        star_cell_size,
                                        pix_size,
                                        cat_filter)

    all_image_names=photo_core.get_image_files(calib_path,tgt_name,filter)

    pixel_mask_data=np.copy(np.array(pix_mask.data))
    mask=pix_mask.copy()

    first=True
    count=0
    for image_name in all_image_names:
        if first:
            plot_this=False
            first=False
        else:
            plot_this=False
        mask.data=np.copy(pixel_mask_data)
        img=photo_core.ESO_image(calib_path,image_name,pix_limit=50000)
        
        img.update_zero(0)
        
        if img.solved==False:
            print("NOT SOLVED, SKIPPING")
            print("-"*10)
            continue
        
        subject_frame=photo_core.colour_calib_frame(img,
                                                    mask,
                                                    edge_pad,
                                                    wide_cat,
                                                    cat_filter)
        
        check = subject_frame.star_fitter(star_cell_size,
                                fwhm_range=1,
                                fwhm_plot=False)
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

term=np.median(grads)
#print(grads)
#print (colour_a + " - " + colour_b + " Colour Term: ",term)


all_colours=[]
all_difs=[]
all_errors=[]
all_offsets=[]

for frame in calib_frames:
    offset = frame.colour_zero(term)
    all_offsets.append(offset)
    all_colours.extend(frame.target_table["cat_colour"])
    all_difs.extend(frame.target_table["colour_dif"]+offset)
    all_errors.extend(frame.target_table["mag_error"])

    #if plot:
        #plt.errorbar(frame.target_table["cat_colour"],frame.target_table["colour_dif"]+offset,yerr=frame.target_table["mag_error"],fmt="k.")

    #plt.plot(np.sort(frame.target_table["cat_colour"]),(term*np.sort(frame.target_table["cat_colour"])),label=frame.frame.header["object"])
    #plt.errorbar(frame.frame_catalogue["rmag"],frame.target_table["mag"]-offset,yerr=frame.target_table["mag_error"],fmt="k.")

#plt.hist(grads,bins=50)
#plt.show()

print ("Per-frame Median term:")
print (colour_a + " - " + colour_b + " Colour Term: ",term)

print("OR")

print ("Per-frame Mean term:")
print (colour_a + " - " + colour_b + " Colour Term: ",np.mean(grads))

print("OR")

print("Total combined term")
fit = fitting.LinearLSQFitter(calc_uncertainties=True)
or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=3, sigma=3.0)
line_init = models.Linear1D()
fitted_line,mask = or_fit(line_init,np.array(all_colours),np.array(all_difs),weights=1/np.array(all_errors))
colour_term = (fitted_line(2)-fitted_line(1))

print (colour_a + " - " + colour_b + " Colour Term: ",colour_term, " +- ",fitted_line.slope.std)

grad_error = fitted_line.slope.std
int_error = fitted_line.intercept.std

if plot:
    xs=np.linspace(np.min(all_colours),np.max(all_colours),100)
    plt.plot(xs,fitted_line(xs))
    plt.fill_between(xs,((fitted_line.slope-grad_error) * xs)+(fitted_line.intercept-int_error),((fitted_line.slope+grad_error) * xs)+(fitted_line.intercept+int_error),alpha=0.2)
    #plt.fill_between(xs,((fitted_line.slope-grad_error) * xs)+(fitted_line.intercept+int_error),((fitted_line.slope+grad_error) * xs)+(fitted_line.intercept-int_error),alpha=0.2)
    plt.scatter(all_colours,np.ma.masked_array(all_difs, mask=mask),c="r")
    plt.errorbar(all_colours,all_difs,yerr=all_errors,fmt="k.")
    

    plt.legend()
    plt.show()

