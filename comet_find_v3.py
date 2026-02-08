#Using random offsets at the start of each run, cycle through the images with the fixed offset, and evaluate some score

import photometry_core as photo_core
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from tqdm import tqdm
import pathlib

rng = np.random.default_rng()

root_dir = pathlib.Path(__file__).resolve().parent
calib_path = pathlib.Path(root_dir/"Data_set_3"/"block_1"/"ALL_FITS"/"PROCESSED FRAMES")
all_fits_path = pathlib.Path(root_dir/"Data_set_3"/"block_1"/"ALL_FITS")

filter_1="V#606"
filter_2="V#606"


#name="137P_Shoemaker-Levy2"
name="146P"
jpl_name="137P"
eph=0

sol_col=0.45

colour_term_1 = -0.4811791650583872
colour_term_2 = -0.17286169884964958
#shift_col= 0.45 #Solar colour, starting point
#shift_col= 0.4085325  

comet = photo_core.study_comet(name,
                                   jpl_name,
                                   filter_2,
                                   calib_path,
                                   eph_code=eph,
                                   plot_stack=True,
                                   comet_pixel_max=10000,
                                   show_frames=False,
                                   cutout_size=50,
                                   pos_offset = [-1,0])

#comet.find_offset(samples=5000,strike_lim=5,search_span=30)
#exit()
comet.get_measures(app_rad=5)
plt.errorbar(comet.t,comet.mags,yerr=comet.errors,fmt="k.")
print(np.median(comet.mags))

plt.ylim(19,25)
plt.show()


"""
#ORIGINAL TEST

image_names=photo_core.get_image_files(calib_path,name,filter_2)[0:]
print("Images of Comet "+name+": ",len(image_names))
if len(image_names)==0:
    exit()

comet_pics=[]
zeros = []
err=[]
times = []
cutouts = []
measure=[]

def assess_img(image):
    global shift
    mag,snr,mag_error = image.comet_ap_phot(3,1.1,1.5,shift=shift)
    return snr,mag


first = True
for image_name in (image_names):
    img=photo_core.ESO_image(calib_path,image_name)
    if img.solved==False:
        print("ERROR! NOT SOLVED!")
        continue
    comet_pic=photo_core.comet_frame(809,jpl_name,img,comet_pixel_max=5000)
   
    good_find = comet_pic.find_comet(eph_code=eph)
    
    if good_find == False:
        continue

    comet_pic.comet_pix_location[0]+=-19
    comet_pic.comet_pix_location[1]+=-17

    #plt.plot(comet_pic.img.data[int(comet_pic.comet_pix_location[1]),int(comet_pic.comet_pix_location[0]-20):int(comet_pic.comet_pix_location[0]+20)])
    #plt.show()

    comet_pic.cutout_comet(cutout_size=50)
    
    comet_pics.append(comet_pic)
    cutouts.append(comet_pic.cutout)
    
    mag,snr,mag_error = comet_pic.comet_ap_phot(4,1.5,2)
    mag=mag+comet_pic.img.get_zero()
    measure.append(mag)
    err.append(mag_error)
    #print(snr)
    zeros.append(comet_pic.img.get_zero())
    times.append(comet_pic.date)
    #comet_pic.cutout[comet_pic.cutout>3000] = 0
    #plt.imshow(comet_pic.cutout,norm=LogNorm(),origin="lower")
    #plt.show()

#cutout_stack = photo_core.composite_comet(cutouts)
#plt.imshow(cutout_stack,norm=LogNorm(),origin="lower")
#plt.errorbar(times,measure,yerr=err,fmt=".")
#plt.ylim(22,25)
bootstraps=[]
for k in range(0,3000):
    sample=rng.choice(measure,np.shape(measure))
    bootstraps.append(np.mean(sample))
plt.hist(bootstraps,bins=50)
print(np.mean(bootstraps))
print(np.median(measure))
print(3*np.std(bootstraps))

print(np.std(measure))
plt.show()
exit()

    

unmodded_pics = comet_pics.copy()

samples = 3000
span = 30
strike_lim = 0

x_offsets = rng.integers(np.ones(samples)*-1*span,np.ones(samples)*span)
y_offsets = rng.integers(np.ones(samples)*-1*span,np.ones(samples)*span)



snr_stddev = []
mags_stddev = []
all_mags = []

dxs=[]
dys=[]
rank=[]

for dx,dy in tqdm(zip(x_offsets,y_offsets)):
    shift = [dx,dy]
    strikes = 0
    snrs=[]
    mags=[]
    for i in (range(0,len(zeros))):
        snr,mag = assess_img(unmodded_pics[i])
        mag += zeros[i]
        if mag > 30 and strikes <=strike_lim:
            strikes+=1
            snrs.append(snr)
            mags.append(mag)
        elif mag > 30 and strikes > strike_lim:
            mags=[]
            snrs=[]
            break
        else:
            snrs.append(snr)
            mags.append(mag)
    if len(mags) == 0:
        continue
    else: 
        all_mags.append(mags)
        snr_stddev.append((np.std(snrs)))
        mag_med = np.median(mags)
        mags_stddev.append(np.std(mags-mag_med))
        dxs.append(dx)
        dys.append(dy)
        mags=np.array(mags)
        rank.append(len(mags[mags < 30]))

print(np.nanmin(snr_stddev))
print(np.nanmin(mags_stddev))

print(len(mags_stddev))
print("-"*10)

snr_min_i = np.argmin(snr_stddev)
mags_min_i = np.argmin(mags_stddev)
rank_max_i = np.argmax(rank)


mags_1=np.array(all_mags)[snr_min_i]
mags_2=np.array(all_mags)[mags_min_i]
mags_3=np.array(all_mags)[rank_max_i]

print("USE THIS ONE:")
print(np.min(mags_stddev))
print(np.median(mags_2))
dx = np.array(dxs)[mags_min_i]
dy = np.array(dys)[mags_min_i]
print([dx,dy])

print("-"*10)


dx = np.array(dxs)[snr_min_i]
dy = np.array(dys)[snr_min_i]
print([dx,dy])

dx = np.array(dxs)[rank_max_i]
dy = np.array(dys)[rank_max_i]
print([dx,dy])



plt.scatter(dxs,dys,c=rank)
plt.colorbar()
plt.show()


#plt.scatter(times,mags_1)
plt.scatter(times,mags_2)
#plt.scatter(times,mags_3)
plt.ylim(19,26)

plt.show()
  
"""