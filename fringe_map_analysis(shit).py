import data_organisation_tools as DOT
import calibration_tools as CT

import pathlib
from ccdproc import ImageFileCollection
import ccdproc as ccdp
from astropy.nddata import CCDData
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import astropy.units as u

rng = np.random.default_rng(seed=123456)

root_dir = pathlib.Path(__file__).resolve().parent
all_fits_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS")
calib_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS"/"PROCESSED FRAMES")

filter="i#705"

fringe_path = pathlib.Path(calib_path / str("FRINGE_MAP_"+filter+".fits"))
image_path = pathlib.Path(calib_path / "2009AU16_i#705_54859.15342363.fits") 

fringe=CCDData.read(fringe_path)
data_fringe = CCDData.read(image_path)

data_fringe.data[fringe.mask==1]=np.nan



lows=rng.integers(50,700,10)
xs=rng.integers(50,800,10)

length=75

mins=[]
maxs=[]

def compare_peaks(low,high,x):
    smooth_map,map=CT.analyse_fringe(low,high,x,fringe,iters=50,window_size=7)
    smooth_image,image=CT.analyse_fringe(low,high,x,data_fringe,iters=50,window_size=7)

    min=np.nanmin(smooth_image)
    max=np.nanmax(smooth_image)
    img_dif=max-min

    min=np.nanmin(smooth_map)
    max=np.nanmax(smooth_map)
    map_dif=max-min

    scale=img_dif/map_dif

    actual=smooth_image-(scale*smooth_map)

    #plt.plot(actual)
    #plt.show()
    #print(scale)



def scale_match(low,high,x,scales):
        difs=[]
        for scale in scales:
            smooth_map,map=CT.analyse_fringe(low,high,x,fringe,iters=50,window_size=7)
            smooth_image,image=CT.analyse_fringe(low,high,x,data_fringe,iters=50,window_size=7)
            actual=smooth_image-(scale*smooth_map)

            plt.plot(actual)

            min=np.min(actual)
            max=np.max(actual)

            dif=max-min
            
            difs.append(dif)

            #mins.append(np.nanmin(actual))
            #maxs.append(np.nanmax(actual))
        plt.show()
        return difs


#obs_pix=(fringe_scale*fringe)+actual_pix

for low,x in zip(lows,xs):
    smooth_map,map=CT.analyse_fringe(low,low+length,x,fringe,iters=50,window_size=7)
    #smooth_image,image=CT.analyse_fringe(low,low+length,x,data_fringe,iters=10,window_size=5)

    ratio=smooth_map

    #plt.plot(ratio)
    mins.append(np.nanmin(ratio))
    maxs.append(np.nanmax(ratio))
#plt.show()
print(np.nanmin(mins))

min=np.nanmin(mins)

fringe.data=fringe.data-min
fringe.data[fringe.data<0]=0

mins=[]
maxs=[]
for low,x in zip(lows,xs):
    smooth_map,map=CT.analyse_fringe(low,low+length,x,fringe,iters=20,window_size=5)
    smooth_image,image=CT.analyse_fringe(low,low+length,x,data_fringe,iters=20,window_size=5)

    img=smooth_image

    scale=1050
    map_scaled=smooth_map*(scale)
    offset=map_scaled[0]-img[0]


    plt.plot(img,label="IMAGE")
    plt.plot(map_scaled-offset,label="MAP")
    plt.legend()
    plt.show()
    mins.append(np.nanmin(ratio))
    maxs.append(np.nanmax(ratio))

#plt.show()

max_mean=np.nanmean(maxs)

arg=maxs<2*max_mean

lows=lows[arg]
xs=xs[arg]

scales=rng.random(400)
#scales=np.append(scales,0)


low=50
high=750
x=150
best_scales=[]
difs=[]

#xs=np.linspace(50,800,10,dtype="int")

for low,x in zip(lows,xs):
     high=low+length
     compare_peaks(low,high,x)



"""
for low,x in zip(lows,xs):
    high=low+length
    difs=[]
    difs=scale_match(low,high,x,scales)
    #plt.scatter(scales,difs,marker=".")
    best_scale=scales[difs==np.nanmin(difs)]
    best_scales.append(best_scale)
    print(best_scale)
#plt.show()

plt.plot(best_scales)
plt.show()
"""

scale=200
img_modded=data_fringe.data-(scale*fringe.data)
img_modded[fringe.mask==1] = np.nan

#plt.imshow(img_modded.data,origin="lower",cmap='gray',norm=LogNorm())
#plt.show()

#smooth_image,image=CT.analyse_fringe(100,250,400,data_fringe,iters=30,window_size=4)



#plt.plot(image/map)
#plt.plot(image)
#plt.plot(image)
#plt.plot(smooth_image)
#plt.show()

#plt.plot(map)
#plt.plot(smooth_map)
#plt.show()
