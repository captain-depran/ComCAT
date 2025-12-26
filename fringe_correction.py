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

fringe_coords=np.loadtxt("fringe_points.txt",delimiter=",")
print(fringe_coords)

root_dir = pathlib.Path(__file__).resolve().parent
all_fits_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS")
calib_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS"/"PROCESSED FRAMES")

filter="i#705"

fringe_path = pathlib.Path(calib_path / str("FRINGE_MAP_"+filter+".fits"))
image_path = pathlib.Path(calib_path / "93P_i#705_54859.37043187.fits") 

fringe = CCDData.read(fringe_path)
data = CCDData.read(image_path)
ratios=[]

fringe.data=fringe.data-np.nanmin(fringe.data)

for pairs in fringe_coords:
    x1=int(pairs[1])
    y1=int(pairs[0])
    x2=int(pairs[3])
    y2=int(pairs[2])

    

    map_dif=np.abs(fringe.data[x2,y2]-fringe.data[x1,y1])
    frame_dif=np.abs(data.data[x2,y2]-data.data[x1,y1])

    ratios.append(frame_dif/map_dif)

print(ratios)
scale=np.median(ratios)
print(scale)

med=np.nanmedian(fringe.data)
fringe.data[fringe.mask==1] = med
reduced_img=data.data-(fringe.data*scale)
reduced_img[data.mask==1]=data.data[data.mask==1]

plt.imshow((fringe.data*scale),norm=LogNorm(),origin="lower",cmap="grey")
plt.show()

image_path = pathlib.Path(calib_path / str("fringe_begone_"+image_path.name)) 

data.data=reduced_img
data.write(image_path,overwrite=True)