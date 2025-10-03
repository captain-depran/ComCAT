import data_organisation_tools as DOT
import calibration_tools as CT

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

import pathlib

from ccdproc import ImageFileCollection
import ccdproc as ccdp

from photutils.detection import IRAFStarFinder,find_peaks
from photutils.aperture import CircularAperture

from astropy.visualization import SqrtStretch,SinhStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.nddata import CCDData
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats as sig
import astropy.units as u
import astropy.coordinates as coords

from astroquery.vizier import Vizier

def get_image_data(calib_path,tgt,filter):
    file=get_image_file(calib_path,tgt,filter)
    with fits.open(calib_path/file) as img:
        CT.show_image(img[0].data)
        data=img[0].data
    return data

def get_image_file(calib_path,tgt,filter):
    criteria={'object' : tgt_name, "ESO INS FILT1 NAME".lower():filter}
    science=ImageFileCollection(calib_path,keywords='*',glob_include=tgt_name+"_"+filter+"*")
    science_files=science.files_filtered(**criteria)
    return science_files[0]

def bound_legal(value,shift,hi_limit,lw_limit):
    value=value+shift
    if value > hi_limit:
        value=hi_limit
    elif value < lw_limit:
        value=lw_limit
    return int(value)

def local_peak(data,mask,cat_pos,width,edge):
    """
    Function to take a catalogue position, and find the local peak within a box of 2 x 'width' centered on that position
    """
    low_x=bound_legal(cat_pos[0],-1*width,edge,0)
    high_x=bound_legal(cat_pos[0],width,edge,0)
    low_y=bound_legal(cat_pos[1],-1*width,edge,0)
    high_y=bound_legal(cat_pos[1],width,edge,0)
    

    clip_data=data[low_y:high_y,low_x:high_x]
    mask=mask[low_y:high_y,low_x:high_x]

    mean, median, std = sig(clip_data, sigma=3.0)
    clip_data[mask]=median
    star=find_peaks(clip_data,threshold=1.*std ,box_size=len(clip_data[0]),npeaks=1)
    #star = IRAFStarFinder(fwhm=7.0,threshold=1.*std,xycoords=[[(high_x-low_x)/2,(high_y-low_y)/2]])
    #star = star(clip_data-median,mask[low_y:high_y,low_x:high_x])
    print(star)
    print(median)
    star_pos=np.transpose([star['x_peak'], star['y_peak']])[0]

    apertures = CircularAperture(star_pos,r=3)
    ps1_apps = CircularAperture([(high_x-low_x)/2,(high_y-low_y)/2],r=4)
    
    plt.imshow(clip_data,cmap='grey', origin='lower', norm=LogNorm())
    apertures.plot(color='blue', lw=1.5, alpha=0.5)
    ps1_apps.plot(color='green',lw=1.5,alpha=0.5)
    plt.show()
    

    return (star_pos-[(high_x-low_x)/2,(high_y-low_y)/2])+cat_pos





root_dir = pathlib.Path(__file__).resolve().parent
calib_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS"/"PROCESSED FRAMES")
all_fits_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS")
vizier = Vizier() # this instantiates Vizier with its default parameters
Vizier.clear_cache()


tgt_name="93P"
filter="V#641"

mask=CT.load_bad_pixel_mask(calib_path)
hdu_path=pathlib.Path(calib_path/get_image_file(calib_path,tgt_name,filter))
with fits.open(hdu_path) as img:
    data=img[0].data
    hdu=img[0]
    img_wcs=WCS(img[0].header)

ccd_data=CCDData.read(calib_path/get_image_file(calib_path,tgt_name,filter))

#plt.imshow(mask,origin="lower")
#plt.show()

field_span=len(data[:,1])



mean, median, std = sig(data, sigma=3.0)
data[mask]=median
print("Searching Catalogue")
catalogue = vizier.query_region(img_wcs.pixel_to_world(416,416),width="4m",catalog="II/349/ps1")[0]
#catalogue = vizier.query_region(img_wcs.pixel_to_world(416,416),width="4m",catalog="I/355/gaiadr3")[0]
print("Catalogue Imported")

ps1_pos=np.transpose((catalogue["RAJ2000"],catalogue["DEJ2000"]))
big_ps1_pix=np.transpose(img_wcs.world_to_pixel(coords.SkyCoord(ra=ps1_pos[:,0],dec=ps1_pos[:,1],unit="deg",frame="fk5")))

ps1_pix=[]
for n in range(0,len(big_ps1_pix)):
    if big_ps1_pix[n,0] < 0 or big_ps1_pix[n,0] > field_span or big_ps1_pix[n,1] < 0 or big_ps1_pix[n,1] > field_span:
        pass
    else:
        ps1_pix.append(big_ps1_pix[n,:])
ps1_pix=np.array(ps1_pix)

ps1_apps=CircularAperture(ps1_pix,r=3)

"""
IRAFfind = IRAFStarFinder(fwhm=7.0, sigma_radius=1.3, threshold=5.*std,sharplo=0.3,sharphi=4)
sources = IRAFfind(data-median)
"""
positions=[]
for coord in ps1_pix:
    star_pos=local_peak(data,mask,coord,10,field_span)
    positions.append(star_pos)
    
positions=np.array(positions)
#positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
apertures = CircularAperture(positions, r=4.0)

"""
for col in sources.colnames:
    if col not in ('id', 'npix'):
        sources[col].info.format = '%.2f'  # for consistent table output
#print (sources)
"""
#norm = ImageNormalize(stretch=SinhStretch(),clip=False)


plt.imshow(data, cmap='grey', origin='lower', norm=LogNorm())
apertures.plot(color='blue', lw=1.5, alpha=0.5)
ps1_apps.plot(color='green',lw=1.5,alpha=0.5)
plt.xlim(0,824)
plt.ylim(0,824)
plt.show()

"""
plt.plot(data[:,256])
plt.show()

"""