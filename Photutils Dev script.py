import data_organisation_tools as DOT
import calibration_tools as CT

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

import pathlib

from ccdproc import ImageFileCollection
import ccdproc as ccdp

from photutils.detection import DAOStarFinder
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

root_dir = pathlib.Path(__file__).resolve().parent
calib_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS"/"PROCESSED FRAMES")
all_fits_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS")
vizier = Vizier() # this instantiates Vizier with its default parameters
Vizier.clear_cache()


tgt_name="93P"
filter="V#641"

hdu_path=pathlib.Path(calib_path/get_image_file(calib_path,tgt_name,filter))
with fits.open(hdu_path) as img:
    hdu=img[0]
    data=img[0].data
    img_wcs=WCS(img[0].header)




mean, median, std = sig(data, sigma=3.0)

daofind = DAOStarFinder(fwhm=7.0, threshold=6.*std)
sources = daofind(data - median)


for col in sources.colnames:
    if col not in ('id', 'npix'):
        sources[col].info.format = '%.2f'  # for consistent table output
#print (sources)


positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
apertures = CircularAperture(positions, r=4.0)



catalogue = vizier.query_region(img_wcs.pixel_to_world(416,416),width="4m",catalog="II/349/ps1")[0]
#catalogue = vizier.query_region(img_wcs.pixel_to_world(416,416),width="4m",catalog="I/355/gaiadr3")[0]

ps1_pos=np.transpose((catalogue["RAJ2000"],catalogue["DEJ2000"]))
ps1_pix=np.transpose(img_wcs.world_to_pixel(coords.SkyCoord(ra=ps1_pos[:,0],dec=ps1_pos[:,1],unit="deg",frame="fk5")))

ps1_apps=CircularAperture(ps1_pix,r=3)

#norm = ImageNormalize(stretch=SinhStretch(),clip=False)

"""
plt.imshow(data-400, cmap='grey', origin='lower', norm=LogNorm())
apertures.plot(color='blue', lw=1.5, alpha=0.5)
ps1_apps.plot(color='green',lw=1.5,alpha=0.5)
plt.xlim(0,824)
plt.ylim(0,824)
plt.show()
"""

plt.plot(data[:,256])
plt.show()

