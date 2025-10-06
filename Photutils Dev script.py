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
from photutils.background import Background2D, MedianBackground
from photutils.psf import fit_2dgaussian as fit_gauss

from astropy.visualization import SqrtStretch,SinhStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.nddata import CCDData
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats as sig
import astropy.units as u
import astropy.coordinates as coords

from astroquery.vizier import Vizier



from astropy.stats import sigma_clipped_stats, SigmaClip





def get_image_data(calib_path,tgt,filter):
    file=get_image_file(calib_path,tgt,filter)
    with fits.open(calib_path/file) as img:
        CT.show_image(img[0].data)
        data=img[0].data
    return data

def get_image_file(calib_path,tgt,filter):
    criteria={'object' : tgt, "ESO INS FILT1 NAME".lower():filter}
    science=ImageFileCollection(calib_path,keywords='*',glob_include=tgt+"_"+filter+"*")
    science_files=science.files_filtered(**criteria)
    return science_files[1]

def catalogue_filter(catalogue,min_sep):
    min_sep=min_sep/60/60
    removed_ids=[]
    filter="imag"

    catalogue[filter].fill_value = 99

    for n,entry in zip(range(0,len(catalogue.values())),catalogue.iterrows("RAJ2000","DEJ2000",filter,"ID")):
        compare_cat=catalogue.copy()
        catalogue[filter].fill_value = 99
        compare_cat.keep_columns(["RAJ2000","DEJ2000",filter,"ID"])
        compare_cat.remove_row(n)
        compare_cat["RAJ2000"]-=entry[0]
        compare_cat["DEJ2000"]-=entry[1]
        distance=np.sqrt((compare_cat["RAJ2000"].value**2)+(compare_cat["DEJ2000"].value**2))
        compare_cat.add_column(distance,name="dist")
        compare_cat.sort("dist")

        for star in compare_cat.filled():
            if star[filter]==99:
                removed_ids.append(star["ID"])

        compare_cat = compare_cat[compare_cat["dist"] < min_sep]

        for star in compare_cat.filled():
            if star[filter] > entry[2]:
                removed_ids.append(star["ID"])
            elif star[filter] < entry[2]:
                removed_ids.append(entry[3])
                break
            elif star[filter]==entry[2]:
                print("UNEXPECTED CASE: IDENTICAL MAGNTIUDE")
    removed_ids=np.unique(removed_ids)
    exclude_cat=catalogue[np.isin(catalogue["ID"],removed_ids)]
    catalogue=catalogue[np.isin(catalogue["ID"],removed_ids,invert=True)]
    return catalogue,exclude_cat
        

def catalogue_pos_parse(catalogue,wcs,field_span):
    ids=[]
    ps1_pos=np.transpose((catalogue["RAJ2000"],catalogue["DEJ2000"]))
    big_ps1_pix=np.transpose(wcs.world_to_pixel(coords.SkyCoord(ra=ps1_pos[:,0],dec=ps1_pos[:,1],unit="deg",frame="fk5")))
    ps1_pix=[]
    for id,n in zip(catalogue["ID"].value,range(0,len(big_ps1_pix))):
        if big_ps1_pix[n,0] < 0 or big_ps1_pix[n,0] > field_span or big_ps1_pix[n,1] < 0 or big_ps1_pix[n,1] > field_span:
            pass
        else:
            ps1_pix.append(big_ps1_pix[n,:])
            ids.append(id)
    return np.array(ps1_pix),np.array(ids)
    


def bound_legal(value,shift,hi_limit,lw_limit):
    """
    Function to shift a value up or down (shift is sign dependent) and clip it to a given upper or lower bound. Primaily for selecting a 2d array section
    """
    value=value+shift
    if value > hi_limit:
        value=hi_limit
    elif value < lw_limit:
        value=lw_limit
    return int(value)

def local_peak(data,cat_pos,width,edge):
    """
    Function to take a catalogue position, and find the local peak within a box of 2 x 'width' centered on that position
    """
    low_x=bound_legal(cat_pos[0],-1*width,edge,0)
    high_x=bound_legal(cat_pos[0],width,edge,0)
    low_y=bound_legal(cat_pos[1],-1*width,edge,0)
    high_y=bound_legal(cat_pos[1],width,edge,0)
    
    if (high_x-low_x)%2 == 0:
        high_x+=1
    if (high_y-low_y)%2 == 0:
        high_y+=1


    center=[(high_x-low_x)/2,(high_y-low_y)/2]


    clip_data=data[low_y:high_y,low_x:high_x]

    result=fit_gauss(clip_data,fwhm=7.0,xypos=center).results
    star_pos=np.array([result["x_fit"][0],result["y_fit"][0]])
    """
    print(star_pos)

    apertures = CircularAperture(star_pos,r=3)
    ps1_apps = CircularAperture(center,r=4)
    
    plt.imshow(clip_data,cmap='grey', origin='lower', norm=LogNorm())
    apertures.plot(color='blue', lw=1.5, alpha=0.5)
    ps1_apps.plot(color='green',lw=1.5,alpha=0.5)
    plt.show()
    """


    return (star_pos-center)+cat_pos






root_dir = pathlib.Path(__file__).resolve().parent
calib_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS"/"PROCESSED FRAMES")
all_fits_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS")
vizier = Vizier() # this instantiates Vizier with its default parameters
Vizier.clear_cache()


tgt_name="31P"
filter="R#642"
pix_size=0.24  #size of a pixel in arcseconds
star_cell_size=10 #half width of the cell used for star detection around a PS1 entry

#Load pixel mask and target image
mask=CT.load_bad_pixel_mask(calib_path)  
hdu_path=pathlib.Path(calib_path/get_image_file(calib_path,tgt_name,filter))
with fits.open(hdu_path) as img:
    data=img[0].data
    hdu=img[0]
    img_wcs=WCS(img[0].header)
ccd_data=CCDData.read(calib_path/get_image_file(calib_path,tgt_name,filter))


field_span=len(data[:,1]) #calulate the width of the image in pixels


#Background estimation, using Photutils
mean, median, std = sig(data, sigma=3.0)
data[mask]=median

sigma_clip = SigmaClip(sigma=3.0)
bkg_estimator = MedianBackground()
bkg = Background2D(data, (20, 20), filter_size=(3, 3),
                    sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
data=data-bkg.background


#Catalogue query for PS1 stars in the image
print("Searching Catalogue")
catalogue = vizier.query_region(img_wcs.pixel_to_world(416,416),width="4m",catalog="II/349/ps1")[0]
catalogue.add_column(np.linspace(0,len(catalogue)-1,num=len(catalogue)),name="ID")
print("Catalogue Imported")


#Scan the catalogue for incidents where two entries would have two overlaping star-finder cells, and select the brightest
catalogue,exclude_cat=catalogue_filter(catalogue,2*star_cell_size*pix_size)  

#Generate pixel coordinates for the included and excluded catalogue entries
ps1_pix,in_ids=catalogue_pos_parse(catalogue,img_wcs,field_span)
exc_pix,ex_ids=catalogue_pos_parse(exclude_cat,img_wcs,field_span)
catalogue=catalogue[np.isin(catalogue["ID"],in_ids)]

#Generate circular apertures around each catalgoue entry
ps1_apps=CircularAperture(ps1_pix,r=3)
exc_apps=CircularAperture(exc_pix,r=3)


positions=[]

for coord in ps1_pix:
    star_pos=local_peak(data,coord,star_cell_size,field_span)
    positions.append(star_pos)


    
positions=np.array(positions)
apertures = CircularAperture(positions, r=4.0)


#norm = ImageNormalize(stretch=SinhStretch(),clip=False)


plt.imshow(data+bkg.background, cmap='grey', origin='lower', norm=LogNorm())
apertures.plot(color='blue', lw=1.5, alpha=0.5)
ps1_apps.plot(color='green',lw=1.5,alpha=0.5)
exc_apps.plot(color='red',lw=1.5,alpha=0.5)
plt.xlim(0,824)
plt.ylim(0,824)
plt.show()

