import data_organisation_tools as DOT
import calibration_tools as CT

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from tqdm import tqdm
import pathlib

from ccdproc import ImageFileCollection
import ccdproc as ccdp

from photutils.aperture import CircularAperture,aperture_photometry,CircularAnnulus,ApertureStats
from photutils.background import Background2D, MedianBackground
from photutils.psf import fit_2dgaussian as fit_gauss

from astropy.table import Table
from astropy.nddata import CCDData
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats as sig
import astropy.units as u
import astropy.coordinates as coords

from astroquery.vizier import Vizier



from astropy.stats import sigma_clipped_stats, SigmaClip

class Not_plate_solved(Exception):
    pass


def mag_error(count_error,count,exptime):
    err = 2.5/np.log(10) * (count_error/count)
    return np.abs(err)


def get_image_data(calib_path,tgt,filter):
    file=get_image_file(calib_path,tgt,filter)
    with fits.open(calib_path/file) as img:
        CT.show_image(img[0].data)
        data=img[0].data
    return data

def get_image_files(calib_path,tgt,filter):
    criteria={'object' : tgt, "ESO INS FILT1 NAME".lower():filter}
    science=ImageFileCollection(calib_path,keywords='*',glob_include=tgt+"_"+filter+"*")
    science_files=science.files_filtered(**criteria)
    return science_files

def get_image_file(calib_path,tgt,filter):
    criteria={'object' : tgt, "ESO INS FILT1 NAME".lower():filter}
    science=ImageFileCollection(calib_path,keywords='*',glob_include=tgt+"_"+filter+"*")
    science_files=science.files_filtered(**criteria)
    return science_files[0]

def catalogue_filter(catalogue,min_sep_deg):
    min_sep=min_sep_deg/60/60
    removed_ids=[]
    filter="rmag"

    catalogue[filter].fill_value = 99

    for n,entry in zip(range(0,len(catalogue.values())),catalogue.iterrows("RA_ICRS","DE_ICRS",filter,"ID")):
        compare_cat=catalogue.copy()
        catalogue[filter].fill_value = 99
        compare_cat.keep_columns(["RA_ICRS","DE_ICRS",filter,"ID"])
        compare_cat.remove_row(n)
        compare_cat["RA_ICRS"]-=entry[0]
        compare_cat["DE_ICRS"]-=entry[1]
        distance=np.sqrt((compare_cat["RA_ICRS"].value**2)+(compare_cat["DE_ICRS"].value**2))
        compare_cat.add_column(distance,name="dist")
        compare_cat.sort("dist")
        if entry[2]==99:
            removed_ids.append(entry[3])
        else:
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
        

def catalogue_pos_parse(catalogue,wcs,field_span,edge_pad):
    ids=[]
    field_span-=edge_pad

    ps1_pos=np.transpose((catalogue["RA_ICRS"],catalogue["DE_ICRS"]))
    big_ps1_pix=np.transpose(wcs.world_to_pixel(coords.SkyCoord(ra=ps1_pos[:,0],dec=ps1_pos[:,1],unit="deg",frame="fk5")))
    ps1_pix=[]
    for id,n in zip(catalogue["ID"].value,range(0,len(big_ps1_pix))):
        if big_ps1_pix[n,0] < edge_pad or big_ps1_pix[n,0] > field_span or big_ps1_pix[n,1] < edge_pad or big_ps1_pix[n,1] > field_span:
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
    if high_y>edge:
        high_y=edge
        low_y-=1
    if high_x>edge:
        high_x=edge
        low_x-=1
    

    center=[(high_x-low_x)/2,(high_y-low_y)/2]



    clip_data=data[low_y:high_y,low_x:high_x]

    result=fit_gauss(clip_data,fix_fwhm=False,xypos=center).results
    star_pos=np.array([result["x_fit"][0],result["y_fit"][0]])
    fwhm_fit=result["fwhm_fit"][0]
    """
    print(star_pos)

    apertures = CircularAperture(star_pos,r=3)
    ps1_apps = CircularAperture(center,r=4)
    
    plt.imshow(clip_data,cmap='grey', origin='lower', norm=LogNorm())
    apertures.plot(color='blue', lw=1.5, alpha=0.5)
    ps1_apps.plot(color='green',lw=1.5,alpha=0.5)
    plt.show()
    """
    return ((star_pos-center)+cat_pos),fwhm_fit


def cosmic_ray_reduce(ccd_image,read_noise):
    new_ccd=ccdp.cosmicray_lacosmic(ccd_image,readnoise=read_noise,sigclip=10)
    return new_ccd



root_dir = pathlib.Path(__file__).resolve().parent
calib_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS"/"PROCESSED FRAMES")
all_fits_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS")
vizier = Vizier(row_limit=-1) # this instantiates Vizier with its default parameters
Vizier.clear_cache()


tgt_name="149P"
filter="R#642"
pix_size=0.24  #size of a pixel in arcseconds
star_cell_size=10 #half width of the cell used for star detection around a PS1 entry

app_rad = 6
ann_in = 1.5
ann_out = 2

edge_pad=2 * ann_out * app_rad #introduces padding at the edge scaled to aperture size

#Load pixel mask and target image
R_r=[]
gr=[]
R_r_err=[]

first_pass=True
pix_mask=CT.load_bad_pixel_mask(calib_path)
mask=pix_mask
pixel_mask_data=np.array(pix_mask.data)
for file in tqdm(get_image_files(calib_path,tgt_name,filter)):
    mask.data=pixel_mask_data
    hdu_path=pathlib.Path(calib_path/file)
    with fits.open(hdu_path) as img:
        data=img[0].data
        hdu=img[0]
        img_wcs=WCS(img[0].header)
        read_noise=float(hdu.header["HIERARCH ESO DET OUT1 RON".lower()])
        gain=float(hdu.header["HIERARCH ESO DET OUT1 CONAD".lower()])
    try:
        if hdu.header["plate_solved"]==False:
            raise Not_plate_solved({"message":"ERROR: IMAGE HAS NOT BEEN PLATE SOLVED","img_name" : str(hdu_path.name)})
    except Not_plate_solved as exp:
        detail = exp.args[0]
        print(detail["message"]+" - "+detail["img_name"])
        exit()

    exptime=hdu.header["exptime"]

    ccd_data=CCDData.read(calib_path/get_image_file(calib_path,tgt_name,filter))

    ccdp.gain_correct(ccd_data, gain * u.electron / u.adu)

    #CT.show_image(ccd_data,log_plt=True)
    cosmic_rayless=cosmic_ray_reduce(ccd_data,read_noise)




    field_span=len(data[:,1]) #calulate the width of the image in pixels


    #Background estimation, using Photutils
    mean, median, std = sig(data, sigma=3.0)
    data[mask]=median
    data[data<0]=median

    sigma_clip = SigmaClip(sigma=3.0)
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (20, 20), filter_size=(3, 3),
                        sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    data_bkgsub=data-bkg.background
    pix_err = np.sqrt(data)

    #Catalogue query for PS1 stars in the image
    #print("Searching Catalogue")
    catalogue = vizier.query_region(img_wcs.pixel_to_world(412,412),
                                    width="4m",
                                    catalog="J/ApJ/867/105/refcat2",
                                    column_filters={'gmag': '!=','rmag': '!='})[0]
    catalogue.add_column(np.linspace(0,len(catalogue)-1,num=len(catalogue)),name="ID")

    #print("Catalogue Imported")


    #Scan the catalogue for incidents where two entries would have two overlaping star-finder cells, and select the brightest
    catalogue,exclude_cat=catalogue_filter(catalogue,2*star_cell_size*pix_size)  

    #Generate pixel coordinates for the included and excluded catalogue entries
    ps1_pix,in_ids=catalogue_pos_parse(catalogue,img_wcs,field_span,edge_pad)
    exc_pix,ex_ids=catalogue_pos_parse(exclude_cat,img_wcs,field_span,edge_pad)
    catalogue=catalogue[np.isin(catalogue["ID"],in_ids)]

    #Generate circular apertures around each catalgoue entry
    #ps1_apps=CircularAperture(ps1_pix,r=3)
    #exc_apps=CircularAperture(exc_pix,r=3)


    positions=[]
    fwhm=[]
    for coord in ps1_pix:
        star_pos,fwhm_fit=local_peak(data_bkgsub,coord,star_cell_size,field_span)
        positions.append(star_pos)
        fwhm.append(fwhm_fit)
    positions=np.array(positions)


    target_table=Table()
    target_table["id"] = in_ids
    target_table["pix_x"] = positions[:,0]
    target_table["pix_y"] = positions[:,1]
    target_table["fwhm"] = fwhm



    apertures = CircularAperture(positions, r=app_rad)
    bkg_annulus = CircularAnnulus(positions, r_in = app_rad * ann_in, r_out = app_rad * ann_out)

    no_app_mask=mask.data
    

    for app_mask in apertures.to_mask():
        app_mask=app_mask.to_image(mask.data.shape,dtype = bool)
        mask.data = np.ma.mask_or(mask.data,app_mask)

        #mask.data = app_mask.cutout(mask.data,fill_value=True)

    phot_table = aperture_photometry(data,apertures,mask=no_app_mask,error=pix_err)
    bkg_app_stats = ApertureStats(data,bkg_annulus,error=pix_err,mask=mask.data)


    bkg_mean  = bkg_app_stats.mean
    aperture_area = apertures.area_overlap(data,mask=no_app_mask)
    total_bkg=bkg_mean*aperture_area

    target_table["app_sum"] = phot_table["aperture_sum"] - total_bkg

    target_table["mag"] = -2.5*np.log10(target_table["app_sum"]/exptime)

    target_table["R-r"] = target_table["mag"] - catalogue["rmag"]
    target_table["g-r"] = catalogue["gmag"] - catalogue["rmag"]
    if first_pass==True:
        offset = np.median(target_table["R-r"])
        first_pass=False
    target_table["Scaled R-r"]=target_table["R-r"]-offset
    target_table["R-r error"]=mag_error((phot_table['aperture_sum_err']+(bkg_app_stats.std/np.sqrt(bkg_app_stats.sum_aper_area.value))),
                                        target_table["app_sum"],
                                        exptime)

    for entry_R,entry_gr,R_err in zip(target_table["Scaled R-r"],target_table["g-r"],target_table["R-r error"]):
        R_r.append(entry_R)
        gr.append(entry_gr)
        R_r_err.append(R_err)


plt.errorbar(gr,R_r,yerr=R_r_err,fmt='.')
plt.xlabel("g - r (Mag)")
plt.ylabel("R - r (Mag)")
plt.show()
#print(target_table.more())
#norm = ImageNormalize(stretch=SinhStretch(),clip=False)


plt.imshow(data, cmap='grey', origin='lower', norm=LogNorm())
apertures.plot(color='blue', lw=1.5, alpha=0.5)
bkg_annulus.plot(color='blue', lw=1.5, alpha=0.5)
#ps1_apps.plot(color='green',lw=1.5,alpha=0.5)
#exc_apps.plot(color='red',lw=1.5,alpha=0.5)
plt.xlim(0,824)
plt.ylim(0,824)
plt.show()

"""
plt.hist(fwhm,bins=30,range=[2,15])
plt.show()
"""