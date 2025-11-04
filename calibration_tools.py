import data_organisation_tools as DOT
from astroquery.astrometry_net import AstrometryNet as ANet

import pathlib
import os
import warnings


import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats as sig
import astropy.units as u
from astropy.nddata import CCDData
from ccdproc import ImageFileCollection, Combiner, combine
import ccdproc as ccdp
from astropy import wcs
from ccdproc import wcs_project
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=UserWarning)


root_dir = pathlib.Path(__file__).resolve().parent

def inv_median(a):
    return 1/np.median(a)

def set_trim(image_path,extra_clip,prescan_x_tag,prescan_y_tag,overscan_x_tag,overscan_y_tag):
    with fits.open(image_path) as img:
        low_x=img[0].header[prescan_x_tag]+extra_clip
        high_x=img[0].header[overscan_x_tag]+extra_clip
        low_y=img[0].header[prescan_y_tag]+extra_clip
        high_y=img[0].header[overscan_y_tag]+extra_clip
        x_span=img[0].header["NAXIS1"]
        y_span=img[0].header["NAXIS2"]
    return (np.s_[low_x:x_span-high_x,low_y:y_span-high_y])

def create_master_bias(in_path,out_path):
    ifc=ImageFileCollection(in_path,keywords='*',glob_exclude="bias_sub_*")
    ifc.sort(["object","mjd-obs"])
    filters={'object':'BIAS'}
    bias=ifc.filter(**filters)
    c=Combiner(bias.ccds(ccd_kwargs={'unit': 'adu'}))
    avg_bias=c.median_combine()

    out_path=pathlib.Path(out_path/("BIAS_MASTER.fits"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    avg_bias.write(str(out_path),overwrite=True)

    return (avg_bias)


def create_master_flat(img_filter,img_filter_tag, flat_type, bias, trim, in_path, out_path):
    print("CREATING FLAT: "+flat_type+" | FILTER: "+img_filter)
    sort_critera={'object' : flat_type,img_filter_tag : img_filter}
    flats=ImageFileCollection(in_path,keywords='*',glob_exclude="bias_sub_*").filter(**sort_critera)
    n=0
    for ccd,file in flats.ccds(ccd_kwargs={'unit': 'adu'},return_fname=True):
        ccd=ccdp.trim_image(ccd[trim])
        ccdp.subtract_bias(ccd,bias)
        ccd.write(str(in_path)+"/bias_sub_"+file,overwrite=True)
    
    flats=ImageFileCollection(in_path,keywords='*',glob_include="bias_sub_*").filter(**sort_critera)
    flats.sort(["object","mjd-obs"])
    c=Combiner(flats.ccds())
    c.scaling=inv_median
    avg_flat=c.median_combine()

    out_path=pathlib.Path(out_path/(flat_type+"_"+img_filter+".fits"))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    avg_flat.write(str(out_path),overwrite=True)

    img=fits.open(out_path,mode="update")
    img[0].header.set('HIERARCH ESO INS FILT1 NAME',img_filter)
    img.close()

    print("FLAT FIELD SAVED AT: "+str(out_path))

    return avg_flat


def load_bad_pixel_mask(dir):
    mask_path=pathlib.Path(dir/"bad_pixel_map.fits")
    if mask_path.is_file():
        print("LOADED BAD PIXEL MASK")
        mask = CCDData.read(mask_path,unit=u.dimensionless_unscaled)
        mask.data=mask.data.astype('bool')
        return mask
    else:
        print("NO BAD PIXEL MASK FOUND")

def generate_bad_pixel_mask(all_fits_path,out_path):

    print("GENERATING BAD PIXEL MASK")
    flats=ImageFileCollection(all_fits_path,keywords='*',glob_include="*bias_sub*")
    flats.sort("exptime")
    crit={"ESO INS FILT1 NAME".lower():"R#642"}

    files=flats.files_filtered(**crit)
    first_img=CCDData.read(all_fits_path/files[0])
    last_img=CCDData.read(all_fits_path/files[-1])

    ratio = last_img.divide(first_img)
    mask = ccdp.ccdmask(ratio)

    mask_as_ccd = CCDData(data=mask.astype('uint8'), unit=u.dimensionless_unscaled)
    mask_as_ccd.header['imagetyp'] = 'flat mask'

    out_path=pathlib.Path(out_path/("bad_pixel_map.fits"))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mask_as_ccd.write(str(out_path),overwrite=True)

    print("MASK SAVED AT: "+str(out_path))
    return mask

def cosmic_ray_mask(ccd_image,gain,read_noise):
    new_ccd=ccdp.cosmicray_lacosmic(ccd_image,readnoise=(1/gain)*read_noise,sigclip=10)
    return new_ccd.mask


def reduce_img(tgt_path,out_path,trim,bias,flat,tgt_name,img_filter, mask = None, *args, **kwargs ):
    with fits.open(tgt_path) as img:
        read_noise=float(img[0].header["HIERARCH ESO DET OUT1 RON".lower()])
        gain=float(img[0].header["HIERARCH ESO DET OUT1 CONAD".lower()])

    tgt=CCDData.read(tgt_path,unit='adu')
    tgt=ccdp.trim_image(tgt[trim])
    tgt=ccdp.subtract_bias(tgt,bias)
    tgt=ccdp.flat_correct(tgt,flat)

    mask = mask | cosmic_ray_mask(tgt,gain,read_noise)
    tgt.mask=tgt.mask | mask
    if mask is not None:
        tgt.mask = tgt.mask | mask


    out_path=pathlib.Path(out_path/(tgt_name+"_"+img_filter+"_"+str(tgt.header["mjd-obs"])+".fits"))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tgt.write(str(out_path),overwrite=True)

    return tgt


def show_image(img,log_plt=True,*args, **kwargs):
    """
    Convience function to display the contents of a CCD image object
    """

    if log_plt==True:
        plt.imshow(img.data,origin="lower",cmap='gray',norm=LogNorm())
    else:
        plt.imshow(img.data,origin="lower",cmap='gray')
    plt.show()

def plate_solve(img_path,px_scale):
    ra_tag="RA"
    dec_tag="DEC"
    solved_flag=False
    with fits.open(img_path) as img:
        ra_cent=img[0].header[ra_tag]
        dec_cent=img[0].header[dec_tag]
    

    ast=ANet()
    ast.api_key="irjhrsszlsratohk"


    plate_wcs=ast.solve_from_image(str(img_path),
                                   verbose=False,
                                   positional_error=20,
                                   scale_units='arcsecperpix',
                                   detect_threshold=8,
                                   fwhm=7,
                                   scale_est=px_scale,
                                   scale_type="ev",
                                   scale_err=5,
                                   crpix_center=True,
                                   center_ra=ra_cent,
                                   center_dec=dec_cent,
                                   radius=5,
                                   ra_dec_units=("degree","degree"))
    if plate_wcs=={}:
        print("PLATE SOLVING FAILED/TIMEDOUT: "+img_path.name)
    else:
        solved_flag=True

    return plate_wcs,solved_flag

def batch_plate_solve(dir,file_list,px_scale):
    print ("PLATE SOLVING ",len(file_list)," IMAGES")
    file_n=len(file_list)
    fails=0
    failed_files=[]
    for file in file_list:
        tgt_path=pathlib.Path(dir/file)
        plate_wcs,solved=plate_solve(tgt_path,px_scale)
        if solved==False:
            with fits.open(tgt_path,mode="update") as img:
                img[0].header.update({"plate_solved" : solved})
            fails+=1
            failed_files.append(tgt_path.name)
        else:
            with fits.open(tgt_path,mode="update") as img:
                img[0].header.update(plate_wcs)
                img[0].header.update({"plate_solved" : solved})
    print("PLATE SOLVING COMPLETE! | ",fails," FAILS | ",file_n-fails, " SUCCESSES")
    if len(failed_files)!=0:
        print("--- FAILED FILES ---")
        for file in failed_files:
            print(file)
            os.remove(pathlib.Path(dir/file))
        print("-> Failed Files Deleted from Output Directory <-")


def allign_and_avgstack(images,wcs):
    reprojected=[]
    n_max=99
    n=0

    for img in images.ccds():
        n+=1
        new_image=wcs_project(img,wcs)
        reprojected.append(new_image)
        if n>=n_max:
            break

    c = Combiner(reprojected)
    stack= c.average_combine()
    return stack

def avgstack(images):
    c=Combiner(images.ccds())
    c.scaling = inv_median
    stack = c.median_combine()
    return stack

def make_fringe_map(calib_path,filter):
    criteria={"ESO INS FILT1 NAME".lower():filter}
    fringe_comp=ImageFileCollection(calib_path,keywords='*',glob_include="fringe_comp*").filter(**criteria)
    comp_files=fringe_comp.files


    fringe_map=avgstack(fringe_comp)

    #mean, median, std = sig(fringe_map.data, sigma=3.0)

    #fringe_map.data=fringe_map.data-mean

    out_path=pathlib.Path(calib_path/("FRINGE_MAP_"+filter+".fits"))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fringe_map.write(str(out_path),overwrite=True)

    img=fits.open(out_path,mode="update")
    img[0].header.set('HIERARCH ESO INS FILT1 NAME',filter)
    img.close()

    print("FRINGE MAP SAVED AT: "+str(out_path))
    for file in comp_files:
        os.remove(calib_path/file)

    return fringe_map

    



#---> BELOW IS THE PIT, A PILE OF OLD DEVELOPMENT CODE KEPT FOR REFERENCE AND DEBUGGING PURPOSES, OR IN CASE I FORGET HOW TO DO SOMETHING <---

#--> RUN THIS LINE IF FILES NOT ORGANISED INTO FOLDER STRUCTURE YET <--
#DOT.organise_files(root_dir/"Data_set_1"/"unpacked_data")


#Experimenting with IFC and Astropy
#for FITS headers in use with CCDProc, ensure lowercase and leave off 'Hierarch'



#scale of pixel is 0.24" /px in 2x2 binning mode for EFOSC2
"""
px_scale=0.24
tgt_name="P113"
filter="R#642"

trim_tags=["HIERARCH ESO DET OUT1 PRSCX","HIERARCH ESO DET OUT1 PRSCY","HIERARCH ESO DET OUT1 OVSCX","HIERARCH ESO DET OUT1 OVSCY"]
ref_image=pathlib.Path(root_dir/"Data_set_1"/"block_1"/"BIAS"/"FREE"/"EFOSC.2009-01-27T21_00_47.752.fits")
extra_clip=100

all_fits_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS")
calib_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS"/"PROCESSED FRAMES")

trim=set_trim(ref_image,extra_clip,*trim_tags)

avg_bias = create_master_bias(all_fits_path,calib_path)
avg_bias = ccdp.trim_image(avg_bias[trim])

avg_flat=create_master_flat(filter,
                            "ESO INS FILT1 NAME".lower(),
                            "SKY,FLAT",
                            avg_bias,
                            trim,
                            all_fits_path,
                            calib_path)




lights=ImageFileCollection(all_fits_path,keywords='*',glob_exclude="bias_sub_*")
lights.sort(["object","mjd-obs"])
criteria={'object' : tgt_name, "ESO INS FILT1 NAME".lower():filter}
tgt_lights=lights.files_filtered(**criteria)

for file in tgt_lights:
    img=reduce_img(pathlib.Path(all_fits_path/file),
                calib_path,
                trim,
                avg_bias,
                avg_flat,
                tgt_name,
                filter)
    


science=ImageFileCollection(calib_path,keywords='*',glob_include=tgt_name+"_"+filter+"*")
science_files=science.files_filtered(**criteria)
batch_plate_solve(calib_path,
                  science_files,
                  px_scale)
"""



"""
Stacking, kind of redundant with comets that move-
"""
"""
science=ImageFileCollection(calib_path,keywords='*',glob_include=tgt_name+"_"+filter+"*")
science.sort(["object","mjd-obs"])
files=science.files_filtered(**criteria)
with fits.open(pathlib.Path(calib_path/files[0])) as img:
    tgt_wcs=wcs.WCS(img[0].header)
reprojected=[]
n_max=9
n=0

for img in science.ccds():
    n+=1
    new_image=wcs_project(img,tgt_wcs)
    reprojected.append(new_image)
    if n>=n_max:
        break

c = Combiner(reprojected)
stack= c.median_combine()
show_image(stack)
"""


