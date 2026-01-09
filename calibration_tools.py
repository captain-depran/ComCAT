import data_organisation_tools as DOT
from astroquery.astrometry_net import AstrometryNet as ANet

import pathlib
import os
import warnings
import time

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

from photutils.aperture import CircularAperture,aperture_photometry,CircularAnnulus,ApertureStats
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground
from astropy.stats import sigma_clipped_stats, SigmaClip,sigma_clip

from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)
warnings.simplefilter('ignore', category=UserWarning)

from astroquery.astrometry_net import conf
conf.api_key = 'irjhrsszlsratohk'

root_dir = pathlib.Path(__file__).resolve().parent

def inv_median(a):
    return 1/np.median(a)

def scale_min_max(a):
    min=np.nanmin(a)
    max=np.nanmax(a)
    
    scaled = (a-min)/(max-min)

    print (np.nanmin(scaled))
    print (np.nanmax(scaled))
    return scaled

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

    col_sum=np.sum(mask_as_ccd.data,axis=0)

    for col in range (0,len(col_sum)):
        if col_sum[col] > 100:
            mask_as_ccd.data[:,col]=1

    mask=mask_as_ccd.data
    #show_image(mask_as_ccd)

    out_path=pathlib.Path(out_path/("bad_pixel_map.fits"))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mask_as_ccd.write(str(out_path),overwrite=True)

    print("MASK SAVED AT: "+str(out_path))
    return mask

def cosmic_ray_mask(ccd_image,gain,read_noise):
    new_ccd=ccdp.cosmicray_lacosmic(ccd_image,readnoise=(1/gain)*read_noise,sigclip=10)
    return new_ccd.mask


def reduce_img(tgt_path,out_path,trim,bias,flat,tgt_name,img_filter, 
               fringe_map = None, fringe_points = None, mask = None, *args, **kwargs ):
    with fits.open(tgt_path) as img:
        read_noise=float(img[0].header["HIERARCH ESO DET OUT1 RON".lower()])
        gain=float(img[0].header["HIERARCH ESO DET OUT1 CONAD".lower()])

    tgt=CCDData.read(tgt_path,unit='adu')
    tgt=ccdp.trim_image(tgt[trim])
    tgt=ccdp.subtract_bias(tgt,bias)
    tgt=ccdp.flat_correct(tgt,flat)

    

    if fringe_map is not None and fringe_points is not None:
        tgt.data = fringe_correction(tgt, fringe_points, fringe_map)

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
                                   positional_error=100,
                                   scale_units='arcsecperpix',
                                   detect_threshold=10,
                                   fwhm=5,
                                   scale_est=px_scale,
                                   scale_type="ev",
                                   scale_err=5,
                                   crpix_center=True,
                                   center_ra=ra_cent,
                                   center_dec=dec_cent,
                                   radius=10,
                                   ra_dec_units=("degree","degree"))
    if plate_wcs=={}:
        print("PLATE SOLVING FAILED/TIMEDOUT: "+img_path.name)
    else:
        solved_flag=True

    return plate_wcs,solved_flag

def source_list_plate_solve(img_path,sources,px_scale):
    ra_tag="RA"
    dec_tag="DEC"
    solved_flag=False
    with fits.open(img_path) as img:
        ra_cent=img[0].header[ra_tag]
        dec_cent=img[0].header[dec_tag]
        xsize=img[0].header["naxis1"]
        ysize=img[0].header["naxis2"]
    

    ast=ANet()
    ast.api_key="irjhrsszlsratohk"

    try_again=True
    submission_id=None

    while try_again:
        try:
            if not submission_id:
                plate_wcs=ast.solve_from_source_list(sources["xcentroid"],
                                            sources["ycentroid"],
                                            xsize,
                                            ysize,
                                            verbose=False,
                                            parity=2,
                                            scale_units='arcsecperpix',
                                            scale_est=px_scale,
                                            scale_type="ev",
                                            scale_err=5,
                                            crpix_center=True,
                                            center_ra=ra_cent,
                                            center_dec=dec_cent,
                                            radius=1,
                                            submission_id=submission_id)
                
            else:
                print("Monitoring Solve...")
                plate_wcs = ast.monitor_submission(submission_id, solve_timeout=30)
            
            #time.sleep(10)
        except TimeoutError as e:
            submission_id = e.args[1]
        else:
            try_again=False

    if plate_wcs=={}:
        print("PLATE SOLVING FAILED: "+img_path.name+" ["+str(len(sources["xcentroid"]))+" Sources]")
    else:
        print("PLATE SOLVING SUCCESFUL: "+img_path.name+" ["+str(len(sources["xcentroid"]))+" Sources]")
        solved_flag=True

    return plate_wcs,solved_flag


def sextractor(img_path,fwhm,thresh,mask):
    """
    Manual source extraction in an attempt to better plate solve the image
    """
    ccd = CCDData.read(img_path)
    data = ccd.data
    mean, median, std = sigma_clipped_stats(data, sigma=5.0, maxiters=5)
    daofind = DAOStarFinder(fwhm=fwhm,threshold=thresh * std,min_separation=8)
    table=daofind.find_stars(data,mask=mask)
    #table=table[np.abs(table["roundness2"])<0.35]

    table.sort("flux")
    table.reverse()
    #table=table[:45]
    
    """
    #This is here to debug source extraction
    coords_x=np.array(table["xcentroid"])
    coords_y=np.array(table["ycentroid"])
    coords=np.stack((coords_x,coords_y),axis=-1)
    apertures = CircularAperture(coords, r=8)
    plt.imshow(data,origin="lower",norm=LogNorm())
    apertures.plot(color='blue', lw=1.5, alpha=0.5)
    plt.show()
    """
    
    table['xcentroid'] += 1
    table['ycentroid'] += 1


    return table


def create_bkg_sub_copy(dir,file,mask):
    frame=CCDData.read(str(dir/file))
    #with fits.open(dir/file) as img:
        #frame=img[0]
    data=frame.data
    mean, median, std = sig(data, sigma=5.0)

    data[mask]=median
    data[data<0]=median

    sigma_clip = SigmaClip(sigma=5.0,maxiters=10)
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (20, 20), filter_size=(3, 3),
                        sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    data_bkgsub=data-bkg.background
    data_bkgsub[data_bkgsub<0]=0
    frame.data=data_bkgsub
    path=pathlib.Path(dir/ str("bkg_sub_"+file))
    frame.write(str(path),overwrite=True)



def batch_plate_solve(dir,file_list,px_scale,mask,fwhm,thresh):
    print ("PLATE SOLVING ",len(file_list)," IMAGES")
    file_n=len(file_list)
    fails=0
    failed_files=[]
    print("-"*10)
    #print("ETC: ",(len(file_list)*10)," SECONDS")
    for file in file_list:

        create_bkg_sub_copy(dir,file,mask)

        tgt_path=pathlib.Path(dir/file)
        ref_tgt_path=pathlib.Path(dir/str("bkg_sub_"+file))

        sources=sextractor(ref_tgt_path,fwhm,thresh,mask)
        #plate_wcs,solved=_plate_solve(ref_tgt_path,px_scale)
        plate_wcs,solved=source_list_plate_solve(tgt_path,sources,px_scale)
        if solved==False:
            with fits.open(tgt_path,mode="update") as img:
                img[0].header.update({"plate_solved" : solved})
            fails+=1
            failed_files.append(tgt_path.name)
        else:
            with fits.open(tgt_path,mode="update") as img:
                img[0].header.update(plate_wcs)
                img[0].header.update({"plate_solved" : solved})
        os.remove(ref_tgt_path)
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

def make_fringe_map(calib_path,filter,mask):
    criteria={"ESO INS FILT1 NAME".lower():filter}
    fringe_comp=ImageFileCollection(calib_path,keywords='*',glob_include="fringe_comp*").filter(**criteria)
    comp_files=fringe_comp.files


    fringe_map=avgstack(fringe_comp)
    fringe_map.data=fringe_map.data
    fringe_map.mask=mask
    #fringe_map.data[mask==1]=np.nan
    #mean, median, std = sig(fringe_map.data, sigma=3.0)

    #fringe_map.data=fringe_map.data-(np.nanmin(fringe_map.data))

    #fringe_map.data=scale_min_max(fringe_map.data)


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

def load_fringe_data(calib_path ,fringe_point_path, filter):
    fringe_points=np.loadtxt(fringe_point_path,delimiter=",")

    fringe_map_path = pathlib.Path(calib_path/("FRINGE_MAP_"+filter+".fits"))

    fringe = CCDData.read(fringe_map_path)
    return fringe_points,fringe

def fringe_correction(data,points,in_fringe):
    ratios=[]
    clean_data=in_fringe
    fringe=clean_data
    fringe.data=fringe.data-np.nanmin(fringe.data)
    fringe.data=fringe.data/np.nanmax(fringe.data)

    for pairs in points:
        x1=int(pairs[0])
        y1=int(pairs[1])
        x2=int(pairs[2])
        y2=int(pairs[3])

        map_dif=np.abs(fringe.data[y2,x2]-fringe.data[y1,x1])
        frame_dif=np.abs(data.data[y2,x2]-data.data[y1,x1])

        ratios.append(frame_dif/map_dif)

    scale=np.median(ratios)

    #med=np.nanmedian(fringe.data)
    fringe.data[fringe.mask==1] = 0
    reduced_img=data.data-(fringe.data*scale)
    #reduced_img[data.mask==1]=data.data[data.mask==1]

    return reduced_img

def moving_avg(data,window_size):
    #odd_shift=(window_size%2)
    pad=int(window_size/2)
    if (window_size%2)==0:
        print("WINDOW MUST BE ODD!")

    elif (window_size%2)==1:
        data=np.insert(data,0,(data[0]*np.ones(pad+1)))
        data=np.insert(data,-1,(data[-1]*np.ones(pad)))
    cumsum_vec=np.cumsum(data)
    ma_vec=(cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    return ma_vec




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


