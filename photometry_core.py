import calibration_tools as CT

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from tqdm import tqdm
import pathlib
import time

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
from astropy.stats import sigma_clipped_stats, SigmaClip,sigma_clip
from astropy.modeling import fitting,models
from astropy.time import Time

from astroquery.vizier import Vizier
from astroquery.jplhorizons import Horizons



class Not_plate_solved(Exception):
    pass


def mag_error(count_error,count):
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
    if len(science_files)==0:
        print("ERROR: No Image Found")
        return 9999
    else:
        return science_files[0]

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
    return ((star_pos-center)+cat_pos),fwhm_fit


def cosmic_ray_reduce(ccd_image,read_noise):
    new_ccd=ccdp.cosmicray_lacosmic(ccd_image,readnoise=read_noise,sigclip=10)
    return new_ccd


def check_job_size(comet_names,filter_names,calib_path):
    file_count=0
    for comet in comet_names:
        for filter in filter_names:
            file_names=get_image_files(calib_path,comet,filter)
            file_count+=len(file_names)
    return file_count

class ESO_image:
    def __init__(self,folder,file_name):
        self.file_name=file_name
        self.image_path=pathlib.Path(folder/file_name)
        with fits.open(pathlib.Path(folder/file_name)) as hdu:
            self.header=hdu[0].header
            self.data=hdu[0].data
            self.wcs=WCS(hdu[0].header)
        self.read_noise=float(self.header["HIERARCH ESO DET OUT1 RON".lower()])
        self.gain=float(self.header["HIERARCH ESO DET OUT1 CONAD".lower()])
        self.exptime = self.header["exptime"]
        self.solved=True
        try:
            if self.header["plate_solved"]==False:
                raise Not_plate_solved({"message":"ERROR: IMAGE HAS NOT BEEN PLATE SOLVED","img_name" : str(file_name)})
        except Not_plate_solved as exp:
            detail = exp.args[0]
            print(detail["message"]+" - "+detail["img_name"])
            self.solved=False


class field_catalogue:
    def __init__(self,ref_img,field_width,star_cell_size,pix_size,ref_filter):
        img_wcs=WCS(ref_img.header)
        mid=int(len(ref_img.data[:,1])/2)
        vizier = Vizier(row_limit=-1) # this instantiates Vizier with its default parameters
        Vizier.clear_cache()
        print("Loading Wide Area Catalogue...")
        catalogue = vizier.query_region(img_wcs.pixel_to_world(mid,mid),
                                    width=field_width,
                                    catalog="J/ApJ/867/105/refcat2",
                                    column_filters={'gmag': '!=','rmag': '!=','imag': '!='})[0]
        catalogue.add_column(np.linspace(0,len(catalogue)-1,num=len(catalogue)),name="ID")
        print(len(catalogue["ID"])," Valid Targets Found...")
        self.whole_field,self.excluded_stars=self.catalogue_filter(catalogue,
                                                                   (3*star_cell_size*pix_size),
                                                                   ref_filter)
        self.cat_pos=np.transpose((self.whole_field["RA_ICRS"],self.whole_field["DE_ICRS"]))
        print("Done!")

    def catalogue_filter(self,catalogue,min_sep_AS,filter_band):
        """
        Removes stars from the catalogue that are too close together
        """
        min_sep=min_sep_AS/60/60

        removed_ids=[]
        filter=filter_band

        catalogue[filter].fill_value = 99
        

        for n,entry in zip(range(0,len(catalogue["ID"].value)),catalogue.iterrows("RA_ICRS","DE_ICRS",filter,"ID")):
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
    
    def catalogue_pos_parse(self,wcs,field_span,edge_pad):
        ids=[]
        field_span-=edge_pad

        big_cat_pix=np.transpose(wcs.world_to_pixel(coords.SkyCoord(ra=self.cat_pos[:,0],dec=self.cat_pos[:,1],unit="deg",frame="fk5")))
        cat_pix=[]
        for id,n in zip(self.whole_field["ID"].value,range(0,len(big_cat_pix))):
            if big_cat_pix[n,0] < edge_pad or big_cat_pix[n,0] > field_span or big_cat_pix[n,1] < edge_pad or big_cat_pix[n,1] > field_span:
                pass
            else:
                cat_pix.append(big_cat_pix[n,:])
                ids.append(id)
        return np.array(cat_pix),np.array(ids)

    def field_section(self,img,edge_pad):
        cat_in_frame_pix,cat_in_frame_ids=self.catalogue_pos_parse(WCS(img.header),len(img.data[:,1]),edge_pad)
        catalogue_section=self.whole_field[np.isin(self.whole_field["ID"],cat_in_frame_ids)]
        return cat_in_frame_pix,cat_in_frame_ids,catalogue_section

class colour_calib_frame:
    """
    A class for a single frame used for calibrating the colour term. Your create one per frame used during the calibration process
    """
    def __init__(self,img,mask,edge_pad,field_catalogue,cat_filter,colour_median=0,*args, **kwargs):
        self.mask=mask
        self.cat_filter=cat_filter
        self.cat_pix,self.cat_ids,self.frame_catalogue=field_catalogue.field_section(img,edge_pad)
        self.frame=img  #An instance of the ESO_image object, or different observatory configured image
        self.field_span=len(self.frame.data[:,0])
        self.estimate_bkg()
        self.colour_median=colour_median
        self.invalid_ids=[]
        self.no_stars=False

    def estimate_bkg(self):
        data=self.frame.data
        mean, median, std = sig(data, sigma=3.0)
        data[self.mask]=median
        data[data<0]=median

        sigma_clip = SigmaClip(sigma=3.0)
        bkg_estimator = MedianBackground()
        bkg = Background2D(data, (20, 20), filter_size=(3, 3),
                            sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        self.data_bkgsub=data-bkg.background
        self.pix_err = np.sqrt(data)
        self.frame.data=data
    
    def star_fitter(self,star_cell_size,fwhm_plot=False,fwhm_range=0.1,*args,**kwargs):
        positions=[]
        fwhm=[]
        for coord in self.cat_pix:
            star_pos,fwhm_fit=local_peak(self.data_bkgsub,
                                         coord,
                                         star_cell_size,
                                         self.field_span)
            positions.append(star_pos)
            fwhm.append(fwhm_fit)
        positions=np.array(positions)

        self.target_table=Table()
        self.target_table["id"] = self.cat_ids
        self.target_table["pix_x"] = positions[:,0]
        self.target_table["pix_y"] = positions[:,1]
        self.target_table["pix_pos"] = positions
        self.target_table["fwhm"] = fwhm

        if fwhm_plot:
            plt.hist(fwhm,bins=20)
            plt.show()
        
        self.avg_fwhm=np.median(fwhm)
        self.invalid_ids.extend(self.target_table[np.abs(self.target_table["fwhm"].value-self.avg_fwhm) > fwhm_range ]["id"].value)
        

    def remove_bad_aps(self,apertures):
        app_stats=ApertureStats(self.frame.data,apertures,error=self.pix_err,mask=self.mask.data)
        self.invalid_ids.extend(self.target_table[np.where(app_stats.max > 50000)]["id"].value)  #Discount any aperture that features a near-saturation pixel

    def ap_error(self,app_sum,sky_bkg,app_area):
        gain=self.frame.gain
        t=self.frame.exptime
        n_pix=app_area 
        n_star=app_sum*gain
        n_sky=sky_bkg*gain
        n_r=self.frame.read_noise

        noise=np.sqrt(n_star+n_pix*(n_sky+n_r**2))
        signal=n_star

        return (signal/noise)



    def ap_phot(self,app_rad,ann_in,ann_out,plot_id=9999,plot=False,*args,**kwargs):

        """
        Performs aperture photometry, with local background subtraction. Background is estimated using local sky annulus
        """

        app_rad=self.avg_fwhm*1.5

        positions=self.target_table["pix_pos"]
        apertures = CircularAperture(positions, r=app_rad)
        bkg_annulus = CircularAnnulus(positions, r_in = app_rad * ann_in, r_out = app_rad * ann_out)
        #print("Total Apertures: ",len(apertures))

        self.remove_bad_aps(apertures)

        #print("Discounted Targets: ",len(self.invalid_ids))
        no_app_mask=self.mask.data

        #print(apertures)

        for app_mask in apertures.to_mask():
            app_mask=app_mask.to_image(self.mask.data.shape,dtype = bool)
            self.mask.data = np.ma.mask_or(self.mask.data,app_mask)

        """
        Below is 'emergency' plotting of each frame and its apertures
        """
        if plot==True:
            plt.imshow(self.frame.data, cmap='grey', origin='lower', norm=LogNorm())
            apertures.plot(color='blue', lw=1.5, alpha=0.5)
            bkg_annulus.plot(color='blue', lw=1.5, alpha=0.5)
            plt.xlim(0,824)
            plt.ylim(0,824)
            for x,y,id in zip(self.target_table["pix_x"].value,self.target_table["pix_y"].value,self.target_table["id"].value):
                plt.annotate(str(id),[x,y],color="w")
            plt.show()

        if plot_id!=9999 and plot_id in self.target_table["id"].value:
            plt.imshow(self.frame.data, cmap='grey', origin='lower', norm=LogNorm())
            apertures.plot(color='blue', lw=1.5, alpha=0.5)
            bkg_annulus.plot(color='blue', lw=1.5, alpha=0.5)
            plt.xlim(0,824)
            plt.ylim(0,824)
            for x,y,id in zip(self.target_table["pix_x"].value,self.target_table["pix_y"].value,self.target_table["id"].value):
                plt.annotate(str(id),[x,y],color="w")
            plt.show()


        phot_table = aperture_photometry(self.frame.data,apertures,mask=no_app_mask)
        bkg_app_stats = ApertureStats(self.frame.data,bkg_annulus,mask=self.mask.data)

        bkg_mean  = bkg_app_stats.mean
        aperture_area = apertures.area_overlap(self.frame.data,mask=no_app_mask)

        total_bkg=bkg_mean*aperture_area
       

        self.target_table["app_sum"] = phot_table["aperture_sum"] - total_bkg

        

        self.target_table["SNR"] = self.ap_error(self.target_table["app_sum"],total_bkg,aperture_area)

        

        self.invalid_ids.extend(self.target_table[self.target_table["app_sum"] <= 0]["id"].value) #Mark any observation with a negative aperture sum as invalid

        

        self.invalid_ids=np.unique(self.invalid_ids) #Remove any duplicate invalid IDs

        self.target_table=self.target_table[np.isin(self.target_table["id"],self.invalid_ids,invert=True)] #Filter the invalids IDs from the observation list


        self.frame_catalogue=self.frame_catalogue[np.isin(self.frame_catalogue["ID"],self.invalid_ids,invert=True)] #Filter the invalid IDs from the comparison catalogue

        if len(self.target_table)==0:
            self.no_stars=True
        self.target_table["mag"] = -2.5*np.log10(self.target_table["app_sum"]/self.frame.exptime) #Calculate the magnitude from the aperture counts        
        self.target_table["mag_error"] = (2.5/np.log(10)) * (1/self.target_table["SNR"])

        

    def colour_grad_fit(self,col_a,col_b):

        self.target_table["colour_dif"] = self.target_table["mag"] - self.frame_catalogue[self.cat_filter]
        self.target_table["cat_colour"] = self.frame_catalogue[col_a] - self.frame_catalogue[col_b]
        self.target_table["Scaled colour_dif"]=self.target_table["colour_dif"]-np.median(self.target_table["colour_dif"])

        fit = fitting.LinearLSQFitter()
        or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=3, sigma=3.0)
        line_init = models.Linear1D()

        fitted_line,mask = or_fit(line_init,self.target_table["cat_colour"].value,self.target_table["Scaled colour_dif"].value,weights=1/self.target_table["mag_error"])
        filtered_data=np.ma.masked_array(self.target_table["Scaled colour_dif"].value,mask=mask)
        self.colour_grad = (fitted_line(2)-fitted_line(1))
        return self.target_table["Scaled colour_dif"].value, self.target_table["cat_colour"].value, self.target_table["id"].value, self.colour_grad,filtered_data,self.target_table["mag_error"]

    def colour_zero(self,gradient):
        offset = np.mean(gradient * self.target_table["cat_colour"].value - self.target_table["colour_dif"].value)
        offset = offset - np.median(self.target_table["cat_colour"]*gradient)
        #offset = np.median(gradient * self.target_table["cat_colour"].value)
        self.zero_term=offset
        return offset
    
class comet_frame:
    def __init__(self,_obs_code,_tgt,_img,*args, **kwargs):
        self.img=_img #an instance of an ESO image object, or other observatory configuration
        mjd_date=Time(float(_img.header["mjd-obs"]),format="mjd")
        self.date=float(mjd_date.jd)
        self.name=_tgt
        self.obs_code=_obs_code #observatory code where imagery was taken

    def find_comet(self,eph_code=0,*arg,**kwargs):
        if eph_code!=0:
            jpl_obj=Horizons(id=eph_code,id_type="smallbody",location=self.obs_code,epochs=self.date)
        else:
            jpl_obj=Horizons(id=self.name,id_type="smallbody",location=self.obs_code,epochs=self.date)
        comet_eph=jpl_obj.ephemerides()
        self.tgt_RA=comet_eph["RA"]
        self.tgt_DEC=comet_eph["DEC"]
        self.comet_pix_location=self.img.wcs.world_to_pixel(coords.SkyCoord(ra=self.tgt_RA,dec=self.tgt_DEC,unit="deg",frame="fk5"))
        self.comet_pix_location=np.array(self.comet_pix_location).flatten()
        #print(self.comet_pix_location)
    
    def show_comet(self):
        aperture = CircularAperture(self.comet_pix_location, r=10)        
        plt.imshow(self.img.data, cmap='grey', origin='lower',norm=LogNorm())
        aperture.plot(color='red', lw=1.5, alpha=0.5)
        plt.xlim(0,824)
        plt.ylim(0,824)
        plt.annotate(self.name,self.comet_pix_location,color="w")
        plt.show()


    def cutout_comet(self):
        cutout_size=100
        self.pad=int(cutout_size/2)
        center=self.comet_pix_location
        x_low=int(center[0]-self.pad)
        x_high=int(center[0]+self.pad)
        y_low=int(center[1]-self.pad)
        y_high=int(center[1]+self.pad)
        self.cutout=self.img.data[y_low:y_high+1,x_low:x_high+1]

    def apply_correction(self):
        self.jpl_location=self.comet_pix_location
        self.comet_pix_location=self.comet_pix_location+self.offset
        #print("BEFORE: ",self.jpl_location," | AFTER: ",self.comet_pix_location)

    def refine_lock(self):
        self.offset=lock_comet(self.cutout)
        self.apply_correction()
        self.cutout_comet()
    
    def comet_ap_phot(self,app_rad,ann_in,ann_out):

        position=self.comet_pix_location
        aperture = CircularAperture(position, r=app_rad)
        bkg_annulus = CircularAnnulus(position, r_in = app_rad * ann_in, r_out = app_rad * ann_out)

        phot_table = aperture_photometry(self.img.data,aperture)
        bkg_app_stats = ApertureStats(self.img.data,bkg_annulus)

        bkg_mean  = bkg_app_stats.mean
        aperture_area = aperture.area_overlap(self.img.data)

        total_bkg=bkg_mean*aperture_area
        comet_sum = phot_table["aperture_sum"].value[0]
        comet_counts = comet_sum - total_bkg
        comet_mag = -2.5*np.log10(comet_counts/self.img.exptime)
        print(comet_counts)
        print(total_bkg)
        print(comet_mag)
        print("-"*10)     


        """
        plt.imshow(self.img.data, cmap='grey', origin='lower',norm=LogNorm())
        aperture.plot(color='red', lw=1.5, alpha=0.5)
        bkg_annulus.plot(color='red', lw=1.5, alpha=0.5)
        plt.xlim(0,824)
        plt.ylim(0,824)
        plt.annotate(self.name,self.comet_pix_location,color="w")
        plt.show()   
        """
        return comet_mag


def composite_comet(frames):
    if (isinstance(frames[0],np.ndarray)):
        sum=np.sum(frames,axis=0)
        avg=np.median(frames,axis=0)
        return avg
    else:
        print("ERROR, IMAGES ARE NOT NUMPY ARRAYS")
        return 0
    
def lock_comet(img):
    result=fit_gauss(img,fwhm=6).results
    center=(len(img[0])-1)/2
    jpl_offset=np.array([result["x_fit"][0],result["y_fit"][0]])
    jpl_offset=jpl_offset-center
    return jpl_offset

def mark_target(pos,image_data):
    aperture = CircularAperture(pos, r=len(image_data[0])/50)
    plt.imshow(image_data,origin="lower",cmap="grey")
    aperture.plot(color="red",lw=1.5,alpha=0.5)
    plt.axis("scaled")
    plt.show()