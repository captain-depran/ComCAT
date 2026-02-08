import calibration_tools as CT

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from tqdm import tqdm
import pathlib
import os
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
from photutils.detection import DAOStarFinder

from astroquery.vizier import Vizier
from astroquery.jplhorizons import Horizons

rng = np.random.default_rng()


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
    #print(tgt)
    criteria={'object' : tgt, "ESO INS FILT1 NAME".lower():filter}
    science=ImageFileCollection(calib_path,keywords='*',glob_include=tgt+"_"+filter+"*")
    #science.sort("mjd-obs")
    science_files=science.files_filtered(**criteria)
    if len(science_files) == 0:
        criteria={'object' : tgt, "ESO INS FILT2 NAME".lower():filter}
        science=ImageFileCollection(calib_path,keywords='*',glob_include=tgt+"_"+filter+"*")
        #science.sort("mjd-obs")
        science_files=science.files_filtered(**criteria)
    return science_files

def get_image_file(calib_path,tgt,filter):
    criteria={'object' : tgt, "ESO INS FILT2 NAME".lower():filter}
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

def local_peak(data,cat_pos,width,top_edge,side_edge,bkg_remove=False,*args,**kwargs):
    """
    Function to take a catalogue position, and find the local peak within a box of 2 x 'width' centered on that position
    """
    low_x=bound_legal(cat_pos[0],-1*width,side_edge,0)
    high_x=bound_legal(cat_pos[0],width,side_edge,0)
    low_y=bound_legal(cat_pos[1],-1*width,top_edge,0)
    high_y=bound_legal(cat_pos[1],width,top_edge,0)
    
    if (high_x-low_x)%2 == 0:
        high_x+=1
    if (high_y-low_y)%2 == 0:
        high_y+=1
    if high_y>top_edge:
        high_y=top_edge
        low_y-=1
    if high_x>side_edge:
        high_x=side_edge
        low_x-=1
    

    center=[(high_x-low_x)/2,(high_y-low_y)/2]

    clip_data=data[low_y:high_y,low_x:high_x]

    if bkg_remove:
        mean, median, std = sig(clip_data, sigma=3.0)
        clip_data = clip_data - median

    result=fit_gauss(clip_data,fix_fwhm=False,xypos=center).results
    star_pos=np.array([result["x_fit"][0],result["y_fit"][0]])
    fwhm_fit=result["fwhm_fit"][0]

    if star_pos[0] < 0 or star_pos[1] < 0 or star_pos[1] > len(clip_data[0]) or star_pos[0] > len(clip_data[0]):
        star_pos[0]=center[0]
        star_pos[1]=center[1]

    #plt.plot(clip_data[int(star_pos[0]),:])
    #plt.plot(clip_data[:,int(star_pos[1])])
    #plt.show()

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
    def __init__(self,folder,file_name,pix_limit=50000,*args,**kwargs):
        self.file_name=file_name
        self.image_path=pathlib.Path(folder/file_name)
        with fits.open(pathlib.Path(folder/file_name),memmap=False) as hdu:
            self.header=hdu[0].header
            self.data=hdu[0].data
            self.wcs=WCS(hdu[0].header)
            hdu.close()
        self.read_noise=float(self.header["HIERARCH ESO DET OUT1 RON".lower()])
        self.gain=float(self.header["HIERARCH ESO DET OUT1 CONAD".lower()])
        self.exptime = self.header["exptime"]
        self.pix_limit=pix_limit
        self.solved=True
        try:
            if self.header["plate_solved"]==False:
                raise Not_plate_solved({"message":"ERROR: IMAGE HAS NOT BEEN PLATE SOLVED","img_name" : str(file_name)})
        except Not_plate_solved as exp:
            detail = exp.args[0]
            print(detail["message"]+" - "+detail["img_name"])
            self.solved=False
        except:
            print("ERROR: IMAGE HAS NOT BEEN PLATE SOLVED - " + str(file_name))
            self.solved=False
        
        
    def get_zero(self):
        with fits.open(self.image_path,mode="update") as img:
            try:
                self.zero=img[0].header["zero_point"]
            except:
                self.zero = 0
                self.zero_err = 0
                print("NO ZERO")
            try:
                self.zero_err = img[0].header ["zero_error"]
            except:
                self.zero_err = 0
                print("NO ZERO ERROR")
            img.close()
        return self.zero,self.zero_err

    def update_zero(self,zero_point,err):
        #print(self.image_path)
        with fits.open(self.image_path,mode="update",memmap=False) as img:
            img[0].header.update({"zero_point" : zero_point, "zero_error" : err})
        


class field_catalogue:
    def __init__(self,ref_img,field_width,star_cell_size,pix_size,ref_filter):
        img_wcs=WCS(ref_img.header)
        midy=int(len(ref_img.data[:,1])/2)
        midx=int(len(ref_img.data[1,:])/2)

        vizier = Vizier(row_limit=-1,columns=['RA_ICRS','DE_ICRS','gmag','rmag','imag','e_gmag','e_rmag','e_imag']) # this instantiates Vizier with its default parameters
        Vizier.clear_cache()
        print("Loading Wide Area Catalogue...")
        catalogue = vizier.query_region(img_wcs.pixel_to_world(midx,midy),
                                    width=field_width,
                                    catalog="J/ApJ/867/105/refcat2",
                                    column_filters={'gmag': '!=','rmag': '!=','imag': '!='})[0]
        
        catalogue.add_column(np.linspace(0,len(catalogue)-1,num=len(catalogue)),name="ID")
        print(len(catalogue["ID"])," Valid Targets Found...")
        print("Filtering Targets...")
        
        self.whole_field,self.excluded_stars=self.catalogue_filter(catalogue,
                                                                   (2*star_cell_size*pix_size),
                                                                   ref_filter)
        self.cat_pos=np.transpose((self.whole_field["RA_ICRS"],self.whole_field["DE_ICRS"]))
        print("Done!")

    def catalogue_filter(self,catalogue,min_sep_AS,filter_band):
        """
        Removes stars from the catalogue that are too close together, to faint, or have errors on their colours that are too large
        """
        min_sep=min_sep_AS/60/60

        removed_ids=[]
        filter=filter_band

        catalogue[filter].fill_value = 99
        catalogue=catalogue[catalogue["e_rmag"] < 0.05]
        catalogue=catalogue[catalogue["e_gmag"] < 0.05]
        catalogue=catalogue[catalogue["e_imag"] < 0.05]       
        

        for n,entry in tqdm(zip(range(0,len(catalogue["ID"].value)),catalogue.iterrows("RA_ICRS","DE_ICRS",filter,"ID"))):
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
    
    def catalogue_pos_parse(self,wcs,field_height,field_width,edge_pad):
        ids=[]
        field_width-=edge_pad
        field_height-=edge_pad

        big_cat_pix=np.transpose(wcs.world_to_pixel(coords.SkyCoord(ra=self.cat_pos[:,0],dec=self.cat_pos[:,1],unit="deg",frame="fk5")))
        cat_pix=[]
        for id,n in zip(self.whole_field["ID"].value,range(0,len(big_cat_pix))):
            if big_cat_pix[n,0] < edge_pad or big_cat_pix[n,0] > field_width or big_cat_pix[n,1] < edge_pad or big_cat_pix[n,1] > field_height:
                pass
            else:
                cat_pix.append(big_cat_pix[n,:])
                ids.append(id)
        return np.array(cat_pix),np.array(ids)

    def field_section(self,img,edge_pad):
        cat_in_frame_pix,cat_in_frame_ids=self.catalogue_pos_parse(WCS(img.header),len(img.data[:,1]),len(img.data[1,:]),edge_pad)
        catalogue_section=self.whole_field[np.isin(self.whole_field["ID"],cat_in_frame_ids)]
        return cat_in_frame_pix,cat_in_frame_ids,catalogue_section

class colour_calib_frame:
    """
    A class for a single frame used for calibrating the colour term. You create one per frame used during the calibration process
    """
    def __init__(self,img,mask,edge_pad,field_catalogue,cat_filter,colour_median=0,SNR_thresh = 5,*args, **kwargs):
        self.mask=mask
        self.cat_filter=cat_filter
        self.cat_pix,self.cat_ids,self.frame_catalogue=field_catalogue.field_section(img,edge_pad)
        self.frame=img  #An instance of the ESO_image object, or different observatory configured image
        self.field_height=len(self.frame.data[:,0])
        self.field_width=len(self.frame.data[0,:])
        self.estimate_bkg()
        self.colour_median=colour_median
        self.invalid_ids=[]
        self.SNR_thresh = SNR_thresh
        self.no_stars=False

    def estimate_bkg(self):
        data=self.frame.data
        #mean, median, std = sig(data, sigma=3.0)
        data[self.mask]=0
        data[data<0]=0

        sigma_clip = SigmaClip(sigma=3.0)
        bkg_estimator = MedianBackground()
        bkg = Background2D(data, (20, 20), filter_size=(3, 3),
                            sigma_clip=sigma_clip, bkg_estimator=bkg_estimator,exclude_percentile=20)
        self.data_bkgsub=data-bkg.background
        self.pix_err = np.sqrt(data)
        self.frame.data=data
    
    def star_fitter(self,star_cell_size,fwhm_plot=False,fwhm_range=0.1,*args,**kwargs):
        positions=[]
        fwhm=[]
        if len(self.cat_pix)==0:
            print("EMPTY CATALOGUE")
            return 1
        else:
            for coord in self.cat_pix:
                star_pos,fwhm_fit=local_peak(self.data_bkgsub,
                                            coord,
                                            star_cell_size,
                                            self.field_height,
                                            self.field_width)
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
                #plt.hist(fwhm,bins=20)
                plt.scatter(np.linspace(0,len(fwhm),len(fwhm)),fwhm)
                plt.show()
            
            self.avg_fwhm=np.median(fwhm)
            self.invalid_ids.extend(self.target_table[np.abs(self.target_table["fwhm"].value-self.avg_fwhm) > fwhm_range ]["id"].value)
            return 0

    def remove_bad_aps(self,apertures,limit,*args,**kwargs):
        app_stats=ApertureStats(self.frame.data,apertures,error=self.pix_err,mask=self.mask.data)
        self.invalid_ids.extend(self.target_table[np.where(app_stats.max > limit)]["id"].value)  #Discount any aperture that features a near-saturation pixel

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

        self.remove_bad_aps(apertures,self.frame.pix_limit)

        #print("Discounted Targets: ",len(self.invalid_ids))
        no_app_mask=self.mask.data

        #print(apertures)

        for app_mask in apertures.to_mask():
            app_mask=app_mask.to_image(self.mask.data.shape,dtype = bool)
            if np.sum(app_mask) is None:
                continue
            self.mask.data = np.ma.mask_or(self.mask.data,app_mask)
        

        """
        Below is 'emergency' plotting of each frame and its apertures
        """
        if plot==True:
            plt.imshow(self.frame.data, cmap='grey', origin='lower', norm=LogNorm())
            apertures.plot(color='blue', lw=1.5, alpha=0.5)
            bkg_annulus.plot(color='blue', lw=1.5, alpha=0.5)
            plt.xlim(0,self.field_width)
            plt.ylim(0,self.field_height)
            for x,y,id in zip(self.target_table["pix_x"].value,self.target_table["pix_y"].value,self.target_table["id"].value):
                plt.annotate(str(id),[x,y],color="w")
            plt.show()

        if plot_id!=9999 and plot_id in self.target_table["id"].value:
            plt.imshow(self.frame.data, cmap='grey', origin='lower', norm=LogNorm())
            apertures.plot(color='blue', lw=1.5, alpha=0.5)
            bkg_annulus.plot(color='blue', lw=1.5, alpha=0.5)
            plt.xlim(0,self.field_width)
            plt.ylim(0,self.field_height)
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
        self.invalid_ids.extend(self.target_table[self.target_table["SNR"] <= 0]["id"].value) #Mark any observation with a negative SNR as invalid
        self.invalid_ids.extend(self.target_table[np.isnan(self.target_table["SNR"])]["id"].value) #Mark any observation with a NAN SNR as invalid
        self.invalid_ids.extend(self.target_table[self.target_table["SNR"] <= self.SNR_thresh]["id"].value) #Mark any sub-par SNR stars as invalid


        self.invalid_ids=np.unique(self.invalid_ids) #Remove any duplicate invalid IDs

        

        self.target_table=self.target_table[np.isin(self.target_table["id"],self.invalid_ids,invert=True)] #Filter the invalids IDs from the observation list


        self.frame_catalogue=self.frame_catalogue[np.isin(self.frame_catalogue["ID"],self.invalid_ids,invert=True)] #Filter the invalid IDs from the comparison catalogue

        if len(self.target_table)==0:
            self.no_stars=True
        self.target_table["mag"] = -2.5*np.log10(self.target_table["app_sum"]/self.frame.exptime) #Calculate the magnitude from the aperture counts        
        self.target_table["mag_error"] = (2.5/np.log(10)) * (1/self.target_table["SNR"])

        

    def colour_grad_fit(self,col_a,col_b, col_low = 0, col_high = 1, *args, **kwargs):

        self.target_table["colour_dif"] = self.target_table["mag"] - self.frame_catalogue[self.cat_filter]
        self.target_table["cat_colour"] = self.frame_catalogue[col_a] - self.frame_catalogue[col_b]
        self.target_table["colour_err"] = linear_error(self.frame_catalogue[str("e_"+col_a)],self.frame_catalogue[str("e_"+col_b)])

        self.frame_catalogue=self.frame_catalogue[self.target_table["cat_colour"] > col_low]
        self.target_table=self.target_table[self.target_table["cat_colour"] > col_low]
        self.frame_catalogue=self.frame_catalogue[self.target_table["cat_colour"] < col_high]
        self.target_table=self.target_table[self.target_table["cat_colour"] < col_high]

        self.target_table["Scaled colour_dif"]=self.target_table["colour_dif"]-np.median(self.target_table["colour_dif"])

        if len(self.target_table) <= 1:
            return 0
        

        fit = fitting.LinearLSQFitter()
        or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=3, sigma=3.0)
        line_init = models.Linear1D()
      

        fitted_line,mask = or_fit(line_init,self.target_table["cat_colour"].value,self.target_table["Scaled colour_dif"].value,weights=1/self.target_table["mag_error"])
        filtered_data=np.ma.masked_array(self.target_table["Scaled colour_dif"].value,mask=mask)
        self.colour_grad = (fitted_line(2)-fitted_line(1))
        return self.colour_grad
    
    def apply_colour_grad(self,col_a,col_b,colour_term, col_low = 0, col_high = 1, *args, **kwargs):
        self.target_table["cat_colour"] = self.frame_catalogue[col_a] - self.frame_catalogue[col_b]
        self.target_table["colour_err"] = linear_error(self.frame_catalogue[str("e_"+col_a)],self.frame_catalogue[str("e_"+col_b)])

        self.frame_catalogue=self.frame_catalogue[self.target_table["cat_colour"] > col_low]
        self.target_table=self.target_table[self.target_table["cat_colour"] > col_low]
        self.frame_catalogue=self.frame_catalogue[self.target_table["cat_colour"] < col_high]
        self.target_table=self.target_table[self.target_table["cat_colour"] < col_high]

        
        #zero = self.target_table["colour_dif"] - (colour_term * self.target_table["cat_colour"].value)

        #self.target_table["Scaled colour_dif"]=self.target_table["colour_dif"]-np.median(self.target_table["colour_dif"])

        #print(zero)

        zero=self.colour_zero(colour_term)

        return zero


    def colour_zero(self,gradient):
        #zeros = self.frame_catalogue[self.cat_filter] - self.target_table["mag"] - (gradient * self.target_table["cat_colour"].value)
        
        zeros = self.frame_catalogue[self.cat_filter] - (self.target_table["mag"] + (gradient * self.target_table["cat_colour"].value))
        
        offset = np.median(zeros)
        

        err = bootstrap_error(zeros,50)

        #print("TEST METHOD")
        #print("Median: ",np.median(zeros))
        #print("Mean: ",np.mean(zeros))

        #print("OLD METHOD")
        #offset = np.median(gradient * self.target_table["cat_colour"].value - self.target_table["colour_dif"].value)
        #print("OFFSET 1:", offset)
        #offset = offset - np.median(self.target_table["cat_colour"]*gradient)
        #print("OFFSET 2: ",offset)
        #print(offset)
        #offset = np.median(gradient * self.target_table["cat_colour"].value)
        self.zero_term=offset

        self.frame.update_zero(offset,err)

        #print(offset," +- ",err)

        return offset
    
    
class comet_frame:
    def __init__(self,_obs_code,_tgt,_img,comet_pixel_max=10000,*args, **kwargs):
        self.img=_img #an instance of an ESO image object, or other observatory configuration
        mjd_date=Time(float(_img.header["mjd-obs"]),format="mjd")
        self.date=float(mjd_date.jd)
        self.name=_tgt
        self.obs_code=_obs_code #observatory code where imagery was taken
        self.img.data[self.img.data < 0]=0  #fix any sub-zero values
        #img_min=np.nanmin(self.img.data)
        #self.img.data=self.img.data-img_min
        self.comet_count_cutoff = comet_pixel_max

    def find_comet(self,eph_code=0,*arg,**kwargs):
        if eph_code!=0:
            jpl_obj=Horizons(id=eph_code,id_type="smallbody",location=self.obs_code,epochs=self.date)
        else:
            jpl_obj=Horizons(id=self.name,id_type="smallbody",location=self.obs_code,epochs=self.date)
        comet_eph=jpl_obj.ephemerides()
        self.tgt_RA = comet_eph["RA"]
        self.tgt_DEC = comet_eph["DEC"]
        self.RA_unc = comet_eph["RA_3sigma"]
        self.DEC_unc = comet_eph["DEC_3sigma"]
        self.comet_pix_location=self.img.wcs.world_to_pixel(coords.SkyCoord(ra=self.tgt_RA,dec=self.tgt_DEC,unit="deg",frame="fk5"))
        self.comet_pix_location=np.array(self.comet_pix_location).flatten()
        self.orig_comet_pix_location = np.copy(self.comet_pix_location)
        if self.comet_pix_location[0] < 0 or self.comet_pix_location[1] < 0 or self.comet_pix_location[0] > len(self.img.data[0,:]) or self.comet_pix_location[1] > len(self.img.data[:,0]):
            return False
        else:
            return True 
        #print(self.comet_pix_location)
    
    def show_comet(self):
        aperture = CircularAperture(self.comet_pix_location, r=10)        
        plt.imshow(self.img.data, cmap='grey', origin='lower',norm=LogNorm())
        aperture.plot(color='red', lw=1.5, alpha=0.5)
        #plt.xlim(0,824)
        #plt.ylim(0,824)
        plt.annotate(self.name,self.comet_pix_location,color="w")
        plt.show()


    def cutout_comet(self,cutout_size=100,manual_shift=[0,0],*args,**kwargs):

        self.pad=int(cutout_size/2)
        center=self.comet_pix_location
        x_low=int(center[0]-self.pad)+manual_shift[0]
        x_high=int(center[0]+self.pad)+manual_shift[0]
        y_low=int(center[1]-self.pad)+manual_shift[1]
        y_high=int(center[1]+self.pad)+manual_shift[1]
        self.cutout=self.img.data[y_low:y_high+1,x_low:x_high+1]

        self.cutout_transform = np.array([x_low,y_low])

    def apply_correction(self):
        self.jpl_location=np.copy(self.comet_pix_location)
        self.comet_pix_location=self.comet_pix_location+self.offset
        #print("BEFORE: ",self.jpl_location," | AFTER: ",self.comet_pix_location)

    def refine_lock(self,cutout_size=100,*args,**kwargs):

        self.offset=lock_comet(self.cutout,cutoff=self.comet_count_cutoff)
        self.apply_correction()
        self.cutout_comet(cutout_size=cutout_size)
    
    def ap_error(self,app_sum,sky_bkg,app_area):
        gain=self.img.gain
        t=self.img.exptime
        n_pix=app_area 
        n_star=app_sum*gain
        n_sky=sky_bkg*gain
        n_r=self.img.read_noise

        noise=np.sqrt(n_star+n_pix*(n_sky+n_r**2))
        signal=n_star

        return (signal/noise)
    
    def comet_ap_phot(self,app_rad,ann_in,ann_out,shift=[0,0],*args,**kwargs):

        position=self.comet_pix_location+shift
        aperture = CircularAperture(position, r=app_rad)
        bkg_annulus = CircularAnnulus(position, r_in = app_rad * ann_in, r_out = app_rad * ann_out)

        phot_table = aperture_photometry(self.img.data,aperture)
        bkg_app_stats = ApertureStats(self.img.data,bkg_annulus)

        bkg_mean  = bkg_app_stats.mean
        aperture_area = aperture.area_overlap(self.img.data)

        total_bkg=bkg_mean*aperture_area
        comet_sum = phot_table["aperture_sum"].value[0]
        comet_counts = comet_sum - total_bkg
        if comet_counts <= 0:
            comet_counts = 1
        comet_mag = -2.5*np.log10(comet_counts/self.img.exptime)

        self.SNR = self.ap_error(comet_sum,total_bkg,aperture_area)
        self.mag_error = (2.5/np.log(10)) * (1/self.SNR)

        #print(comet_counts)
        #print(total_bkg)
        #print(comet_mag)
        #print("-"*10)     


        """
        plt.imshow(self.img.data, cmap='grey', origin='lower',norm=LogNorm())
        aperture.plot(color='red', lw=1.5, alpha=0.5)
        bkg_annulus.plot(color='red', lw=1.5, alpha=0.5)
        plt.xlim(0,824)
        plt.ylim(0,824)
        plt.annotate(self.name,self.comet_pix_location,color="w")
        plt.show()   
        """
        return comet_mag,self.SNR,self.mag_error

class study_comet:
    def __init__(self,tgt_name,
                 search_name,
                 filt,
                 calib_path,
                 eph_code=0,
                 obs_code=809,
                 plot_stack=False,
                 comet_pixel_max=10000,
                 cutout_size=100,
                 show_frames=False,
                 pos_offset=[0,0],
                 *args, **kwargs):
        
        self.tgt_name=tgt_name
        self.jpl_name=search_name
        self.filter = filt
        self.calib_path = calib_path
        self.obs_code = obs_code
        self.eph_code = eph_code
        self.pos_offset = pos_offset
        self.comet_pics = []
        self.cutouts = []
        self.t=[]
        self.zeros=[]
        self.zero_errs=[]
        self.mags=[]
        self.snrs=[]
        self.errors=[]
        self.comet_pixel_max=comet_pixel_max
        self.cutout_size=cutout_size
        self.skip_this=False
        self.show_frames=show_frames

        all_image_names = self.load_files()
        if all_image_names is None:
            self.skip_this=True
            return None
        
        for image in all_image_names:
            valid = self.init_frame(image)
        

        print("Images used: ",len(self.t))
        self.t=np.array(self.t)
        if len(self.t !=0):
            self.t=(self.t-self.t[0])*24*60

        self.cutout_stack = composite_comet(self.cutouts)
        if plot_stack:
            self.show_full_stack()

    def get_measures(self,app_rad = 4, app_in = 1.5, app_out = 2,shift = [0,0],*args,**kwargs):
        self.mags=[]
        self.snrs=[]
        self.errors=[]    
        for frame,zero,err in zip(self.comet_pics,self.zeros,self.zero_errs):
            mag,snr,mag_err = frame.comet_ap_phot(app_rad,app_in,app_out,shift=shift)
            self.mags.append(mag+zero)
            self.snrs.append(snr)
            self.errors.append(linear_error(mag_err,err))

    def init_frame(self,image_name):
        img=ESO_image(self.calib_path,image_name)
        if img.solved==False:
            print("ERROR! NOT SOLVED!")
            return 0
        comet_pic=comet_frame(self.obs_code,self.jpl_name,img,comet_pixel_max=5000)

        good_find = comet_pic.find_comet(eph_code=self.eph_code)
        
        if good_find == False:
            print("NO FIND")
            return 0

        comet_pic.offset=self.pos_offset
        comet_pic.apply_correction()


        comet_pic.cutout_comet(cutout_size=50)
        
        self.comet_pics.append(comet_pic)
        self.cutouts.append(comet_pic.cutout)
        zero,zero_err = comet_pic.img.get_zero()
        self.zeros.append(zero)
        self.zero_errs.append(zero_err)
        self.t.append(comet_pic.date)
        if self.show_frames:
            comet_pic.show_comet()

    def load_files(self):
        all_image_names=get_image_files(self.calib_path,self.tgt_name,self.filter)
        print("Images of Comet "+self.tgt_name+": ",len(all_image_names))
        if len(all_image_names)==0:
            print("ERROR: NO FILES")
            return None

        return all_image_names
            
    def find_offset(self,search_span = 30, strike_lim = 3, samples = 3000):
        x_offsets = rng.integers(np.ones(samples)*-1*search_span,np.ones(samples)*search_span)
        y_offsets = rng.integers(np.ones(samples)*-1*search_span,np.ones(samples)*search_span)

        snr_stddev = []
        mags_stddev = []
        all_mags = []

        dxs=[]
        dys=[]
        rank=[]

        unmodded_pics = self.comet_pics.copy()

        for dx,dy in tqdm(zip(x_offsets,y_offsets)):
            shift = [dx,dy]
            strikes = 0
            snrs=[]
            mags=[]
            for i in (range(0,len(self.zeros))):
                mag,snr,mag_error = unmodded_pics[i].comet_ap_phot(3,1.1,1.5,shift=shift)
                mag += self.zeros[i]
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

        snr_min_i = np.argmin(snr_stddev)
        mags_min_i = np.argmin(mags_stddev)
        rank_max_i = np.argmax(rank)

        mags_1=np.array(all_mags)[snr_min_i]    #Offset with Minimum SNR Deviation
        mags_2=np.array(all_mags)[mags_min_i]   #Offset with Minimum Mag Deviation
        mags_3=np.array(all_mags)[rank_max_i]   #Offset with Most Valid Images

        print("- MOST CONSISTENT MAG RESULT -")
        print("Mag Sigma: ",np.min(mags_stddev))
        print("Mag Median: ",np.median(mags_2))
        dx = np.array(dxs)[mags_min_i]
        dy = np.array(dys)[mags_min_i]
        print("OFFSET: DX = ",dx," | DY = ",dy)
        print("-"*10)

        print("- MOST CONSISTENT SNR RESULT -")
        dx = np.array(dxs)[snr_min_i]
        dy = np.array(dys)[snr_min_i]
        print("OFFSET: DX = ",dx," | DY = ",dy)

        print("- MOST UNREJECTED IMAGES RESULT -")
        dx = np.array(dxs)[rank_max_i]
        dy = np.array(dys)[rank_max_i]
        print("OFFSET: DX = ",dx," | DY = ",dy)
        
    def plot_surf_brightness(self,min=1,
                             max=20,
                             y_relative = False,
                             logx=True,
                             logy=True,
                             *args,**kwargs):
        sums,radii=surf_brightness(np.array([0,0])+((len(self.cutout_stack[0])-1)/2),
                           self.cutout_stack,
                           step_size=1,
                           min=min,
                           max=max)
        radii=np.array(radii)

        if logx:
            radii = np.log(radii)

        if y_relative:
            sums=sums/np.max(sums)

        if logy:
            sums = np.log(sums)
        

        plt.plot(radii,sums,label=self.jpl_name)
    
    def show_full_stack(self):
        mark_target([(len(self.cutout_stack[0])-1)/2,(len(self.cutout_stack[0])-1)/2],self.cutout_stack)

    def analyse_path(self):
        jpl_coord_list=[]
        pix_coord_list=[]
        for pic in self.comet_pics:
            pix_coord = pic.img.wcs.pixel_to_world(pic.comet_pix_location[0],pic.comet_pix_location[1])
            jpl_coord = coords.SkyCoord(ra=pic.tgt_RA,dec=pic.tgt_DEC,unit="deg",frame="fk5")

            pix_coord_list.append([pix_coord.ra.to(u.arcsec).value,pix_coord.dec.to(u.arcsec).value])
            jpl_coord_list.append(np.array([jpl_coord.ra.to(u.arcsec).value,jpl_coord.dec.to(u.arcsec).value]).flatten())
            
            dra,ddec = jpl_coord.spherical_offsets_to(pix_coord)
            print((dra.to(u.arcsec)).value / 0.35 , " , ", ddec.to(u.arcsec).value / 0.35)
            print("-"*10)

        pix_coord_list=np.array(pix_coord_list)
        jpl_coord_list=np.array(jpl_coord_list)

        #print((pix_coord_list[:,0]-jpl_coord_list[:,0])/0.35)
        #print((pix_coord_list[:,1]-jpl_coord_list[:,1])/0.35)

        print(np.median(pix_coord_list[:,0]-jpl_coord_list[:,0])/0.35)
        print(np.median(pix_coord_list[:,1]-jpl_coord_list[:,1])/0.35)

        plt.plot(self.t,pix_coord_list[:,1]-jpl_coord_list[0,1],".-",label="Locked On Track")
        plt.plot(self.t,jpl_coord_list[:,1]-jpl_coord_list[0,1],".-",label="JPL Position Track")
        #plt.plot(self.t,(pix_coord_list[:,0]-jpl_coord_list[:,0]),".-",label="RA Offset")
        plt.legend()
        plt.show()

    def coord_sextractor(self,pic):
        data = np.copy(pic.cutout)

        sigma_clip = SigmaClip(sigma=5.0,maxiters=10)
        bkg_estimator = MedianBackground()
        bkg = Background2D(data, (20, 20), filter_size=(3, 3),
                            sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        
        data_bkgsub=data-bkg.background
        data_bkgsub[data_bkgsub<0]=0

        data = data_bkgsub

        pad = int(len(data[0]) / 10)
        mean, median, std = sigma_clipped_stats(data, sigma=5.0, maxiters=5)
        daofind = DAOStarFinder(fwhm=5,threshold=3 * std, min_separation=4)
        table=daofind.find_stars(data)

        coords_x=np.array(table["xcentroid"])
        coords_y=np.array(table["ycentroid"])
        points=np.stack((coords_x,coords_y),axis=-1)

        points=points[points[:,0] > pad]
        points=points[points[:,1] > pad]
        points=points[points[:,0] < (len(data[0])-pad)]
        points=points[points[:,1] < (len(data[0])-pad)]

        points[:,0] += pic.cutout_transform[0]
        points[:,1] += pic.cutout_transform[1]

        #apertures = CircularAperture(points, r=6)
        #plt.imshow(pic.img.data,origin="lower",norm=LogNorm())
        #apertures.plot(color='blue', lw=1.5, alpha=0.5)
        #plt.show()

        sky_points =  coords.SkyCoord.from_pixel(points[:,0],points[:,1],pic.img.wcs)
        points[:,0] = sky_points.ra.to(u.deg).value
        points[:,1] = sky_points.dec.to(u.deg).value

        return points

    def sextractor_check_predict(self,coords,pic):
        data = np.copy(pic.img.data)
        

        sigma_clip = SigmaClip(sigma=5.0,maxiters=10)
        bkg_estimator = MedianBackground()
        bkg = Background2D(data, (20, 20), filter_size=(3, 3),
                            sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        
        data_bkgsub=data-bkg.background
        data_bkgsub[data_bkgsub<0]=0

        data = data_bkgsub

        mean, median, std = sigma_clipped_stats(data, sigma=5.0, maxiters=5)
        daofind = DAOStarFinder(fwhm=5,threshold=1 * std, min_separation=4,xycoords=coords)
        table=daofind.find_stars(data)

        print(table)

        #print (table["flux"])
        #return table["flux"]
            
def composite_comet(frames):
    if (isinstance(frames[0],np.ndarray)):
        valid_frames=[]
        for i in range(0,len(frames)):
            if np.shape(frames[i]) == np.shape(frames[0]):
                valid_frames.append(frames[i])


        #for frame in frames:
            #frame = frame/np.median(frame)
        #avg=np.nansum(frames,axis=0)
        avg=np.median(valid_frames,axis=0)
        return avg
    else:
        print("ERROR, IMAGES ARE NOT NUMPY ARRAYS")
        return 0
    
def lock_comet(og_img,fwhm=4,cutoff=10000,*args,**kwargs):
    img=np.copy(og_img)
    img[img > cutoff] = 0
    #img=img-np.nanmedian(img) 
    img[img < 0] = 0
    result=fit_gauss(img,fwhm=fwhm).results
    center=(len(img[0])-1)/2

    jpl_offset=np.array([result["x_fit"][0],result["y_fit"][0]])
    #mark_target(jpl_offset,img)
    jpl_offset=jpl_offset-center
    return jpl_offset

def mark_target(pos,image_data,rad=0,*args,**kwargs):
    if rad==0:
        rad=1+len(image_data[0])/50
    aperture = CircularAperture(pos, r=rad)
    plt.imshow(image_data,origin="lower",cmap="grey",norm=LogNorm())
    aperture.plot(color="red",lw=1.5,alpha=0.5)
    plt.axis("scaled")
    plt.show()

def surf_brightness(pos,image_data,min=3,max=20,step_size=1,*args,**kwargs):
    sums=[]
    radii=[]
    for rad in range(min,max,step_size):
        #area=np.pi*rad*rad

        bkg_annulus = CircularAnnulus(pos, r_in = 1.1 * max, r_out = 1.5 * max)
        bkg_app_stats = ApertureStats(image_data,bkg_annulus)
        bkg_mean  = bkg_app_stats.mean

        aperture = CircularAperture(pos, r=rad)
        aperture_area = aperture.area_overlap(image_data)

        total_bkg=bkg_mean*aperture_area
        

        
        phot_table = aperture_photometry(image_data,aperture)
        sums.append(((phot_table["aperture_sum"].value[0])-total_bkg)/aperture_area)
        radii.append(np.sqrt(aperture_area/np.pi))

    return sums,radii

def bootstrap_error(data,iters):
    recalcs=[]
    for i in range(0,iters):
        samples=rng.choice(data,len(data))
        recalcs.append(np.median(samples))
    
    return np.std(recalcs)

def linear_error(err1,err2):
    return np.sqrt((err1*err1) + (err2*err2))