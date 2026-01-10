import calibration_tools as CT
import data_organisation_tools as DOT
import utility_functions as util

import pathlib
from ccdproc import ImageFileCollection
import ccdproc as ccdp
from astropy.nddata import CCDData
from astropy.io import fits
import matplotlib.pyplot as plt

import os
import numpy as np

class process_filter:
    def __init__(self,
                 filter,
                 input_path,
                 output_path,
                 ref_image,
                 exclude_tgts=[],
                 include_tgts=[],
                 make_bad_pixel_mask = True,
                 plate_solve = True,
                 fringe_correct = True,
                 fringe_exptime_limit=0,
                 *args,
                 **kwargs):

        self.filter=filter
        self.input_path=input_path
        self.output_path=output_path
        self.plate_solve=plate_solve
        self.fringe_correct=fringe_correct
        self.exptime_limit=fringe_exptime_limit
        all_names=util.report_names(input_path)

        if len(include_tgts)!=0:
            tgt_names=all_names[np.isin(all_names,include_tgts,invert=False)]
        else:
            tgt_names=all_names[np.isin(all_names,exclude_tgts,invert=True)]

        root_dir = pathlib.Path(__file__).resolve().parent
        self.px_scale=0.24

        trim_tags=["HIERARCH ESO DET OUT1 PRSCX","HIERARCH ESO DET OUT1 PRSCY","HIERARCH ESO DET OUT1 OVSCX","HIERARCH ESO DET OUT1 OVSCY"]
        ref_image=pathlib.Path(root_dir/"Data_set_1"/"block_1"/"BIAS"/"FREE"/"EFOSC.2009-01-27T21_00_47.752.fits")
        extra_clip=100


        all_fits_path = input_path
        calib_path = output_path

        self.trim=CT.set_trim(ref_image,extra_clip,*trim_tags)

        avg_bias = CT.create_master_bias(all_fits_path,calib_path)
        self.avg_bias = ccdp.trim_image(avg_bias[self.trim])

        self.avg_flat=CT.create_master_flat(filter,
                                    "ESO INS FILT1 NAME".lower(),
                                    "SKY,FLAT",
                                    self.avg_bias,
                                    self.trim,
                                    all_fits_path,
                                    calib_path)
        

        if make_bad_pixel_mask:
            self.bad_pixel_mask=CT.generate_bad_pixel_mask(all_fits_path,calib_path)
        else:
            self.bad_pixel_mask=CT.load_bad_pixel_mask(calib_path)


        

        filter_tgts=[]
        print("--- COMETS IN FILTER BAND ---")
        for tgt in tgt_names:
            if len(tgt)>10:
                continue
            elif "STD" in tgt:
                continue
                
            lights=ImageFileCollection(input_path,keywords='*',glob_exclude="bias_sub_*")
            criteria={'object' : tgt, "ESO INS FILT1 NAME".lower():filter}
            file_names=lights.files_filtered(**criteria)
            if len(file_names)>0:
                filter_tgts.append(tgt)
                print(tgt)
        print("-"*10)
        
        
        self.filter_tgts=filter_tgts

        

    def run(self):
        for tgt in self.filter_tgts:
            print("COMET: "+tgt)
            self.reduce_and_plate_solve(tgt,plate_solve=self.plate_solve)
            print("-"*10)

    def reduce_and_plate_solve(self,tgt_name,plate_solve=True,*args,**kwargs):
        all_fits_path=self.input_path
        calib_path=self.output_path

        lights=ImageFileCollection(all_fits_path,keywords='*',glob_exclude="bias_sub_*")
        lights.sort(["object","mjd-obs"])
        criteria={'object' : tgt_name, "ESO INS FILT1 NAME".lower():self.filter}
        tgt_lights=lights.files_filtered(**criteria)
        lights=lights.filter(**criteria)
        times=np.array(lights.summary["exptime"])

        for file,time in zip(tgt_lights,times):
            if time > self.exptime_limit  and self.fringe_correct==True:
                if ("i" in self.filter or "R" in self.filter or "V" in self.filter):
                    self.fringe_points,self.fringe_map = CT.load_fringe_data(calib_path,"fringe_points.txt",self.filter) #Load the fringe points and Map
                else:
                    self.fringe_map=None
                    self.fringe_points=None
            else:
                self.fringe_map=None
                self.fringe_points=None
            img=CT.reduce_img(pathlib.Path(all_fits_path/file),
                        calib_path,
                        self.trim,
                        self.avg_bias,
                        self.avg_flat,
                        tgt_name,
                        self.filter,
                        mask=self.bad_pixel_mask,
                        fringe_map=self.fringe_map,
                        fringe_points=self.fringe_points)
        if plate_solve:
            science=ImageFileCollection(calib_path,keywords='*',glob_include=tgt_name+"_"+self.filter+"*")
            science_files=science.files_filtered(**criteria)
            CT.batch_plate_solve(calib_path,
                            science_files,
                            self.px_scale,
                            self.bad_pixel_mask,
                            6,
                            8)

class fringe_correct_existing:
    def __init__(self,                 
                 filter,
                 calib_path,
                 exclude_tgts=[],
                 include_tgts=[],
                 *args,
                 **kwargs):

        self.filter=filter
        
        root_dir = pathlib.Path(__file__).resolve().parent
        self.px_scale=0.24

        #self.bad_pixel_mask=CT.load_bad_pixel_mask(calib_path)


        self.calib_path=calib_path        

    def run(self,time_thresh=0,report=False,*args,**kwargs):
        self.times=[]
        self.scales=[]
        self.ratios=[]
        if report:
            self.measure_fringe()
        else:
            self.correct_fringe(exptime_thresh=time_thresh)

    def correct_fringe(self,exptime_thresh=0,*args,**kwargs):
        lights=ImageFileCollection(self.calib_path,keywords='*')
        criteria={"ESO INS FILT1 NAME".lower():self.filter}
        tgt_lights=lights.files_filtered(**criteria)

        for file in tgt_lights:
            if len(file) < 20 or "MAP" in file:
                continue
            else:
                print(file)
                out_path=self.calib_path
                with fits.open(pathlib.Path(out_path/file)) as img:
                    exptime=img[0].header["exptime"]
                    print(exptime)
                if exptime < exptime_thresh:
                    print("SKIP!")
                    continue
                else:
                    tgt=CCDData.read(pathlib.Path(out_path/file))
                    self.fringe_points,self.fringe_map = CT.load_fringe_data(self.calib_path,"fringe_points.txt",self.filter) #Load the fringe points and Map
                    tgt.data = CT.fringe_correction(tgt, self.fringe_points, self.fringe_map)

                    out_path=pathlib.Path(out_path/file)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    tgt.write(str(out_path),overwrite=True)

    def measure_fringe(self):
        lights=ImageFileCollection(self.calib_path,keywords='*',glob_exclude="bias_sub_*")
        criteria={"ESO INS FILT1 NAME".lower():self.filter}
        tgt_lights=lights.files_filtered(**criteria)
        for file in tgt_lights:
            if len(file) < 20 or "MAP" in file:
                continue
            else:
                print(file)
                out_path=self.calib_path
                tgt=CCDData.read(pathlib.Path(out_path/file))
                self.fringe_points,self.fringe_map = CT.load_fringe_data(self.calib_path,"fringe_points.txt",self.filter) #Load the fringe points and Map
                scale,ratios = CT.fringe_correction(tgt, self.fringe_points, self.fringe_map,report_only=True)

                with fits.open(pathlib.Path(out_path/file)) as img:
                    exptime=img[0].header["exptime"]
                
                self.times.append(exptime)
                self.scales.append(scale)
                self.ratios.append(ratios)

    def plot_fringe_scales(self):
        plt.scatter(self.times,self.scales)
        plt.show()

    def plot_fringe_stats(self):
        ranges=[]
        for ratio_set,scale in zip(self.ratios,self.scales):
            range = np.std(ratio_set)
            ranges.append(range)
        plt.scatter(self.times,ranges)
        plt.show()
