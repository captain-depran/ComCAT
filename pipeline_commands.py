import calibration_tools as CT
import data_organisation_tools as DOT
import utility_functions as util

import pathlib
from ccdproc import ImageFileCollection
import ccdproc as ccdp

import os
import numpy as np

class process_filter:
    def __init__(self,
                 filter,
                 input_path,
                 output_path,
                 ref_image,
                 exclude_tgts):

        self.filter=filter
        self.input_path=input_path
        self.output_path=output_path

        all_names=util.report_names(input_path)
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

        #bad_pixel_mask=CT.generate_bad_pixel_mask(all_fits_path,calib_path)
        self.bad_pixel_mask=CT.load_bad_pixel_mask(calib_path)

        filter_tgts=[]
        print("--- COMETS IN FILTER BAND ---")
        for tgt in tgt_names:
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
            self.reduce_and_plate_solve(tgt)
            print("-"*10)

    def reduce_and_plate_solve(self,tgt_name):
        all_fits_path=self.input_path
        calib_path=self.output_path
        

        lights=ImageFileCollection(all_fits_path,keywords='*',glob_exclude="bias_sub_*")
        lights.sort(["object","mjd-obs"])
        criteria={'object' : tgt_name, "ESO INS FILT1 NAME".lower():self.filter}
        tgt_lights=lights.files_filtered(**criteria)

        for file in tgt_lights:
            img=CT.reduce_img(pathlib.Path(all_fits_path/file),
                        calib_path,
                        self.trim,
                        self.avg_bias,
                        self.avg_flat,
                        tgt_name,
                        self.filter,
                        mask=self.bad_pixel_mask)
            

        science=ImageFileCollection(calib_path,keywords='*',glob_include=tgt_name+"_"+self.filter+"*")
        science_files=science.files_filtered(**criteria)
        CT.batch_plate_solve(calib_path,
                        science_files,
                        self.px_scale,
                        self.bad_pixel_mask,
                        6,
                        8)