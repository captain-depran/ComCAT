import data_organisation_tools as DOT
import calibration_tools as CT

import pathlib
from ccdproc import ImageFileCollection
import ccdproc as ccdp

root_dir = pathlib.Path(__file__).resolve().parent

px_scale=0.24
tgt_name="149P"
filter="R#642"

trim_tags=["HIERARCH ESO DET OUT1 PRSCX","HIERARCH ESO DET OUT1 PRSCY","HIERARCH ESO DET OUT1 OVSCX","HIERARCH ESO DET OUT1 OVSCY"]
ref_image=pathlib.Path(root_dir/"Data_set_1"/"block_1"/"BIAS"/"FREE"/"EFOSC.2009-01-27T21_00_47.752.fits")
extra_clip=100

all_fits_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS")
calib_path = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"ALL_FITS"/"PROCESSED FRAMES")

trim=CT.set_trim(ref_image,extra_clip,*trim_tags)

avg_bias = CT.create_master_bias(all_fits_path,calib_path)
avg_bias = ccdp.trim_image(avg_bias[trim])

avg_flat=CT.create_master_flat(filter,
                            "ESO INS FILT1 NAME".lower(),
                            "SKY,FLAT",
                            avg_bias,
                            trim,
                            all_fits_path,
                            calib_path)

#bad_pixel_mask=CT.generate_bad_pixel_mask(all_fits_path,calib_path)
bad_pixel_mask=CT.load_bad_pixel_mask(calib_path)

fringe_points,fringe_map = CT.load_fringe_data(calib_path,"fringe_points.txt",filter)

lights=ImageFileCollection(all_fits_path,keywords='*',glob_exclude="bias_sub_*")
lights.sort(["object","mjd-obs"])
criteria={'object' : tgt_name, "ESO INS FILT1 NAME".lower():filter}
tgt_lights=lights.files_filtered(**criteria)

for file in tgt_lights:
    img=CT.reduce_img(pathlib.Path(all_fits_path/file),
                calib_path,
                trim,
                avg_bias,
                avg_flat,
                tgt_name,
                filter,
                mask=bad_pixel_mask,
                fringe_map=None,
                fringe_points=None)
    

science=ImageFileCollection(calib_path,keywords='*',glob_include=tgt_name+"_"+filter+"*")
science_files=science.files_filtered(**criteria)
"""
CT.batch_plate_solve(calib_path,
                  science_files,
                  px_scale,
                  bad_pixel_mask,
                  6,
                  8)
"""
