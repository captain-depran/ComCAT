import data_organisation_tools as DOT
import calibration_tools as CT

import pathlib
from ccdproc import ImageFileCollection
import ccdproc as ccdp

root_dir = pathlib.Path(__file__).resolve().parent

px_scale=0.24

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

bad_pixel_mask=CT.generate_bad_pixel_mask(all_fits_path,calib_path)
#bad_pixel_mask=CT.load_bad_pixel_mask(calib_path)

lights=ImageFileCollection(all_fits_path,keywords='*',glob_exclude="bias_sub_*")
lights.sort(["object","mjd-obs"])
criteria={"ESO INS FILT1 NAME".lower():filter}
tgt_lights=lights.files_filtered(**criteria)


tgts=[]
black_list=["DOME","SKY,FLAT","FLAT","BIAS","WAVE","29Psky","OTHER"]

for file in tgt_lights:
    name=DOT.extract_name(pathlib.Path(all_fits_path/file))
    if name in tgts:
        pass
    #elif "seq" in name:
        #pass
    elif name in black_list:
        pass
    elif "STD" in name:
        pass
    else:
        tgts.append(name)
        print(name)
        img=CT.reduce_img(pathlib.Path(all_fits_path/file),
                    calib_path,
                    trim,
                    avg_bias,
                    avg_flat,
                    str("fringe_comp_"+name),
                    filter,
                    mask=bad_pixel_mask)

print("FILES USED FOR FRINGE MAP: ",len(tgts))

fringe=CT.make_fringe_map(calib_path,filter,bad_pixel_mask)


CT.show_image(fringe,log_plt=True)

