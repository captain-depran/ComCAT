import pipeline_commands as ComCAT
import pathlib

filter="i#705"

root_dir = pathlib.Path(__file__).resolve().parent
all_fits_path = pathlib.Path(root_dir/"Data_set_1"/"block_3"/"ALL_FITS")
calib_path = pathlib.Path(root_dir/"Data_set_1"/"block_3"/"ALL_FITS"/"PROCESSED FRAMES")

ref_image = pathlib.Path(root_dir/"Data_set_1"/"block_1"/"BIAS"/"FREE"/"EFOSC.2009-01-27T21_00_47.752.fits")

excluded_tgts=['17P_wht_nt1_cal_seq',
               '17P_wht_nt3_cal_seq',
               '29P_wht_nt2_cal_seq',
               '29P_wht_nt3_cal_seq',
               '29Psky',
               '50P_wht_nt2_cal_seq',
               '50P_wht_nt3_cal_seq',
               'BIAS',
               'DOME',
               'FLAT',
               'HILT600',
               'LTT3218',
               'OTHER',
               'PG0918+029',
               'PG1047+003',
               'PG1323-086',
               'SA95-107',
               'SA98-642',
               'SKY,FLAT',
               'WAVE']

include_tgts=["149P","P2004F3","2009AU16","29P","50P","74P","93P","94P","P113","P2005R2","48P"]
#include_tgts=["29P","67P"]

job=ComCAT.process_filter(filter,
                          all_fits_path,
                          calib_path,
                          ref_image,
                          include_tgts=include_tgts)

job.run()