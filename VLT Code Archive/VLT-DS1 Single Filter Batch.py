import pipeline_commands as ComCAT
import calibration_tools as CT
import pathlib

filter="R_SPECIAL"
block=1

block = str("block_"+str(block))
root_dir = pathlib.Path(__file__).resolve().parent
ref_image=pathlib.Path(root_dir/"Data_set_2"/"block_1"/"BIAS"/"FREE"/"FORS2.2009-01-23T13_02_59.381.FITS")
trim_tags=["HIERARCH ESO DET OUT1 PRSCX","HIERARCH ESO DET OUT1 PRSCY","HIERARCH ESO DET OUT1 OVSCX","HIERARCH ESO DET OUT1 OVSCY"]

all_fits_path = pathlib.Path(root_dir/"Data_set_2"/"block_1"/"ALL_FITS")
calib_path = pathlib.Path(root_dir/"Data_set_2"/"block_1"/"ALL_FITS"/"PROCESSED FRAMES")
trim=CT.set_trim(ref_image,*trim_tags,xlow=250,xhigh=250,yhigh=100)


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
               'FLAT,SKY',
               'WAVE']

excluded_tgts.extend(["130P",
               "132P",
               "138P",
               "139P",
               "149P",
               "152P"])

#include_tgts=["P2004F3","2009AU16","29P","50P","74P","93P","94P","P113","P2005R2","48P","P29"]
#include_tgts=["93P","94P","P113"]

job = ComCAT.plate_solve_existing(calib_path,
                                  all_fits_path,
                                  0.25,
                                  filter,
                                  sext_fwhm=4,
                                  sext_thresh=8,
                                  exclude_tgts = excluded_tgts,
                                  retry_fails=False)
job.run()

"""
job=ComCAT.process_filter(filter,
                          all_fits_path,
                          calib_path,
                          ref_image,
                          trim=trim,
                          px_scale=0.25,
                          exclude_tgts=excluded_tgts,
                          plate_solve=False,
                          fringe_correct=False)

job.run()
"""
"""

job = ComCAT.fringe_correct_existing(filter,
                                     calib_path,
                                     include_tgts=include_tgts)

job.run(time_thresh=62)
"""
