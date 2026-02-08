import pipeline_commands as ComCAT
import calibration_tools as CT
import pathlib

filter="V#606"
block=5

block = str("block_"+str(block))
root_dir = pathlib.Path(__file__).resolve().parent
ref_image=pathlib.Path(root_dir/"Data_set_3"/"block_1"/"BIAS"/"FREE"/"EMMI.2007-05-13T20_18_04.811.FITS")
trim_tags=["HIERARCH ESO DET OUT1 PRSCX","HIERARCH ESO DET OUT1 PRSCY","HIERARCH ESO DET OUT1 OVSCX","HIERARCH ESO DET OUT1 OVSCY"]

all_fits_path = pathlib.Path(root_dir/"Data_set_3"/"block_5"/"ALL_FITS")
calib_path = pathlib.Path(root_dir/"Data_set_3"/"block_5"/"ALL_FITS"/"PROCESSED FRAMES")
trim=CT.set_trim(ref_image,*trim_tags,hdu_index="CCID20-14-5-3",xlow=10,xhigh=250,yhigh=200,ylow=200)


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
               'WAVE',
               'LAMPFLAT',
               'SKYFLAT']


#include_tgts=["P2004F3","2009AU16","29P","50P","74P","93P","94P","P113","P2005R2","48P","P29"]
#include_tgts=["93P","94P","P113"]

job = ComCAT.plate_solve_existing(calib_path,
                                  all_fits_path,
                                  0.35,
                                  filter,
                                  sext_fwhm=4,
                                  sext_thresh=12,
                                  filter_tag="ESO INS FILT2 NAME",
                                  exclude_tgts = excluded_tgts,
                                  retry_fails=False)
job.run()

"""
job=ComCAT.process_filter(filter,
                          all_fits_path,
                          calib_path,
                          ref_image,
                          trim=trim,
                          filter_tag="ESO INS FILT2 NAME",
                          data_hdu="CCID20-14-5-3",
                          px_scale=0.35,
                          flat_type="SKYFLAT",
                          exclude_tgts=excluded_tgts,
                          plate_solve=False,
                          solver_fwhm=4,
                          solver_thresh=12,
                          fringe_correct=False)

job.run()

"""
"""
job = ComCAT.fringe_correct_existing(filter,
                                     calib_path,
                                     include_tgts=include_tgts)

job.run(time_thresh=62)
"""
