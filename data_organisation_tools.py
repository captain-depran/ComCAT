import pathlib
import os
import shutil

import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from ccdproc import ImageFileCollection, Combiner, combine
import ccdproc as ccdp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm



def list_files(p):
    """
    Scans a folder and returns a list of the paths of all files within it

    INPUTS
    - p : (pathlib.Path) The folder to be scanned

    OUTPUTS
    - file_list : (list) A list of the paths of every file in the folder
    """
    file_list=[x for x in p.iterdir() if x.is_file()]
    return file_list

def list_folders(p):
    """
    Scans a folder and returns a list of the paths of all subfolders within it

    INPUTS
    - p : (pathlib.Path) The folder to be scanned

    OUTPUTS
    - folder_list : (list) A list of the paths of every subfolder in the folder
    """
    folder_list=[x for x in p.iterdir() if x.is_dir()]
    return folder_list


def copy_files(file,root,folder_name):
    """
    Utility function to copy a file to a constructed directory. Will create the folder path if none exists

    INPUTS
    - file : (pathlib.Path) The file path of the existing file
    - root : (pathlib.Path) The root folder for the dataset
    - folder_name : (string) The name of the destination folder, within the root
    """
    to_dir=pathlib.Path(root/folder_name/file.name)
    os.makedirs(os.path.dirname(to_dir), exist_ok=True)
    shutil.copy(file,to_dir)

def move_file(file,root,folder_name):
    """
    Utility function to move a file to a constructed directory. Will create the folder path if none exists

    INPUTS
    - file : (pathlib.Path) The file path of the existing file
    - root : (pathlib.Path) The root folder for the dataset
    - folder_name : (string) The name of the destination folder, within the root
    """
    to_dir=pathlib.Path(root/folder_name/file.name)
    os.makedirs(os.path.dirname(to_dir), exist_ok=True)
    shutil.move(file,to_dir)


def sort_block(block,block_n):
    """
    Takes a observational data block and copies it a unique folder

    INPUTS
    - block : (list) The list of files in the block
    - block_n : the numerical identifier for the block within the whole dataset
    """
    for file in block:
        copy_files(file,pathlib.Path(root_dir/"Data_set_1"),"block_"+str(block_n))

def extract_name(path):
    """
    Extract the name from a fits file
    """
    with fits.open(path) as img:
        object=(img[0].header['object'])
    return object

def dataset_split(full_set,threshold):
    """
    Splits a given dump of data into individual observation blocks, based on the time between observations being more than the passed threshold. 
    The result is N number of folders (where N is the number of blocks detected) in the original Data_set parent folder

    INPUTS
    - full_set : (list) the list of files in the data dump
    - threshold : (float) the number of hours to set the cutoff for seperate blocks

    """
    split_set=[]
    all_sets=[]
    last_time=0
    mjd_limit=threshold/24

    total_unsorted=len(full_set)

    for file in full_set:
        with fits.open(file) as hdul:
            current_time = hdul[0].header["mjd-obs"]
        if (current_time - last_time)>mjd_limit and last_time!=0:
            all_sets.append(split_set)
            split_set=[]
            split_set.append(file)
            last_time=0
        else:
            split_set.append(file)
            last_time = current_time
    all_sets.append(split_set)

    sorted_total=0

    for block,block_n in zip(all_sets,range(0,len(all_sets))):
        block_n+=1
        sort_block(block,block_n)
        sorted_total+=len(block)

    if np.abs(total_unsorted-sorted_total)!=0:
        print("ERROR IN BLOCK SORTING PROCESS - Pre/post-sort File amount mismatch")
    print("Sorting Done!")




def filter_sort(folder_path):
    """
    Scans a data set and sorts them into folders by filter

    INPUTS
    - folder_path : (pathlib.Path) The directory containing the unsorted files
    """
    all_files=list_files(folder_path)
    for file in all_files:
        with fits.open(file) as img:
            filter=(img[0].header['HIERARCH ESO INS FILT1 NAME'])
        move_file(file,folder_path,filter)
    
    
def object_sort(folder_path):
    """
    Scans a data set and sorts them into folders by observational object

    INPUTS
    - folder_path : (pathlib.Path) The directory containing the unsorted files
    """
    all_files=list_files(folder_path)
    for file in all_files:
        copy_files(file,folder_path,"ALL_FITS")
        with fits.open(file) as img:
            object=(img[0].header['object'])
        move_file(file,folder_path,object)


def filter_object_sort(block_parent_path):
    """
    Runs through a directory sorted into observational blocks, and per block, sorts by observational object, and within each object's folder, sorts by filter
    (Designed to run after the dataset_split() function)
    INPUTS
    - folder_path : (pathlib.Path) The directory containing the observation blocks
    """
    for block in list_folders(block_parent_path):
        if block.name=="unpacked_data":
            pass
        else:
            object_sort(pathlib.Path(block))
            for object in list_folders(block):
                if object.name=="ALL_FITS":
                    pass
                else:
                    filter_sort(object)



def create_master(img):
    print("CREATING MASTER "+img.type+" | BLOCK: "+img.block_path.name+" | OBJECT: "+img.object+" | FILTER: "+img.filter)
    all_files=list_files(img.block_path/img.object/img.filter)
    for file in all_files:
        if file.name==img.object+"_stacked.npy":
            pass
        else:
            with fits.open(file) as frame:
                if img.type=="FLAT":
                    raw=frame[0].data
                    scale=1/np.median(raw)
                    raw=raw*scale
                    img.all_frames.append(raw)
                else:
                    img.all_frames.append(frame[0].data)
    
    img.stacked=np.median(np.dstack(img.all_frames),axis=-1)
    np.save(img.block_path/img.object/img.filter/(img.object+"_stacked"),img.stacked)
    move_file(img.block_path/img.object/img.filter/(img.object+"_stacked.npy"),img.block_path/"CALIB MASTERS",img.filter)



class flat_field:
    def __init__(self,_object,_filter,_block_path):
        self.block_path=_block_path
        self.filter=_filter
        self.type = "FLAT"
        self.object=_object #(FLAT),(DOME),(SKY,FLAT)
        self.all_frames=[]
        self.stacked=np.empty((2,2))
        


class bias_frame:
    def __init__(self,_filter,_block_path):
        self.block_path=_block_path
        self.filter=_filter
        self.type="BIAS"
        self.object="BIAS"
        self.all_frames=[]
        self.stacked=np.empty((2,2))



class _reduced_image:
    """
    (LEGACY) Experimental manual frame reduction without ccdproc, using file names and paths to construct the correct collection of images. Basically redundant
    """
    def __init__(self,_raw_path):
        self.block_path=_raw_path.parent.parent.parent
        self.target=_raw_path.parent.parent.name
        self.filter=_raw_path.parent.name
        self.raw_path=_raw_path

        self.bias=np.load(self.block_path/"CALIB MASTERS"/"FREE"/"BIAS_STACKED.npy")
        self.flat=np.load(self.block_path/"CALIB MASTERS"/self.filter/"SKY,FLAT_STACKED.npy")

        with fits.open(self.raw_path) as frame:
                self.raw=frame[0].data

        self.processed=((self.raw-self.bias)/self.flat)


#My home made master flat and bias function
def create_np_frames(data_path):
    for block in list_folders(data_path):
        if block.name=="unpacked_data":
            pass
        else:
            for object in list_folders(block):
                for filter in list_folders(object):
                    if object.name=="FLAT" or object.name=="DOME" or object.name =="SKY,FLAT":
                        flat=flat_field(object.name,filter.name,block)
                        create_master(flat)
                    elif object.name=="BIAS":
                        bias=bias_frame(filter.name,block)
                        create_master(bias)


def organise_files(unsorted_dir):
    """
    Organisation of a data dump of .fits files. Will sort as follows:
        1) Split into observational blocks/nights
        2) Each block is sorted into 'object' subfolders, and a copy of all files into 'ALL_FITS' subfolder
        3) Each 'object' folder is sorted internally into filter subfolders
        4) Each filter folder will contain the actual .fits file
    The resultant path for a fits file will be: ".../ Data_set / observation block / imaged object / imaged filter / image.fits"
    This is done for convience for viewing files in DS9 or targetting specifics without needing to import the entire dump into python when searching for sub-group of images
    """
    data_set_dir=unsorted_dir.parent
    file_list=list_files(unsorted_dir)
    dataset_split(file_list,4)
    filter_object_sort(data_set_dir)

root_dir = pathlib.Path(__file__).resolve().parent
#dir=pathlib.Path(root_dir/"Data_set_1"/"unpacked_data")





"""
img=_reduced_image(root_dir/"Data_set_1"/"block_1"/"67P"/"B#639"/"EFOSC.2009-01-28T00_36_06.922.fits")
print(img.processed)
plt.imshow(img.flat,origin="lower")
plt.show()
"""

