# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 08:55:30 2020

@author: SHosseinitabatabaei
"""


import time
import datetime
import numpy as np
import os
import skimage.morphology
import sys
import scipy.io
import matplotlib.pyplot as plt
from skimage.morphology import flood_fill
from skimage import util 
from skimage.morphology import label
from skimage.morphology import remove_small_objects
from skimage.color.colorconv import convert_colorspace
from skimage.morphology import binary_dilation
from skimage.morphology import binary_erosion
from datetime import date
import tkinter as tk
import tkinter.filedialog as fld
from os.path import normpath, basename
import statistics
from skimage.morphology import (square, rectangle, diamond, disk, cube,
                                octahedron, ball, octagon, star) 
import pandas as pd
os.chdir("S:\\Projects_Mahdi\\Time-lapse\\Scripts\\final set of codes for the trial-26112020")
import aim_tl_gs_v6 as atl
import aim_reader_v1 as aim

import csv
#%%

results_list = []
# path = "V:\\1-Adult Phase 2b\\02_DATA GENERATED AT SHC\\3101_Montreal\\310101\\3Dreg\\time lapse radius"

# def time_lapse_batch(path):
start = time.time()
# ==================================
# Select the folder for the patient that contains the inputs and
# store the names and directories

# One test directory is here: "V:\\1-Adult Phase 2b\\02_DATA GENERATED AT SHC\\3101_Montreal\\310101\\3Dreg\\time lapse radius"
# You can try ano other directory from other patients
root = tk.Tk()
root.withdraw()
dirname = fld.askdirectory(parent=root,initialdir="/",title='Please select a directory')
root.destroy()
# dirname = path
all_files = os.listdir(dirname)
img_dir = "V:\\1-Adult Phase 2b\\Time lapse results for the trial\\TL QC image repository"

# Find the anatomical site using the folder name
if ("Rad" in dirname) or ("rad" in dirname):
    Site = "Radius"
elif ("Tib" in dirname) or ("tib" in dirname):
    Site = "Tibia"


# Only count the number of AIM files to determine how many pairs of scans exist
input_files = []
for i in all_files:
    if "AIM" in i:
        input_files.append(i)

# Replace the / with \\ so that python can read the file path
for i in range(len(input_files)):
    input_files[i] = dirname + "?" + input_files[i]
    # input_files[i] = dirname + "*\*\*" + input_files[i]
    tmp1 = input_files[i].replace("/","\*\*")
    tmp2 = tmp1.replace("*","")
    tmp3 = tmp2.replace("?","\\")
    input_files[i] = tmp3
    del tmp1, tmp2
    
# Check the number of inputs files and find how many scans are being registered...
    # Each scan has 4 files
if len(input_files)%4 == 0:
    n = int(len(input_files)/4)
else: 
    print("Incorrect number of files! Moving to the nex patient!")
    # return

if n == 0:
    print("No input files available! Moving to the next patient!")
    # return
    
# Read the first image to obtain patient ID from it's header
gray0,header_gray0 = aim.AIMreader(input_files[3])
Pat_ID = header_gray0["Patient_ID"]
del gray0, header_gray0

print("There are ", n, " scans available for patient",Pat_ID," at ",Site,"!")
now = datetime.datetime.now()
print ("Time of the analysis : ")
print (now.strftime("%Y-%m-%d %H:%M:%S"))
print("Input files are from: " + dirname)
# print(input_files)

# Find and sort the input files
# Formatted strings are used, so the code is scalable to any amount of pairs...
# without the need to write explicit codes for different numbers
cn = 0
for i in range(1,int(n+1)):      
# Initialize names to be iterated as needed
    init_namegs = "aim_gray{}"
    nm_strgs = init_namegs.format(i)
    vars()[nm_strgs] = input_files[cn+3]
    
    init_namect = "aim_cort{}"
    nm_strct = init_namect.format(i)
    vars()[nm_strct] = input_files[cn]
    
    init_nametb = "aim_trab{}"
    nm_strtb = init_nametb.format(i)
    vars()[nm_strtb] = input_files[cn+2]
    
    init_namesg = "aim_seg{}"
    nm_strsg= init_namesg.format(i)
    vars()[nm_strsg] = input_files[cn+1]

# Identify scanner generation to assign proper density and cluster size thresholds
    if i == 1:
        gray1,header_gray1 = aim.AIMreader(vars()[nm_strgs])
        gen = 1 if int(header_gray1["XCT_gen"]) == 1 else  2
        th = 275 if int(header_gray1["XCT_gen"]) == 1 else  300
        cl = 15 if int(header_gray1["XCT_gen"]) == 1 else  10
        
        del gray1, header_gray1
    cn += 4

# Estimate cpu time
if (gen == 1) and (Site == "Radius"):
    print("The analysis will take about ",(n-1)*5," minutes! ")
elif (gen == 2) and (Site == "Radius"):
    print("The analysis will take about ",(n-1)*15," minutes! ")
elif (gen == 1) and (Site == "Tibia"):
    print("The analysis will take about ",(n-1)*14," minutes! ")
elif (gen == 2) and (Site == "Tibia"):
    print("The analysis will take about ",(n-1)*20," minutes! ")


#%%

# Assign variable names using formatted strings to be scalable
for i in range(2,int(n+1)):

    init_nameov = "overlay1{}"
    nameov = init_nameov.format(i)
    init_namefd = "formed_denoised1{}"
    namefd = init_namefd.format(i)
    init_namerd = "resorbed_denoised1{}"
    namerd = init_namerd.format(i)
    init_namemd = "mineralized_denoised1{}"
    namemd = init_namemd.format(i)
    init_namedd = "demineralized_denoised1{}"
    namedd = init_namedd.format(i)
    init_nameci = "combined_img1{}"
    nameci = init_nameci.format(i)
    init_names1 = "seg1_common1{}"
    names1 = init_names1.format(i)
    init_names2 = "seg{}_common1{}"
    names2 = init_names2.format(i,i)
    init_namec1 = "cort1_common1{}"
    namec1 = init_namec1.format(i)
    init_namec2 = "cort{}_common1{}"
    namec2 = init_namec2.format(i,i)
    init_namet1 = "trab1_common1{}"
    namet1 = init_namet1.format(i)
    init_namet2 = "trab{}_common1{}"
    namet2 = init_namet2.format(i,i)
    init_ndays = "n_days1{}"
    ndays = init_ndays.format(i)
    
    init_gs1 = "aim_gray{}"
    ings1 = init_gs1.format(1)
    init_ct1 = "aim_cort{}"
    inct1 = init_ct1.format(1)
    init_tb1 = "aim_trab{}"
    intb1 = init_tb1.format(1)
    init_sg1 = "aim_seg{}"
    insg1 = init_sg1.format(1)
    
    init_gs2 = "aim_gray{}"
    ings2 = init_gs2.format(i)
    init_ct2 = "aim_cort{}"
    inct2 = init_ct2.format(i)
    init_tb2 = "aim_trab{}"
    intb2 = init_tb2.format(i)
    init_sg2 = "aim_seg{}"
    insg2 = init_sg2.format(i)

    init_plot = "overlay_1{}"
    inplot = init_plot.format(i)
    
    init_save = "1{}"
    save_n = init_save.format(i)

# Run the TL analysis
    vars()[nameov], vars()[namefd], vars()[namerd], vars()[namemd], vars()[namedd], vars()[nameci], vars()[names1],\
        vars()[names2], vars()[namec1], vars()[namec2], vars()[namet1], vars()[namet2],voxel_size,vars()[ndays],Patient_ID\
            = atl.aim4time_lapse(vars()[ings1],vars()[ings2],vars()[insg1],vars()[insg2],vars()[inct1],vars()[inct2],vars()[intb1],vars()[intb2],thresh = th, cluster = cl, compartment = "Both")

    # atl.plot_array(vars()[nameov],img_dir,Patient_ID,Site,inplot,thresh=th,cluster=cl,slice_n=40,mode="nipy_spectral")
#%%

# Calculate TL outcomes
    # First, using simple method by Mahdi:
    FBV = np.count_nonzero(vars()[namefd])    # formed volume
    RBV = np.count_nonzero(vars()[namerd])    # resorbed volume
    BSLBV = np.count_nonzero(vars()[names1])  # baseline segmented volume
    FBVTV = FBV*100/BSLBV
    RBVTV = RBV*100/BSLBV
    
    form_dil = np.zeros_like(vars()[namefd])
    resorb_dil = np.zeros_like(vars()[namefd])
    seg1_erod = np.zeros_like(vars()[namefd])
    seg2_erod = np.zeros_like(vars()[namefd])
    
    # Dilate formed/resorbed voxels using disk with radius 1
    # Erode seg1 and seg2
    struc_2d = {"disk": disk(1)}
    for i in range(np.shape(vars()[namefd])[2]):
        form_dil[:,:,i] = binary_dilation(vars()[namefd][:,:,i],struc_2d["disk"])
        resorb_dil[:,:,i] = binary_dilation(vars()[namerd][:,:,i],struc_2d["disk"])
        seg1_erod[:,:,i] = binary_erosion(vars()[names1][:,:,i],struc_2d["disk"])
        seg2_erod[:,:,i] = binary_erosion(vars()[names2][:,:,i],struc_2d["disk"])
    # Get the dilated surface of formed/resorbed and surface of seg and seg2
    surf_form = form_dil - vars()[namefd]
    surf_resorb = resorb_dil - vars()[namerd]
    surf_seg1 = - seg1_erod + vars()[names1]
    surf_seg2 = - seg2_erod + vars()[names2]
    
    # Find the intersect between dilated formed/resorbed surfaces and surface of seg1 and seg2
    form_intersect = surf_form * surf_seg1
    resorb_intersect = surf_resorb * surf_seg2
        
    FBS = np.count_nonzero(form_intersect)
    RBS = np.count_nonzero(resorb_intersect)
    BSLBS = np.count_nonzero(surf_seg1)
    FBSTS = FBS*100/BSLBS
    RBSTS = RBS*100/BSLBS
    
    
    #%% MATLAB method for calculations
    # First, create the union of constant, formed, and resorbed bones
    cortical_both = np.add(vars()[names1],vars()[namefd])
    
    Image_surf_labels = np.zeros_like(cortical_both)
    Image_without_surface = np.zeros_like(cortical_both)
    Image_labels = np.zeros_like(cortical_both)
    
    # Constant bone is seg1 minus resorbed bone
    constant_bone = np.subtract(vars()[names1],vars()[namerd])
    
    # Create the surface of the union (2 layers)
    struc_2d = {"disk": disk(1)}
    for i in range(np.shape(cortical_both)[2]):
        Image_without_surface[:,:,i] = binary_erosion(cortical_both[:,:,i],struc_2d["disk"])
        Image_labels[:,:,i] = binary_erosion(Image_without_surface[:,:,i],struc_2d["disk"])
        
    Image_surf_labels = cortical_both - Image_without_surface
    Image_labels = Image_without_surface - Image_labels
    
    constant_bone_volume = Image_without_surface*constant_bone
    
    Image_surf_constant = np.zeros_like(constant_bone_volume)
    
    # surface area of constant bone, grow 1, check if it´s newly formed
    for i in range(np.shape(constant_bone_volume)[2]):
        Image_surf_constant[:,:,i] = binary_dilation(constant_bone_volume[:,:,i],struc_2d["disk"])
        
    Image_surf_constant = Image_surf_constant - constant_bone_volume
    
    # Finally, crete formation and resorption surfaces
    Formed_surface = np.zeros_like(Image_surf_constant)
    Resorbed_surface = np.zeros_like(Image_surf_constant)
    
    Formed_surface = vars()[namefd] * Image_surf_constant
    Resorbed_surface = vars()[namerd] * Image_labels
    Constant_surface = constant_bone * Image_labels
    
    # Get amount of surface voxels
    FBS_mat = np.count_nonzero(Formed_surface)
    RBS_mat = np.count_nonzero(Resorbed_surface)
    Constant_mat = np.count_nonzero(Constant_surface)
    total_surface = FBS_mat + RBS_mat + Constant_mat
    FBSTS_mat = FBS_mat*100/total_surface
    RBSTS_mat = RBS_mat*100/total_surface
    
    # Volume calculations
    constant_bone_voxels_all_logic = np.zeros_like(constant_bone_volume)
    formed_bone_voxels_all_logic = np.zeros_like(constant_bone_volume)
    resorbed_bone_voxels_all_logic = np.zeros_like(constant_bone_volume)
    
    # calculate volumes
    constant_bone_voxels_all_logic = Image_without_surface * constant_bone
    formed_bone_voxels_all_logic = Image_without_surface * vars()[namefd]
    resorbed_bone_voxels_all_logic = Image_without_surface * vars()[namerd]
    
    # number of voxel in the total volume without the surface
    constant_bone_voxels_all = np.count_nonzero(constant_bone_voxels_all_logic)
    formed_bone_voxels_all = np.count_nonzero(formed_bone_voxels_all_logic)
    resorbed_bone_voxels_all = np.count_nonzero(resorbed_bone_voxels_all_logic)
    
    # number of voxel in the total volume with the surface
    constant_bone_voxels = constant_bone_voxels_all + Constant_mat*0.5;
    formed_bone_voxels = formed_bone_voxels_all + FBS_mat*0.5;
    resorbed_bone_voxels = resorbed_bone_voxels_all + RBS_mat*0.5;
    
    total_volume = constant_bone_voxels + resorbed_bone_voxels
    
    FBVTV_mat = formed_bone_voxels*100/total_volume
    RBVTV_mat = resorbed_bone_voxels*100/total_volume

#%%
    # plt.imshow(vars()[names1][:,:,50], cmap="gray"),plt.show()
    # plt.imshow(vars()[names2][:,:,50], cmap="gray"),plt.show()
    # plt.imshow(vars()[namefd][:,:,50], cmap="gray"),plt.show()
    # plt.imshow(vars()[namerd][:,:,50], cmap="gray"),plt.show()
    # plt.imshow(surf_seg1[:,:,50], cmap="gray"),plt.show()
    # plt.imshow(surf_seg2[:,:,50], cmap="gray"),plt.show()
    # plt.imshow(surf_form[:,:,50], cmap="gray"),plt.show()
    # plt.imshow(surf_resorb[:,:,50], cmap="gray"),plt.show()
    
    # scipy.io.savemat('S:\\Projects_Mahdi\\Time-lapse\\sample_images\\mat_inputs\\seg1_310101.mat', {"seg1_310101":np.uint8(vars()[names1])}, oned_as='row')
    # scipy.io.savemat('S:\\Projects_Mahdi\\Time-lapse\\sample_images\\mat_inputs\\form_310101.mat', {"form_310101":np.uint8(vars()[namefd])}, oned_as='row')
    
# Append results to the results list
    res_tmp = Patient_ID,save_n,Site,FBVTV,RBVTV,FBSTS,RBSTS,FBVTV_mat,RBVTV_mat,FBSTS_mat,RBSTS_mat
    
    results_list.append(res_tmp)
    # return res_tmp
    # plt.imshow(vars()[names1][:,:,40], cmap="gray"),plt.show()
    # plt.imshow(vars()[names2][:,:,40], cmap="gray"),plt.show()
    # plt.imshow(vars()[namefd][:,:,40], cmap="gray"),plt.show()
    # plt.imshow(vars()[namerd][:,:,40], cmap="gray"),plt.show()
    # plt.imshow(surf_form[:,:,40], cmap="gray"),plt.show()
    # plt.imshow(surf_resorb[:,:,40], cmap="gray"),plt.show()
    # plt.imshow(surf_seg1[:,:,40], cmap="gray"),plt.show()
    # plt.imshow(surf_seg2[:,:,40], cmap="gray"),plt.show()
    
    # plt.imshow(cortical_both[:,:,40], cmap="gray"),plt.show()
    # plt.imshow(Image_surf_labels[:,:,40], cmap="gray"),plt.show()
    # plt.imshow(Image_without_surface[:,:,40], cmap="gray"),plt.show()
    # plt.imshow(Image_labels[:,:,40], cmap="gray"),plt.show()
    # plt.imshow(constant_bone_volume[:,:,40], cmap="gray"),plt.show()
    # plt.imshow(Image_surf_constant[:,:,40], cmap="gray"),plt.show()
    
    # plt.imshow(form_intersect[:,:,40], cmap="gray"),plt.show()
    # plt.imshow(resorb_intersect[:,:,40], cmap="gray"),plt.show()
    # plt.imshow(Formed_surface[:,:,40], cmap="gray"),plt.show()
    # plt.imshow(Resorbed_surface[:,:,40], cmap="gray"),plt.show()


    # wb = op.load_workbook('V:\\1-Adult Phase 2b\\20-ICON data transfer\\24-month results - in progress\\time-lapse-results.xlsx')

#     with open("V:\\1-Adult Phase 2b\\Time lapse results for the trial\\time-lapse-results-left.csv", "a", newline='') as fp:
#         wr = csv.writer(fp, dialect='excel')
#         wr.writerow(res_tmp)
# os.chdir(path)
# f= open("check.txt","w+")

elapsed_time = (time.time() - start)
print("Patient done! Elapsed time:", int(elapsed_time), "seconds or",\
      round(int(elapsed_time)/60,1), "minutes")
    #======================================
        
# path = "V:\\1-Adult Phase 2b\\02_DATA GENERATED AT SHC\\3101_Montreal\\310101\\3Dreg\\time lapse radius"

# results = time_lapse_batch(path)