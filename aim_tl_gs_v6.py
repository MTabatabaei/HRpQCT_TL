# -*- coding: utf-8 -*-
"""
Developed by Mahdi Tabatabaei - Shriners Hospital for Children, McGill University, Monreal
mahdi.tabatabaei@mail.mcgill.ca
"""
import time
start = time.time()
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import flood_fill
from skimage import util 
from skimage.morphology import label
from skimage.morphology import remove_small_objects
from skimage.morphology import binary_dilation
from skimage.morphology import binary_erosion
from skimage.morphology import (square, rectangle, diamond, disk, cube,
                                octahedron, ball, octagon, star)
from datetime import date
import aim_reader_v1 as aim



def aim4time_lapse(gray1,gray2,seg1,seg2,cort1,cort2,trab1,trab2,thresh = 225, cluster = 5, compartment = "Both"):
    """
    > function for creating volumes of bone formation, resorption, and constant bone

    Args:
        path to the AIM files of grayscale images, segmented images, cortical masks
        and trabecular masks for two scans to be overlayed
        Compartment: The bone compartment to be analyzed (Cortical, Trabecular, or Both)
    """
    # ==========================================================================
    # Reading the AIM files as numpy arrays and their headers
    gray1,header_gray1 = aim.aim2np(gray1)
    gray2,header_gray2 = aim.aim2np(gray2)
    seg1,header_seg1 = aim.aim2np(seg1)
    seg2,header_seg2 = aim.aim2np(seg2)
    cort1,header_cort1 = aim.aim2np(cort1)
    cort2,header_cort2 = aim.aim2np(cort2)
    trab1,header_trab1 = aim.aim2np(trab1)
    trab2,header_trab2 = aim.aim2np(trab2)
    
    nps = [gray1,gray2,seg1,seg2,cort1,cort2,trab1,trab2]
    headers = [header_gray1,header_gray2,header_seg1,header_seg2,header_cort1,\
               header_cort2,header_trab1,header_trab2]
    
    print("       Calculations start now!")
    
    if header_gray1["Patient_ID"] != header_gray2["Patient_ID"]:
        print("Error! The patient ID does not match between the two scans!")
        # return
    Patient_ID = header_gray1["Patient_ID"]
    # ===========================================================================
    # Finding the number of days between scans. This will swap scans if they
    # are not ordered correctly according to their dates
    n_days = date(int(header_gray2["Date"][2]),int(header_gray2["Date"][1]),int(header_gray2["Date"][0]))\
        - date(int(header_gray1["Date"][2]),int(header_gray1["Date"][1]),int(header_gray1["Date"][0]))
    n_days = n_days.days
    if n_days == 0: n_days = 1
    elif n_days < 0:
        gray1,header_gray1 = aim.aim2np(gray2)
        gray2,header_gray2 = aim.aim2np(gray1)
        seg1,header_seg1 = aim.aim2np(seg2)
        seg2,header_seg2 = aim.aim2np(seg1)
        cort1,header_cort1 = aim.aim2np(cort2)
        cort2,header_cort2 = aim.aim2np(cort1)
        trab1,header_trab1 = aim.aim2np(trab2)
        trab2,header_trab2 = aim.aim2np(trab1)
        
        nps = [gray1,gray2,seg1,seg2,cort1,cort2,trab1,trab2]
        headers = [header_gray1,header_gray2,header_seg1,header_seg2,header_cort1,\
                   header_cort2,header_trab1,header_trab2]
        
    
    # ===========================================================================
    # Determine the voxel size to be used for calculations
    voxel_size = 82 if int(header_gray1["XCT_gen"]) == 1 else  60.7
    # =============================================================================
    # Finding the smallest 3D arrays that fits all of the images and aligning them in x-y plane
    order = ["gray1","gray2","seg1","seg2","cort1","cort2","trab1","trab2"]
    order_row = order[:]
    order_col = order[:]
    order_z = order[:]
    # Getting 3D coordinte of each image in the global coordinate system
    rows = [int(headers[i]["coordinates"][1]) for i in range(len(headers))]
    columns = [int(headers[i]["coordinates"][0]) for i in range(len(headers))]
    zs = [int(headers[i]["coordinates"][2]) for i in range(len(headers))]
    
    # Finding the highest value of x, y, and z coordinates
    rows_max = [int(headers[i]["coordinates"][1]+headers[i]["dimensions"][1])\
                for i in range(len(headers))]
    columns_max = [int(headers[i]["coordinates"][0]+headers[i]["dimensions"][0])\
                   for i in range(len(headers))]
    zs_max = [int(headers[i]["coordinates"][2]+headers[i]["dimensions"][2])\
              for i in range(len(headers))]
        
    # Sort the images based on their locations in x, y, and z axes
    tuples = zip(*sorted(zip(rows, order_row)))
    rows, order_row = [ list(tuple) for tuple in  tuples]
    tuples = zip(*sorted(zip(columns, order_col)))
    columns, order_col = [ list(tuple) for tuple in  tuples]
    tuples = zip(*sorted(zip(zs, order_z)))
    zs, order_z = [ list(tuple) for tuple in  tuples]
    
    row_c = max(rows_max) - rows[0]
    col_c = max(columns_max) - columns[0]
    z_c = max(zs_max) - zs[0]
    
    rows[:] = [elem - rows[0] for elem in rows]
    columns[:] = [elem - columns[0] for elem in columns]
    zs[:] = [elem - zs[0] for elem in zs]
    
    # Create zero numpy arrays with the common size for all images
    gray1_common = np.zeros((row_c,col_c,z_c), dtype="int16")
    gray2_common = np.zeros((row_c,col_c,z_c), dtype="int16")
    seg1_common = np.zeros((row_c,col_c,z_c), dtype="uint8")
    seg2_common = np.zeros((row_c,col_c,z_c), dtype="uint8")
    cort1_common = np.zeros((row_c,col_c,z_c), dtype="uint8")
    cort2_common = np.zeros((row_c,col_c,z_c), dtype="uint8")
    trab1_common = np.zeros((row_c,col_c,z_c), dtype="uint8")
    trab2_common = np.zeros((row_c,col_c,z_c), dtype="uint8")
    
    # Map the images to the common array
    gray1_common[rows[order_row.index("gray1")]:(rows[order_row.index("gray1")]+headers[0]["dimensions"][1])\
          ,columns[order_col.index("gray1")]:(columns[order_col.index("gray1")]+headers[0]["dimensions"][0])\
              ,zs[order_z.index("gray1")]:zs[order_z.index("gray1")]+headers[0]["dimensions"][2]] = np.int16(gray1)
        
    gray2_common[rows[order_row.index("gray2")]:(rows[order_row.index("gray2")]+headers[1]["dimensions"][1])\
          ,columns[order_col.index("gray2")]:(columns[order_col.index("gray2")]+headers[1]["dimensions"][0])\
              ,zs[order_z.index("gray2")]:zs[order_z.index("gray2")]+headers[1]["dimensions"][2]] = np.int16(gray2)
        
    seg1_common[rows[order_row.index("seg1")]:(rows[order_row.index("seg1")]+headers[2]["dimensions"][1])\
          ,columns[order_col.index("seg1")]:(columns[order_col.index("seg1")]+headers[2]["dimensions"][0])\
              ,zs[order_z.index("seg1")]:zs[order_z.index("seg1")]+headers[2]["dimensions"][2]] = np.uint(seg1)
        
    seg2_common[rows[order_row.index("seg2")]:(rows[order_row.index("seg2")]+headers[3]["dimensions"][1])\
          ,columns[order_col.index("seg2")]:(columns[order_col.index("seg2")]+headers[3]["dimensions"][0])\
              ,zs[order_z.index("seg2")]:zs[order_z.index("seg2")]+headers[3]["dimensions"][2]] = np.uint8(seg2)
        
    cort1_common[rows[order_row.index("cort1")]:(rows[order_row.index("cort1")]+headers[4]["dimensions"][1])\
          ,columns[order_col.index("cort1")]:(columns[order_col.index("cort1")]+headers[4]["dimensions"][0])\
              ,zs[order_z.index("cort1")]:zs[order_z.index("cort1")]+headers[4]["dimensions"][2]] = np.uint8(cort1)
        
    cort2_common[rows[order_row.index("cort2")]:(rows[order_row.index("cort2")]+headers[5]["dimensions"][1])\
          ,columns[order_col.index("cort2")]:(columns[order_col.index("cort2")]+headers[5]["dimensions"][0])\
              ,zs[order_z.index("cort2")]:zs[order_z.index("cort2")]+headers[5]["dimensions"][2]] = np.uint8(cort2)
        
    trab1_common[rows[order_row.index("trab1")]:(rows[order_row.index("trab1")]+headers[6]["dimensions"][1])\
          ,columns[order_col.index("trab1")]:(columns[order_col.index("trab1")]+headers[6]["dimensions"][0])\
              ,zs[order_z.index("trab1")]:zs[order_z.index("trab1")]+headers[6]["dimensions"][2]] = np.uint8(trab1)
        
    trab2_common[rows[order_row.index("trab2")]:(rows[order_row.index("trab2")]+headers[7]["dimensions"][1])\
          ,columns[order_col.index("trab2")]:(columns[order_col.index("trab2")]+headers[7]["dimensions"][0])\
              ,zs[order_z.index("trab2")]:zs[order_z.index("trab2")]+headers[7]["dimensions"][2]] = np.uint8(trab2)

    # =====================================================================
    # Masking the grayscale images based on the compartment to be analyzed
    for a,s in np.nditer([cort1_common,trab1_common],op_flags=['readwrite']):  # make sure that the cort and trab masks dont overlap due to rotation
        if a != 0 and s != 0:
            a[...] = 0
            
    for a,s in np.nditer([cort2_common,trab2_common],op_flags=['readwrite']):  # make sure that the cort and trab masks dont overlap due to rotation
        if a != 0 and s != 0:
            a[...] = 0
    
    cort1_common[np.where(cort1_common != 0)] = 1
    cort2_common[np.where(cort2_common != 0)] = 1
    trab1_common[np.where(trab1_common != 0)] = 1 
    trab2_common[np.where(trab2_common != 0)] = 1    
    seg1_common[np.where(seg1_common != 0)] = 1    
    seg2_common[np.where(seg2_common != 0)] = 1
    
    if compartment == "Both":
        mask1_common = cort1_common + trab1_common
        mask2_common = cort2_common + trab2_common
        mask1_common = util.invert(flood_fill(mask1_common, (1, 1, 1), 1)) + 2 + mask1_common
        # mask1_common[np.where(mask1_common != 0)] = 1
        struc_2d = {"disk": disk(4)}
        for i in range(np.shape(mask1_common)[2]):
            mask1_common[:,:,i] = binary_dilation(mask1_common[:,:,i],struc_2d["disk"])
        # mask2_common = util.invert(flood_fill(mask2_common, (1, 1, 1), 127)) + 128 + mask2_common
        # mask2_common[np.where(mask2_common != 0)] = 1
        # struc_2d = {"disk(7)": disk(6)}
        # for i in range(np.shape(mask2_common)[2]):
        #     mask2_common[:,:,i] = binary_dilation(mask2_common[:,:,i],struc_2d["disk(7)"])
    elif compartment == "Cortical":
        mask1_common = cort1_common
        mask1_common[np.where(mask1_common != 0)] = 1
        mask2_common = cort2_common
        mask2_common[np.where(mask2_common != 0)] = 1
    elif compartment == "Trabecular":
        mask1_common = trab1_common
        mask1_common[np.where(mask1_common != 0)] = 1
        mask2_common = trab2_common
        mask2_common[np.where(mask2_common != 0)] = 1

    gray1_common = (mask1_common) * gray1_common
    gray2_common = (mask1_common) * gray2_common

    # ========================================================================
    # Converting gray native values to bone density mgHA/cm3
    if (header_gray1["slope"] == header_gray2["slope"]) and (header_gray1["intercept"] == header_gray2["intercept"]):
        print("       Both scans have the same calibration!")
        threshold = thresh*float(header_gray1["Mu_scaling"])/header_gray1["slope"]
        histomorph = np.zeros_like(gray1_common)  
        tmp = gray1_common - gray2_common
        histomorph[np.where(tmp > threshold)] = 500
        histomorph[np.where(tmp < -threshold)] = 300
        check = ((header_gray1["Mu_scaling"])*(-header_gray1["intercept"]+20))/header_gray1["slope"]
        # histomorph[np.where(gray1_common < check) and np.where(gray2_common < check)] = 0
    
        x, y, z = np.shape(gray1_common)
        for i in range(x):
            for j in range(y):
                for k in range (z):
                    if gray1_common[i,j,k] < check and gray2_common[i,j,k] < check:
                        histomorph[i,j,k] == 0                                                                                                       
        # for a,s,p in np.nditer([histomorph,gray1_common,gray2_common],op_flags=['readwrite']):
        #     if s < check and p < check:
        #         a[...] = 0
    else:
        print("       Scans have different calibrations!")
        x, y, z = np.shape(gray1_common)
        for i in range(x):
            for j in range(y):
                for k in range (z):
                    gray1_common[i,j,k] = gray1_common[i,j,k]/(float(header_gray1["Mu_scaling"]))*header_gray1["slope"] + header_gray1["intercept"]
                    gray2_common[i,j,k] = gray2_common[i,j,k]/(float(header_gray2["Mu_scaling"]))*header_gray2["slope"] + header_gray2["intercept"]
        tmp = gray1_common - gray2_common
        histomorph = np.zeros_like(gray1_common)  
        histomorph[np.where(tmp > thresh)] = 500
        histomorph[np.where(tmp < -thresh)] = 300
        
        for a,s,p in np.nditer([histomorph,gray1_common,gray2_common],op_flags=['readwrite']):
            if s < 20 and p < 20:
                a[...] = 0
    # ===========================================================
    # Trimming the extra empty padding in all axes
    check_0 = ~(seg1_common==0).all((1,2))
    check_1 = ~(seg1_common==0).all((0,2))
    check_2 = ~(seg1_common==0).all((0,1))
    array_range_0 = np.where(check_0 == True)
    array_range_1 = np.where(check_1 == True)
    array_range_2 = np.where(check_2 == True)
    start0 = array_range_0[0][0]
    stop0 = array_range_0[0][-1]
    start1 = array_range_1[0][0]
    stop1 = array_range_1[0][-1]
    start2 = array_range_2[0][0]
    stop2 = array_range_2[0][-1]
    combined_img = histomorph[start0:stop0+1,start1:stop1+1,start2:stop2+1]
    seg1_common_tr = seg1_common[start0:stop0+1,start1:stop1+1,start2:stop2+1]
    seg2_common_tr = seg2_common[start0:stop0+1,start1:stop1+1,start2:stop2+1]
    cort1_common_tr = cort1_common[start0:stop0+1,start1:stop1+1,start2:stop2+1]
    cort2_common_tr = cort2_common[start0:stop0+1,start1:stop1+1,start2:stop2+1]
    trab1_common_tr = trab1_common[start0:stop0+1,start1:stop1+1,start2:stop2+1]
    trab2_common_tr = trab2_common[start0:stop0+1,start1:stop1+1,start2:stop2+1]
    
    resorbed_bone = np.zeros_like(combined_img)
    resorbed_bone = resorbed_bone.astype("uint8")
    formed_bone = np.zeros_like(combined_img)
    formed_bone = formed_bone.astype("uint8")
    constant_bone = np.zeros_like(combined_img)
    constant_bone = constant_bone.astype("uint8")
    
    demineral_bone = np.zeros_like(combined_img)
    demineral_bone = demineral_bone.astype("uint8")
    mineral_bone = np.zeros_like(combined_img)
    mineral_bone = mineral_bone.astype("uint8")
    # Creating separate arrays for resorption and formation
    resorbed_bone[np.where(combined_img == 500)] = 1
    formed_bone[np.where(combined_img == 300)] = 1
    
    # Filtering out the noise on bone formation and resrption arrays based on the
    # size of the connected voxels
    resorbed_denoised = np.zeros_like(resorbed_bone)
    resorbed_label_tuple = label(resorbed_bone, background=None, return_num=True, connectivity=1)
    resorbed_small_clusters = remove_small_objects(resorbed_label_tuple[0], min_size=cluster, in_place=False)
    resorbed_denoised[np.where(resorbed_small_clusters != 0)] = 1
            
    formed_denoised = np.zeros_like(formed_bone)
    formed_label_tuple = label(formed_bone, background=None, return_num=True, connectivity=1)
    formed_small_clusters = remove_small_objects(formed_label_tuple[0], min_size=cluster, in_place=False)
    formed_denoised[np.where(formed_small_clusters !=0)] = 1
    
    # Making sure that formed/resorbed and seg1/seg2 are mutually exclusive
    mineral_bone[np.where(formed_denoised + seg1_common_tr == 2)] = 1
    formed_denoised[np.where(mineral_bone == 1)] = 0
    
    demineral_bone[np.where(resorbed_denoised + seg2_common_tr == 2)] = 1
    resorbed_denoised[np.where(demineral_bone == 1)] = 0
    
    overlay = np.zeros_like(formed_bone)
    overlay = overlay.astype(np.uint16)
    overlay[np.where(seg1_common_tr == 1)] = 600
    overlay[np.where(formed_denoised == 1)] = 300
    overlay[np.where(resorbed_denoised == 1)] = 500
    
    
    return overlay,formed_denoised, resorbed_denoised, mineral_bone, demineral_bone, combined_img, seg1_common_tr,\
        seg2_common_tr, cort1_common_tr, cort2_common_tr, trab1_common_tr, trab2_common_tr,voxel_size,n_days,Patient_ID

def np_array_type(array):
    output = np.float64(array)
    return output

def plot_array_show(imvtk,slice_n=50,mode="gray"):
    array = np_array_type(imvtk)
    plt.figure()
    plt.imshow(array[:,:,slice_n], cmap=mode),plt.show()
    return

def plot_array(imvtk,save_dir,Patient_ID,Site,time_point,thresh=225,cluster=5,slice_n=50,mode="gray"):
    file_name = Patient_ID + "_" + Site + "_" + str(time_point)
    save = save_dir + '\\' + file_name
    array = np_array_type(imvtk)
    plt.figure()
    plt.imshow(array[:,:,slice_n], cmap=mode),plt.savefig(save),plt.close()
    return

def array_overlay(array1,array2,slice_n):
    """
    > function for overlaying a plotting two numpy arrays

    Args:
        array1,array2: the two arrays to be overlayed
        slice_n: the slice number to be ploted
    """
    array1 = np_array_type(array1)
    array2 = np_array_type(array2)
    plt.figure()
    plt.imshow(array1[:,:,slice_n], cmap="gray")
    plt.imshow(array2[:,:,slice_n], cmap="jet", alpha=0.4)
    plt.show()