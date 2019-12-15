# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:11:22 2019

@author: Tan
"""
"""
对模型预测做后处理,删除非极大连通域,之后进行膨胀腐蚀运算进行去空洞平滑等作用
"""
import nibabel as nib
import numpy as np
from skimage import morphology
import glob
import os

def get_max_label(arr):
    res=dict()
    for i in arr.flatten():
        if i in res:
            res[i]+=1
        else:
            res[i]=1
    res=sorted(res.items(),key=lambda x:x[1],reverse=True)
    return res[1]

def remove_nonmax_area(arr,label,count):
    morphology.remove_small_objects(arr, min_size=count, connectivity=1, in_place=True)
    arr//=label
    return arr

pre_fpath='/media/laomaotao/0A9AD66165F33762/ty/prostate_3d_segmentation/result/d3-unet/2019-12-3_all/pred'
post_path='/media/laomaotao/0A9AD66165F33762/ty/prostate_3d_segmentation/result/d3-unet/2019-12-3_all/pred/post_pred_close'
if os.path.exists(post_path) is False:
    os.makedirs(post_path)
    
pre_dir=glob.glob(os.path.join(pre_fpath, '*.nii'))
for i in pre_dir:
    name=i[len(os.path.dirname(i))+1:].split('_')[0]+'_post.nii'
    img=nib.load(i)
    hdr = nib.Nifti1Header()
    re_affine = img.affine
    arr=img.get_fdata()
    
    post_arr=morphology.label(arr, neighbors=8,return_num=False)
    label,count=get_max_label(post_arr)
    remove_nonmax_area(post_arr,label,count)
    post_arr.astype(np.uint8)
    post_arr=morphology.binary_closing(post_arr, selem=None, out=None)
    out=nib.Nifti1Image(post_arr, re_affine,hdr)
    out.to_filename(os.path.join(post_path, name))