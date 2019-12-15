# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:11:22 2019

@author: Tan
"""
import nibabel as nib
import numpy as np
from keras.utils import np_utils
import os
import glob

def test_dice_coef(truth, pred,smooth=1.0):
#对预测后的npy ,在做了拼接为nii后进行单个的dice计算
    truth = truth.flatten()
    pred = pred.flatten()
    intersection = np.sum(truth * pred)
    dice = (2 * intersection + smooth) / (np.sum(truth) + np.sum(pred) + smooth)
    return dice

pred_path = '/media/laomaotao/0A9AD66165F33762/ty/prostate_3d_segmentation/result/d3-unet/2019-12-3_all/pred/post_pred_close'
truth_path = '/media/laomaotao/0A9AD66165F33762/ty/data/test_nii/mask'

pred_dirs = glob.glob(os.path.join(pred_path, '*.nii'))
truth_dirs = os.listdir(truth_path)

nb_classes = 2
all_dice = np.zeros((len(pred_dirs),nb_classes-1))
curr_id = 0
f = open('/media/laomaotao/0A9AD66165F33762/ty/prostate_3d_segmentation/result/d3-unet/2019-12-3_all/pred/post_dice_close.txt','a')
for pred_dir in pred_dirs:
    pred = nib.load(pred_dir).get_fdata()
    pred = np_utils.to_categorical(pred,nb_classes)
    truth = nib.load(truth_path+'/'+truth_dirs[curr_id]).get_fdata()
    truth = np_utils.to_categorical(truth,nb_classes)
    f.write(pred_dir[len(os.path.dirname(pred_dir))+1:].split('_')[0]+': ')
    for label in range(1,nb_classes):
        all_dice[curr_id,label-1] = test_dice_coef(truth[:,:,:,label], pred[:,:,:,label],smooth=1.0)
        print(all_dice[curr_id,label-1])
        f.write('label{} {:.4f}; '.format(label,all_dice[curr_id,label-1]))
    f.write('\n')    
    curr_id=curr_id+1

f.write('average dice: ')
aver_dice = np.sum(all_dice,axis=0)/len(pred_dirs)
for label in range(1,nb_classes):
    f.write('label{} {:.4f}; '.format(label,aver_dice[label-1]))
print('dice is done!')
f.close()