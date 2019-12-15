# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:13:27 2019

@author: tan
"""
#用训练模型分割测试集
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.models import load_model
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from metrics import (
        dice_coefficient_loss, get_label_dice_coefficient_function, 
        dice_coefficient
        )
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
#config.gpu_options.per_process_gpu_memory_fraction = 0.1
##config=tf.ConfigProto(device_count={'cpu':0})
session = tf.Session(config=config)
set_session(session)
            
if __name__ == '__main__':
    model = load_model(r'I:\ty\prostate_3d_segmentation\result\2019-5-4\deconv_d3-unet(all_patch)_update_22_3layer_noise_rotate\3d_unet.hdf5',
                       custom_objects={'dice_coefficient_loss': dice_coefficient_loss,'clas_0_dice':dice_coefficient,
                       'clas_1_dice':dice_coefficient})
    save_path = r'I:\ty\prostate_3d_segmentation\result\2019-5-4\deconv_d3-unet(all_patch)_update_22_3layer_noise_rotate\pred_model'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    test_img = np.load(
            'I:/ty/data/train_test-patch(64-32)/test_img_32_16.npy')
    test_pred = model.predict(test_img,batch_size=8, verbose=1)
    np.save(save_path+'/test_pred.npy_all',test_pred)
    label_pred = np.argmax(test_pred,axis=-1)
    np.save(save_path+'/test_pred_argmax.npy',np.expand_dims(label_pred,axis=-1))
    print('prediction is done!')