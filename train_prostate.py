# -*- coding: utf-8 -*-
"""
Created on Mon May  6 21:03:03 2019

@author: Tan
"""
import os
#设置Gpu
multi_gpu = False
if multi_gpu:
    multi_gpu_num = 2
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    multi_gpu_num = None

from random import shuffle
import nibabel as nib
import numpy as np

from build_model import Unet
from generators import get_training_and_validation_generators
from training import load_old_model,train_model

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配,config.gpu_options.per_process_gpu_memory_fraction = 0.6  #进行配置，使用50%的GPU
session = tf.Session(config=config)
set_session(session)

config = dict()
config["patch_shape"] = (64,64,32)
config['input_size'] = tuple(list(config["patch_shape"])+[1])
config['nb_classes'] = 2
config['initial_filter_num'] = 32
config['l2_regular'] = 1e-5
config['bn'] = True
config['deconv'] = True
config['drop'] = 0.0 #dropout_rate
config['deeper_4'] = False
config['activation_fun'] = 'sigmoid'
if config['nb_classes']>1:
    config['activation_fun'] = 'softmax'
config["batch_size"] = 8
config["validation_batch_size"] = 8 #在多个GPU下，保证val_num/batch_size>gpu_num
config["n_epochs"] = 30
config["early_stop"] = 3 ## training will be stopped after this many epochs without the validation loss improving
config["initial_lr"] = 0.005 #与模型选择的optimizer对应，该处使用SGD
config["learning_rate_drop"] =0.8
config["learning_rate_epochs"] = 2
config["use_Adam"]=False
config["validation_split"] = 0.125

#训练数据的随机切块参数
config["binary_num_rate"] = 0.025
config["min_point"] = 800
config["pos_num_rate"] = 0.5
config["nag_num_rate"] = 0.5

config["validation_patch_overlap"]=(32,32,16)
config["flip"] = False  # augments the data by randomly flipping an axis during
config["distort"] = False  # switch to None if you want no distortion
config["augment"] = config["flip"] or config["distort"]

config["save_fpath"] = '/media/laomaotao/0A9AD66165F33762/ty/prostate_3d_segmentation/result/d3-unet/2019-12-3_1_new'
if not os.path.exists(config["save_fpath"]):
    os.mkdir(config['save_fpath'])
config["data_fpath"] = '/media/laomaotao/0A9AD66165F33762/ty/data/preprocess_nii_new/train_nii'
config["model_file"] = None#载入预训练模型
config["training_config_file"] = config['save_fpath']+'/confing.txt'
config_file=open(config['training_config_file'],'w')
for key,value in config.items():
    config_file.writelines(key+':'+str(value)+'\n')
config_file.close()

def get_data(data_fpath,split_rate,shuffle_list=True):
    image_indexs=os.listdir(data_fpath+'/image')
    image_indexs = sorted(image_indexs, key=str.lower)
    label_indexs=os.listdir(data_fpath+'/mask')
    label_indexs = sorted(label_indexs, key=str.lower)
    sample_list=list(range(len(image_indexs)))
    
    #获取训练验证的数据索引
    datas=[];train_name=[];val_name=[]
    if shuffle_list:
        shuffle(sample_list)
    n_training = int(len(sample_list) * split_rate)
    val_list = sample_list[:n_training]
    train_list = sample_list[n_training:]
    
    for i in range(len(image_indexs)):
        name=image_indexs[i].split('.')[0]
        file=nib.load(os.path.join(data_fpath, 'image', image_indexs[i]))
        image=np.expand_dims(file.get_fdata(),axis=-1)
        affine=file.affine
        
        truth=nib.load(os.path.join(data_fpath, 'mask', label_indexs[i]))
        truth=np.expand_dims(truth.get_fdata(),axis=-1).astype(np.uint8)
        datas.append(tuple([name,image,truth,affine]))
        print(image_indexs[i],label_indexs[i])
        if i in val_list:
            val_name.append(name)
        else:train_name.append(name)
        
    return datas,train_list,val_list,train_name,val_name
    
def main():
    print('loading training data.....')
    data_file_opened,train_list,val_list,train_name,val_name= get_data(config['data_fpath'],
                                                                       split_rate=config["validation_split"],shuffle_list=True)
    config_file=open(config['training_config_file'],'a')
    config_file.writelines('train_nii: '+' / '.join(train_name))
    config_file.writelines('\n')
    config_file.writelines('val_nii: '+' / '.join(val_name))
    del train_name,val_name
    config_file.close()
    
    if config["model_file"]:
        print('load pre_train_model....')
        model = load_old_model(config["model_file"])
    else:
        print('build the new mode.....')
        model = Unet(input_size=config['input_size'],first_filter_num=config['initial_filter_num'],
                     weight_decay=config['l2_regular'],batch_norm=config['bn'],deconv=config['deconv'],
                     dropout_rate=config['drop'],deeper=config['deeper_4'],nb_classes=config['nb_classes'],
                     initial_learning_rate=config['initial_lr'],activate_fun=config['activation_fun'],
                     multi_gpu=multi_gpu_num,use_Adam=config["use_Adam"])
        
    # get training and testing generators
    print('get training and testing generators....')
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
            data_file_opened,patch_shape=config["patch_shape"],
            batch_size=config["batch_size"],
            n_labels=config['nb_classes'],
            binary_num_rate=config["binary_num_rate"],
            min_point=config["min_point"],
            pos_num_rate=config["pos_num_rate"],
            nag_num_rate=config["nag_num_rate"],
            validation_list=val_list,
            training_list=train_list,
            validation_batch_size=config["validation_batch_size"],
            validation_patch_overlap=config["validation_patch_overlap"],
            augment=config["augment"],
            augment_flip=config["flip"],
            augment_distortion_factor=config["distort"])
    # run training
    print('training is going on...')
    model = train_model(model=model,save_path=config['save_fpath'],
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=config["initial_lr"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_epochs=config['learning_rate_epochs'],
                n_epochs=config["n_epochs"],
                early_stopping_patience=config["early_stop"])
if __name__ == "__main__":
    main()
    print('finisn training....')