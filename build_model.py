# -*- coding: utf-8 -*-
"""
Created on Fri May  3 22:21:52 2019

@author: Tan
"""
from keras.utils import multi_gpu_model,plot_model
import os
os.environ["PATH"] += os.pathsep +'D:/Graphviz2.38/bin/'
from keras.models import Model
from keras.layers import (Input,Conv3D,MaxPooling3D,UpSampling3D,concatenate,
                          Deconvolution3D,SpatialDropout3D,BatchNormalization,LeakyReLU)
from keras.optimizers import Adam,SGD
from keras.regularizers import l2
from metrics import (dice_coefficient_loss, get_label_dice_coefficient_function,dice_coefficient)

#模型组件
def double_conv(inputs,filter_num_1,filter_num_2,weight_decay=0.0,batch_norm=False):
    conv1 = Conv3D(filter_num_1, 3, padding = 'same',kernel_initializer = 'he_normal',
                   kernel_regularizer = l2(weight_decay))(inputs)
    if batch_norm:
        conv1 = BatchNormalization(axis=-1)(conv1)
    conv1 = LeakyReLU(alpha=0.3)(conv1)
    conv2 = Conv3D(filter_num_2, 3,padding = 'same',kernel_initializer = 'he_normal',
                   kernel_regularizer = l2(weight_decay))(conv1)
    if batch_norm:
        conv2 = BatchNormalization(axis=-1)(conv2)
    conv2 = LeakyReLU(alpha=0.3)(conv2)
    return conv2

def get_up_convolution(inputs,filter_num,deconv=False,weight_decay=0.0,batch_norm=False):
    if deconv:
        deconv = Deconvolution3D(filters=filter_num, kernel_size=(2,2,2), strides=(2,2,2),
                               kernel_initializer = 'he_normal',kernel_regularizer = l2(weight_decay))(inputs)
        if batch_norm:
            deconv = BatchNormalization(axis=-1)(deconv)
        return deconv
    else:
        return UpSampling3D(size=(2,2,2))(inputs)
    
def Unet(input_size,first_filter_num,weight_decay,batch_norm=False,deconv=False,dropout_rate=0.3,deeper=False,
         nb_classes=1,initial_learning_rate=0.005,activate_fun='sigmoid',metrics=[dice_coefficient],multi_gpu=None,use_Adam=False):
    inputs = Input(input_size)
    conv1 = double_conv(inputs, filter_num_1=first_filter_num, filter_num_2=2*first_filter_num,
                        weight_decay=weight_decay, batch_norm=batch_norm)#32-64
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    
    conv2 = double_conv(pool1, filter_num_1=2*first_filter_num, filter_num_2=4*first_filter_num,
                        weight_decay=weight_decay, batch_norm=batch_norm)#64-128
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    
    conv3 = double_conv(pool2, filter_num_1=4*first_filter_num, filter_num_2=8*first_filter_num,
                        weight_decay=weight_decay, batch_norm=batch_norm)#128-256
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    
    conv4 = double_conv(pool3, filter_num_1=8*first_filter_num, filter_num_2=16*first_filter_num,
                        weight_decay=weight_decay, batch_norm=batch_norm)#256-512
    if dropout_rate!=0.0:
        drop4 = SpatialDropout3D(dropout_rate)(conv4)
        conv4 = drop4
    
    #是否加深到4个skip-connection,参数量
    if deeper:
        pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
        
        conv0 = double_conv(pool4, filter_num_1=16*first_filter_num, filter_num_2=32*first_filter_num,
                            weight_decay=weight_decay, batch_norm=batch_norm)#512-1024
        if dropout_rate!=0.0:
            drop0 = SpatialDropout3D(dropout_rate)(conv0)
            conv0 = drop0

        up0 = get_up_convolution(conv0,filter_num=32*first_filter_num, deconv=deconv,weight_decay=weight_decay,
                             batch_norm=batch_norm)#up:1024
        merge0 = concatenate([conv4,up0], axis = 4)
        conv0 = double_conv(merge0, filter_num_1=16*first_filter_num, filter_num_2=16*first_filter_num,
                            weight_decay=weight_decay, batch_norm=batch_norm)#512-512
        conv4 = conv0 #方便up5输入命名，无需更改

    up5 = get_up_convolution(conv4,filter_num=16*first_filter_num,deconv=deconv,weight_decay=weight_decay,
                             batch_norm=batch_norm)#up:512
    merge5 = concatenate([conv3,up5], axis = 4)
    conv5 = double_conv(merge5, filter_num_1=8*first_filter_num, filter_num_2=8*first_filter_num,
                        weight_decay=weight_decay, batch_norm=batch_norm)#256-256

    up6 = get_up_convolution(conv5,filter_num=8*first_filter_num,deconv=deconv,weight_decay=weight_decay,
                             batch_norm=batch_norm)#up:256
    merge6 = concatenate([conv2,up6], axis = 4)
    conv6 = double_conv(merge6, filter_num_1=4*first_filter_num, filter_num_2=4*first_filter_num,
                        weight_decay=weight_decay, batch_norm=batch_norm)#128-128

    up7 = get_up_convolution(conv6,filter_num=4*first_filter_num, deconv=deconv,weight_decay=weight_decay,
                             batch_norm=batch_norm)#up:128
    merge7 = concatenate([conv1,up7], axis = 4)
    conv7 = double_conv(merge7, filter_num_1=2*first_filter_num, filter_num_2=2*first_filter_num,
                        weight_decay=weight_decay, batch_norm=batch_norm)#64-64

    conv8 = Conv3D(nb_classes, 1, activation=activate_fun, kernel_initializer = 'he_normal',
                   kernel_regularizer = l2(weight_decay))(conv7)

    model = Model(inputs = inputs, outputs = conv8)
    if multi_gpu:
        model = multi_gpu_model(model, multi_gpu)

    if not isinstance(metrics, list):
        metrics = [metrics]

    if nb_classes > 1: #注意还要修改metrics函数中loss的默认nb_class
        label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(0,nb_classes)]
        metrics = label_wise_dice_metrics
    if not use_Adam:
        model.compile(optimizer = SGD(initial_learning_rate), loss=dice_coefficient_loss, metrics=metrics)
        print('use the SGD optimizer...')
    else:
        model.compile(optimizer = Adam(lr=initial_learning_rate,beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                      loss = dice_coefficient_loss, metrics = metrics)
        print('use the Adam optimizer....')
#    model.compile(optimizer = SGD(initial_learning_rate), loss = connection_loss, metrics = [dice_coef])
#     la=[layer for layer in model.layers]
#     print(la)
#     model.summary()
    plot_model(model, to_file='Model.png',show_shapes=True)
    return model