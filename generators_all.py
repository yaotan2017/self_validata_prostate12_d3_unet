# -*- coding: utf-8 -*-
"""
Created on Mon May  6 21:03:03 2019

@author: Tan
"""
"""
这里训练是在线切完块后，按块的数量划分验证和训练块,与generat_all对应使用
"""
import copy
from random import shuffle
import itertools

import numpy as np
from keras.utils import np_utils

from get_patch_from_train import random_cut_patch
from patches import compute_patch_indices, get_patch_from_3d_data
from augment import augment_data


def get_training_and_validation_generators(data_file, patch_shape, batch_size, n_labels, training_list, validation_list,
                                           val_split,binary_num_rate, min_point, pos_num_rate, nag_num_rate,
                                           augment=False, augment_flip=False, augment_distortion_factor=None,
                                           validation_batch_size=None, use_gauss=False, rot_xy=False):
    """
    :return: Training data generator, validation data generator, number of training steps, number of validation steps
    """
    if not validation_batch_size:
        validation_batch_size = batch_size
    if validation_list==None:
        index_list = create_patch_index_list(data_file, training_list, patch_shape,binary_num_rate,
                                             min_point, pos_num_rate, nag_num_rate)
        print(len(training_list),len(index_list))
        shuffle(index_list)
        n_training = int(len(index_list) * val_split)
        val_list = index_list[:n_training]
        train_list = index_list[n_training:]

    training_generator = data_generator(data_file, train_list,
                                        batch_size=batch_size,
                                        n_labels=n_labels,
                                        patch_shape=patch_shape,
                                        augment=augment,
                                        augment_flip=augment_flip,
                                        augment_distortion_factor=augment_distortion_factor,
                                        use_gauss=use_gauss, rot_xy=rot_xy)
    num_training_steps = get_number_of_steps(len(train_list), batch_size)
    print("Number of training steps: ", num_training_steps)

    validation_generator = data_generator(data_file, val_list,
                                          batch_size=validation_batch_size,
                                          n_labels=n_labels,
                                          patch_shape=patch_shape,
                                          augment=False,
                                          augment_flip=False,
                                          augment_distortion_factor=None,
                                          use_gauss=False, rot_xy=False)

    num_validation_steps = get_number_of_steps(len(val_list),validation_batch_size)
    print("Number of validation steps: ", num_validation_steps)

    return training_generator, validation_generator, num_training_steps, num_validation_steps


def data_generator(data_file, index_list, batch_size, n_labels=1, patch_shape=None, augment=False, augment_flip=False,
                   augment_distortion_factor=None, use_gauss=False, rot_xy=False, shuffle_index_list=True):
    orig_index_list = index_list
    while True:
        x_list = list()
        y_list = list()
        index_list = copy.copy(orig_index_list)
        
        if shuffle_index_list:
            shuffle(index_list)
        while len(index_list) > 0:
            index = index_list.pop()
            add_data(x_list, y_list, data_file, index, patch_shape=patch_shape,
                     augment=augment, augment_flip=augment_flip,
                     augment_distortion_factor=augment_distortion_factor)

            if len(x_list) == batch_size or (len(index_list) == 0 and len(x_list) > 0):
                x, y = convert_data(x_list, y_list, n_labels=n_labels)
                yield x, y
                x_list = list()
                y_list = list()

def create_patch_index_list(datas, index_list, patch_shape,binary_num_rate=0, min_point=0,
                            pos_num_rate=0, nag_num_rate=0):
    patch_index = list()
    for index in index_list:
        # 训练数据采用随机撒点进行采样
        truth = datas[index][2][:, :, :, 0]
        patches = random_cut_patch(truth, patch_shape=patch_shape,
                                    binary_num_rate=binary_num_rate,
                                    min_point=min_point, pos_num_rate=pos_num_rate,
                                    nag_num_rate=nag_num_rate)
        patch_index.extend(itertools.product([index], patches))
    return patch_index


def get_number_of_steps(n_samples, batch_size):
    if n_samples <= batch_size:
        return n_samples
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples // batch_size
    else:
        return n_samples // batch_size + 1


def add_data(x_list, y_list, data_file, index, patch_shape, augment=False, augment_flip=False,
             augment_distortion_factor=None, use_gauss=False, rot_xy=False):
    """
    Adds data from the data file to the given lists of feature and target data
    :param x_list: list of data to which data from the data_file will be appended.
    :param y_list: list of data to which the target data from the data_file will be appended.
    :param data_file: [(name1,data1,truth1,affine1),(name1,data1,truth1,affine1),...]
    :param index: index of the data file from which to extract the data.
    :param augment: if True, data will be augmented according to the other augmentation parameters (augment_flip and
    augment_distortion_factor)
    :param augment_flip: if True and augment is True, then the data will be randomly flipped along the x, y and z axis
    :param augment_distortion_factor: if augment is True, this determines the standard deviation from the original
    that the data will be distorted (in a stretching or shrinking fashion). Set to None, False, or 0 to prevent the
    augmentation from distorting the data in this way.
    :return:
    """
    data, truth = get_data_from_file(data_file, index, patch_shape=patch_shape)
    if augment:
        if patch_shape is not None:
            affine = data_file[index[0]][-1]
        else:
            affine = data_file[index][-1]

        data, truth = augment_data(np.squeeze(data, axis=-1), np.squeeze(truth, axis=-1), affine, flip=augment_flip,
                                   scale_deviation=augment_distortion_factor, add_gauss=use_gauss, rotation=rot_xy)
        data = np.expand_dims(data, axis=-1)
        truth = np.expand_dims(truth, axis=-1)
    x_list.append(data)
    y_list.append(truth)


def get_data_from_file(data_file, index, patch_shape=None):
    if patch_shape:
        index, patch_index = index
        data, truth = get_data_from_file(data_file, index, patch_shape=None)
        x = get_patch_from_3d_data(data, patch_shape, patch_index)
        y = get_patch_from_3d_data(truth, patch_shape, patch_index)
    else:
        x, y = data_file[index][1], data_file[index][2]
    return x, y


def convert_data(x_list, y_list, n_labels=1):
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    if n_labels > 1:
        y = np_utils.to_categorical(y[..., 0], n_labels)
    #        y = get_multi_class_labels(y, n_labels=n_labels, labels=labels)
    return x, y

# def get_multi_class_labels(data, n_labels, labels=None):
#    """
#    Translates a label map into a set of binary labels.
#    :param data: numpy array containing the label map with shape: (n_samples, 1, ...).
#    :param n_labels: number of labels.
#    :param labels: integer values of the labels.
#    :return: binary numpy array of shape: (n_samples, n_labels, ...)
#    """
#    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
#    y = np.zeros(new_shape, np.int8)
#    for label_index in range(n_labels):
#        if labels is not None:
#            y[:, label_index][data[:, 0] == labels[label_index]] = 1
#        else:
#            y[:, label_index][data[:, 0] == (label_index + 1)] = 1
#    return y