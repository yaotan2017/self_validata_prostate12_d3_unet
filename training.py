# -*- coding: utf-8 -*-
"""
Created on Mon May  6 21:03:03 2019

@author: Tan
"""
import os
import math
from functools import partial

from keras.callbacks import (ModelCheckpoint, CSVLogger, LearningRateScheduler,
                             ReduceLROnPlateau, EarlyStopping,TensorBoard)
from keras.models import load_model

from metrics import (dice_coefficient, dice_coefficient_loss,weighted_dice_coefficient_loss, weighted_dice_coefficient)
from history_plot import LossHistory

# learning rate schedule
#def step_decay(epoch, initial_lrate, drop, epochs_drop):
#    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))

#自定义learning rate schedule,每隔2个epoch，学习率减小为原来的1/10
def step_decay(epoch, initial_lrate, drop, epochs_drop):
    if epoch%epochs_drop==0 and epoch!=0:
        print("lr changed to {}".format(initial_lrate * drop))
        return initial_lrate * drop
    else:
        return initial_lrate

def get_callbacks(save_path,initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None,
                  learning_rate_patience=None,verbosity=1,early_stopping_patience=None):
    callbacks = list()
    callbacks.append(CSVLogger(save_path+'/training.log', append=True))
    if learning_rate_epochs:
        callbacks.append(LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate,
                                                       drop=learning_rate_drop, epochs_drop=learning_rate_epochs)))
    else:
        callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity))
    if early_stopping_patience:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=early_stopping_patience,mode='auto'))
    if not os.path.exists(save_path+'/model_hdf5'):
        os.mkdir(save_path+'/model_hdf5')
    filepath =save_path+"/model_hdf5/model.{epoch:02d}-{loss:.2f}.hdf5"
    model_checkpoint = ModelCheckpoint(filepath, monitor= 'loss',verbose=1, save_weights_only=False, save_best_only=True, period=1)
    
    tbCallBack = TensorBoard(log_dir=save_path+'/logs', histogram_freq=0, write_graph=True,
                             write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    history = LossHistory()
    callbacks.extend([model_checkpoint,tbCallBack,history])
    return callbacks


def load_old_model(model_file):
    print("Loading pre-trained model")
    '''custom_objects = {'dice_coefficient_loss': dice_coefficient_loss, 'dice_coefficient': dice_coefficient,
                      'clas_1_dice': dice_coefficient, 'weighted_dice_coefficient': weighted_dice_coefficient,
                      'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss}'''
    custom_objects = {'dice_coefficient_loss': dice_coefficient_loss, 'clas_0_dice': dice_coefficient,
                      'clas_1_dice': dice_coefficient}
#    try:
#        from keras_contrib.layers import InstanceNormalization
#        custom_objects["InstanceNormalization"] = InstanceNormalization
#    except ImportError:
#        pass
#    try:
#        return load_model(model_file, custom_objects=custom_objects)
#    except ValueError as error:
#        if 'InstanceNormalization' in str(error):
#            raise ValueError(str(error) + "\n\nPlease install keras-contrib to use InstanceNormalization:\n"
#                                          "'pip install git+https://www.github.com/keras-team/keras-contrib.git'")
#        else:
#            raise error
    return load_model(model_file, custom_objects=custom_objects)


def train_model(model, save_path, training_generator, validation_generator, steps_per_epoch, validation_steps,
                initial_learning_rate=0.001, learning_rate_drop=0.5, learning_rate_epochs=None, n_epochs=500,
                learning_rate_patience=None, early_stopping_patience=None):
    """
    Train a Keras model.
    :param early_stopping_patience: If set, training will end early if the validation loss does not improve after the
    specified number of epochs.
    :param learning_rate_patience: If learning_rate_epochs is not set, the learning rate will decrease if the validation
    loss does not improve after the specified number of epochs. (default is 20)
    :param model: Keras model that will be trained.
    :param model_file: Where to save the Keras model.
    :param training_generator: Generator that iterates through the training data.
    :param validation_generator: Generator that iterates through the validation data.
    :param steps_per_epoch: Number of batches that the training generator will provide during a given epoch.
    :param validation_steps: Number of batches that the validation generator will provide during a given epoch.
    :param initial_learning_rate: Learning rate at the beginning of training.
    :param learning_rate_drop: How much at which to the learning rate will decay.
    :param learning_rate_epochs: Number of epochs after which the learning rate will drop.
    :param n_epochs: Total number of epochs to train the model.
    :return: 
    """
    print('Fitting model....')
    callbacks = get_callbacks(save_path,initial_learning_rate=initial_learning_rate,
                  learning_rate_drop=learning_rate_drop,
                  learning_rate_epochs=learning_rate_epochs,
                  learning_rate_patience=learning_rate_patience,
                  early_stopping_patience=early_stopping_patience)
    model.fit_generator(generator=training_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=n_epochs,
                        verbose = 1,workers=0,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        callbacks=callbacks)
    model.save(save_path+'/final_model.hdf5')
    #绘制acc-loss曲线
    callbacks[-1].log_write('epoch',save_path)
    callbacks[-1].loss_plot('epoch',save_path)
    return model