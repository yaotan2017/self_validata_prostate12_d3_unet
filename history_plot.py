# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:17:50 2019

@author: Tan
"""
import keras
import matplotlib.pyplot as plt


# 写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy_0 = {'batch': [], 'epoch': []}
        self.accuracy_1 = {'batch': [], 'epoch': []}

        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc_0 = {'batch': [], 'epoch': []}
        self.val_acc_1 = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        #        self.accuracy_0['batch'].append(logs.get('clas_0_dice'))
        self.accuracy_1['batch'].append(logs.get('clas_1_dice'))

        self.val_loss['batch'].append(logs.get('val_loss'))
        #        self.val_acc_0['batch'].append(logs.get('val_clas_0_dice'))
        self.val_acc_1['batch'].append(logs.get('val_clas_1_dice'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        #        self.accuracy_0['epoch'].append(logs.get('clas_0_dice'))
        self.accuracy_1['epoch'].append(logs.get('clas_1_dice'))

        self.val_loss['epoch'].append(logs.get('val_loss'))
        #        self.val_acc_0['epoch'].append(logs.get('val_clas_0_dice'))
        self.val_acc_1['epoch'].append(logs.get('val_clas_1_dice'))

    #    def loss_plot(self, loss_type):
    #        iters = range(len(self.losses[loss_type]))
    #        fig = plt.figure()
    #        ax1 = fig.add_subplot(111)
    #        ax2 = ax1.twinx()
    #
    #        # train acc and loss
    #        ax1.plot(iters, self.losses[loss_type], 'g', label='train loss')
    #        ax2.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
    #
    #        if loss_type == 'epoch':
    #            # val_loss
    #            ax1.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
    #            # val_acc
    #            ax2.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
    #
    #        fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    #        plt.grid(True)
    #        plt.xlabel(loss_type)
    #        ax1.set_ylabel('train_val_loss')
    #        ax2.set_ylabel('train_val_acc')
    #        plt.savefig('acc_loss.png')
    #        plt.show()

    def loss_plot(self, loss_type, save_path):
        iters = range(len(self.losses[loss_type]))
        fig1 = plt.figure()
        # train and val loss
        plt.plot(iters, self.losses[loss_type], 'r', label='train loss')
        if loss_type == 'epoch':
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'b', label='val loss')
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        fig1.legend(loc=1)
        plt.savefig(save_path + '/loss.png')
        plt.show()

        fig2 = plt.figure()
        plt.plot(iters, self.accuracy_1[loss_type], 'r', label='train dice')
        if loss_type == 'epoch':
            plt.plot(iters, self.val_acc_1[loss_type], 'b', label='val dice')
        plt.xlabel(loss_type)
        plt.ylabel('dice')
        fig2.legend(loc=1)
        plt.savefig(save_path + '/dice.png')
        plt.show()

    def log_write(self, loss_type, save_path):
        with open(save_path + '/train_loss.txt', 'a', encoding='utf-8') as f:
            loss = [str(i) + '\n' for i in self.losses[loss_type]]
            f.writelines(loss)
        # with open(save_path+'/train_acc_0.txt','a',encoding='utf-8') as f:
        #            acc0 = [str(j)+'\n' for j in self.accuracy_0[loss_type]]
        #            f.writelines(acc0)
        with open(save_path + '/train_acc_1.txt', 'a', encoding='utf-8') as f:
            acc1 = [str(j) + '\n' for j in self.accuracy_1[loss_type]]
            f.writelines(acc1)

        with open(save_path + '/val_loss.txt', 'a', encoding='utf-8') as f:
            val_loss = [str(k) + '\n' for k in self.val_loss[loss_type]]
            f.writelines(val_loss)
        # with open(save_path+'/val_acc_0.txt','a',encoding='utf-8') as f:
        #            val_acc0 = [str(m)+'\n' for m in self.val_acc_0[loss_type]]
        #            f.writelines(val_acc0)
        with open(save_path + '/val_acc_1.txt', 'a', encoding='utf-8') as f:
            val_acc1 = [str(m) + '\n' for m in self.val_acc_1[loss_type]]
            f.writelines(val_acc1)
