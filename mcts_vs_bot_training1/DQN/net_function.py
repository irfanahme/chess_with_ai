# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 09:48:20 2020

@author: ashba
"""

import numpy as np 
from keras.layers import Dense,  Conv2D, Add, Flatten, BatchNormalization
from keras.layers import   LeakyReLU,Activation
from keras import regularizers
import tensorflow as tf
state_dim   = 32
num_actions = 1968


def conv_block(x_image,filters=256, kernel_size=(4, 4), reg_const =0.00001,scale=False):
    
    block = Conv2D(filters, kernel_size, data_format='channels_last',
                   padding='same',use_bias=False, activation='linear', 
                   kernel_regularizer=regularizers.l2( reg_const))(x_image)
    
    block = BatchNormalization(epsilon=1e-5, axis=1, center=True,scale=scale)(block)
    block = LeakyReLU()(block)
    return block

def residual_block(x, filters=256):
    x_ = x
    block = conv_block(x, filters)
    block2 = conv_block(block, filters)
    block3 = conv_block(block2, filters)
 
    #taking residual after 3 convolutional blocks 
    added= Add()([x_, block3])
    block = Activation('relu')(added)
    return block


def  creat_network_3d(states, myscope, state_shape= [-1, 18, 8, 8],residual_tower = 2, 
                      output_dim =  num_actions, reg_const =0.01 ):  
    with tf.variable_scope(myscope): 
        x = states[np.newaxis,:] 
        x_image = tf.reshape(x,state_shape )  #    x_image = tf.reshape(x, [-1, 3, 3, 1])
        x_image =tf.dtypes.cast(x_image,dtype=float)
        
        block = conv_block(x_image)
        for i in range(residual_tower):
            block = residual_block(block)
#            print(i, "\t", block.shape)a
    
        #probabilty output head
        block = conv_block(block)
        print("DOne wth residul block")
        block = Flatten()(block)
        block = Dense(units=output_dim*2, use_bias=False, activation='softmax',
                          kernel_regularizer=regularizers.l2(reg_const), name='policy_head')(block)
        try:
            Wo = tf.get_variable("Wo", [output_dim*2, output_dim],initializer=tf.random_normal_initializer(stddev=0.1))
            bo = tf.get_variable("bo", [output_dim],initializer=tf.constant_initializer(0))
        except:
            Wo = tf.get_variable("iWo", [output_dim*2, output_dim],initializer=tf.random_normal_initializer(stddev=0.1))
            bo = tf.get_variable("ibo", [output_dim],initializer=tf.constant_initializer(0))
    
        p = tf.matmul(block, Wo) + bo
    return p







