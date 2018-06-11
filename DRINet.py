from __future__ import division
import os
import tensorflow as tf
import numpy as np

def conv2d(input_, output_channel, kernel_size=3, stride=1, name='conv'):
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', [kernel_size, kernel_size,
            input_.get_shape()[-1], output_channel],
            initializer=tf.truncated_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input_, weights,
            strides=[1, stride, stride, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_channel],
            initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv

def max_pool(input_, kernel_size=2, stride=2, name='pool'):
    with tf.variable_scope(name):
        pool = tf.nn.max_pool(input_, ksize=[1,kernel_size,kernel_size,1],
            strides=[1,stride,stride,1], padding='SAME', name=name)
        return pool

def bn_relu(input_, is_train=True):
    bn = tf.layers.batch_normalization(input_, training=is_train)
    relu = tf.nn.relu(bn)
    return relu

def dense_conv(input_, growth_rate, name='dense_conv', is_train=True):
    with tf.variable_scope(name):
        [b, w, h, c] = input_.get_shape().as_list()
        bn_relu1 = bn_relu(input_, is_train)
        conv1 = conv2d(bn_relu1, growth_rate, name='conv1')
        concat1 = tf.concat((input_, conv1), axis=3)
        bn_relu2 = bn_relu(concat1, is_train)
        conv2 = conv2d(bn_relu2, growth_rate, name='conv2')
        concat2 = tf.concat((concat1, conv2), axis=3)
        bn_relu3 = bn_relu(concat2, is_train)
        conv3 = conv2d(bn_relu3, growth_rate, name='conv3')
        concat3 = tf.concat((concat2, conv3), axis=3)
        bn_relu4 = bn_relu(concat3, is_train)
        conv4 = conv2d(bn_relu4, c+3*growth_rate, 1, name='conv4')
        return conv4

def deconv2d(input_, output_shape=None, kernel_size=3, stride=1, name='deconv',
    bn=True, act_fn=True, is_train=True):
    with tf.variable_scope(name):
        if not output_shape:
            output_shape = input_.get_shape().as_list()
        weights = tf.get_variable('weights',
            [kernel_size, kernel_size, output_shape[-1],
            input_.get_shape()[-1]],
            initializer=tf.random_normal_initializer(stddev=0.01))
        deconv = tf.nn.conv2d_transpose(input_, weights,
            output_shape=output_shape, strides=[1, stride, stride, 1])
        biases = tf.get_variable('biases', [output_shape[-1]],
            initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if bn:
            deconv = tf.layers.batch_normalization(deconv,training=is_train)
        if act_fn:
            deconv = tf.nn.relu(deconv)
        return deconv

def deconv2d_atrous(input_, output_shape=None, rate=2, kernel_size=3,
    name='deconv_atrous', bn=True, act_fn=True, is_train=True):
    with tf.variable_scope(name):
        if not output_shape:
            output_shape = input_.get_shape().as_list()
        weights = tf.get_variable('weights',
            [kernel_size, kernel_size, output_shape[-1],
            input_.get_shape()[-1]],
            initializer=tf.random_normal_initializer(stddev=0.01))
        deconv = tf.nn.atrous_conv2d_transpose(input_, weights,
            output_shape=output_shape, rate=rate, padding='SAME')
        biases = tf.get_variable('biases', [output_shape[-1]],
            initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if bn:
            deconv = tf.layers.batch_normalization(deconv, training=is_train)
        if act_fn:
            deconv = tf.nn.relu(deconv)
        return deconv

def res_inc_deconv(input_, name='res_inc_deconv', is_train=True):
    with tf.variable_scope(name):
        [b, w, h, c] = input_.get_shape().as_list()
        deconv1_1 = deconv2d(input_, [b,w,h,c//4], 1,
            name='deconv1_1', is_train=is_train)
        deconv2_1 = deconv2d(input_, [b,w,h,c//4], 1,
            name='deconv2_1', is_train=is_train)
        deconv2_2 = deconv2d(deconv2_1, [b,w,h,c//2],
            name='deconv2_2', is_train=is_train)
        deconv3_1 = deconv2d(input_, [b,w,h,c//4], 1,
            name='deconv3_1', is_train=is_train)
        deconv3_2 = deconv2d_atrous(deconv3_1, [b,w,h,c//4],
            name='deconv3_2', is_train=is_train)
        concat = tf.concat((deconv1_1, deconv2_2, deconv3_2), axis=3)
        deconv = deconv2d(concat, [b,w,h,c], 1, name='deconv',
            act_fn=False, is_train=is_train)
        fuse = tf.add(input_,deconv)
        return fuse

def unpool(input_, name='unpool', is_train=True):
    with tf.variable_scope(name):
        [b, w, h, c] = input_.get_shape().as_list()
        deconv1_1 = deconv2d(input_, [b,w,h,c//2], 1,
            name='deconv1_1', is_train=is_train)
        deconv1_2 = deconv2d(deconv1_1, [b,w*2,h*2,c//4], 3, 2,
            name='deconv1_2', is_train=is_train)
        deconv2_1 = deconv2d(input_, [b,w,h,c//2], 1,
            name='deconv2_1', is_train=is_train)
        deconv2_2 = deconv2d_atrous(deconv2_1, [b,w,h,c//2],
            name='deconv2_2', is_train=is_train)
        deconv2_3 = deconv2d(deconv2_2, [b,w*2,h*2,c//4], 3, 2,
            name='deconv2_3', is_train=is_train)
        concat = tf.concat((deconv1_2, deconv2_3), axis=3)
        return concat

def DRINet(images, num_class, growth_rate, name='DRINet',
    is_train=True, reuse=False):
    '''
    Input arguments:

        - images in shape [batch, height, width, channel]

        - num_class: a scalar, number of objects of segment excluding background

        - growth_rate: a list of 4 entries

        - name: scope name of the network

        - is_train: True for training and False for testing

        - reuse: True or False

    Output:

        - segmentation maps in shape [batch, height, width, num_class+1]
    '''
    [b, h, w, c] = images.get_shape().as_list()
    with tf.variable_scope(name, reuse=reuse) as scope:
        conv1_1 = conv2d(images, 32, name='conv1_1')
        conv1_2 = dense_conv(conv1_1, growth_rate[0], 'conv1_2', is_train)
        pool1 = max_pool(conv1_2, name='pool1')
        conv2 = dense_conv(pool1, growth_rate[1], 'conv2', is_train)
        pool2 = max_pool(conv2, name='pool2')
        conv3 = dense_conv(pool2, growth_rate[2], 'conv3', is_train)
        pool3 = max_pool(conv3, name='pool3')
        conv4_1 = dense_conv(pool3, growth_rate[3], 'conv4_1', is_train)
        bn_relu4_1 = bn_relu(conv4_1, is_train=is_train)
        deconv4_2 = res_inc_deconv(bn_relu4_1, 'deconv4_2', is_train)
        deconv4_3 = res_inc_deconv(deconv4_2, 'deconv4_3', is_train)
        deconv4_4 = res_inc_deconv(deconv4_3, 'deconv4_4', is_train)
        up4 = unpool(deconv4_4, 'up4', is_train)
        deconv5_1 = res_inc_deconv(up4, 'deconv5_1', is_train)
        deconv5_2 = res_inc_deconv(deconv5_1, 'deconv5_2', is_train)
        deconv5_3 = res_inc_deconv(deconv5_2, 'deconv5_3', is_train)
        up5 = unpool(deconv5_3, 'up5', is_train)
        deconv6_1 = res_inc_deconv(up5, 'deconv6_1', is_train)
        deconv6_2 = res_inc_deconv(deconv6_1, 'deconv6_2', is_train)
        deconv6_3 = res_inc_deconv(deconv6_2, 'deconv6_3', is_train)
        up6 = unpool(deconv6_3, 'up6', is_train)
        deconv7_1 = res_inc_deconv(up6, 'deconv7_1', is_train)
        deconv7_2 = res_inc_deconv(deconv7_1, 'deconv7_2', is_train)
        deconv7_3 = res_inc_deconv(deconv7_2, 'deconv7_3', is_train)
        deconv7_4 = deconv2d(deconv7_3, [b,h,w,num_class+1], 1,
            name='deconv7_4', bn=False, act_fn=False, is_train=is_train)
        return deconv7_4
