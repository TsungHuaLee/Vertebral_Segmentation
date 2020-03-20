from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Activation, concatenate, Dropout, Conv2DTranspose, LeakyReLU, Add
from tensorflow.keras.layers import GlobalAveragePooling2D, Lambda, GlobalMaxPooling2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization as BN
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import numpy as np
import cv2

from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import backend as K

width = 400
height = 1200

class GroupNormalization(Layer):
    def __init__(self,
                 groups=4,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                             'multiple of the number of channels (' +
                             str(dim) + ').')

        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)
        tensor_input_shape = K.shape(inputs)

        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(1, self.groups)

        reshape_group_shape = K.shape(inputs)
        group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        group_axes[self.axis] = input_shape[self.axis] // self.groups
        group_axes.insert(1, self.groups)

        # reshape inputs to new group shape
        group_shape = [group_axes[0], self.groups] + group_axes[2:]
        group_shape = K.stack(group_shape)
        inputs = K.reshape(inputs, group_shape)

        group_reduction_axes = list(range(len(group_axes)))

        mean = K.mean(inputs, axis=group_reduction_axes, keepdims=True)
        variance = K.var(inputs, axis=group_reduction_axes, keepdims=True)

        inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))

        # prepare broadcast shape
        inputs = K.reshape(inputs, group_shape)

        outputs = inputs

        # In this case we must explicitly broadcast all parameters.
        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma

        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta

        # finally we reshape the output back to the input shape
        outputs = K.reshape(outputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

def convBlock(n_filter, kernel_size, name, with_gn=True):
    n_filter = int(n_filter)
    conv_name = name + '_conv_out'
    gn_name = name + '_conv_gn'
    act_name = name + '_conv_act'
    if(n_filter>16):
        gn_groups = 16
    else:
        gn_groups = 4
    def call(x):
        if(with_gn):
            conv_out = Conv2D(filters=n_filter, kernel_size=kernel_size, padding='same', name=conv_name)(x)
            conv_gn = GroupNormalization(groups = gn_groups, name = gn_name)(conv_out)
            conv_act = LeakyReLU(alpha='0.3', name = act_name)(conv_gn)
        else:
            conv_out = Conv2D(filters=n_filter, kernel_size=kernel_size, padding='same', name=conv_name)(x)
            conv_act = LeakyReLU(alpha='0.3', name = act_name)(conv_out)
        return conv_act
    return call

# output_n = growth_rate * 8
def denseBlock(growth_rate, output_n, name):
    # conv_1 = convBlock(n_filter=output_n/2, kernel_size=(1,1))
    # conv_2 = convBlock(n_filter=growth_rate, kernel_size=(3,3))
    def call(x):
        conv1_1 = convBlock(n_filter=output_n/2, kernel_size=(1,1), name=name+'_b11')(x)
        conv1_2 = convBlock(n_filter=growth_rate, kernel_size=(3,3), name=name+'_b12')(conv1_1)
        conv1_out = concatenate([x, conv1_2], axis = 3)

        conv2_1 = convBlock(n_filter=output_n/2, kernel_size=(1,1), name=name+'_b21')(conv1_out)
        conv2_2 = convBlock(n_filter=growth_rate, kernel_size=(3,3), name=name+'_b22')(conv2_1)
        conv2_out = concatenate([conv1_out, conv2_2], axis = 3)

        conv3_1 = convBlock(n_filter=output_n/2, kernel_size=(1,1), name=name+'_b31')(conv2_out)
        conv3_2 = convBlock(n_filter=growth_rate, kernel_size=(3,3), name=name+'_b32')(conv3_1)
        conv3_out = concatenate([conv2_out, conv3_2], axis = 3)

        conv4_1 = convBlock(n_filter=output_n/2, kernel_size=(1,1), name=name+'_b41')(conv3_out)
        conv4_2 = convBlock(n_filter=growth_rate, kernel_size=(3,3), name=name+'_b42')(conv4_1)
        conv4_out = concatenate([conv3_out, conv4_2], axis = 3, name=name+'_output')

        return conv4_out
    return call

def dense_unet(growth_rate = 4, n_fm = 32, input_size = (512,512,1)):
    inputs = Input(input_size)
    # contracting path
    num = 1
    encoder1_1 = convBlock(n_filter = n_fm/2, kernel_size=(3,3), name = 'en'+str(num))(inputs)
    encoder1_2 = denseBlock(growth_rate=2**(num-1)*growth_rate, output_n=2**(num-1)*n_fm, name='en'+str(num))(encoder1_1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(encoder1_2)
    num += 1
    encoder2 = denseBlock(growth_rate=2**(num-1)*growth_rate, output_n=2**(num-1)*n_fm, name='en'+str(num))(pool1)

    pool2 = MaxPooling2D(pool_size=(2, 2))(encoder2)
    num += 1
    encoder3 = denseBlock(growth_rate=2**(num-1)*growth_rate, output_n=2**(num-1)*n_fm, name='en'+str(num))(pool2)

    pool3 = MaxPooling2D(pool_size=(2, 2))(encoder3)
    num += 1
    encoder4 = denseBlock(growth_rate=2**(num-1)*growth_rate, output_n=2**(num-1)*n_fm, name='en'+str(num))(pool3)

    pool4 = MaxPooling2D(pool_size=(2, 2))(encoder4)
    num += 1
    buttom = denseBlock(growth_rate=2**(num-1)*growth_rate, output_n=2**(num-1)*n_fm, name='buttom')(pool4)

    # expansive path
    num -= 1
    u4 = UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(buttom)
    u4 = convBlock(n_filter = 2**(num-1)*n_fm, kernel_size=(1,1), name = 'de_up4'+str(num))(u4)
    add4 = Add(name= 'add'+str(num))([encoder4, u4])
    decoder4_1 = convBlock(n_filter = 2**(num-2)*n_fm, kernel_size=(1,1), name = 'de'+str(num))(add4)
    decoder4_2 = denseBlock(growth_rate=2**(num-1)*growth_rate, output_n=2**(num-1)*n_fm, name='de'+str(num))(decoder4_1)

    num -= 1
    u3 = UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(decoder4_2)
    u3 = convBlock(n_filter = 2**(num-1)*n_fm, kernel_size=(1,1), name = 'de_up3'+str(num))(u3)
    add3 = Add(name= 'add'+str(num))([encoder3, u3])
    decoder3_1 = convBlock(n_filter = 2**(num-2)*n_fm, kernel_size=(1,1), name = 'de'+str(num))(add3)
    decoder3_2 = denseBlock(growth_rate=2**(num-1)*growth_rate, output_n=2**(num-1)*n_fm, name='de'+str(num))(decoder3_1)

    num -= 1
    u2 = UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(decoder3_2)
    u2 = convBlock(n_filter = 2**(num-1)*n_fm, kernel_size=(1,1), name = 'de_up2'+str(num))(u2)
    add2 = Add(name= 'add'+str(num))([encoder2, u2])
    decoder2_1 = convBlock(n_filter = 2**(num-2)*n_fm, kernel_size=(1,1), name = 'de'+str(num))(add2)
    decoder2_2 = denseBlock(growth_rate=2**(num-1)*growth_rate, output_n=2**(num-1)*n_fm, name='de'+str(num))(decoder2_1)

    num -= 1
    u1 = UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(decoder2_2)
    u1 = convBlock(n_filter = 2**(num-1)*n_fm, kernel_size=(1,1), name = 'de_up1'+str(num))(u1)
    add1 = Add(name= 'add'+str(num))([encoder1_2, u1])
    decoder1_1 = convBlock(n_filter = 2**(num-2)*n_fm, kernel_size=(1,1), name = 'de'+str(num))(add1)
    decoder1_2 = denseBlock(growth_rate=2**(num-1)*growth_rate, output_n=2**(num-1)*n_fm, name='de'+str(num))(decoder1_1)

    unet_output = Conv2D(filters=1, kernel_size=(1,1), padding='same')(decoder1_2)
    unet_output = Activation('sigmoid', name='seg')(unet_output)
    model = Model(inputs = [inputs], outputs = [unet_output])
    return model

def dice_metric(y_true, y_pred):
    dice = lambda x: 2 * x[0] / (x[1] + x[2] + 1e-6)
    reshape = lambda x: K.reshape(x, (-1, height * width, 1))
    div_tmp_AiB = K.sum(reshape(y_true * y_pred), axis=1)
    div_tmp_A = K.sum(reshape(y_true), axis=1)
    div_tmp_B = K.sum(reshape(y_pred), axis=1)
    div_dice = dice([div_tmp_AiB, div_tmp_A, div_tmp_B])
    return div_dice

def construct_model():
    model = dense_unet(growth_rate = 8, n_fm = 64, input_size = (height,width,1))
    return model

def load_model(weight_file):
    print("loading model")
    loaded_model = construct_model()
    loaded_model.load_weights(weight_file)
    print("loaded model")
    return loaded_model

from scipy.ndimage import label, generate_binary_structure
def seperated_aspine(original_label, flag = True):
    label_cat = []
    s = generate_binary_structure(2,2)
    labeled_array, num_features = label(original_label, structure=s)
    print('num_features:',num_features)
    if(flag):
        upperbound = 20
    else:
        upperbound = num_features
    for area_number in range(0,upperbound):
        one = np.where(labeled_array == area_number+1,1,0)
        label_cat.append(one)
    label_cat = np.array(label_cat)
    print(label_cat.shape)
    print(label_cat.shape)
    return label_cat, np.array(labeled_array)

def calculate_dice(y_true, y_pred):
    dice = lambda x: 2 * x[0] / (x[1] + x[2] + 1e-6)
    reshape = lambda x: np.reshape(x, (-1, height * width, 1))
    div_tmp_AiB = np.sum(reshape(y_true * y_pred), axis=1)
    div_tmp_A = np.sum(reshape(y_true), axis=1)
    div_tmp_B = np.sum(reshape(y_pred), axis=1)
    div_dice = dice([div_tmp_AiB, div_tmp_A, div_tmp_B])
    return div_dice

def all_dice(predict_sep, gt_sep):
    sep_dice_result = np.zeros(20)
    for idx, sep in enumerate(gt_sep):
        max_dice = 0
        for jdx, sepy in enumerate(predict_sep):
            re = calculate_dice(sep, sepy)
            if(re > max_dice):
                max_dice = re
        sep_dice_result[idx] = max_dice
    return (sep_dice_result)

def run(model_file_name, image_file_name, gt_file_name):
    unet_model = load_model(model_file_name)

    source = cv2.imread(image_file_name, 0)
    source = source/255
    source = source[:, 50:450]
    gt = cv2.imread(gt_file_name, 0)
    gt = gt/255
    gt = gt[:, 50:450]

    temp = np.expand_dims(source, axis=0)
    temp = np.expand_dims(temp, axis=-1)
    predict = unet_model.predict(temp).reshape(height, width)
    predict = np.where(predict > 0.8, 1, 0)
    # seperate aspine
    predict_sep, x = seperated_aspine(predict, False)
    gt_sep, y = seperated_aspine(gt.reshape(height,width))
    # dice
    sep_dice_result = all_dice(predict_sep, gt_sep)
    dice_result = calculate_dice(predict, gt.reshape(height,width))

    temp = predict.astype(np.uint8)
    contours, hierarchy = cv2.findContours(temp,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    empty = np.zeros([height, width, 3])
    cv2.drawContours(empty,contours,-1,(1,0,0),3)

    source = np.expand_dims(source, axis=-1)
    ori = np.repeat(source, 3, axis=-1).astype(np.float64)
    img_add = cv2.addWeighted(ori, 0.5, empty, 0.5, 0)

    import matplotlib
    matplotlib.image.imsave("./output.png", img_add)

    return sep_dice_result, dice_result

if __name__ == '__main__':
    run('./fold1.h5', './data/f01/image/0001.png', './data/f01/label/0001.png')
