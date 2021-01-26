# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
# 마스크가 있는 부분을 중심으로 변환시켜야한다. 즉, 다른 부분은 변환 x (이상적)
# 되도록 마스크(입, 코) 부분만을 중점적으로 변환 (다른 부분도 약간은 변할 수 있다.)
# 모델을 그러면 부분적으로 짜야한다. (지금 다운받고 있는 데이터는 CASIA 데이터인데, 이것이 잘 될지 안될지는 미지수이다.)

Conv2D = tf.keras.layers.Conv2D
BatchNorm = tf.keras.layers.BatchNormalization
ReLU = tf.keras.layers.ReLU
LeakyReLU = tf.keras.layers.LeakyReLU
l2 = tf.keras.regularizers.l2

def conv_relu_bn(inputs,
                 filters,
                 kernel_size,
                 strides,
                 padding,
                 weight_decay):

    h = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding=padding,
               use_bias=False,
               kernel_regularizer=weight_decay)(inputs)
    h = BatchNorm()(h)
    h = ReLU()(h)

    return h

class gabor_filters(tf.keras.layers.Layer):
    # https://thinkpiece.tistory.com/304
    # v - Gaussian distribution
    # Theta - Kernel 방향성 (추출하는 edge 방향 결정, 0일때는 PI --> 수직방향)
    # Lambda - 반복주기 결정
    # Psi - Gabor filter의 중간값 결정
    # Gamma - Filter의 가로 세로 비율 (1이면 같은 비율, 값이 작아지면 타원형으로 변환됨)

    # 식 - exp( (-x^2 + gamma^2*y^2) / 2*v) * cos(2*PI*x/Lambda + Psi)
    # x = x*cos(theta) + y*sin(theta)
    # y = -x*sin(theta) + y*cos(theta)

    # 모든 파라미터를 learnable 하게 만들자
    # https://stackoverflow.com/questions/43247620/using-gabor-filter-in-tensorflow-or-any-other-filter-instead-of-default-one
    def __init__(self, filters, kernel):
        super(gabor_filters, self).__init__()
        self.kernel = kernel
        self.filters = filters
        self.PI = 3.14159

    def build(self, input_shapes):
        # filters shape = [3, 3, input filters, output filters]
        self.gaussian_distri = self.add_weight(shape=[input_shapes[3], self.filters],
                                               initializer="random_normal",
                                               trainable=True)
        self.lamb = self.add_weight(shape=[input_shapes[3], self.filters],
                                    initializer="random_normal",
                                    trainable=True)
        self.gamma = self.add_weight(shape=[input_shapes[3], self.filters],
                                     initializer="random_normal",
                                     trainable=True)
        self.theta = self.add_weight(shape=[input_shapes[3], self.filters],
                                     initializer="random_normal",
                                     trainable=True)
        self.sigma = self.add_weight(shape=[input_shapes[3], self.filters],
                                     initializer="random_normal",
                                     trainable=True)
        self.Psi = self.add_weight(shape=[input_shapes[3], self.filters],
                                   initializer="random_normal",
                                   trainable=True)

        self.gaborkernel_ = np.zeros([self.kernel, self.kernel, input_shapes[3], self.filters])
        for x in range(self.kernel):
            for y in range(self.kernel):
                x_ = x * tf.cos(self.theta) + y * tf.sin(self.theta)
                y_ = -x * tf.sin(self.theta) + y * tf.cos(self.theta)
                self.gaborkernel_[x, y, :, :] = tf.exp(-tf.pow(x_, 2)+tf.pow(self.gamma, 2)*tf.pow(y_, 2) \
                    / 2*tf.pow(self.gaussian_distri, 2)) * tf.cos(2*self.PI*x_/self.lamb + self.Psi)
        
        # 이걸 랜덤하게 어떻게 만들면될까??

    def call(self, inputs, **kwargs):
        h = Conv2D(filters=self.filters,
                   kernel_size=self.kernel,
                   padding="same",
                   kernel_initializer=tf.keras.initializers.Constant(self.gaborkernel_))(inputs)

        return h

def generator(input_shape=(256, 256, 3), weight_decay=0.000005):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = conv_relu_bn(inputs=h,
                     filters=64,
                     kernel_size=7,
                     strides=1,
                     padding="valid",
                     weight_decay=l2(weight_decay)) # [256, 256, 64]
    h = conv_relu_bn(inputs=h,
                     filters=64,
                     kernel_size=3,
                     strides=1,
                     padding="same",
                     weight_decay=l2(weight_decay)) # [256, 256, 64]
    h = gabor_filters(64, kernel=3)(h)  # [256, 256, 64]
    
    h = tf.keras.layers

    h = conv_relu_bn(inputs=h,
                     filters=128,
                     kernel_size=3,
                     strides=1,
                     padding="same",
                     weight_decay=l2(weight_decay)) # [128, 128, 128]
    h = conv_relu_bn(inputs=h,
                     filters=128,
                     kernel_size=3,
                     strides=1,
                     padding="same",
                     weight_decay=l2(weight_decay)) # [128, 128, 128]
    h = conv_relu_bn(inputs=h,
                     filters=128,
                     kernel_size=3,
                     strides=1,
                     padding="same",
                     weight_decay=l2(weight_decay)) # [128, 128, 128]
    h = gabor_filters(128, kernel=3)(h)  # [128, 128, 128]


    return tf.keras.Model(inputs=inputs, outputs=h)

model = generator()
model.summary()