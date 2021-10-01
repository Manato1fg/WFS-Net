"""
MIT License

Copyright (c) 2021 Manato Yo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import tensorflow as tf
from . import util
import numpy as np

"""
input:
n -> the number of images you use as a hint

output -> the Model of WFS-Net
"""
def get_model(n):
    inputs = tf.keras.layers.Input(shape=[None, None, 3 + n])

    initializer = tf.random_normal_initializer(0., 0.02)
    conv1 = tf.keras.layers.Conv2D(64, 5, strides=2, kernel_initializer=initializer, padding="same")(inputs)
    lrelu1 = tf.keras.layers.LeakyReLU(alpha=0.1)(conv1)
    conv2 = tf.keras.layers.Conv2D(128, 5, strides=2, kernel_initializer=initializer, padding="same")(lrelu1)
    lrelu2 = tf.keras.layers.LeakyReLU(alpha=0.1)(conv2)
    conv3 = tf.keras.layers.Conv2D(256, 5, strides=2, kernel_initializer=initializer, padding="same")(lrelu2)
    lrelu3 = tf.keras.layers.LeakyReLU(alpha=0.1)(conv3)
    conv4 = tf.keras.layers.Conv2D(512, 5, strides=2, kernel_initializer=initializer, padding="same")(lrelu3)
    lrelu4 = tf.keras.layers.LeakyReLU(alpha=0.1)(conv4)
    conv5 = tf.keras.layers.Conv2D(1024, 5, strides=2, kernel_initializer=initializer, padding="same")(lrelu4)
    lrelu5 = tf.keras.layers.LeakyReLU(alpha=0.1)(conv5)

    deconv1 = tf.keras.layers.Conv2DTranspose(512, 5, strides=2, kernel_initializer=initializer, activation="relu", padding="same")(lrelu5)
    deconv2 = tf.keras.layers.Conv2DTranspose(256, 5, strides=2, kernel_initializer=initializer, activation="relu", padding="same")(deconv1)
    deconv3 = tf.keras.layers.Conv2DTranspose(128, 5, strides=2, kernel_initializer=initializer, activation="relu", padding="same")(deconv2)
    deconv4 = tf.keras.layers.Conv2DTranspose(64, 5, strides=2, kernel_initializer=initializer, activation="relu", padding="same")(deconv3)
    deconv5 = tf.keras.layers.Conv2DTranspose(3, 5, strides=2, kernel_initializer=initializer, activation="relu", padding="same")(deconv4)
    return tf.keras.Model(inputs=inputs, outputs=deconv5)

def get_optimizer():
    return tf.keras.optimizers.Adam(learning_rate=1e-4)

"""
input:
x -> an output image with shape (batch, 128, 128, channel)
y -> the original image with shape (batch, 128, 128, channel)

output -> tf.Tensor object of the loss function
"""
def loss(x, y):
    _, w, h, c = x.shape
    l2 = tf.square(y - x) / (w * h * c)
    l2 = tf.reduce_sum(l2)
    gray_x = util.gray(x)
    gray_y = util.gray(y)
    t_x = util.lbp(gray_x)
    t_y = util.lbp(gray_y)
    t_x = (t_x - tf.reduce_min(t_x, axis=[1,2])) / (tf.reduce_max(t_x, axis=[1,2]) - tf.reduce_min(t_x, axis=[1,2]))
    t_y = (t_y - tf.reduce_min(t_y, axis=[1,2])) / (tf.reduce_max(t_y, axis=[1,2]) - tf.reduce_min(t_y, axis=[1,2]))
    l = tf.square(t_y - t_x)
    l = tf.reduce_sum(l)
    loss = l2 + l
    return loss

"""
input:
img -> masked input image with shape (batch, 128, 128, channel)
database -> numpy array of grayscaled dataset Φ with shape (N, 128, 128)

output -> Weighted Similar Face Set with shape (n, 128, 128)
"""
def get_wsfs(img, database, n):
    V1 = 6.5025 # (255 * 0.01) ^2
    V2 = 58.5225 # (255 * 0.03) ^2
    
    mask_ary = util.binarize_image(img)
    gray = util.gray(img)
    Cs = []
    for i in range(img.shape[0]):
        mask = mask_ary[i]
        mask = tf.stack([mask for _ in range(len(database))]) 
        masked_database = tf.multiply(mask, database)

        u_i = tf.reduce_mean(masked_database, axis=[1, 2])
        sigma_i = tf.square(masked_database - tf.reshape(u_i, (u_i.shape[0], 1, 1))) 
        sigma_i = tf.reduce_sum(sigma_i, axis=[1,2]) / (sigma_i.shape[1] * sigma_i.shape[2])
        u_i_2 = tf.square(u_i)

        u_d = tf.reduce_mean(gray[i], axis=[0, 1])
        sigma_d = tf.reduce_sum(tf.square(gray[i] - u_d))
        u_d_2 = tf.square(u_d)

        sigma_d_i = tf.multiply(masked_database - tf.reshape(u_i, (u_i.shape[0], 1, 1)), gray[i] - u_d)
        sigma_d_i = tf.reduce_sum(sigma_d_i, axis=[1, 2]) / (sigma_d_i.shape[1] * sigma_d_i.shape[2])

        s = (2 * u_d + V1) *  (2 * sigma_d_i + V2) / ((u_d_2 + u_i_2 + V1) * (sigma_i + sigma_d + V2))
        s -= tf.reduce_min(s)
        # why tf.argsort does not work properly ?
        s = s.numpy()
        s_arg = np.argsort(s)[::-1]

        rn = database[s_arg]
        rn = rn[:n]
        w = s[s_arg][:n]
        w /= np.sum(s)
        w = w.reshape((w.shape[0], 1, 1))
        C = rn * w
        Cs.append(C)
    Cs = np.array(Cs)
    Cs = np.transpose(Cs, (0,2,3,1))
    return Cs