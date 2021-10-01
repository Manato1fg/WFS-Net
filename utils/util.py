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

import numpy as np
import tensorflow as tf

def binarize_image(img, r=0, g=0, b=0):
    flag = False
    if tf.rank(img) == 3:
        img = tf.reshape(img, [1, img.shape[0], img.shape[1], img.shape[2]])
        flag = True
    ZERO = tf.zeros((img.shape[0], img.shape[1], img.shape[2]))
    ONE = tf.ones((img.shape[0], img.shape[1], img.shape[2]))
    i = tf.where((img[:, :, :, 0] == r) & (img[:, :, :, 1] == g) & (img[:, :, :, 2] == b), ZERO, ONE)
    if flag:
        i = tf.reshape(i, (i.shape[1], i.shape[2]))
    return i

def nega2posi(img):
    return 255 - img

def gray(img):
    flag = False
    if tf.rank(img) == 3:
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        flag = True
    rgbL = pow(img/255.0, 2.2)
    r, g, b = rgbL[:,:,:,0], rgbL[:,:,:,1], rgbL[:,:,:,2]
    grayL = 0.299 * r + 0.587 * g + 0.114 * b  # BT.601
    gray = pow(grayL, 1.0/2.2)*255
    if flag:
        gray = gray.reshape((gray.shape[1], gray.shape[2]))
    return gray

def lbp(img):
    flag = False
    if tf.rank(img) == 2:
        img = img.reshape((1, img.shape[0], img.shape[1]))
        flag = True
    paddings = tf.constant([[0,0], [1, 1], [1, 1]])
    img = tf.pad(img, paddings)
    b, w, h = img.shape
    t = tf.zeros((b, w - 2, h - 2))
    ONE = tf.ones((b, w - 2, h - 2))
    ZERO = tf.zeros((b, w - 2, h - 2))

    i0 = img[:, 1:w-1, 1:h-1]
    i1 = img[:, 1:w-1, 0:h-2]
    i2 = img[:, 0:w-2, 0:h-2]
    i3 = img[:, 0:w-2, 1:h-1]
    i4 = img[:, 0:w-2,   2:h]
    i5 = img[:, 1:w-1,   2:h]
    i6 = img[:, 2:w,     2:h]
    i7 = img[:, 2:w,   1:h-1]
    i8 = img[:, 2:w,   0:h-2]

    t += tf.where(i1 - i0 >= 0, ONE, ZERO) * 1
    t += tf.where(i2 - i0 >= 0, ONE, ZERO) * 2
    t += tf.where(i3 - i0 >= 0, ONE, ZERO) * 4
    t += tf.where(i4 - i0 >= 0, ONE, ZERO) * 8
    t += tf.where(i5 - i0 >= 0, ONE, ZERO) * 16
    t += tf.where(i6 - i0 >= 0, ONE, ZERO) * 32
    t += tf.where(i7 - i0 >= 0, ONE, ZERO) * 64
    t += tf.where(i8 - i0 >= 0, ONE, ZERO) * 128
    if flag:
        t = tf.reshape(t, (t.shape[1], t.shape[2]))
    return t

def mask(mask, img):
    if tf.rank(img) == 3:
        masked = tf.multiply(mask, img)
    else:
        masked = tf.where(tf.equal(tf.stack([mask for _ in range(img.shape[3])], axis=3), 1), img, tf.zeros_like(img))
    return masked