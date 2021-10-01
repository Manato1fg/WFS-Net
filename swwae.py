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

"""
This file is unused currently but in the future, 
this method may be one of the options which replaces 
the method to pick up WSFS.
"""

import argparse
from PIL import Image
import numpy as np
import glob

def get_model():
    return tf.keras.applications.VGG19()

def main(args):
    database = []
    database_folder = args.dataset.removesuffix("/")
    for f in glob.glob(database_folder+"/*.jpg"):
        im = Image.open(f)
        database.append(np.array(im))
    database = np.array(database).astype("float32")
    if database.ndim == 3:
        database = database.reshape([database.shape[0],database.shape[1], database.shape[2], 1])
    
    if database.shape[3] == 1:
        database = np.concatenate([database for _ in range(3)], axis=3)

    _, w, h, _ = database.shape
    pad_left   = (224 - w) // 2
    pad_right  = pad_left if w % 2 == 0 else pad_left + 1
    pad_top    = (224 - h) // 2
    pad_bottom = pad_top if h % 2 == 0 else pad_top + 1
    database = np.pad(database, [[0, 0], [pad_left, pad_right], [pad_top, pad_bottom], [0, 0]]) # (batch, 224, 224, 3)

    model = get_model()

    y_database = model.predict(database)

    data = np.random.randn(1, 224, 224, 3)
    y_data = model.predict(data)
    y = np.square(y_database - y_data)
    y = np.sum(y, axis=1)
    index_y = np.argsort(y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', help='the name of a folder which includes the dataset images used as WSFS. All JPEG files under this folder will be loaded as a dataset for WSFS.', required=True, default="database")
    args = parser.parse_args()
    import tensorflow as tf

    main(args)

