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
import time
import numpy as np
from PIL import Image
import random
import glob
import argparse
from matplotlib import pyplot as plt

def main(args):
    original_folder = args.original.removesuffix("/")
    masked_folder = args.masked.removesuffix("/")
    IMAGE_SIZE = 128

    K1 = args.k1
    N = len(glob.glob(original_folder + "/*.jpg"))
    im_x_arr = [None for _ in range(N * K1)]
    im_y_arr = [None for _ in range(N * K1)]
    p = np.random.permutation(N * K1)
    for i in range(N):
        j = i + 1
        im_y = Image.open(original_folder + "/"+str(j)+".jpg")
        im_m = Image.open(masked_folder + "/"+str(j)+".jpg")

        w, h = im_y.size
        if w >= h:
            t = (int(w / h * IMAGE_SIZE), IMAGE_SIZE)
        else:
            t = (IMAGE_SIZE, int(h / w * IMAGE_SIZE))
        im_y = np.array(im_y.resize(t, Image.LANCZOS))
        im_x = np.array(im_m.resize(t, Image.LANCZOS))

        x_size = IMAGE_SIZE
        y_size = IMAGE_SIZE
        x_max = im_x.shape[0] - IMAGE_SIZE
        if x_max < 0:
            x_max = 0
            x_size = im_x.shape[0]
        y_max = im_x.shape[1] - IMAGE_SIZE
        if y_max < 0:
            y_max = 0
            y_size = im_x.shape[1]
        for k in range(K1):
            x = random.randint(0, x_max)
            y = random.randint(0, y_max)

            im_x_cropped = im_x[x:x+x_size, y:y+y_size, :]
            im_x_arr[p[K1 * i + k]] = im_x_cropped

            im_y_cropped = im_y[x:x+x_size, y:y+y_size, :]
            im_y_arr[p[K1 * i + k]] = im_y_cropped

        print("\r"+str(j)+"/"+str(N)+" wait until all images are loaded...", end="")

    BATCH_SIZE = args.batch
    X = tf.data.Dataset.from_tensor_slices(np.array(im_x_arr).astype("float32")).batch(BATCH_SIZE)
    Y = tf.data.Dataset.from_tensor_slices(np.array(im_y_arr).astype("float32")).batch(BATCH_SIZE)

    del im_x_arr
    del im_y_arr

    database = []
    database_folder = args.database.removesuffix("/")
    K2 = args.k2
    for f in glob.glob(database_folder+"/*.jpg"):
        im = Image.open(f)
        w, h = im.size
        if w >= h:
            t = (int(w / h * IMAGE_SIZE), IMAGE_SIZE)
        else:
            t = (IMAGE_SIZE, int(h / w * IMAGE_SIZE))
        im_ary = np.array(im.resize(t, Image.LANCZOS))
        x_size = IMAGE_SIZE
        y_size = IMAGE_SIZE
        x_max = im_ary.shape[0] - IMAGE_SIZE
        if x_max < 0:
            x_max = 0
            x_size = im_ary.shape[0]
        y_max = im_ary.shape[1] - IMAGE_SIZE
        if y_max < 0:
            y_max = 0
            y_size = im_ary.shape[1]
        for k in range(K2):
            x = random.randint(0, x_max)
            y = random.randint(0, y_max)

            im_cropped = im_ary[x:x+x_size, y:y+y_size, :]
            database.append(im_cropped)
    
    database = np.array(database).astype("float32")
    database = util.gray(database)
    print("Loading the model...")

    G = network.get_model(args.n)
    optimzer = network.get_optimizer()

    checkpoint_path = "./checkpoints2"

    ckpt = tf.train.Checkpoint(G=G, optimzer=optimzer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')
    print("Done.")

    def train_step(x, y, n):
        loss = None
        with tf.GradientTape(persistent=True) as tape:
            wsfs = network.get_wsfs(x, database, n)
            inp = tf.concat([x, wsfs], 3)
            out = G(inp, training=True)
            loss = network.loss(out, y)
        gradients = tape.gradient(loss, G.trainable_variables)
        optimzer.apply_gradients(zip(gradients, G.trainable_variables))
        return loss


    EPOCH = args.epoch
    losses = []
    for e in range(EPOCH):
        print("epoch {0} started.".format(e + 1))
        start = time.time()
        losses.append(0)
        n = 1
        for x, y in tf.data.Dataset.zip((X, Y)):
            loss = train_step(x, y, args.n)
            losses[e] += loss.numpy() / (500 * K1)
            print("\r{:.2f}% ".format(n * BATCH_SIZE / (5 * K1)), end="")
            n += 1
        print(" ... Done")
        if (e + 1) % 10 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(e+1, ckpt_save_path))
        print("Time taken for epoch {0} was {1}. Loss: {2}".format(e+1, time.time() - start, losses[e]))
    
    y = np.array(losses)
    x = np.array(range(EPOCH))
    plt.plot(x, y)
    plt.savefig("losses.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', help='default=\"original\". The name of a folder which includes the original images. The images will be resized as the longer edge become 128 px conserving its aspect, then k1 pieces of 128x128 regions will be randomly cropped. IMPORTANT: The images under this folder must be named in order. ex) 12.jpg.', default="original")
    parser.add_argument('--masked', help='default=\"masked\". The name of a folder which includes the masked images. The images will be resized as the longer edge become 128 px conserving its aspect, then k1 pieces of 128x128 regions will be randomly cropped. IMPORTANT: The images under this folder must be named in order. ex) 12.jpg.', default="masked")
    parser.add_argument('--k1', help="default=4. The number of regions cropped from original and masked images.",default=4, type=int)
    parser.add_argument('--database', help='default\"database\". The name of a folder which includes the dataset images used as WSFS. The images will be resized as the longer edge become 128 px conserving its aspect, then k pieces of 128x128 regions will be randomly cropped.', default="database")
    parser.add_argument('--k2', help="default=1. The number of regions cropped from original and masked images.",default=1, type=int)
    parser.add_argument('--n', help="default=10. The size of WSFS",default=10, type=int)
    parser.add_argument('--epoch', help='default=100. The number of epoch', default=100, type=int)
    parser.add_argument('--checkpoints', help='default=\"checkpoints\". The name of a folder in which checkpoints files will be created.', default="checkpoints")
    parser.add_argument('--batch', help='default=1. The batch size', default=1, type=int)
    args = parser.parse_args()
 
    import tensorflow as tf
    from utils import network
    from utils import util

    main(args)