# -*- coding:utf-8 -*-
from absl import flags
from random import random, shuffle
from model_V4 import *

import matplotlib.pyplot as plt
import os
import sys

#gpus = tf.config.experimental.list_physical_devices("GPU")
#if gpus:
#    try:
#        tf.config.experimental.set_memory_growth(gpus[0], True)
#    except RuntimeError as e:
#        print(e)

flags.DEFINE_string("txt_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/train80_test20/train_1.txt", "Trainaing text path")

flags.DEFINE_string("img_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/Crop_dlib/", "Training image path")

flags.DEFINE_string("te_txt_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/train80_test20/test_1.txt", "Test text path")

flags.DEFINE_string("te_img_path", "D:/[1]DB/[1]second_paper_DB/original_MORPH/Crop_dlib/", "Test image path")

flags.DEFINE_integer("img_size", 256, "Training and testing image size")

flags.DEFINE_integer("num_classes", 54, "Number of classes")

flags.DEFINE_integer("epochs", 500, "Total epochs during training")

flags.DEFINE_integer("batch_size", 8, "Training batch size")

flags.DEFINE_float("lr", 0.0001, "Leanring rate")

flags.DEFINE_bool("pre_checkpoint", False, "True or False")

flags.DEFINE_string("pre_checkpoint_path", "", "Restored or test checkpoint path")

flags.DEFINE_string("save_checkpoint", "", "Save checkpoint path")

flags.DEFINE_bool("train", True, "True or False")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

optim = tf.keras.optimizers.Adam(FLAGS.lr)

def func_(img_path, lab_path):

    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])

    if random() > 0.5:
        img = tf.image.flip_left_right(img)

    img = tf.image.per_image_standardization(img)

    if lab_path == 74:
        label = (lab_path - 2) - 16
        label = tf.one_hot(label, FLAGS.num_classes)
    elif lab_path == 75:
        label = (lab_path - 2) - 16
        label = tf.one_hot(label, FLAGS.num_classes)
    elif lab_path == 76:
        label = (lab_path - 2) - 16
        label = tf.one_hot(label, FLAGS.num_classes)
    elif lab_path == 77:
        label = (lab_path - 2) - 16
        label = tf.one_hot(label, FLAGS.num_classes)
    else:
        label = lab_path - 16
        label = tf.one_hot(label, FLAGS.num_classes)

    return img, label

def te_func_(img_path, lab_path):

    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])

    img = tf.image.per_image_standardization(img)

    if lab_path == 74:
        label = (lab_path - 2) - 16
    elif lab_path == 75:
        label = (lab_path - 2) - 16
    elif lab_path == 76:
        label = (lab_path - 2) - 16
    elif lab_path == 77:
        label = (lab_path - 2) - 16
    else:
        label = lab_path - 16

    return img, label

@tf.function
def cal_loss(model, images, labels):

    with tf.GradientTape() as tape:

        logits = model(images, True)

        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(labels, logits)

    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return loss

@tf.function
def cal_mae(model, images, labels):

    logits = model(images, False)
    proba = tf.nn.softmax(logits, 1)
    predict = tf.argmax(proba, 1, output_type=tf.int32)

    ae = tf.reduce_sum(tf.abs(predict[0] - labels[0]))

    return ae

def main():
    
    model = age_estimation_model(input_shape=(FLAGS.img_size, FLAGS.img_size, 3),
                                 num_classes=FLAGS.num_classes)
    model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model,
                                   optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored the latest checkpoint files!!!")

    if FLAGS.train:
        count = 0

        ######################################################################
        # loss는 어떻게 정해야할지 고민해보자 (일단은 그냥 돌려보고 성능의 개선이)
        # 기존보다 괜찮다면 바꾸고
        # 성능의 개선이 없다 싶으면 모델을 수정하면된다.
        ######################################################################
        img_data = np.loadtxt(FLAGS.txt_path, dtype="<U100", skiprows=0, usecols=0)
        img_data = [FLAGS.img_path + img for img in img_data]
        lab_data = np.loadtxt(FLAGS.txt_path, dtype=np.int32, skiprows=0, usecols=1)

        te_img_data = np.loadtxt(FLAGS.te_txt_path, dtype="<U100", skiprows=0, usecols=0)
        te_img_data = [FLAGS.te_img_path + img for img in te_img_data]
        te_lab_data = np.loadtxt(FLAGS.te_txt_path, dtype=np.int32, skiprows=0, usecols=1)        

        for epoch in range(FLAGS.epochs):

            A = list(zip(img_data, lab_data))
            shuffle(A)
            img_data, lab_data = zip(*A)
            img_data, lab_data = np.array(img_data), np.array(lab_data)

            gener = tf.data.Dataset.from_tensor_slices((img_data, lab_data))
            gener = gener.shuffle(len(img_data))
            gener = gener.map(func_)
            gener = gener.batch(FLAGS.batch_size)
            gener = gener.prefetch(tf.data.experimental.AUTOTUNE)

            te_gener = tf.data.Dataset.from_tensor_slices((te_img_data, te_lab_data))
            te_gener = te_gener.map(te_func_)
            te_gener = te_gener.batch(2)
            te_gener = te_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_iter = iter(gener)
            tr_idx = len(img_data)  // FLAGS.batch_size

            for step in range(tr_idx):
                batch_images, batch_labels = next(tr_iter)

                loss = cal_loss(model, batch_images, batch_labels)

                if count % 10 == 0:
                    print("Epoch: {} [{}/{}] total_loss = {}".format(
                        epoch, step + 1, tr_idx, loss))

                if count % 1000 == 0:
                    num_ = int(count // 1000)
                    model_dir = "%s/%s" % (FLAGS.save_checkpoint, num)
                    if not os.path.isdir(model_dir):
                        os.makedirs(model_dir)
                        print("Make {} files to save checkpoint".format(num_))

                    ckpt = tf.train.Checkpoint(model=model,
                                               optim=optim)
                    ckpt_dir = model_dir + "/" + "paper_4_age_estimation_{}.ckpt".format(count)
                    ckpt.save(ckpt_dir)

                if count % 5500 == 0 and count != 0:
                    te_iter = iter(te_gener)
                    te_idx = len(te_img_data) // 2
                    ae = 0
                    for j in range(te_idx):
                        images, labels = next(te_iter)
                        ae += cal_mae(model, images, labels)

                        if j % 1000 == 0:
                            print("processing mae (test step {}) = {}".format(j + 1, ae / (j + 1)))

                    mae = ae / len(te_img_data)
                    print("MAE ({} steps) = {}".format(count + 1, mae))

                count += 1

if __name__ == "__main__":
    main()
