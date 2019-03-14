# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""CIFAR dataset input module.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math


def build_input(dataset, data_path, batch_size, mode):
    """Build CIFAR image and labels.

    Args:
      dataset: Either 'cifar10' or 'cifar100'.
      data_path: Filename for data.
      batch_size: Input batch size.
      mode: Either 'train' or 'eval'.
    Returns:
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
    Raises:
      ValueError: when the specified dataset is not supported.
    """
    image_size = 32
    if dataset == 'cifar10':
        label_bytes = 1
        label_offset = 0
        num_classes = 10
    elif dataset == 'cifar100':
        label_bytes = 1
        label_offset = 1
        num_classes = 100
    else:
        raise ValueError('Not supported dataset %s', dataset)

    depth = 3
    image_bytes = image_size * image_size * depth
    record_bytes = label_bytes + label_offset + image_bytes

    data_files = tf.gfile.Glob(data_path)
    file_queue = tf.train.string_input_producer(data_files, shuffle=True)
    # Read examples from files in the filename queue.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, value = reader.read(file_queue)

    # Convert these examples to dense labels and processed images.
    record = tf.reshape(tf.decode_raw(value, tf.uint8), [record_bytes])
    label = tf.cast(tf.slice(record, [label_offset], [label_bytes]), tf.int32)
    # Convert from string to [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(tf.slice(record, [label_offset + label_bytes], [image_bytes]),
                             [depth, image_size, image_size])
    # Convert from [depth, height, width] to [height, width, depth].
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

    if mode == 'train':
        image = tf.image.resize_image_with_crop_or_pad(
            image, image_size + 4, image_size + 4)
        image = tf.random_crop(image, [image_size, image_size, 3])
        image = tf.image.random_flip_left_right(image)
        # Brightness/saturation/constrast provides small gains .2%~.5% on cifar.
        # image = tf.image.random_brightness(image, max_delta=63. / 255.)
        # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        image = tf.image.per_image_standardization(image)

        example_queue = tf.RandomShuffleQueue(
            capacity=16 * batch_size,
            min_after_dequeue=8 * batch_size,
            dtypes=[tf.float32, tf.int32],
            shapes=[[image_size, image_size, depth], [1]])
        num_threads = 16
    else:
        image = tf.image.resize_image_with_crop_or_pad(
            image, image_size, image_size)
        image = tf.image.per_image_standardization(image)

        example_queue = tf.FIFOQueue(
            3 * batch_size,
            dtypes=[tf.float32, tf.int32],
            shapes=[[image_size, image_size, depth], [1]])
        num_threads = 1

    example_enqueue_op = example_queue.enqueue([image, label])
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
        example_queue, [example_enqueue_op] * num_threads))

    # Read 'batch' labels + images from the example queue.
    images, labels = example_queue.dequeue_many(batch_size)
    labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    labels = tf.sparse_to_dense(
        tf.concat(values=[indices, labels], axis=1),
        [batch_size, num_classes], 1.0, 0.0)

    assert len(images.get_shape()) == 4
    assert images.get_shape()[0] == batch_size
    assert images.get_shape()[-1] == 3
    assert len(labels.get_shape()) == 2
    assert labels.get_shape()[0] == batch_size
    assert labels.get_shape()[1] == num_classes

    # Display the training images in the visualizer.
    tf.summary.image('images', images)
    return images, labels


def eval_data_input(eval_data_path, EVAL_NUM, show_images=True):
    tf.logging.info('Loading the eval data from {}'.format(eval_data_path))
    label_bytes = 1  # 2 for CIFAR-100
    height = 32
    width = 32
    depth = 3
    TRAIN_NUM = 10000
    image_bytes = height * width * depth + 1
    batch_bytes = TRAIN_NUM * image_bytes
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    with open(eval_data_path, 'rb') as file:
        byte_stream = file.read(batch_bytes)
        data = np.frombuffer(byte_stream, dtype=np.uint8).reshape((TRAIN_NUM, image_bytes))
        # print('The data shape is {}'.format(data.shape))
        image = data[:, 1:].reshape((TRAIN_NUM, depth, height, width)).transpose((0, 2, 3, 1))
        label = data[:, 0]

    num_classes = 10
    data_format = 'channels_last'
    batch_size = 1

    tf.logging.info(
        'Select part of the data from 0 to {0} and change the label into one_hot encoding,and normalized'.format(
            EVAL_NUM))

    X_test = image[0:EVAL_NUM]
    Y_test = np.eye(num_classes)[label[0:EVAL_NUM]]
    if show_images == True:
        show_eval_images(X_test, Y_test, EVAL_NUM)

    tf.logging.info('Preprocessing the eval image:\n \
    We do not using the queue runner pipeline,just use the numpy to load data and do stardardization for each image!!!')
    mean = np.expand_dims(X_test.reshape((EVAL_NUM, -1)).mean(axis=1), axis=1)
    std = np.expand_dims(X_test.reshape((EVAL_NUM, -1)).std(axis=1), axis=1)
    # print(mean.shape, std.shape)
    images_std = ((X_test.reshape((EVAL_NUM, -1)) - mean) / std).reshape((EVAL_NUM, height, width, depth))
    return images_std, Y_test


def show_eval_images(images, labels, EVAL_NUM):
    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    label_dict = {}
    for key, value in enumerate(label_names):
        label_dict[key] = value

    tf.logging.info('Display the image from the eval data')
    # index = 1
    # print('The label of {0}th image is {1}:{2}'.format(index, label[index], label_dict[label[index]]))
    plt.figure(figsize=(16, 16))
    for index in range(EVAL_NUM):
        if index < 16:
            plt.subplot(math.ceil(16 / 4), 4, index + 1)
            plt.imshow(images[index])
            plt.axis('off')
            plt.title(label_dict[np.argmax(labels[index])])
        else:
            pass
    plt.show()
