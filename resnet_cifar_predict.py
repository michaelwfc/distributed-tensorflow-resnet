from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import matplotlib.pylab as plt

import tensorflow as tf
import resnet_model
import resnet_model_official
import math


eval_data_path = r'D:\wangfeicheng\Tensorflow\cifar10-tensorflow\cifar-10-binary\cifar-10-batches-bin\test_batch.bin'
train_dir = r'D:\wangfeicheng\Tensorflow\docker-multiple\ResNet\resnet50-cifar-ckpt-20190218\model.ckpt-107738'
eval_dir = r'D:\wangfeicheng\Tensorflow\docker-multiple\ResNet\resnet50-cifar-eval'

FLAGS = tf.app.flags.FLAGS
flags = tf.app.flags
flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100 or imagenet.')
flags.DEFINE_string('mode', 'eval', 'train or eval.')
flags.DEFINE_integer('EVAL_NUM', 100, 'The number of images we want to predict .')
flags.DEFINE_string('train_data_path','',
                           'Filepattern for training data.')
flags.DEFINE_string('eval_data_path',eval_data_path,
                           'Filepattern for eval data')
flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.app.flags.DEFINE_string('train_dir', train_dir,
                           'Directory to keep training outputs.')
flags.DEFINE_string('eval_dir', eval_dir,
                           'Directory to keep eval outputs.')
flags.DEFINE_integer('eval_batch_count', 50,
                            'Number of batches to eval.')
flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')
flags.DEFINE_string('log_root', '',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
flags.DEFINE_integer('num_gpus', 0,
                            'Number of gpus used for training. (0 or 1)')

FLAGS = flags.FLAGS

# Dimensions of the images in the CIFAR-10 dataset.
# See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
# input format.
'''
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 
There are 50000 training images and 10000 test images. 

The dataset is divided into five training batches and one test batch, each with 10000 images. 
The test batch contains exactly 1000 randomly-selected images from each class. 
The training batches contain the remaining images in random order, but some training batches may contain more images from one class 
than another. Between them, the training batches contain exactly 5000 images from each class. 
'''

tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.info('Loading the eval data from {}'.format(FLAGS.eval_data_path))
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

with open(FLAGS.eval_data_path, 'rb') as file:
    byte_stream = file.read(batch_bytes)
    data = np.frombuffer(byte_stream, dtype=np.uint8).reshape((TRAIN_NUM, image_bytes))
    # print('The data shape is {}'.format(data.shape))
    image = data[:, 1:].reshape((TRAIN_NUM, depth, height, width)).transpose((0, 2, 3, 1))
    label = data[:, 0]

label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
label_dict = {}
for key, value in enumerate(label_names):
    label_dict[key] = value

# print(data[0,:])
# print(image.shape)
# print(label.shape)



num_classes = 10
data_format = 'channels_last'
batch_size = 1

tf.logging.info('Select part of the data from 0 to {0} and change the label into one_hot encoding,and normalized'.format(FLAGS.EVAL_NUM))

X_test = image[0:FLAGS.EVAL_NUM]
Y_test = np.eye(num_classes)[label[0:FLAGS.EVAL_NUM]]


tf.logging.info('Resetting the default graph')
tf.reset_default_graph()
g = tf.Graph()


class ResNet(object):
    """ResNet model."""

    def __init__(self, hps, images, labels, mode):
        '''
        ResNet constructor.
        Args:
          hps: Hyperparameters.
          images: Batches of images. [batch_size, image_size, image_size, 3]
          labels: Batches of labels. [batch_size, num_classes]
          mode: One of 'train' and 'eval'.
        '''
        self.hps = hps
        self._images = images
        self.labels = labels
        self.mode = mode

        self._extra_train_ops = []

    def build_graph(self, istrain=True):
        """Build a whole graph for the model."""
        self.global_step = tf.train.get_or_create_global_step()
        self._build_model(istrain)
        if self.mode == 'train':
            self._build_train_op()
        self.summaries = tf.summary.merge_all()

    def _build_model(self, istrain):
        """Build the core model within the graph."""
        if FLAGS.dataset == 'cifar10':
            network = resnet_model_official.cifar10_resnet_v2_generator(resnet_size=50,
                                                                        num_classes=self.hps.num_classes,
                                                                        data_format=data_format)
        elif FLAGS.dataset == 'imagenet':
            network = resnet_model_official.imagenet_resnet_v2(resnet_size=50, num_classes=self.hps.num_classes,
                                                               data_format=data_format)

        logits = network(self._images, istrain)
        self.predictions = tf.nn.softmax(logits)
        cross_entropy = tf.losses.softmax_cross_entropy(
            logits=logits, onehot_labels=self.labels
        )
        tf.identity(cross_entropy, name='cross_entropy')
        tf.summary.scalar('cross_entropy', cross_entropy)

        # Add weight decay to the loss.
        self.cost = cross_entropy + self.hps.weight_decay_rate * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

        tf.summary.scalar('cost', self.cost)


hps = resnet_model.HParams(num_classes=num_classes,\
                         lrn_rate=0.1,
                         weight_decay_rate = 0.0002,
                         optimizer='mom')


def create_config_proto():
    """Returns session config proto.
    Args:
    params: Params tuple, typically created by make_params or
            make_params_from_flags.
    """
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    return config




summary_writer = tf.summary.FileWriter(eval_dir)


with tf.Session(config=create_config_proto(),graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    tf.logging.info('Building the graph:\n \
    We bulid the model by script,but give the inputs as placeholder, then we can use feed_dict to feed the evaluation data to predict, without using the data pipeline')
    # images, labels = cifar_input.build_input(dataset, eval_data_dir, batch_size, mode)
    # model = ResNet(hps, images, labels, mode)

    X = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3), name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='Y')
    model = ResNet(hps, X, Y, FLAGS.mode)
    model.build_graph(istrain=False)

    '''
    try:
        ckpt_state = tf.train.get_checkpoint_state(train_dir)
    except tf.errors.OutOfRangeError as e:
        tf.logging.error('Cannot restore checkpoint: %s', e)
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
        tf.logging.info('No model to eval yet at %s', train_dir)
    tf.logging.info('Loading the checkpoint from train_dir:{0}'.format(ckpt_state.model_checkpoint_path))
    '''
    tf.logging.info('Loading the checkpoint from train_dir:{0}'.format(FLAGS.train_dir))

    saver = tf.train.Saver()


    #saver.restore(sess, ckpt_state.model_checkpoint_path)
    # Assuming model_checkpoint_path looks something like:
    #   /my-favorite-path/cifar10_train/model.ckpt-0,

    #just restoring the ckpt data
    #saver.restore(sess,tf.train.latest_checkpoint(train_dir))

    saver.restore(sess, FLAGS.train_dir)

    tf.logging.info('Preprocessing the eval image:\n \
    We do not using the queue runner pipeline,just use the numpy to load data and do stardardization for each image!!!')
    mean = np.expand_dims(X_test.reshape((FLAGS.EVAL_NUM,-1)).mean(axis=1),axis=1)
    std = np.expand_dims(X_test.reshape((FLAGS.EVAL_NUM,-1)).std(axis=1),axis=1)
    #print(mean.shape, std.shape)
    images_std = ((X_test.reshape((FLAGS.EVAL_NUM,-1)) - mean)/std).reshape((FLAGS.EVAL_NUM,height,width,depth))
    #print(images_std.shape)

    (summaries, loss, predictions, truth, train_step) = sess.run(
        [model.summaries, model.cost, model.predictions,model.labels, model.global_step],
        feed_dict={X:images_std,Y:Y_test})

    truth = np.argmax(truth, axis=1)
    predictions = np.argmax(predictions, axis=1)
    correct_num = np.sum(truth == predictions)
    precision = correct_num/FLAGS.EVAL_NUM
    tf.logging.info('The truth:{0},\nThe label:{1},\nThe prediction:{2},\nThe correct num:{3},\nThe precision:{4}'\
          .format(truth, label[0:FLAGS.EVAL_NUM], predictions, correct_num, precision))

    tf.logging.info('Display the image from the eval data')
    # index = 1
    # print('The label of {0}th image is {1}:{2}'.format(index, label[index], label_dict[label[index]]))
    plt.figure(figsize=(16, 16))
    for index in range(FLAGS.EVAL_NUM):
        plt.subplot(math.ceil(FLAGS.EVAL_NUM /10),10,  index+1)
        plt.imshow(X_test[index])
        plt.axis('off')
        if label[index] == predictions[index]:
            plt.title(label_names[label[index]])
        else:
            plt.title(label_names[label[index]] + '!=' + label_names[predictions[index]],color='red')
    plt.show()


