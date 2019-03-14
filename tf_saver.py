import time
import six
import sys

import os
import cifar_input
import numpy as np
import resnet_model
import tensorflow as tf

eval_data_path = r'D:\wangfeicheng\Tensorflow\cifar10-tensorflow\cifar-10-binary\cifar-10-batches-bin'
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_string('mode', 'eval', 'train or eval.')
tf.app.flags.DEFINE_string('train_data_path', '',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path', eval_data_path,
                           'Filepattern for eval data')
tf.app.flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.app.flags.DEFINE_string('train_dir', '',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 50,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', '',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 0,
                            'Number of gpus used for training. (0 or 1)')
tf.flags.DEFINE_string('data_format', 'channels_first',
                           'channels_first for cuDNN, channels_last for MKL')
tf.flags.DEFINE_integer("num_intra_threads", 0,
                     "Number of threads to use for intra-op parallelism. If set" 
                     "to 0, the system will pick an appropriate number.")
tf.flags.DEFINE_integer("num_inter_threads", 0,
                     "Number of threads to use for inter-op parallelism. If set" 
                     "to 0, the system will pick an appropriate number.")


from tensorflow.python import pywrap_tensorflow

ckpt_dir = r'D:\wangfeicheng\Tensorflow\docker-multiple\ResNet\resnet50-cifar-ckpt-20190218'
file = os.path.join(ckpt_dir, 'checkpoint')
with open(file) as ckpt:
    lines = ckpt.readlines()
    # for x in lines:
    #     print(x)
print(lines)

# 查看TensorFlow checkpoint文件中的变量名和对应值
latest_ckpt_dir = os.path.join(ckpt_dir, 'model.ckpt-107738')
# print(latest_ckpt_dir)
reader = pywrap_tensorflow.NewCheckpointReader(latest_ckpt_dir)
var_to_shape_map = reader.get_variable_to_shape_map()
print('Lenth of variables: %d' % len(var_to_shape_map))


# for key in var_to_shape_map:
#     print("tensor_name: ", key)
#     print(reader.get_tensor(key))


def create_config_proto():
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    return config


num_classes=10
hps = resnet_model.HParams(num_classes=num_classes,
                           lrn_rate=0.1,
                           weight_decay_rate=0.0002,
                           optimizer='mom')



'''
# with graph.as_default():
    # Loading the graph first,then it can restore() the variable for this graph
    #graph_dir = os.path.join(ckpt_dir, 'model.ckpt-107738.meta')
    #with tf.device('/cpu:0'):
    #   saver = tf.train.import_meta_graph(graph_dir)
import_meta_graph appends the network defined in .meta file to the current graph. So, this will create the 
graph/network for you but we still need to load the value of the parameters that we had trained on this graph.

InvalidArgumentError (see above for traceback): Restoring from checkpoint failed. This is most likely due to a 
mismatch between the current graph and the graph from the checkpoint. Please ensure that you have not altered the 
graph expected based on the checkpoint. Original error:
Cannot assign a device for operation IteratorToStringHandle: Operation was explicitly assigned to 
/job:worker/task:0/device:GPU:0 but available devices are [ /job:localhost/replica:0/task:0/device:CPU:0 ].
 Make sure the device specification refers to a valid device.
 [[node IteratorToStringHandle (defined at D:/wangfeicheng/Tensorflow/tensorflow-learning/tf_saver.py:39)  = 
 IteratorToStringHandle[_device="/job:worker/task:0/device:GPU:0"](OneShotIterator)]]
'''


tf.reset_default_graph()
graph = tf.Graph()
batch_size = 128
batch_size = 100
with graph.as_default():
    # images, labels = cifar_input.build_input(
    #     FLAGS.dataset, FLAGS.eval_data_path, batch_size, FLAGS.mode)
    X = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3), name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='Y')
    model = resnet_model.ResNet(hps, X, Y, FLAGS.mode)
    # Build the inference graph again
    model.build_graph(False)
    saver = tf.train.Saver()  # Gets all variables in `graph`.


with tf.Session(graph=graph, config=create_config_proto()) as sess:
    # restore all the values for the variables
    saver.restore(sess, latest_ckpt_dir)
    trainable_variables = tf.trainable_variables()
    print(type(trainable_variables))
    print(len(trainable_variables))

    for var in trainable_variables:
        print(var)

    # labels = graph.get_tensor_by_name('labels')
    # predictions = graph.get_tensor_by_name('predictions')
    #
    # truth = tf.argmax(labels,axis=1)
    # prediction = tf.argmax(predictions,axis=1)
    # accuracy = tf.reduce_mean(tf.to_float(tf.equal(truth,prediction)))
    dense_bias = graph.get_tensor_by_name('dense/bias:0')
    print("dense_bias:{}".format(sess.run(dense_bias)))

    conv2d_51= graph.get_operation_by_name('conv2d_51/kernel')
    print("conv2d_51:{}:".format(sess.run(conv2d_51)))


