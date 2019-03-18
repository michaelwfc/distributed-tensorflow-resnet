# Freeze model from checkpoint file
'''

利用 pd 文件(evaluation graph + ckpt)做预测:
1. 读取 pd 文件
2. 指定输入，输出节点
3. 读取数据
4. 在session中使用feeding模式运行
'''

import os
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from cifar_input import eval_data_input, display_eval_images
import resnet_model
import matplotlib.pyplot as plt
import math
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('train_data_path', '',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path', '',
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
tf.flags.DEFINE_string('data_format', 'channels_last',
                       'channels_first for cuDNN, channels_last for MKL')
tf.flags.DEFINE_integer("num_intra_threads", 0,
                        "Number of threads to use for intra-op parallelism. If set"
                        "to 0, the system will pick an appropriate number.")
tf.flags.DEFINE_integer("num_inter_threads", 0,
                        "Number of threads to use for inter-op parallelism. If set"
                        "to 0, the system will pick an appropriate number.")
# 输出需要评估的数据
EVAL_NUM = 100

# ckpt保存位置
ckpt_dir = r'D:\wangfeicheng\Tensorflow\docker-multiple\ResNet\resnet50-cifar-ckpt-20190218'

# eval数据保存位置
eval_data_path = r'D:\wangfeicheng\Tensorflow\cifar10-tensorflow\cifar-10-binary\cifar-10-batches-bin\test_batch.bin'

# pd文件
output_eval_pb = os.path.join(ckpt_dir, "resnet50_cifar_frozen_model_eval.pb")


# 利用 pd 文件(evaluation grsph + ckpt)做预测:
# Accessing Frozen Models
def load_graph(frozen_filename):
    with tf.gfile.GFile(frozen_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # graph_def = data_format_changed(graph_def)
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, name="prefix")
    return graph


# 1. 读取 pd 文件
graph = load_graph(output_eval_pb)

# 2. 指定输入，输出节点
# 得到 input tensor
x = graph.get_tensor_by_name("prefix/X:0")
y = graph.get_tensor_by_name("prefix/Y:0")
# 得到 output tensor
prediction_tensor = graph.get_tensor_by_name("prefix/predictions:0")
precision_tensor = graph.get_tensor_by_name("prefix/precision:0")

# 3. 读取数据
images, labels, images_org = eval_data_input(eval_data_path, EVAL_NUM)


def create_config_proto():
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    return config


# 4. 在session中使用feeding模式运行
with tf.Session(graph=graph, config=create_config_proto()) as sess:
    predictions_value, precision_value = sess.run([prediction_tensor, precision_tensor],
                                                  feed_dict={x: images, y: labels})
    print("The prediction:{0}\nThe label:{1}\nThe precision:{2}" \
          .format(predictions_value, np.argmax(labels, axis=1), precision_value))

    display_eval_images(images_org,labels,predictions_value,EVAL_NUM)
