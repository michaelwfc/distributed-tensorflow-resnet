# Freeze model from checkpoint file
'''

Save to  .pd with evaluation model
1. 重建 evaluation graph（必须同 training 时候所建模型一致）:
2. 将 evaluation graph 保存到 .meta
   上诉两步实际可以在训练的时候提前将 evaluation graph保存为 meta_graph
3. 读取ckpt文件中的variable，将其与 meta_graph 合并，形成 .pd 文件
4. 利用 pd 文件做预测


freeze_graph.py是怎么做的呢？
首行它先加载模型文件，再从checkpoint文件读取权重数据初始化到模型里的权重变量，
再将权重变量转换成权重 常量 （因为 常量 能随模型一起保存在同一个文件里），
然后再通过指定的输出节点将没用于输出推理的Op节点从图中剥离掉，再重新保存到指定的文件里（用write_graphdef或Saver）

方法1：从train保存下来的graph.pdtxt 和 .meta 文件中恢复图模型
        1. 需要消除 device
        2. AvgPool 的 data_format='NCHW' OR '' 报错
        3. 需要去除 train_op 做图优化
方法2：重新用eval过程建图，然后保存为 .meta，然后使用 frezze_graph保存为 .pd，最后用于prediction
        1. data_format 需要根据cpu或者gpu计算而改变，否则会和方法1出现同样的AvgPool 的 data_format 报错
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

# ckpt保存位置
ckpt_dir = r'D:\wangfeicheng\Tensorflow\docker-multiple\ResNet\test\resnet50-cifar-ckpt-20190218'

# eval数据保存位置
eval_data_path = r'D:\wangfeicheng\Tensorflow\cifar10-tensorflow\cifar-10-binary\cifar-10-batches-bin\test_batch.bin'

num_classes = 10
hps = resnet_model.HParams(num_classes=num_classes,
                           lrn_rate=0.1,
                           weight_decay_rate=0.0002,
                           optimizer='mom')

# 1. 重建 evaluation graph（必须同 training 时候所建模型一致）:
tf.reset_default_graph()

with tf.Graph().as_default() as eval_graph:
    X = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3), name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='Y')
    model = resnet_model.ResNet(hps, X, Y, mode='eval')
    model.build_graph(istrain=False)
    truth = tf.argmax(model.labels, axis=1, name='truth')
    predictions = tf.argmax(model.predictions, axis=1, name='predictions')
    precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)), name='precision')

# 2. 将 evaluation graph 保存到 .meta
output_meta_graph = os.path.join(ckpt_dir, "resnet50_cifar_eval_graph.meta")
if not os.path.exists(output_meta_graph):
    tf.logging.info("save to  meta_graph with evaluation model:%s" % output_meta_graph)
    tf.train.export_meta_graph(output_meta_graph, graph=eval_graph)
else:
    tf.logging.info("we have saved to  meta_graph with evaluation model:%s" % output_meta_graph)


# 3. 读取ckpt文件中的variable，将其与 meta_graph 合并，形成 .pd 文件
# 最新的ckpt文件
latest_ckpt_dir = os.path.join(ckpt_dir, 'model.ckpt-107738')
# evaluation graph 输出的节点名称（需要和在建图的时候一致）
output_node_names = "predictions,precision"

restore_op = "save/restore_all"
filename_tensor = "save/Const:0"

output_eval_pb = os.path.join(ckpt_dir, "resnet50_cifar_frozen_model_eval.pb")

# 使用 freeze_graph.freeze_graph() 函数将meta_graph 和ckpt 文件结合，并形成 pd 文件
if not os.path.exists(output_eval_pb):
    freeze_graph.freeze_graph(input_graph=None,
                              input_saver="",
                              input_binary=True,
                              input_checkpoint=latest_ckpt_dir,
                              output_node_names=output_node_names,
                              restore_op_name=restore_op,
                              filename_tensor_name=filename_tensor,
                              output_graph=output_eval_pb,
                              clear_devices=True,
                              initializer_nodes="",
                              input_meta_graph=output_meta_graph)
    tf.logging.info("We create the .pb file in %s" % output_eval_pb)
else:
    tf.logging.info("We have alread created the .pb file in %s" % output_eval_pb)

tf.logging.info("find out the input and output node of your model")

# with tf.gfile.GFile(output_eval_pb, 'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#
# output_node = [n.name + '=>' + n.op
#                for n in graph_def.node if
#                n.op in ('Softmax', 'ArgMax', 'predictions', 'Placeholder', 'IteratorGetNext',)]
# print(output_node)

'''
def data_format_changed(graph_def):
    from tensorflow.core.framework import attr_value_pb2
    from tensorflow.python.framework import dtypes
    from tensorflow.python.framework import tensor_util

    # new_data_format = tf.constant('NHWC',dtype=tf.string)
    # _ = tf.import_graph_def(graph_def, input_map={"DecodeJpeg:0": tf_new_image})
    # InvalidArgumentError (see above for traceback): Default AvgPoolingOp only supports NHWC on device type CPU
    #  [[node prefix/average_pooling2d/AvgPool (defined at D:/wangfeicheng/Tensorflow/docker-multiple/ResNet/resnet_cifar_frozen_model.py:61)
    #  = AvgPool[T=DT_FLOAT, data_format="NCHW", ksize=[1, 1, 8, 8], padding="VALID", strides=[1, 1, 1, 1], _
    #  device="/job:localhost/replica:0/task:0/device:CPU:0"](prefix/Relu_48)]]
    
    attr_NHWC = attr_value_pb2.AttrValue(s=b"NHWC")
    avgpool_nodes = [n for n in graph_def.node if n.op in ('AvgPool')]
    print(avgpool_nodes)
    for avgpool in avgpool_nodes:
        avgpool.attr.get('data_format').CopyFrom(attr_NHWC)
    return graph_def
'''

# 4. 利用 pd 文件做预测
#Accessing Frozen Models
def load_graph(frozen_filename):
    with tf.gfile.GFile(frozen_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # graph_def = data_format_changed(graph_def)
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, name="prefix")
    return graph


# 从pd文件中加载 graph
graph = load_graph(output_eval_pb)
# 得到 input tensor
x = graph.get_tensor_by_name("prefix/X:0")
y = graph.get_tensor_by_name("prefix/Y:0")

# 得到 output tensor
prediction_tensor =  graph.get_tensor_by_name("prefix/predictions:0")
precision_tensor = graph.get_tensor_by_name("prefix/precision:0")


# 输出需要评估的数据
EVAL_NUM = 100

images, labels,images_org = eval_data_input(eval_data_path, EVAL_NUM)


def create_config_proto():
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    return config


with tf.Session(graph=graph, config=create_config_proto()) as sess:
    predictions_value, precision_value = sess.run([prediction_tensor,precision_tensor], feed_dict={x: images, y: labels})
    print("The prediction:{0}\nThe label:{1}\nThe precision:{2}"\
          .format(predictions_value, np.argmax(labels, axis=1), precision_value))

    display_eval_images(images_org, labels, predictions_value, 100)

