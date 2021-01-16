import tensorflow as tf
import numpy as np
import os
import sys
rootPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(rootPath)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.enable_eager_execution()

# 将预训练数据.data转为 train.tfrecord,valid.tfrecord,test.tfrecord
# All raw values should be converted to a type compatible with tf.Example. Use
# the following functions to do these convertions.
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def embed(seq, mapper, worddim):
    mat = np.asarray([mapper[element] if element in mapper else np.random.rand(worddim)*2-1 for element in seq])
    return mat

def onehot(index):
    if index == 0:
        return [1, 0]
    else:
        return [0, 1]


def write_record(writer_train, write_valid, datasetpath):
    count_all = 0
    if os.path.exists(datasetpath):
        mapper = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [1, 1, 1, 1]}
        with open(datasetpath) as seqfile:
            seqdata = []
            label = []
            for x in zip(seqfile):
                x_str = "".join(x)
                seqdata.append(list(x_str.strip().split()[1]))
                label.append(float(x_str.strip().split()[2]))
                count_all += 1
            seqdata = np.asarray(seqdata)
            labels = np.asarray(label)

            train_data_num = int(count_all*8/10)
            valid_data_num = count_all - train_data_num
            index = 0
            for seq, label in zip(seqdata, labels):
                mat = embed(seq, mapper, len(mapper['A']))
                result = mat.reshape(-1)
                result = result.astype(int)
                # result = mat.transpose().reshape(-1)
                tf_example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'label': _int64_feature(value=onehot(label)),
                        'sequence': _int64_feature(value=result)
                    }))
                index += 1
                # Write the serialized example to a record file.
                if index <= train_data_num:
                    writer_train.write(tf_example.SerializeToString())
                else:
                    write_valid.write(tf_example.SerializeToString())
    return train_data_num, valid_data_num

def write_test_record(writer_test, datasetpath):
    count_all = 0
    if os.path.exists(datasetpath):
        mapper = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [1, 1, 1, 1]}
        with open(datasetpath) as seqfile:
            seqdata = []
            label = []
            for x in zip(seqfile):
                x_str = "".join(x)
                seqdata.append(list(x_str.strip().split()[1]))
                label.append(float(x_str.strip().split()[2]))
                count_all += 1
            seqdata = np.asarray(seqdata)
            labels = np.asarray(label)

            index = 0
            for seq, label in zip(seqdata, labels):
                mat = embed(seq, mapper, len(mapper['A']))
                result = mat.reshape(-1)
                result = result.astype(int)
                # result = mat.transpose().reshape(-1)
                tf_example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'label': _int64_feature(value=onehot(label)),
                        'sequence': _int64_feature(value=result)
                    }))
                index += 1
                # Write the serialized example to a record file.
                writer_test.write(tf_example.SerializeToString())
    return count_all

def read_record(tfrecords_filename):
    # Use dataset API to import date directly from TFRecord file.
    dataset = tf.data.TFRecordDataset(tfrecords_filename)

    # Create a dictionary describing the features. The key of
    # the dict should be the same with the key in writing function.
    image_feature_description = {'label': tf.FixedLenFeature([2], tf.int64),
                'sequence': tf.FixedLenFeature([404], tf.int64),
                }

    # Define the parse function to extract a single example as a dict.
    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.parse_single_example(example_proto, image_feature_description)

    dataset = dataset.map(_parse_image_function)
    iterator = dataset.make_one_shot_iterator()
    label, seq = iterator.get_next()


if __name__ == "__main__":
    filedir = os.path.join("pretrain_data", "pretrain_sequences")
    tfrecorddir = os.path.join("pretrain_data", "pretrain_tfrecords")
    files = os.listdir(filedir)
    count = 0

    tfrecords_filename = os.path.join(rootPath, tfrecorddir, 'train.tfrecords')
    tfrecords_valid_filename = os.path.join(rootPath, tfrecorddir, "valid.tfrecords")
    tfrecords_test_filename = os.path.join(rootPath, tfrecorddir, "test.tfrecords")
    tfrecords_dir = os.path.join(rootPath, tfrecorddir)

    writer_train = tf.io.TFRecordWriter(tfrecords_filename)
    write_valid = tf.io.TFRecordWriter(tfrecords_valid_filename)
    write_test = tf.io.TFRecordWriter(tfrecords_test_filename)
    if not os.path.exists(tfrecords_dir):
        os.makedirs(tfrecords_dir)
    for file in files:
        if os.path.isdir(os.path.join(rootPath, filedir, file)):
            datasets = os.listdir(os.path.join(rootPath, filedir, file))
            for dataset in datasets:

                if dataset == "test.data":
                    datasetpath = os.path.join(rootPath, filedir, file, dataset)
                    train_num = write_test_record(write_test, datasetpath)
                if dataset == "train.data":
                    datasetpath = os.path.join(rootPath, filedir, file, dataset)
                    train_num, valid_num = write_record(writer_train, write_valid, datasetpath)
            print(file, "count:", count)
            count += 1
    writer_train.close()
    write_valid.close()
    write_test.close()
