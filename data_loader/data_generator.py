import tensorflow as tf
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]

class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.input_files = [os.path.join(self.config._path, 'train.tfrecords')]
        self.train_num = self.get_record_num(self.input_files[0])
        self.valid_files = [os.path.join(self.config._path, 'valid.tfrecords')]
        self.valid_num = self.get_record_num(self.valid_files[0])
        self.test_files = [os.path.join(self.config._path, 'test.tfrecords')]
        self.test_num = self.get_record_num(self.test_files[0])


    def parse(self, record):
        features = tf.parse_single_example(
            record,
            features={'label': tf.FixedLenFeature([2], tf.int64),
                      'sequence': tf.FixedLenFeature([404], tf.int64),
                      }
        )
        sequence = tf.reshape(features['sequence'], [101, 4])
        label = features["label"]
        return label, sequence

    def get_record_num(self, valid_data_path):
        dataset_nums = 0
        for record in tf.python_io.tf_record_iterator(valid_data_path):
            dataset_nums += 1
        return dataset_nums
