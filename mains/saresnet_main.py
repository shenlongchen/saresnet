import tensorflow as tf
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import warnings
warnings.filterwarnings("ignore")
import sys
from shutil import copyfile
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from data_loader.data_generator import DataGenerator
from models.saresnet_model import SAResnetModel
from trainers.example_trainer import ExampleTrainer
from utils_project.config import process_config
from utils_project.dirs import create_dirs
from utils_project.logger import Logger
# from utils_project.utils import get_args


def main(is_pretrain, tfrecords_files, kernal_channel=64, fc_num=32, count_stop=10):

    config = process_config(os.path.join(rootPath, "configs/example.json"))

    for train_data_file in sorted(os.listdir(tfrecords_files)):

        _path = os.path.join(tfrecords_files, train_data_file)
        if os.path.isdir(_path):
            config.train_data_file = train_data_file
            config.summary_dir = os.path.join(rootPath, "result", "saresnet", train_data_file, "summary")
            config.checkpoint_dir = os.path.join(rootPath, "result", "saresnet", train_data_file, "checkpoint")
            config.basicmodel_dir = os.path.join(rootPath, "result", "basic_model")
            config.pred_result_dir = os.path.join(rootPath, "result", "pred_result")
            config.result_csv_dir = os.path.join(rootPath, "result", "result_csv")
            config.is_pretrain = is_pretrain
            if not os.path.exists(config.pred_result_dir):
                os.makedirs(config.pred_result_dir)
            if not os.path.exists(config.result_csv_dir):
                os.makedirs(config.result_csv_dir)
            config._path = _path
            config.train_file_name = train_data_file
            config.count_stop = count_stop
            create_dirs([config.summary_dir, config.checkpoint_dir, config.basicmodel_dir])

            file_name = os.path.join(config.checkpoint_dir, "checkpoint")
            file_name_before = os.path.join(config.basicmodel_dir, "checkpoint")

            if not is_pretrain:
                if not os.path.exists(file_name):
                    copyfile(os.path.join(config.basicmodel_dir, "saresnet_basic.data-00000-of-00001"),
                             os.path.join(config.checkpoint_dir, "saresnet_basic.data-00000-of-00001"))
                    copyfile(os.path.join(config.basicmodel_dir, "saresnet_basic.index"),
                             os.path.join(config.checkpoint_dir, "saresnet_basic.index"))
                    copyfile(os.path.join(config.basicmodel_dir, "saresnet_basic.meta"),
                             os.path.join(config.checkpoint_dir, "saresnet_basic.meta"))

                    with open(file_name_before) as f_before, open(file_name, "w") as f:
                        lines = f_before.readlines()
                        for line in lines:
                            line_ = line.replace("%s", train_data_file) + "\n"
                            f.write(line_)

            # gpu_options = tf.GPUOptions(allow_growth=True)
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.28)
            tf_config = tf.ConfigProto(gpu_options=gpu_options)

            sess = tf.Session(config=tf_config)
            data = DataGenerator(config)
            model = SAResnetModel(config, kernal_channel=kernal_channel, fc_num=fc_num)

            logger = Logger(sess, config)
            trainer = ExampleTrainer(sess, model, data, config, logger)
            model.load(sess)
            trainer.train()

            tf.reset_default_graph()


if __name__ == '__main__':
    root_server = "/home/shenlc/PycharmProjects"
    kernal_channel = 64
    fc_num = 32

    is_pretrain = True
    count_stop = 8
    if is_pretrain:
        tfrecords_files = root_server + "/DataSet/tfrecords_pretrain"
    else:
        tfrecords_files = root_server + "/DataSet/tfrecords"
    main(is_pretrain, tfrecords_files, kernal_channel, fc_num, count_stop=count_stop)

    is_pretrain = False
    count_stop = 5
    if is_pretrain:
        tfrecords_files =  root_server+"/DataSet/tfrecords_pretrain"
    else:
        tfrecords_files = root_server + "/DataSet/tfrecords"
    main(is_pretrain, tfrecords_files, kernal_channel, fc_num, count_stop=count_stop)
