import numpy as np
import tensorflow as tf
from shutil import copyfile
import math
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import os
import csv
from datetime import datetime
from utils_project.dirs import create_dirs
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]

class ExampleTrainer():
    def __init__(self, sess, model, data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data = data
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)
        self.train_num = self.data.train_num
        self.test_num = self.data.test_num
        self.valid_num = self.data.valid_num

    def train(self):
        best_loss_model = 1000000  # big
        best_epoch = 0
        best_auc = 0
        count_stop = self.config.count_stop
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            best_loss_model, count_stop, best_epoch, best_auc = \
                self.train_epoch(cur_epoch, best_loss_model, best_auc, count_stop, best_epoch)
            self.sess.run(self.model.increment_cur_epoch_tensor)
            if count_stop < 0:
                print("best is ", best_auc, "epoch is ", best_epoch)
                break
        print("best is ", best_auc, "epoch is ", best_epoch)

    def train_epoch(self, cur_epoch, best_loss_model, best_auc, count_stop, best_epoch):
        #
        start_time = datetime.now()
        #
        input_file = self.data.input_files
        feed_dict = {self.model.tfrecord_path: input_file, self.model.is_training: True}
        self.sess.run(self.model.iterator.initializer, feed_dict=feed_dict)

        loss_trains = []
        pred_numerics_train = []
        reals_train = []
        loop_train = math.ceil(self.train_num / self.config.batch_size)
        for _ in range(loop_train):
            loss_, real, pred_numeric = self.train_step(input_file)
            loss_trains.append(loss_)
            pred_numerics_train.extend(list(pred_numeric[:, 1]))
            reals_train.extend(list(real))

        loss_train = np.mean(loss_trains)
        end_time = datetime.now()
        print("train_time：%.2f ### cur_epoch: %s #### train loss: %s "
              % ((end_time-start_time).seconds/60, cur_epoch, loss_train))

        # valid
        valid_file = self.data.valid_files
        feed_dict = {self.model.tfrecord_path: valid_file, self.model.is_training: False}
        self.sess.run(self.model.iterator.initializer, feed_dict=feed_dict)

        loss_valids = []
        pred_numerics_valid = []
        reals_valid = []
        loop_valid = range(math.ceil(self.valid_num / self.config.batch_size))

        #
        start_time=datetime.now()
        #
        for _ in loop_valid:
            loss_valid_, real, pred_numeric = self.valid_step(valid_file)
            loss_valids.append(loss_valid_)
            pred_numerics_valid.extend(list(pred_numeric[:, 1]))
            reals_valid.extend(list(real))
        loss_valid = np.mean(loss_valids)
        roc_auc_valid = self.auc_computing(reals_valid, pred_numerics_valid)
        #
        end_time=datetime.now()
        print("valid_time：%.2f ### cur_epoch: %s #### valid loss: %s , #### valid AUC: %s" %
              ((end_time-start_time).seconds/60, cur_epoch, loss_valid, roc_auc_valid))

        # test start, every epoch one time
        test_file = self.data.test_files
        feed_dict = {self.model.tfrecord_path: test_file, self.model.is_training: False}
        self.sess.run(self.model.iterator.initializer, feed_dict=feed_dict)
        #
        start_time=datetime.now()
        #
        loss_tests = []
        pred_numerics_test = []
        reals_test = []
        loop_test = range(math.ceil(self.test_num / self.config.batch_size))
        for _ in loop_test:
            loss_test_, real, pred_numeric = self.test_step(test_file)
            loss_tests.append(loss_test_)
            pred_numerics_test.extend(list(pred_numeric[:, 1]))
            reals_test.extend(list(real))
        loss_test = np.mean(loss_tests)
        roc_auc_test = self.auc_computing(reals_test, pred_numerics_test)

        end_time=datetime.now()
        print("test_time：%.2f ### cur_epoch: %s #### test loss: %s,####### test AUC: %s " %
              ((end_time-start_time).seconds/60, cur_epoch, loss_test, roc_auc_test))
        print("---------------------------------------------------")

        # recording
        result_csv = os.path.join(self.config.result_csv_dir, self.config.train_file_name + "_result.csv")
        if cur_epoch == 0:
            with open(result_csv, 'w+', newline="") as f:
                csv_write = csv.writer(f)
                csv_head = ["epoch", "loss_train", "loss_valid", "loss_test", "AUC_valid", "AUC_test"]
                csv_write.writerow(csv_head)
        with open(result_csv, 'a+', newline="") as f:
            csv_write = csv.writer(f)
            data_row = [cur_epoch, loss_train, loss_valid, loss_test, roc_auc_valid, roc_auc_test]
            csv_write.writerow(data_row)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss_train,
            'auc': roc_auc_valid,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        if loss_valid <= best_loss_model:
            if self.config.is_pretrain:
                self.model.save(self.sess)
            print("save best model")
            best_loss_model = loss_valid
            best_auc = roc_auc_test
            best_epoch = cur_epoch
            count_stop = self.config.count_stop

            pred_result_path = os.path.join(self.config.pred_result_dir, self.config.train_file_name, "basic_model")
            create_dirs([pred_result_path])
            with open(os.path.join(pred_result_path, "bestiter.pred"), 'w+') as f:
                for line in pred_numerics_test:
                    f.write(str(line))
                    f.write('\n')
            self.copy_to_basic_model()
        else:
            count_stop -= 1
        # valid_result and test_result
        if self.config.is_pretrain:
            pred_result_path = os.path.join(self.config.pred_result_dir, self.config.train_file_name, "valid_result")
            create_dirs([pred_result_path])
            with open(os.path.join(pred_result_path, "bestiter.pred"), 'w+') as f:
                for index in range(len(pred_numerics_valid)):
                    f.write(str(pred_numerics_valid[index]) + " " + str(reals_valid[index]))
                    f.write('\n')

            pred_result_path = os.path.join(self.config.pred_result_dir, self.config.train_file_name, "test_result")
            create_dirs([pred_result_path])
            with open(os.path.join(pred_result_path, "bestiter.pred"), 'w+') as f:
                for index in range(len(pred_numerics_test)):
                    f.write(str(pred_numerics_test[index])+" "+str(reals_test[index]))
                    f.write('\n')

        return best_loss_model, count_stop, best_epoch, best_auc


    def train_step(self, input_file):
        feed_dict = {self.model.tfrecord_path: input_file, self.model.is_training: True}
        _, loss, batch_y, pred_numeric = self.sess.run(
            [self.model.train_step, self.model.cross_entropy, self.model.input_y, self.model.pred_numeric],
            feed_dict=feed_dict)
        real = np.reshape(batch_y[:, 1], [-1])
        return loss, real, pred_numeric

    def valid_step(self, input_file):
        feed_dict = {self.model.tfrecord_path: input_file, self.model.is_training: False}
        loss, batch_y, pred_numeric = self.sess.run(
            [self.model.cross_entropy, self.model.input_y, self.model.pred_numeric], feed_dict=feed_dict)
        real = np.reshape(batch_y[:, 1], [-1])
        return loss, real, pred_numeric

    def test_step(self, test_file):
        feed_dict = {self.model.tfrecord_path: test_file, self.model.is_training: False}
        loss, batch_y, pred_numeric = self.sess.run([self.model.cross_entropy,
                                                     self.model.input_y, self.model.pred_numeric], feed_dict=feed_dict)
        real = np.reshape(batch_y[:, 1], [-1])
        return loss, real, pred_numeric

    def auc_computing(self, real, pred_numerics):
        for i in range(len(pred_numerics)):
            if np.isnan(pred_numerics[i]):
                pred_numerics[i] = 0.5
        fpr, tpr, thresholds = roc_curve(real, pred_numerics)
        roc_auc = auc(fpr, tpr)
        return roc_auc

    def copy_to_basic_model(self):
        if self.config.is_pretrain:
            file_name = os.path.join(self.config.checkpoint_dir, "checkpoint")
            file_name_before = os.path.join(self.config.basicmodel_dir, "checkpoint")
            with open(file_name_before, "w") as f_before, open(file_name) as f:

                lines = f.readlines()
                line_1 = lines[0].replace(self.config.train_data_file, "%s")
                basic_model_name_befor = line_1.split('/')[-1].split('"')[0]
                line_1 = line_1.replace(basic_model_name_befor, 'saresnet_basic')
                line_2 = lines[-1].replace(self.config.train_data_file, "%s")
                line_2 = line_2.replace(basic_model_name_befor, "saresnet_basic")
                f_before.write(line_1)
                f_before.write(line_2)
                copyfile(os.path.join(self.config.checkpoint_dir, basic_model_name_befor + ".data-00000-of-00001"),
                         os.path.join(self.config.basicmodel_dir, "saresnet_basic.data-00000-of-00001"))
                copyfile(os.path.join(self.config.checkpoint_dir, basic_model_name_befor + ".index"),
                         os.path.join(self.config.basicmodel_dir, "saresnet_basic.index"))
                copyfile(os.path.join(self.config.checkpoint_dir, basic_model_name_befor + ".meta"),
                         os.path.join(self.config.basicmodel_dir, "saresnet_basic.meta"))