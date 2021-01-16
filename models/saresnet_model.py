import tensorflow as tf
import os
import utils_project.self_attention as NonLocal

class SAResnetModel():
    def __init__(self, config, kernal_channel, fc_num):
        self.config = config
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()
        self.k_c = kernal_channel
        self.fc_num = fc_num
        self.channels = 4
        self.height = 1
        self.weight = 101
        self.dropout_rate = 0.7
        self.build_model()
        self.init_saver()

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, os.path.join(self.config.checkpoint_dir, "saresnet"), self.global_step_tensor)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')


    def identity_block(self, X_input, kernel_size, in_filter, out_filters, stage, block, training):
        # defining name basis
        block_name = 'res' + str(stage) + block
        f1, f2 = out_filters
        with tf.variable_scope(block_name):
            X_shortcut = X_input

            #first
            X = tf.layers.batch_normalization(X_input, axis=-1, training=training)
            X = tf.nn.elu(X)
            W_conv1 = self.weight_variable([1, kernel_size, in_filter, f1])
            X = tf.nn.conv2d(X, W_conv1, strides=[1, 1, 1, 1], padding='SAME')

            #second
            X = tf.layers.batch_normalization(X, axis=-1, training=training)
            X = tf.nn.elu(X)
            W_conv2 = self.weight_variable([1, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')

            #final step
            add_result = tf.add(X, X_shortcut)
            
        return add_result


    def convolutional_block(self, X_input, kernel_size, in_filter, out_filters, stage, block, training, stride=2):
        # defining name basis
        block_name = 'res' + str(stage) + block
        with tf.variable_scope(block_name):
            f1, f2 = out_filters

            x_shortcut = X_input
            #first
            X = tf.layers.batch_normalization(X_input, axis=3, training=training)
            X = tf.nn.elu(X)            
            W_conv1 = self.weight_variable([1, kernel_size, in_filter, f1])
            X = tf.nn.conv2d(X, W_conv1, strides=[1, 1, stride, 1], padding='SAME')

            #second
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.elu(X)
            W_conv2 = self.weight_variable([1, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')

            #shortcut path
            W_shortcut = self.weight_variable([1, 1, in_filter, f2])
            x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, 1, stride, 1], padding='SAME')

            #final
            add_result = tf.add(x_shortcut, X)
            
        return add_result

    def parse(self, record):
        features = tf.parse_single_example(
            record,
            features={'label': tf.FixedLenFeature([2], tf.int64),
                      'sequence': tf.FixedLenFeature([self.weight*self.channels], tf.int64),
                      }
        )
        sequence = tf.reshape(features['sequence'], [self.weight, self.channels])
        label = features["label"]
        return label, sequence

    def build_model(self):
        # here you build the tensorflow graph of any model you want and also define the loss.
        self.tfrecord_path = tf.placeholder(tf.string, shape=[None], name="tfrecord_path")
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        dataset = tf.data.TFRecordDataset(self.tfrecord_path)
        dataset = dataset.map(self.parse, num_parallel_calls=8)

        dataset = dataset.batch(self.config.batch_size, drop_remainder=False)  # 每100 一个 batch
        dataset = dataset.prefetch(buffer_size=self.config.batch_size*10)
        dataset = dataset.repeat(count=1)
        self.iterator = dataset.make_initializable_iterator()
        self.input_y, self.input_x = self.iterator.get_next()
        self.input_y = tf.to_float(self.input_y)
        self.input_x = tf.to_float(self.input_x)
        self.input_x = tf.reshape(self.input_x, [-1, 1, self.weight, self.channels])

        # network architecture
        w_conv1 = self.weight_variable([1, 7, self.channels, self.k_c])
        x = tf.nn.conv2d(self.input_x, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.elu(x)
        x = tf.layers.max_pooling2d(x, pool_size=[1, 3], strides=[1, 2])
        # stage 2
        x = NonLocal.NonLocalBlockND(x, in_channels=self.k_c, sub_sample=False, block='1')
        x = self.convolutional_block(x, 3, self.k_c, [self.k_c, self.k_c], 2, 'a', self.is_training, stride=1)
        x = self.identity_block(x, 3, self.k_c, [self.k_c, self.k_c], stage=2, block='b', training=self.is_training)
        x = self.identity_block(x, 3, self.k_c, [self.k_c, self.k_c], stage=2, block='c', training=self.is_training)
        x = NonLocal.NonLocalBlockND(x, in_channels=self.k_c, sub_sample=False, block='2')

        # stage 3
        x = self.convolutional_block(x, 3, self.k_c, [self.k_c, self.k_c], 3, 'a', self.is_training)
        x = self.identity_block(x, 3, self.k_c, [self.k_c, self.k_c], 3, 'b', training=self.is_training)
        x = self.identity_block(x, 3, self.k_c, [self.k_c, self.k_c], 3, 'c', training=self.is_training)
        x = self.identity_block(x, 3, self.k_c, [self.k_c, self.k_c], 3, 'd', training=self.is_training)
        x = NonLocal.NonLocalBlockND(x, in_channels=self.k_c, sub_sample=False, block='3')
        # stage 4
        x = self.convolutional_block(x, 3, self.k_c, [self.k_c, self.k_c], 4, 'a', self.is_training)
        x = self.identity_block(x, 3, self.k_c, [self.k_c, self.k_c], 4, 'b', training=self.is_training)
        x = self.identity_block(x, 3, self.k_c, [self.k_c, self.k_c], 4, 'c', training=self.is_training)
        x = self.identity_block(x, 3, self.k_c, [self.k_c, self.k_c], 4, 'd', training=self.is_training)
        x = self.identity_block(x, 3, self.k_c, [self.k_c, self.k_c], 4, 'e', training=self.is_training)
        x = self.identity_block(x, 3, self.k_c, [self.k_c, self.k_c], 4, 'f', training=self.is_training)

        # stage 5
        x = self.convolutional_block(x, 3, self.k_c, [self.k_c, self.k_c], 5, 'a', self.is_training)
        x = self.identity_block(x, 3, self.k_c, [self.k_c, self.k_c], 5, 'b', training=self.is_training)
        x = self.identity_block(x, 3, self.k_c, [self.k_c, self.k_c], 5, 'c', training=self.is_training)

        x = tf.nn.avg_pool(x, [1, 1, 7, 1], strides=[1, 1, 1, 1], padding='VALID')
        flatten = tf.layers.flatten(x)

        with tf.name_scope('dropout'):
            x = tf.layers.dropout(inputs=flatten, rate=self.dropout_rate, training=self.is_training)
        x = tf.layers.dense(x, units=self.fc_num, activation=None)
        with tf.name_scope('dropout'):
            x = tf.layers.dropout(inputs=x, rate=self.dropout_rate, training=self.is_training)
        logits = tf.layers.dense(x, units=2, activation=None)
        
        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y,
                                                                                        logits=logits))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if self.config.is_pretrain:
                    self.train_step = tf.train.AdamOptimizer(
                        learning_rate=self.config.learning_rate_pretrain).minimize(
                        self.cross_entropy, global_step=self.global_step_tensor)
                else:
                    self.train_step = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize \
                        (self.cross_entropy, global_step=self.global_step_tensor)

            output = tf.nn.softmax(logits)
            # train accuracy
            correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
            self.pred_numeric = output
            tf.add_to_collection('pred_numeric', self.pred_numeric)
            tf.add_to_collection('accuracy', self.accuracy)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
