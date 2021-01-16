import tensorflow as tf


def NonLocalBlockND(input, in_channels, inter_channels=None, sub_sample=True, block=''):

    if inter_channels is None:
        inter_channels = in_channels // 2
        if inter_channels == 0:
            inter_channels = 1

    x = tf.layers.batch_normalization(input, axis=-1)
    x = tf.nn.elu(x)
    g_x = tf.layers.conv2d(inputs=x, filters=inter_channels, kernel_size=[1, 1], strides=[1, 1], padding='SAME',
                           name='g_x'+block)
    if sub_sample:
        g_x = tf.layers.max_pooling2d(g_x, pool_size=[1, 2], strides=[1, 2])
    g_x = tf.reshape(g_x, [-1, g_x.shape[1]*g_x.shape[2], inter_channels]) 

    theta_x = tf.layers.conv2d(inputs=x, filters=inter_channels, kernel_size=[1, 1], strides=[1, 1], padding='SAME',
                                 name='theta_x'+block)
    theta_x = tf.reshape(theta_x, [-1, theta_x.shape[1]*theta_x.shape[2], inter_channels])

    phi_x = tf.layers.conv2d(inputs=x, filters=inter_channels, kernel_size=[1, 1], strides=[1, 1], padding='SAME')
    if sub_sample:
        phi_x = tf.layers.max_pooling2d(phi_x, pool_size=[1, 2], strides=[1, 2])
    phi_x = tf.transpose(phi_x, [0, 3, 1, 2])
    phi_x = tf.reshape(phi_x, [-1, inter_channels, phi_x.shape[2]*phi_x.shape[3]])
    f = tf.matmul(theta_x, phi_x)

    N = tf.size(f, out_type=tf.int32)
    N = tf.cast(N, dtype=tf.float32)
    f = f/N
    y = tf.nn.softmax(f)
    y = tf.matmul(y, g_x)
    y = tf.reshape(y, [-1, input.shape[1], input.shape[2], inter_channels])

    W_y = tf.layers.batch_normalization(y, axis=-1)
    W_y = tf.nn.elu(W_y)
    W_y = tf.layers.conv2d(inputs=W_y, filters=in_channels, kernel_size=[1, 1], strides=[1, 1], padding='SAME')
    sita = tf.Variable(initial_value=0.0, dtype=tf.float32)
    z = sita*W_y + input
    return z

# if __name__ == "__main__":
#     img = tf.zeros([100, 1, 101, 64])
#     out = NonLocalBlockND(img, in_channels=64, inter_channels=1, dimension=2)
#     print(out)
