import csv
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.decomposition import PCA


def load_csv_data(path):
    result = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            result.append(row)
    return result


def read_content(data):
    if data[0][1] == 'target':
        ids = [d[0] for d in data[1:]]
        targets = np.array([d[1] for d in data[1:]]).astype(np.float32)
        features = np.array([d[2:] for d in data[1:]]).astype(np.float32)
        return ids, targets, features
    else:
        ids = [d[0] for d in data[1:]]
        features = np.array([d[1:] for d in data[1:]]).astype(np.float32)
        return ids, features


def cal_loss(preds, y):
    return np.sqrt(np.average(np.power(np.log(preds + 1) - np.log(y + 1), 2)))


def write_csv(idx, target, path):
    with open(path, 'w') as f:
        f.write('ID,target\n')
        for i, id, t in enumerate(zip(idx, target)):
            f.write(id+','+str(t))
            if i + 1 < len(idx):
                f.write('\n')


# Applying standardize, normalize, PCA, whitening
def standardize(x):
    return preprocessing.scale(x)


def normalize(x, y=None):
    return preprocessing.Normalizer.fit_transform(x, y)


def decomposing(x, n, flag):
    pca = PCA(n_components=n, whiten=flag)
    return pca.fit_transform(x)


class NN:
    def __init__(self, batch_size, x_size, size, learning_rate):
        self._lr = learning_rate

        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=self.config)

        self.x = tf.placeholder([batch_size, x_size], tf.float32)
        self.y = tf.placeholder([batch_size], tf.float32)
        self.w = []
        self.b = []
        with tf.variable_scope('neuron', reuse=True):
            for i, s in enumerate(size):
                if i == 0:
                    self.w.append(tf.get_variable('weight'+str(i), [x_size, s], tf.float32,
                                                  initializer=tf.random_normal_initializer()))
                else:
                    self.w.append(tf.get_variable('weight' + str(i), [size[i-1], s], tf.float32,
                                                  initializer=tf.random_normal_initializer()))
                self.b.append(tf.get_variable('bias'+str(i), [s], tf.float32,
                                              initializer=tf.constant_initializer(1)))
            self.w.append(tf.get_variable('output weight', [size[-1], 1], tf.float32,
                                          initializer=tf.random_normal_initializer()))
            self.w.append(tf.get_variable('output bias', [1], tf.float32,
                                          initializer=tf.constant_initializer(1)))

        self.output = tf.Variable('output', [batch_size, x_size], tf.float32)
        self.output = self.x
        for one_w, one_b in zip(self.w, self.b):
            self.output = tf.add(tf.matmul(self.output, one_w), one_b)

        self.loss = tf.sqrt(tf.losses.mean_squared_error(self.y, self.output))

        self.optimizer = tf.train.AdamOptimizer(self._lr)

        self.train = self.optimizer.minimize(self.loss)

    def set_loss_func(self, new_loss):
        self.loss = new_loss


if __name__ == '__main__':
    print("This is the utils program, only running for testing.")
    #print(load_csv_data('data/test.csv')[:10])
