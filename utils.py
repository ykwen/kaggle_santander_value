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
        targets = np.array([[d[1]] for d in data[1:]]).astype(np.float32)
        features = np.array([d[2:] for d in data[1:]]).astype(np.float32)
        return ids, targets, features
    else:
        ids = [d[0] for d in data[1:]]
        features = np.array([d[1:] for d in data[1:]]).astype(np.float32)
        return ids, features


def cal_loss(preds, y):
    r = []
    if type(y[0]) == 'list':
        y = [yy[0] for yy in y]
    for p, yy in zip(preds, y):
        r.append(np.power(np.log(p + 1) - np.log(yy + 1), 2))
    return np.sqrt(np.average(r))


def write_csv(idx, target, path):
    with open(path, 'w+') as f:
        f.write('ID,target\n')
        for i, id, t in enumerate(zip(idx, target)):
            f.write(id+','+str(t))
            if i + 1 < len(idx):
                f.write('\n')


def get_order_batch(data, batch_size):
    if len(data) >= batch_size:
        return data[:batch_size], data[batch_size:]
    else:
        placeholder = np.zeros(len(data[0]))
        return data + [placeholder for _ in range(batch_size - len(data))], []


# Applying standardize, normalize, PCA, whitening
def standardize(x):
    return preprocessing.StandardScaler().fit_transform(x)


def normalize(x, y=None):
    return preprocessing.normalize(x)


def decomposing(x, n, flag):
    pca = PCA(n_components=n, whiten=flag)
    return pca.fit_transform(x)


def fit_test(model, x_train, x_test, y_train, y_test):
    y_train = [y[0] for y in y_train]
    y_test = [y[0] for y in y_test]
    model.fit(x_train, y_train)
    p = model.predict(x_test)
    return cal_loss(p, y_test), model


def test_models(model, x_test, y_test):
    preds = []
    for m in model:
        preds.append(m.predict(x_test))
    pred = np.mean(preds, axis=0)
    return cal_loss(pred, y_test)


class NN:
    def __init__(self, batch_size, x_size, size, learning_rate, suumary_path):
        self._lr = learning_rate
        self.summary_path = suumary_path

        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=self.config)

        self.x = tf.placeholder(tf.float32, [batch_size, x_size])
        self.y = tf.placeholder(tf.float32, [batch_size, 1])
        self.keep_prob = tf.placeholder(tf.float32, [1])
        self.w = []
        self.b = []
        with tf.variable_scope('neuron', reuse=tf.AUTO_REUSE):
            for i, s in enumerate(size):
                if i == 0:
                    self.w.append(tf.get_variable('weight'+str(i), [x_size, s], tf.float32,
                                                  initializer=tf.random_normal_initializer()))
                else:
                    self.w.append(tf.get_variable('weight' + str(i), [size[i-1], s], tf.float32,
                                                  initializer=tf.random_normal_initializer()))
                self.b.append(tf.get_variable('bias'+str(i), [s], tf.float32,
                                              initializer=tf.constant_initializer(1)))
            self.w.append(tf.get_variable('output_weight', [size[-1], 1], tf.float32,
                                          initializer=tf.random_normal_initializer()))
            self.b.append(tf.get_variable('output_bias', [1], tf.float32,
                                          initializer=tf.constant_initializer(1)))
        with tf.variable_scope('result', reuse=tf.AUTO_REUSE):
            self.output = tf.get_variable('output', [batch_size, x_size], tf.float32)
        self.output = self.x
        for one_w, one_b in zip(self.w, self.b):
            # Using ReLU neuron
            self.output = tf.add(tf.max(tf.matmul(self.output, one_w), 0), one_b)

        self.output = tf.nn.dropout(self.output, keep_prob=self.keep_prob)

        self.loss = tf.sqrt(tf.losses.mean_squared_error(self.y, self.output))

        self.optimizer = tf.train.AdamOptimizer(self._lr)

        self.train = self.optimizer.minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        tf.summary.scalar('loss', self.loss)
        self.summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.summary_path)

    def set_loss_func(self, new_loss):
        self.loss = new_loss


if __name__ == '__main__':
    print("This is the utils program, only running for testing.")
    #print(load_csv_data('data/test.csv')[:10])
