from utils import *
from sklearn.model_selection import train_test_split

train_file = 'data/train.csv'
test_file = 'data/test.csv'
model_check_path = 'model/nn/nn0'
summary_path = 'model/nn/summary'
predict_file = 'data/predict.csv'

train_data = load_csv_data(train_file)
_, y, x = read_content(train_data)

x = standardize(np.log(x+1))
#y = normalize(y)

theta = 0.2
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=theta)

batch_size = 1
nn_size = [512, 128, 16]
learning_rate = 0.5
keep_prob = 0.8


def train(x, y):
    with tf.device('/gpu:0'):
        model = NN(batch_size, len(x[0]), nn_size, learning_rate, summary_path)
        i = 0
        while len(x) > 0:
            xx, x = get_order_batch(x, batch_size)
            yy, y = get_order_batch(y, batch_size)
            feed_dict = {model.x: xx, model.y: yy, model.keep_prob: keep_prob}
            _, summary = model.sess.run([model.train, model.summary], feed_dict=feed_dict)
            model.writer.add_summary(summary, i * batch_size)
            i += 1
        model.saver.save(model.sess, model_check_path)


def test(x, y):
    with tf.device('/gpu:0'):
        model = NN(batch_size, len(x[0]), nn_size, learning_rate, summary_path)
        model.saver.restore(model.sess, model_check_path)

        loss = []
        while len(x) > 0:
            xx, x = get_order_batch(x, batch_size)
            yy, y = get_order_batch(y, batch_size)
            feed_dict = {model.x: xx, model.y: yy, model.keep_prob: 1.0}
            loss.append(model.sess.run(model.loss, feed_dict=feed_dict))
    return np.mean(loss)


if __name__ == '__main__':
    score, model = nn(x_train, x_test, y_train, y_test, [128, 32, 8])
    #train(x_train, y_train)
    #score = test(x_train, y_train)
    print(score)
    '''
    ids, x_final = read_content(load_csv_data(test_file))
    x_final = np.log(x_final + 1)
    preds = model.predict(x_final)
    write_csv(ids, preds, predict_file)
    '''

