import numpy as np
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split

def load(filename):
    with open(filename + '.pickle', 'rb') as f:
        return pickle.load(f)

data, labels = load('train_spins'), load('train_labels')
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)


num_inputs = 16 * 16
num_classes = 2

regularization = 2 * 1e-5

print_every = 400

with tf.name_scope("inputs"):
    x = tf.placeholder(tf.float32, [None, num_inputs], name="x-input")
    y = tf.placeholder(tf.float32, [None, num_classes], name="y-input")

with tf.name_scope("model"):
    W = tf.Variable(tf.zeros([num_inputs, num_classes]), name="W")
    b = tf.Variable(tf.zeros([num_classes]), name="b")

y_pred = tf.sigmoid(tf.matmul(x, W) + b)

with tf.name_scope("loss-function"):
    loss = tf.losses.log_loss(labels=y, predictions=y_pred)
    loss += regularization * tf.nn.l2_loss(W)

with tf.name_scope("hyperparameters"):
    regularization = tf.placeholder(tf.float32, name="regularization")
    learning_rate = tf.placeholder(tf.float32, name="learning-rate")

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

with tf.name_scope("score"):
    correct_prediction = tf.equal(tf.to_float(y_pred > 0.5), y)
    accuracy = tf.reduce_mean(tf.to_float(correct_prediction), name="accuracy")


with tf.Session() as sess:

    tf.global_variables_initializer().run()
    for epoch in range(50000):

        perm = np.arange(len(x_train))
        np.random.shuffle(perm)
        x_train = x_train[perm]
        y_train = y_train[perm]

        feed = {x: x_train, y: y_train, learning_rate: 1e-2, regularization: 2 * 1e-5}
        sess.run(train_op, feed_dict=feed)

        if epoch % print_every == 0:
            train_accuracy, loss_value = sess.run([accuracy, loss], feed_dict=feed)
            print("step: %4d, loss: %.3f, training accuracy: %.3f" % (epoch, loss_value, train_accuracy))

    print("Test Accuracy:", "{:.3f}".format(sess.run(accuracy, feed_dict={x: x_test, y: y_test})))

