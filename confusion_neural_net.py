import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Seed random number generator
random_state = 42
np.random.seed(random_state)
tf.set_random_seed(random_state)

# Function to load training and test data
def load(filename):
    with open(filename + '.pickle', 'rb') as f:
        return pickle.load(f)

# Loading data
data_x = load('train_spins')
temp = load('temperature')
dim = len(temp)

# Parameters
learning_rate = 1e-2
l2 = 2 * 1e-5
training_epochs = 2000
display_step = 100

# Network Parameters
n_hidden_1 = 100 # 1st layer number of features
n_input = 16 * 16 # 2D Ising lattice
n_classes = 2 # high and low phase

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden_1])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes]))
}

# Create models
def single_sigmoid(x, weights, biases):
    layer_1 = tf.sigmoid(tf.matmul(x, weights['h1']) + biases['b1'])
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

def initialize_model(x, weights, biases):

    # Construct model, define cost and optimizer
    pred = single_sigmoid(x, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
    regularizer = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['out'])
    cost = tf.reduce_mean(cost + l2 * regularizer)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    return pred, cost, optimizer

step = 25 # Step in confusion scheme
accuracy = []

# Loop over proposed critical temperatures
for pos in range(0, dim, step):

    print('step {}'.format(pos))

    # Initializing the variables
    pred, cost, optimizer = initialize_model(x, weights, biases)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Generate label for training
    low = np.array([[0] * (dim - pos), [1] * (dim - pos)]).T
    high = np.array([[1] * pos, [0] * pos]).T
    data_y = np.vstack((low, high))

    # Take subset of data for training
    train_x, _, train_y, _ = train_test_split(data_x, data_y, train_size=0.1, random_state=random_state)

    for epoch in range(training_epochs):

        # Randomly shuffle training data
        perm = np.arange(len(train_x))
        np.random.shuffle(perm)
        train_x, train_y = train_x[perm], train_y[perm]

        # Train the network
        _, c = sess.run([optimizer, cost], feed_dict={x: train_x, y: train_y})

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "Cost =", "{:.3f}".format(c))

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy_tmp = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy.append(accuracy_tmp.eval({x: data_x, y: data_y}))
    sess.close()

plt.plot(np.linspace(1., 3.5, len(accuracy)), accuracy, '-o', color="orange")
plt.xlabel('Temperature', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.grid()
plt.savefig('confusion_NN.png')
