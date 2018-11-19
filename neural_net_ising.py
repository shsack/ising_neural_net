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
data_x, data_y = load('train_spins'), load('train_labels')
T = load('temperature')

# Split data into training and test set
train_x, test_x, train_y, test_y, train_T, test_T = train_test_split(data_x, data_y, T, test_size=0.1, random_state=random_state)

# Parameters
learning_rate = 1e-2
l2 = 2 * 1e-5
training_epochs = 5000
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

# Create model
def multilayer_sigmoid(x, weights, biases):
    layer_1 = tf.sigmoid(tf.matmul(x, weights['h1']) + biases['b1'])
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

# Construct model
pred = multilayer_sigmoid(x, weights, biases)

# Define cost and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
regularizer = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['out'])
cost = tf.reduce_mean(cost + l2 * regularizer)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Training loop
for epoch in range(training_epochs):

    # Randomly shuffle training data
    perm = np.arange(len(train_x))
    np.random.shuffle(perm)
    train_x, train_y = train_x[perm], train_y[perm]
    train_T = train_T[perm]

    # Training step
    _, c = sess.run([optimizer, cost], feed_dict={x: train_x, y: train_y})

    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "Cost =", "{:.3f}".format(c))

# Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

# Calculate and print accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print('Network trained!')
print('Accuracy: {:.3f}'.format(accuracy.eval({x: test_x, y: test_y})))

# Define NN output
output = tf.nn.softmax(multilayer_sigmoid(tf.cast(test_x, tf.float32), weights, biases))

# Plotting output of NN
plt.plot(test_T, abs(output[:, 0].eval()), '+', color="green", label='high T')
plt.plot(test_T, abs(output[:, 1]).eval(), 'v', color="red", label='low T')
plt.ylim(-0.05, 1.05)
plt.xlabel("Temperature", fontsize=15)
plt.ylabel("Output ", fontsize=15)
plt.grid()
plt.legend()
plt.savefig('magnetization_NN.png')