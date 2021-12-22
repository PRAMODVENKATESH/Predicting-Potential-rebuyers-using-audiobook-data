##Import Necessary libraries##
import numpy as np
import tensorflow as tf

##Load the files containing the datasets##
npz = np.load('/content/drive/MyDrive/Udemy /train_data/train_data.npz')

# extract the inputs using the keyword under which we saved them
train_inputs = npz['inputs'].astype(np.float)
# targets must be int because of sparse_categorical_crossentropy (we want to be able to smoothly one-hot encode them)
train_targets = npz['targets'].astype(np.int)

# Load the validation data in the temporary variable
npz = np.load('/content/drive/MyDrive/Udemy /validation_data/validaton_data.npz')
# Load the inputs and the targets in the same line
validation_inputs, validation_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)

#Load the test data in the temporary variable
npz = np.load('/content/drive/MyDrive/Udemy /test_data/test_data.npz')
#Create 2 variables that will contain the test inputs and the test targets
test_inputs, test_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)

##MODEL##
# Set the input and output sizes
input_size = 10
output_size = 2
# Use same hidden layer size for both hidden layers. Not a necessity.
hidden_layer_size1 = 100
hidden_layer_size2 = 70
hidden_layer_size3 = 30
    
# define how the model will look like
model = tf.keras.Sequential([
    # tf.keras.layers.Dense is basically implementing: output = activation(dot(input, weight) + bias)
    # it takes several arguments, but the most important ones for us are the hidden_layer_size and the activation function
    tf.keras.layers.Dense(hidden_layer_size1, activation='relu'), # 1st hidden layer
    tf.keras.layers.Dense(hidden_layer_size2, activation='relu'), # 2nd hidden layer
    tf.keras.layers.Dense(hidden_layer_size3, activation='relu'), # 2nd hidden layer
    # the final layer is no different, we just make sure to activate it with softmax
    tf.keras.layers.Dense(output_size, activation='sigmoid') # output layer
])


### Choose the optimizer and the loss function

# we define the optimizer we'd like to use, 
# the loss function, 
# and the metrics we are interested in obtaining at each iteration
opt = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])#if we wanna use sigmoid as activation chanege output node to 1 and loss to binary_crossentropy

### Training
# That's where we train the model we have built.

# set the batch size
batch_size = 60

# set a maximum number of training epochs
max_epochs = 200

# set an early stopping mechanism
# let's set patience=2, to be a bit tolerant against random validation loss increases
early_stopping = tf.keras.callbacks.EarlyStopping(patience=15)

