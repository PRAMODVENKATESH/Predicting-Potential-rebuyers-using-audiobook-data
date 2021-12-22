import numpy as np

# use the sklearn preprocessing library, as it will be easier to standardize the data.
from sklearn import preprocessing

# Load the data
raw_csv_data = np.loadtxt('Audiobooks_data.csv',delimiter=',')

# The inputs are all columns in the csv, except for the first one [:,0]
# Target Variable is the last column [:,-1] (which is our targets)

unscaled_inputs_all = raw_csv_data[:,1:-1]

# The targets are in the last column. That's how datasets are conventionally organized.
targets_all = raw_csv_data[:,-1]

##Schuffel the data##-This step is crucial in order to prevent the optimizer from being stuck at the local minima and converge at global minima.
# Since we will be batching, we want the data to be as randomly spread out as possible
shuffled_indices = np.arange(unscaled_inputs_all.shape[0])
np.random.shuffle(shuffled_indices)

# Use the shuffled indices to shuffle the inputs and targets.
unscaled_inputs_all = unscaled_inputs_all[shuffled_indices]
targets_all = targets_all[shuffled_indices]
# Count how many targets are 1,s because there are many zeros and lesser 1's [0's represent customer who did not make repurchase, 1 indicates customers who made a repurchase]
num_one_targets = int(np.sum(targets_all))

# Set a counter for targets that are 0 (meaning that the customer did not convert)
zero_targets_counter = 0

# We want to create a "balanced" dataset, so we will have to remove some input/target pairs.
# Declare a variable that will do that:
indices_to_remove = []

# Count the number of targets that are 0. 
# Once there are as many 0s as 1s, mark entries where the target is 0.
for i in range(targets_all.shape[0]):
    if targets_all[i] == 0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)

# Create two new variables, one that will contain the inputs, and one that will contain the targets.
# We delete all indices that we marked "to remove" in the loop above.
unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)

##Standardizing the input##
# We will take advantage of its preprocessing capabilities of sklearn to standardize the dataset
# The line of code standardizes the inputs.
scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)

##Splitting the data into train test validation datset##
# Count the total number of samples
samples_count = shuffled_inputs.shape[0]

# Count the samples in each subset (assuming we want 80-10-10 distribution of training, validation, and test).
# Naturally, the numbers are integers.
train_samples_count = int(0.8 * samples_count)
validation_samples_count = int(0.1 * samples_count)

# The 'test' dataset contains all remaining data.
test_samples_count = samples_count - train_samples_count - validation_samples_count

# Create variables that record the inputs and targets for training
# In the shuffled dataset, they are the first "train_samples_count" observations
train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

# Create variables that record the inputs and targets for validation.
# They are the next "validation_samples_count" observations, folllowing the "train_samples_count" we already assigned
validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

# Create variables that record the inputs and targets for test.
test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]

# Data set is balanced to be 50-50 (for targets 0 and 1), but the training, validation, and test were 
# Print the number of targets that are 1s, the total number of samples, and the proportion for training, validation, and test.
print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)
print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)

##Saving the datasets zipped .np file##
np.savez('Audiobooks_data_train', inputs=train_inputs, targets=train_targets)
np.savez('Audiobooks_data_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('Audiobooks_data_test', inputs=test_inputs, targets=test_targets)
