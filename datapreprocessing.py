"""
    The data from the audiobooks store.
    The headers is as fallow:
    ID, Book length overall, book length average, price overall, price average, review, review 10/10, minutes listened, completion, support request, last visited minus purchase data, targets
    """

import pandas as pd
from sklearn import preprocessing
import numpy as np

# data reading
raw_data = np.loadtxt('Audiobooks_data.csv', delimiter=',')

unscaled_input_all = raw_data[:, 1:-1]
targets_all = raw_data[:, -1]

# balance the dataset

num_one_targets = int(np.sum(targets_all))  # counting ones in targets
zero_targets_counter = 0
indices_to_remove = []

for i in range(targets_all.shape[0]):
    if (targets_all[i] == 0):
        zero_targets_counter += 1
        if(zero_targets_counter > num_one_targets):
            indices_to_remove.append(i)

unscaled_inputs_balanced = np.delete(
    unscaled_input_all, indices_to_remove, axis=0)
targets_balanced = np.delete(
    targets_all, indices_to_remove, axis=0)

# scaling
scaled_inputs = preprocessing.scale(unscaled_inputs_balanced)

# shuffle the data
shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)
shuffled_inputs = scaled_inputs[shuffled_indices, :]
shuffled_targets = targets_balanced[shuffled_indices]

# split into validation and test

samples_count = shuffled_inputs.shape[0]
train_samples_count = int(0.8 * samples_count)
validation_samples_count = int(0.1 * samples_count)
test_count = samples_count - train_samples_count - validation_samples_count

train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

validation_inputs = shuffled_inputs[train_samples_count:
                                    train_samples_count + validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:
                                      train_samples_count + validation_samples_count]

test_inputs = shuffled_inputs[train_samples_count +
                              validation_samples_count:]
test_targets = shuffled_targets[train_samples_count +
                                validation_samples_count:]


# save to
np.savez('Audiobooks_data_train', inputs=train_inputs, targets=train_targets)
np.savez('Audiobooks_data_validation',
         inputs=validation_inputs, targets=validation_targets)
np.savez('Audiobooks_data_test', inputs=test_inputs, targets=test_targets)
