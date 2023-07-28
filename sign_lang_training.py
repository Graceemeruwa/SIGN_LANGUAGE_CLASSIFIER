# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 10:13:11 2023

@author: USER
"""

# Import Dependencies
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import warnings
warnings.filterwarnings("ignore")

#importing the datasets
train = pd.read_csv('train_data.csv')
test =pd.read_csv('test_data.csv')


# See how many labels we have in the dataset
train['label'].nunique()

# Let's check, if we have samples with label 9 and 25.
len(train.loc[train['label'] == 9, :])  # 0
len(train.loc[train['label'] == 25, :])  # 0


## Data Preprocessing

# Convert the datasets into numpy arrays for efficiency.
train_data = np.array(train, dtype='float32')
test_data = np.array(test, dtype='float32')

# Define class labels for easy interpretation
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
len(class_names)

# Sanity check - plot a few images and labels
i = random.randint(1, train.shape[0])
fig1, ax1 = plt.subplots(figsize=(2, 2))
plt.imshow(train_data[i, 1:].reshape((28, 28)), cmap='gray')
plt.show()
print("Label for the image is: ", class_names[int(train_data[i, 0])])

# Data distribution visualization -> Dataset seems to be fairly balanced. No balancing operation is needed.
fig = plt.figure(figsize=(18, 18))
# sns.set_theme(style = "darkgrid")
ax = sns.countplot(x="label", data=train)
ax.set_ylabel('Count')
ax.set_title('Label')
plt.show()

# Normalize / scale X values
X_train = train_data[:, 1:] /255.
X_test = test_data[:, 1:] /255.

# Convert y to categorical if planning on using categorical_crossentropy. No need to do this if using sparse_categorical_crossentropy.
y_train = train_data[:, 0]
# y_train_cat = to_categorical(y_train, num_classes=24)

y_test = test_data[:, 0]
# y_test_cat = to_categorical(y_test, num_classes=24)

# Reshape for the neural network
X_train = X_train.reshape(X_train.shape[0], *(28, 28, 1))
X_test = X_test.reshape(X_test.shape[0], *(28, 28, 1))

# Take a look at some samples from train dataset
plt.figure(figsize=(9, 7))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(X_train[i], cmap='gray')
    plt.xlabel(np.argmax(y_train[i]))
plt.show()


## Model Building

# Define epochs for all models
epochs = 10

# Model1
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(25, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

# Model summary
model.summary()

# Plot model architecture
plot_model(model, to_file='results/model_plot.png', show_shapes=True, show_layer_names=True)


# Train the model and measure model speed
start = time.time()
history = model.fit(X_train,
                    y_train,
                    validation_data=(X_test, y_test),
                    batch_size=128,
                    epochs=epochs,
                    verbose=1,
                    #callbacks=callbacks_list
                    )
end = time.time()
print(f'The time taken to execute is {round(end-start,2)} seconds.')

# Save the model
model.save('sign_lang_model.hdf5')
print("model saved")
