# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 00:50:15 2021

@author: doguilmak

dataset: https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope

"""
#%%
# 1. Importing Libraries

from sklearn.impute import SimpleImputer
from keras.models import load_model
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import time
import warnings
warnings.filterwarnings('ignore')

#%%
# 2. Data Preprocessing

# 2.1. Uploading data
start = time.time()
data = pd.read_csv('magic04.data', header=None)

# 2.2. Looking for anomalies and duplicated datas
print(data.isnull().sum())
print("\n", data.head(10))
print("\n", data.describe().T)
print("\n{} duplicated.".format(data.duplicated().sum()))

# 2.3. Looking for '?' mark
from sklearn.preprocessing import LabelEncoder
data = data.apply(LabelEncoder().fit_transform)
print("data:\n", data)
data.replace('?', -999999, inplace=True)

imputer = SimpleImputer(missing_values= -999999, strategy='mean')
newData = imputer.fit_transform(data)

# 2.3. Determination of dependent and independent variables
X = newData[:, 0:10]
y = newData[:, 10]

#%%
# 3 Artificial Neural Network

# 3.1 Loading Created Model
model = load_model('model.h5')

# 3.2 Checking the Architecture of the Model
model.summary()

"""
# 3.1. Creating layers
model = Sequential()
# Input layer
# First hidden layer:
model.add(Dense(16, init="uniform", activation="relu", input_dim=10))
# Second hidden layer:
model.add(Dense(32, init="uniform", activation="relu"))
# Third hidden layer:
model.add(Dense(16, init="uniform", activation="relu"))
# Output layer:
model.add(Dense(1, init="uniform", activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model_history = model.fit(X, y, epochs=128, batch_size=32, validation_split=0.13)

# Plot accuracy and val_accuracy
print(model_history.history.keys())
model.summary()
model.save('model.h5')
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 12))
sns.set_style('whitegrid')
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('ANN Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
"""
# Predicting class
predict = np.array([23.8277,11.8989,2.4393,0.4655,0.2891,11.1013,11.5776,6.8613,35.3166,152.072]).reshape(1, 10)
print(f'Model predicted class as {model.predict_classes(predict)}.')

end = time.time()
cal_time = end - start
print("\nProcess took {} seconds.".format(cal_time))
