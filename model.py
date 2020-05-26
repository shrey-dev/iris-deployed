# Keras Model for IRIS dataset

# Importing libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
import joblib

# Importing dataset
iris= pd.read_csv(r'C:\Users\Shrey\Downloads\ML Datasets\IRIS.csv')
x = iris.drop('species', axis=1)
y = iris['species']

# Feature Scaling
lb = LabelBinarizer()
y = lb.fit_transform(y)
sc = MinMaxScaler()
scaled_x = sc.fit_transform(x)

# Creating Keras Model
model = Sequential()
model.add(Dense(units=4, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(scaled_x, y, epochs=150)

# Saving Model and Data
model.save("final_iris_model.h5")
joblib.dump(sc, 'iris_scaler.pkl')