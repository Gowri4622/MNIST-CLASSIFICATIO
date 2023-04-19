
# <p align="center"> Convolutional Deep Neural Network for Digit Classification </p>

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

![image](https://user-images.githubusercontent.com/75235293/190975763-7d3b7c0f-9458-41e9-a35c-aa063c4977da.png)


## Neural Network Model

![image](https://user-images.githubusercontent.com/75235293/190976591-e7be8aea-1886-4181-b490-2abde39f2ff6.png)


## DESIGN STEPS
### STEP-1:
Import tensorflow and preprocessing libraries
### STEP 2:
Download and load the dataset
### STEP 3:
Scale the dataset between it's min and max values
### STEP 4:
Using one hot encode, encode the categorical values
### STEP-5:
Split the data into train and test
### STEP-6:
Build the convolutional neural network model
### STEP-7:
Train the model with the training data
### STEP-8:
Plot the performance plot
### STEP-9:
Evaluate the model with the testing data
### STEP-10:
Fit the model and predict the single input

## PROGRAM

```python
# Developed By:Gowri M
# Register Number:212220230019
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image= X_train[25440]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train[25440]
y_train.shape
X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled.min()
X_train_scaled.max()
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape
single_image = X_train[2376]
plt.imshow(single_image,cmap='gray')
y_train_onehot[2376]
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)
model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.summary()

# Choose the appropriate parameters
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
model.fit(X_train_scaled ,y_train_onehot, epochs=50,batch_size=64, validation_data=(X_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))

#Prediction for a single input

from tensorflow.keras.preprocessing import image
img=image.load_img('8nom.png')
img_tensor=tf.convert_to_tensor(np.asarray(img))
img_28=tf.image.resize(img_tensor,(28,28))
img_28_gray=tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled=img_28_gray.numpy()/255.0
x_single_prediction = np.argmax(model.predict(img_28_gray_scaled.reshape(1,28,28,1)),axis=1)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='Dark2')
print(x_single_prediction)
img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
plt.imshow(img_28_gray_inverted_scaled.reshape(28,28),cmap='gray')
x_single_prediction = np.argmax(model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),axis=1)
print(x_single_prediction)
```
</br>
</br>

</br>

</br>

</br>

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![WhatsApp Image 2022-09-22 at 7 02 58 PM (5)](https://user-images.githubusercontent.com/75235455/191963555-f4ac0a2e-775b-4273-89e0-55e50f877e29.jpeg)


### Classification Report
![WhatsApp Image 2022-09-22 at 7 02 58 PM (3)](https://user-images.githubusercontent.com/75235455/191963673-b8a66b7e-869e-49fa-b45f-56d1d112687e.jpeg)



### Confusion Matrix

![WhatsApp Image 2022-09-22 at 7 02 58 PM (4)](https://user-images.githubusercontent.com/75235455/191963743-30d43828-5aaa-48f6-8c43-c86964d53453.jpeg)



### New Sample Data Prediction
![WhatsApp Image 2022-09-22 at 7 02 58 PM (2)](https://user-images.githubusercontent.com/75235455/191963970-79451889-c73a-4a9c-930d-82748f7f6d4c.jpeg)
![WhatsApp Image 2022-09-22 at 7 02 58 PM (1)](https://user-images.githubusercontent.com/75235455/191963989-26732ffd-1230-4129-b949-26bef5994ce7.jpeg)
![WhatsApp Image 2022-09-22 at 7 02 58 PM](https://user-images.githubusercontent.com/75235455/191964022-579b82f2-1098-40a5-83da-ce2e86f11f67.jpeg)



## RESULT
Thus a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully
