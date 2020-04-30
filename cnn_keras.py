from keras.layers import *
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# MNIST Data - 10 Classes (0-9)
def create_model():
    model = Sequential()
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
    model.add(MaxPool2D((2,2)))
    model.add(Conv2D(64,(3,3),activation='relu')) 
    model.add(MaxPool2D((2,2)))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    model.summary()

    return model;

def preprocess_data(X,Y):
    X = X.reshape((-1,28,28,1))
    X = X/255.0
    Y = to_categorical(Y)
    return X,Y


if __name__ == '__main__':

    model = create_model()
    ((XTrain,YTrain),(XTest,YTest)) = mnist.load_data()
    XTrain,YTrain = preprocess_data(XTrain,YTrain)
    XTest,YTest = preprocess_data(XTest,YTest)

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    hist = model.fit(XTrain,YTrain,epochs=20,validation_split=0.1,batch_size=128)

    model.evaluate(XTest,YTest)
