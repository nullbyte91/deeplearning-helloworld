from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def LeNet(numChannels, imgRows, imgCols, numClasses,
		activation="relu", weightsPath=None):
    # initialize the model
		model = Sequential()
		inputShape = (imgRows, imgCols, numChannels)

		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
			inputShape = (numChannels, imgRows, imgCols)

		# define the first set of CONV => ACTIVATION => POOL layers
		model.add(Conv2D(20, 5, padding="same",
			input_shape=inputShape))
		model.add(Activation(activation))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# define the second set of CONV => ACTIVATION => POOL layers
		model.add(Conv2D(50, 5, padding="same"))
		model.add(Activation(activation))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# define the first FC => ACTIVATION layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation(activation))

		# define the second FC layer
		model.add(Dense(numClasses))

		# lastly, define the soft-max classifier
		model.add(Activation("softmax"))

		# if a weights path is supplied (inicating that the model was
		# pre-trained), then load the weights
		if weightsPath is not None:
			model.load_weights(weightsPath)

		# return the constructed network architecture
		return model

def testing(number):
    for i in np.random.choice(np.arange(0, len(testLabels)), size=(number,)):
        # classify the digit
        probs = model.predict(testData[np.newaxis, i])
        prediction = probs.argmax(axis=1)

        # extract the image from the testData if using "channels_first"
        # ordering
        if K.image_data_format() == "channels_first":
            image = (testData[i][0] * 255).astype("uint8")

        # otherwise we are using "channels_last" ordering
        else:
            image = (testData[i] * 255).astype("uint8")

        # merge the channels into one image
        image = cv2.merge([image] * 3)

        # resize the image from a 28 x 28 image to a 96 x 96 image so we
        # can better see it
        image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)

        # show the image and prediction
        cv2.putText(image, str(prediction[0]), (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
            np.argmax(testLabels[i])))
        cv2.imshow("Digit", image)
        cv2.waitKey(0)

# grab the MNIST dataset (if this is your first time running this
# script, the download may take a minute -- the 55MB MNIST dataset
# will be downloaded)
print("[INFO] downloading MNIST...")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

# if we are using "channels first" ordering, then reshape the
# design matrix such that the matrix is:
# num_samples x depth x rows x columns
if K.image_data_format() == "channels_first":
	trainData = trainData.reshape((trainData.shape[0], 1, 28, 28))
	testData = testData.reshape((testData.shape[0], 1, 28, 28))
# otherwise, we are using "channels last" ordering, so the design
# matrix shape should be: num_samples x rows x columns x depth
else:
	trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
	testData = testData.reshape((testData.shape[0], 28, 28, 1))

# scale data to the range of [0, 1]
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

# transform the training and testing labels into vectors in the
# range [0, classes] -- this generates a vector for each label,
# where the index of the label is set to `1` and all other entries
# to `0`; in the case of MNIST, there are 10 class labels
trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)

#Now we will split training data into training data and validation data
trainData, validaData, trainLabels, validLabels = train_test_split(trainData, trainLabels, test_size = 0.1, random_state=42)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet(numChannels=1, imgRows=28, imgCols=28, numClasses=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
history = model.fit(trainData, trainLabels, validation_data=(validaData, validLabels), batch_size=128, epochs=20, verbose=1)

#Model testing
#testing(1)

#Model evaluation
#Loss plot
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, color='red', label='Training loss')
plt.plot(epochs, val_loss, color='green', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Accuracy Plot
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, color='red', label='Training acc')
plt.plot(epochs, val_acc, color='green', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Prediction with valid dataset
prediction=model.evaluate(validaData,validLabels)
print("Val:accuaracy", str(prediction[1]*100))
print("Val:Total loss",str(prediction[0]*100))

#Evaluvating with test dataset
prediction=model.evaluate(testData, testLabels)
print("Test:accuaracy", str(prediction[1]*100))
print("Test:Total loss",str(prediction[0]*100))