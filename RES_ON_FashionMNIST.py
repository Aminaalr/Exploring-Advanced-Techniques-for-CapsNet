import os
from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Reshape, Flatten, Lambda, Multiply
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import LeakyReLU
import tensorflow.keras.activations as activations
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks
from tensorflow.keras.layers import PReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras import callbacks
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
import time
import datetime
from capsUtils import plotLog

def conditional_prelu(inputs):
  features, predictions = inputs  # Split input into features and predictions
  pos = K.relu(features)
  neg = predictions * K.relu(-features)
  return pos + neg

def fix_gpu():
    # Set GPU memory growth option
    config = tf.config.experimental.ConfigProto()
    config.gpu_options.allow_growth = True

    # Create TensorFlow session using the updated config
    tf.Session(config=config)

    # Clear Keras session to ensure compatibility
    K.clear_session()


# Image Directory Location
pathImg = 'images'

# Image Size
image_size = 28




# Normalization of Data
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def loadDataset():
    # Load Fashion MNIST dataset from Keras
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Resize training images to (32, 32) and convert to RGB
    x_train_resized_rgb = np.array([np.array(Image.fromarray(image).resize((32, 32))) for image in x_train])
    x_train_resized_rgb = np.repeat(x_train_resized_rgb[..., np.newaxis], 3, axis=-1)

    # Resize testing images to (32, 32) and convert to RGB
    x_test_resized_rgb = np.array([np.array(Image.fromarray(image).resize((32, 32))) for image in x_test])
    x_test_resized_rgb = np.repeat(x_test_resized_rgb[..., np.newaxis], 3, axis=-1)

    # Rescale the images to [0, 1] range
    x_train_resized_rgb = x_train_resized_rgb.astype('float32') / 255.
    x_test_resized_rgb = x_test_resized_rgb.astype('float32') / 255.

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    return (x_train_resized_rgb, y_train), (x_test_resized_rgb, y_test), class_names
# depend on this equation
import tensorflow as tf
from tensorflow.keras import layers


def squash(vectors, axis=-1):
    # Non-linear activation function (squashing)
    s_squared_norm = tf.reduce_sum(tf.square(vectors / 5), axis, keepdims=True)
    # s_squared_norm2= tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = (0.5 * s_squared_norm) / (1 + 0.5 * s_squared_norm) / tf.sqrt(s_squared_norm + K.epsilon())
    # Layer normalization
    epsilon = 1e-6
    mean, variance = tf.nn.moments(vectors, axes=-1, keepdims=True)
    vectors = (vectors - mean) / tf.sqrt(variance + epsilon)

    return scale * vectors
# Build Model for VGG16 with CapsuleNet
def res_capsNet_model(n_routings=3):
    # Initialize the VGG16 Model
    res = tf.keras.applications.ResNet152V2(include_top=False, weights="imagenet",
                                            input_shape=(image_size, image_size, 3))
    # CapsuleNet as primary Capsule Convolution
    xCpas = Conv2D(filters=8 * 32, kernel_size=1, strides=2, padding='valid', name='primarycap_conv2')(vgg.output)
    # Reshape the Primary Capsule
    xCpas = Reshape(target_shape=[-1, 8], name='primarycap_reshape')(xCpas)
    # Squash for Primary Capsule
    xCpas = Lambda(squash, name='primarycap_squash')(xCpas)
    # Normalization of Momentum as 0.8
    xCpas = BatchNormalization(momentum=0.8)(xCpas)
    # Flatten Layer
    xCpas = Flatten()(xCpas)
    # Initialization of Dense
    vgg_Caps = Dense(160, kernel_initializer='he_normal', bias_initializer='zeros', name='uhat_digitcaps')(xCpas)
    # Repeat Dense for Number of Routings
    for i in range(n_routings):
        # Activation of Softmax
        cSoft = Activation('softmax', name='softmax_digitcaps' + str(i))(vgg_Caps)
        # Assign Dense as 160
        cSoft = Dense(160)(cSoft)
        cSoft = Lambda(squash, name='squash2')(cSoft)
        # Multiply
        vggCaps = Multiply()([vgg_Caps, cSoft])
        # Enhance PRelu activation function
        # Implement conditional prelu using Lambda
        sJ = Lambda(conditional_prelu)([vggCaps, cSoft])
    # Output Dense as 32
    vggCaps = Dense(32, activation='relu')(sJ)
    # Predict Model
    pred = Dense(len(classNames), activation='softmax')(vggCaps)
    return Model(res.input, pred)

# load data
(x_train, y_train), (x_test, y_test), (x_val, y_val), classNames = loadDataset()

resCapsNet_Model = res_capsNet_model()
resCapsNet_Model.summary()
resCapsNet_Model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
tf.config.run_functions_eagerly(True)

# callbacks
lr = 0.001
lr_decay = 0.9
log = callbacks.CSVLogger('./result' + '/log.csv')
checkpoint = callbacks.ModelCheckpoint('./result' + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                       save_best_only=True, save_weights_only=True, verbose=1)
lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: lr * (lr_decay ** epoch))

tensorboard = TensorBoard(log_dir='logs')
checkpoint = ModelCheckpoint("effnet.h5", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_delta=0.001, mode='auto', verbose=1)
history = resCapsNet_Model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, verbose=1, batch_size=100,
                               callbacks=[log, checkpoint, reduce_lr])

y_pred = resCapsNet_Model.predict([x_test], batch_size=100)

# Confusion matrix
cm = confusion_matrix(np.argmax(y_test, 1), np.argmax(y_pred, 1))
# Overall Performance
performance_metrics(cm, classNames)
plotLog('./result' + '/log.csv', showPlot=True)
print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])
end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time} seconds")
