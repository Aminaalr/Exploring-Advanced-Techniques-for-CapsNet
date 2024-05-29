import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Reshape, Flatten, Lambda, Multiply, Activation,Dropout
from capsUtils import combineIMG
from capsUtils import plotLog
from PIL import Image, ImageOps
from capsLayers4 import CapsuleLayer, PrimaryCap, Length, Mask,squash
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img
from os import listdir
from os.path import isfile, join
from numpy import asarray

tf.keras.datasets.fashion_mnist.load_data()

K.set_image_data_format('channels_last')
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.layers import LeakyReLU
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import EarlyStopping

start_time = time.time()


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)



import tensorflow as tf

def swish(x, alpha=2, beta=0.5):

    return (alpha * x) * tf.math.sigmoid(beta * x)


fix_gpu()

# Image Directory Location
pathImg = 'images'

# Image Size
image_size = 32
# Function to create Capsule Network (CapsNet)
from keras.layers import Add


def CapsNet(input_shape, n_class, routings, batch_size):
    x = layers.Input(shape=input_shape, batch_size=batch_size)

    # Base Filter Substructure (BFS)
    conv1 = layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same',  activation=swish, name='conv1')(x)
    bn1 = layers.BatchNormalization()(conv1)

    # SqueezeNet Structure (SNS)
    def fire_module(x, squeeze_filters, expand_filters):
        squeeze = layers.Conv2D(squeeze_filters, (1, 1), padding='same')(x)
        squeeze = layers.BatchNormalization()(squeeze)
        squeeze = layers.Activation(activation=swish)(squeeze)
        expand1 = layers.Conv2D(expand_filters, (1, 1), padding='same')(squeeze)
        expand1 = layers.BatchNormalization()(expand1)
        expand1 = layers.Activation(activation=swish)(expand1)
        expand3 = layers.Conv2D(expand_filters, (3, 3), padding='same')(squeeze)
        expand3 = layers.BatchNormalization()(expand3)
        expand3 = layers.Activation( activation=swish)(expand3)
        return layers.Concatenate()([expand1, expand3])

    fire2 = fire_module(bn1, 16, 64)
    fire3 = fire_module(fire2, 16, 64)
    fire4 = fire_module(fire3, 32, 128)
    fire4= layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same',  activation=swish, name='conv13')(fire4)
    fire4=Add()([bn1, fire4])
    conv11 = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same',  activation=swish)(fire4)
    fire5 = fire_module(conv11, 64, 256)
    fire6 = fire_module(fire5, 96, 384)
    fire6 = layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='same',  activation=swish,name='conv222')(fire6)
    # Residual connection from the output of conv11 to the output of fire6
    fire6 = Add()([conv11, fire6])
    bn2 = layers.BatchNormalization()(fire6)
    conv2 = layers.Conv2D(filters=128, kernel_size=1, strides=1, padding='same',  activation=swish, name='conv2')(bn2)
    conv2 = Lambda(squash, name='squash1')(conv2)
    fire7 = fire_module(conv2, 192, 384)
    fire7=layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='same',  activation=swish, name='conv22')(fire7)

    # Residual Connection
    fire7_residual = Add()([bn2, fire7])

    # CapsNet Substructure (CNS)
    primarycaps = PrimaryCap(fire7_residual, dim_capsule=8, n_channels=32, kernel_size=3, strides=2, padding='valid')
    primarycaps = Lambda(squash, name='primarycap_squash2')(primarycaps)
    # Residual Connection in Primary Capsule Layer
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, name='digitcaps')(primarycaps)
    out_caps = Length(name='capsnet')(digitcaps)
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])
    masked = Mask()(digitcaps)

    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512,activation=swish, input_dim=16 * n_class))
    decoder.add(layers.BatchNormalization())
    decoder.add(Dropout(0.3))
    decoder.add(layers.Dense(1024,  activation=swish))
    #decoder.add(layers.Dense(2048, activation='swish'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='swish'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))

    return train_model, eval_model, manipulate_model

def margin_loss(y_true, y_pred):
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))
    return tf.reduce_mean(tf.reduce_sum(L, 1))

# Performance Matrics
def performance_metrics(cnf_matrix, class_names):
    # Confusion Matrix Plot
    cmd = ConfusionMatrixDisplay(cnf_matrix, display_labels=class_names)
    cmd.plot(cmap='Greens')
    cmd.ax_.set(xlabel='Predicted', ylabel='Actual')
    # Find All Parameters
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)
    # F1-Score accuracy for each class
    FScore = 2 * (PPV * TPR) / (PPV + TPR)
    # Overall accuracy for each class
    ACC = (TP + TN) / (TP + FP + TN + FN)
    print('\n\nClassName\tTP\tFP\tFN\tTN\tPrecision\tSensitivity\tSpecificity\tF-Score\t\tAccuracy')
    for i in range(len(class_names)):
        print(class_names[i] + "\t\t{0:.0f}".format(TP[i]) + "\t{0:.0f}".format(FP[i]) + "\t{0:.0f}".format(
            FN[i]) + "\t{0:.0f}".format(TN[i]) + "\t{0:.4f}".format(PPV[i]) + "\t\t{0:.4f}".format(
            TPR[i]) + "\t\t{0:.4f}".format(TNR[i]) + "\t\t{0:.4f}".format(FScore[i]) + "\t\t{0:.4f}".format(ACC[i]))


# training part of Model
def train(model, data, class_names, args):
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
    #early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)  # Add early stopping

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)  # 800 total batch size
        while 1:
            x_batch, y_batch = generator.next()
            yield (x_batch, y_batch), (y_batch, x_batch)

    # Training with data augmentation. If shift_fraction=0., no augmentation.
    model.fit(train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
              steps_per_epoch=int(y_train.shape[0] / args.batch_size),
              epochs=args.epochs,
              validation_data=((x_test, y_test), (y_test, x_test)), batch_size=args.batch_size,
              callbacks=[log, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    y_pred, x_recon = model.predict((x_test, y_test), batch_size=100)

    # Confusion matrix
    cm = confusion_matrix(np.argmax(y_test, 1), np.argmax(y_pred, 1))
    # Overall Performance
    performance_metrics(cm, class_names)
    plotLog(args.save_dir + '/log.csv', showPlot=True)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])
    return model


def test(model, data, args):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-' * 30 + 'Begin: test' + '-' * 30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])

    img = combine_images(np.concatenate([x_test[:50], x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    plt.show()


def manipulate_latent(model, data, args):
    print('-' * 30 + 'Begin: manipulate' + '-' * 30)
    x_test, y_test = data
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:, :, dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
    print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
    print('-' * 30 + 'End: manipulate' + '-' * 30)


# Normalization of Data
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def loadDataset():
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

  # Preprocess data (e.g., normalize pixel values)
  x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
  x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

  # Convert labels to one-hot encoding
  y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

  # Update class names for MNIST digits (0-9)
  class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

  return (x_train, y_train), (x_test, y_test), class_names



# arguments for Caps Net Parameters
if __name__ == "__main__":
    import os
    import argparse
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on Dataset.")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    (x_train, y_train), (x_test, y_test), classNames = loadDataset()


    # define model
    model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape[1:],
                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                  routings=args.routings,
                                                  batch_size=args.batch_size)
    model.summary()

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), class_names=classNames, args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        manipulate_latent(manipulate_model, (x_test, y_test), args)
        test(model=eval_model, data=(x_test, y_test), args=args)

end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time} seconds")
