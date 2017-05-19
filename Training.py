from keras.datasets import mnist
import numpy as np
import random
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

# read mnist data from keras
(x_train, _), (x_test, _) = mnist.load_data()
# duplicate original train data
ori_train = np.concatenate([x_train, x_train])
# generate 120k modified add-noise train data
noi_train = x_train[:]
for i in range(x_train.shape[0]):
    for j in range(x_train.shape[1]):
        for k in range(x_train.shape[2]):
            # filp pixel value
            if random.randint(1, 100) < 4:
                noi_train[i][j][k] = 255 - ori_train[i][j][k]
noi_train = np.concatenate([noi_train, noi_train])
# generate add-noise test data with same shape as x_test
noi_test = x_test[:]
for i in range(x_test.shape[0]):
    for j in range(x_test.shape[1]):
        for k in range(x_test.shape[2]):
            if random.randint(1, 100) < 4:
                noi_test[i][j][k] = 255 - x_test[i][j][k]
# normalize train and test data
noi_train = noi_train.astype('float32') / 255.
noi_test = noi_test.astype('float32') / 255.
ori_train = ori_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# adapt this if using `channels_first` image data format
input_img = Input(shape=(28, 28, 1))

# first CNN architecture
# build encoder function
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['mse'])
# adapt this if using `channels_first` image data format
ori_train = np.reshape(ori_train, (len(ori_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
noi_train = np.reshape(noi_train, (len(noi_train), 28, 28, 1))
noi_test = np.reshape(noi_test, (len(noi_test), 28, 28, 1))
# train noise data in autoencoder
autoencoder.fit(noi_train, ori_train,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(noi_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder', histogram_freq=0, write_graph=False)], verbose=2)
# plot test error
test_error1 = [0.050, 0.046, 0.045, 0.044, 0.043, 0.042, 0.041, 0.041, 0.042, 0.040]
x_axis = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
plt.plot(x_axis, test_error1)
plt.xlabel("number of epoch")
plt.ylabel("test set error")
plt.title("First CNN architecture")
plt.show()
x_predict1 = autoencoder.predict(noi_test)
# display best 10 digits
mse1 = np.argsort(np.mean(np.square(x_predict1 - x_test), axis=(1, 2, 3)))
for i in range(10):
    # display original
    ax = plt.subplot(2, 10, i + 1)
    list = mse1[:10]
    plt.imshow(x_test[list[i]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2, 10, i + 10 + 1)
    plt.imshow(x_predict1[list[i]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
# display 10 worst digits
for i in range(10):
    # display original
    ax = plt.subplot(2, 10, i + 1)
    list = mse1[::-1][:10]
    plt.imshow(x_test[list[i]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2, 10, i + 10 + 1)
    plt.imshow(x_predict1[list[i]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
# second CNN architecture
# build encoder function
x2 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x2 = MaxPooling2D((2, 2), padding='same')(x2)
x2 = Conv2D(16, (3, 3), activation='relu', padding='same')(x2)
x2 = MaxPooling2D((2, 2), padding='same')(x2)
x2 = Conv2D(16, (3, 3), activation='relu', padding='same')(x2)
encoded2 = MaxPooling2D((2, 2), padding='same')(x2)
x2 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded2)
x2 = UpSampling2D((2, 2))(x2)
x2 = Conv2D(8, (3, 3), activation='relu', padding='same')(x2)
x2 = UpSampling2D((2, 2))(x2)
x2 = Conv2D(16, (3, 3), activation='relu')(x2)
x2 = UpSampling2D((2, 2))(x2)
decoded2 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x2)
autoencoder2 = Model(input_img, decoded2)
autoencoder2.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['mse'])
# train noise data in autoencoder
autoencoder2.fit(noi_train, ori_train,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(noi_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder', histogram_freq=0, write_graph=False)], verbose=2)
# plot test error
test_error2 = [0.0533, 0.0478, 0.0435, 0.0433, 0.0416, 0.0412, 0.0408, 0.0395, 0.0398, 0.0395]
x_axis = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
plt.plot(x_axis, test_error2)
plt.xlabel("number of epoch")
plt.ylabel("test set error")
plt.title("Second CNN architecture")
plt.show()
x_predict2 = autoencoder2.predict(noi_test)
# display best 10 digits
mse2 = np.argsort(np.mean(np.square(x_predict2 - x_test), axis=(1, 2, 3)))
for i in range(10):
    # display original
    ax = plt.subplot(2, 10, i + 1)
    list = mse2[:10]
    plt.imshow(x_test[list[i]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2, 10, i + 10 + 1)
    plt.imshow(x_predict2[list[i]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
# display 10 worst digits
for i in range(10):
    # display original
    ax = plt.subplot(2, 10, i + 1)
    list = mse2[::-1][:10]
    plt.imshow(x_test[list[i]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2, 10, i + 10 + 1)
    plt.imshow(x_predict2[list[i]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
# Third CNN architecture
# build encoder function
x3 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x3 = MaxPooling2D((2, 2), padding='same')(x3)
x3 = Conv2D(16, (3, 3), activation='relu', padding='same')(x3)
x3 = MaxPooling2D((2, 2), padding='same')(x3)
x3 = Conv2D(16, (3, 3), activation='relu', padding='same')(x3)
encoded3 = MaxPooling2D((2, 2), padding='same')(x3)
x3 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded3)
x3 = UpSampling2D((2, 2))(x3)
x3 = Conv2D(8, (3, 3), activation='relu', padding='same')(x3)
x3 = UpSampling2D((2, 2))(x3)
x3 = Conv2D(16, (3, 3), activation='relu')(x3)
x3 = UpSampling2D((2, 2))(x3)
decoded3 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x3)
autoencoder3 = Model(input_img, decoded3)
autoencoder3.compile(optimizer='adadelta', loss='mse', metrics=['mse'])
# train noise data in autoencoder
autoencoder3.fit(noi_train, ori_train,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(noi_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder', histogram_freq=0, write_graph=False)], verbose=2)
# plot test error
test_error3 = [0.0588, 0.0511, 0.0468, 0.0455, 0.0432, 0.0426, 0.0413, 0.0423, 0.0416, 0.0413]
x_axis = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
plt.plot(x_axis, test_error3)
plt.xlabel("number of epoch")
plt.ylabel("test set error")
plt.title("Third CNN architecture")
plt.show()
x_predict3 = autoencoder3.predict(noi_test)
# display best 10 digits
mse3 = np.argsort(np.mean(np.square(x_predict3 - x_test), axis=(1, 2, 3)))
for i in range(10):
    # display original
    ax = plt.subplot(2, 10, i + 1)
    list = mse3[:10]
    plt.imshow(x_test[list[i]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2, 10, i + 10 + 1)
    plt.imshow(x_predict3[list[i]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
# display 10 worst digits
for i in range(10):
    # display original
    ax = plt.subplot(2, 10, i + 1)
    list = mse3[::-1][:10]
    plt.imshow(x_test[list[i]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2, 10, i + 10 + 1)
    plt.imshow(x_predict3[list[i]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
# fourth CNN architecture
# build encoder function
x4 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x4 = MaxPooling2D((2, 2), padding='same')(x4)
x4 = Conv2D(32, (3, 3), activation='relu', padding='same')(x4)
x4 = MaxPooling2D((2, 2), padding='same')(x4)
x4 = Conv2D(32, (3, 3), activation='relu', padding='same')(x4)
encoded4 = MaxPooling2D((2, 2), padding='same')(x4)
x4 = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded4)
x4 = UpSampling2D((2, 2))(x4)
x4 = Conv2D(16, (3, 3), activation='relu', padding='same')(x4)
x4 = UpSampling2D((2, 2))(x4)
x4 = Conv2D(16, (3, 3), activation='relu')(x4)
x4 = UpSampling2D((2, 2))(x4)
decoded4 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x4)
autoencoder4 = Model(input_img, decoded4)
autoencoder4.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['mse'])
# train noise data in autoencoder
autoencoder4.fit(noi_train, ori_train,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(noi_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder', histogram_freq=0, write_graph=False)], verbose=2)
# plot test error
test_error3 = [0.0493, 0.0440, 0.0403, 0.0389, 0.0382, 0.0372, 0.0369, 0.0368, 0.0365, 0.0354]
x_axis = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
plt.plot(x_axis, test_error3)
plt.xlabel("number of epoch")
plt.ylabel("test set error")
plt.title("Third CNN architecture")
plt.show()
x_predict4 = autoencoder4.predict(noi_test)
# display best 10 digits
mse4 = np.argsort(np.mean(np.square(x_predict4 - x_test), axis=(1, 2, 3)))
for i in range(10):
    # display original
    ax = plt.subplot(2, 10, i + 1)
    list = mse4[:10]
    plt.imshow(x_test[list[i]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2, 10, i + 10 + 1)
    plt.imshow(x_predict4[list[i]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
# display 10 worst digits
for i in range(10):
    # display original
    ax = plt.subplot(2, 10, i + 1)
    list = mse4[::-1][:10]
    plt.imshow(x_test[list[i]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2, 10, i + 10 + 1)
    plt.imshow(x_predict4[list[i]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()







