import tensorflow as tf
from main import IMAGE_SIZE


(x_train,_),(x_test,_)=tf.keras.datasets.mnist.load_data()
x_train=x_train.astype('float32')/255.
x_test=x_test.astype('float32')/255.
x_train=x_train.reshape(x_train.shape[0],*IMAGE_SIZE)
x_test=x_test.reshape(x_test.shape[0],*IMAGE_SIZE)