import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

class Sampling(layers.Layer):
  def __init__(self,**kwargs):
    super().__init__(**kwargs)
  def call(self,inputs):
    z_mean,z_log_var=inputs
    batch=tf.shape(z_mean)[0]
    dim=tf.shape(z_mean)[1]
    epsilon=tf.keras.backend.random_normal(shape=(batch,dim))
    return z_mean+tf.exp(0.5*z_log_var)*epsilon

def build_encoder(image_size,latent_dim_1,latent_dim_2):
    encoder_inputs=keras.Input(shape=image_size)
    x=layers.Conv2D(32,3,activation='relu',strides=2,padding='same')(encoder_inputs)
    x=layers.Conv2D(64,3,activation='relu',strides=2,padding='same')(x)
    x=layers.Conv2D(128,3,activation='relu',strides=2,padding='same')(x)
    x=layers.Conv2D(256,3,activation='relu',strides=2,padding='same')(x)
    x=layers.Flatten()(x)
    x=layers.Dense(512,activation='relu')(x)
    z1_mean=layers.Dense(latent_dim_1,name='z1_mean')(x) # Apply the Dense layer to 'x'
    z1_log_var=layers.Dense(latent_dim_1,name='z1_log_var')(x) # Apply the Dense layer to 'x'
    z1=Sampling(name='z1_sampling')([z1_mean,z1_log_var])
    x_for_z2=layers.Dense(256,activation='relu')(x)
    z2_mean=layers.Dense(latent_dim_2,name='z2_mean')(x_for_z2)
    z2_log_var=layers.Dense(latent_dim_2,name='z2_log_var')(x_for_z2)
    z2=Sampling(name='z2_sampling')([z2_mean,z2_log_var])
    return keras.Model(encoder_inputs,[z1_mean,z1_log_var,z1,z2_mean,z2_log_var,z2],name='encoder')
    
def build_decoder(latent_dim_1,latent_dim_2):
    latent_inputs_z1=keras.Input(shape=(latent_dim_1,))
    latent_inputs_z2=keras.Input(shape=(latent_dim_2,))
    z_combined=layers.concatenate([latent_inputs_z1,latent_inputs_z2]) # Corrected concatenation
    decoder_hidden=layers.Dense(7*7*256,activation='relu')(z_combined)
    decoder_hidden=layers.Reshape((7,7,256))(decoder_hidden)
    x_decoded=layers.Conv2DTranspose(128,3,activation='relu',strides=2,padding='same')(decoder_hidden) # Adjusted filters and removed one layer
    x_decoded=layers.Conv2DTranspose(64,3,activation='relu',strides=2,padding='same')(x_decoded) # Adjusted filters and removed one layer
    decoder_outputs=layers.Conv2DTranspose(1,3,activation='sigmoid',padding='same')(x_decoded)
    return keras.Model([latent_inputs_z1,latent_inputs_z2],decoder_outputs,name='decoder')