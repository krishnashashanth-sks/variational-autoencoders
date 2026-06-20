from tensorflow import keras
import tensorflow as tf
import numpy as np

class HVAE(keras.Model):
  def __init__(self,encoder,decoder,image_size,**kwargs):
    super().__init__(**kwargs)
    self.encoder=encoder
    self.decoder=decoder
    self.image_size=image_size
    self.total_loss_tracker=keras.metrics.Mean(name='total_loss')
    self.reconstruction_loss_tracker=keras.metrics.Mean(name='reconstruction_loss')
    self.kl_loss_z1_tracker=keras.metrics.Mean(name='kl_loss_z1')
    self.kl_loss_z2_tracker=keras.metrics.Mean(name='kl_loss_z2')
  @property
  def metrics(self):
    return[
        self.total_loss_tracker,
        self.reconstruction_loss_tracker,
        self.kl_loss_z1_tracker,
        self.kl_loss_z2_tracker
    ]
  def call(self,inputs):
      z1_mean,z1_log_var,z1,z2_mean,z2_log_var,z2=self.encoder(inputs)
      reconstruction=self.decoder([z1,z2])
      return reconstruction
  def train_step(self,data):
    with tf.GradientTape() as tape:
      z1_mean,z1_log_var,z1,z2_mean,z2_log_var,z2=self.encoder(data)
      reconstruction=self.decoder([z1,z2])
      reconstruction_loss=tf.reduce_mean(
          keras.losses.binary_crossentropy(data,reconstruction),
      )*np.prod(self.image_size)
      kl_loss_z1=-0.5*(1+z1_log_var-tf.square(z1_mean)-tf.exp(z1_log_var)) # Added missing *
      kl_loss_z1=tf.reduce_mean(tf.reduce_sum(kl_loss_z1,axis=1))
      kl_loss_z2=-0.5*(1+z2_log_var-tf.square(z2_mean)-tf.exp(z2_log_var)) # Corrected z1_log_var to z2_log_var
      kl_loss_z2=tf.reduce_mean(tf.reduce_sum(kl_loss_z2,axis=1))
      total_loss=reconstruction_loss+kl_loss_z1+kl_loss_z2
    grads=tape.gradient(total_loss,self.trainable_weights) # Fixed typo
    self.optimizer.apply_gradients(zip(grads,self.trainable_weights))
    self.total_loss_tracker.update_state(total_loss)
    self.reconstruction_loss_tracker.update_state(reconstruction_loss) # Fixed typo
    self.kl_loss_z1_tracker.update_state(kl_loss_z1)
    self.kl_loss_z2_tracker.update_state(kl_loss_z2)
    return {
        "loss":self.total_loss_tracker.result(),
        "reconstruction_loss":self.reconstruction_loss_tracker.result(),
        "kl_loss_z1":self.kl_loss_z1_tracker.result(),
        "kl_loss_z2":self.kl_loss_z2_tracker.result()
    }