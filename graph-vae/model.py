from tensorflow import keras
import tensorflow as tf

class GraphVAE(keras.Model):
  def __init__(self,encoder,decoder,**kwargs):
    super().__init__(**kwargs)
    self.encoder=encoder
    self.decoder=decoder
    self.total_loss_tracker=keras.metrics.Mean(name='total_loss')
    self.reconstruction_loss_tracker=keras.metrics.Mean(name='reconstruction_loss')
    self.kl_loss_tracker=keras.metrics.Mean(name='kl_loss')
  @property
  def metrics(self):
    return [
        self.total_loss_tracker,
        self.reconstruction_loss_tracker,
        self.kl_loss_tracker
    ]
  def train_step(self,data):
    x_input, a_true, a_hat_input = data
    with tf.GradientTape() as tape:
      mu,log_var,z=self.encoder([tf.expand_dims(x_input,axis=0),tf.expand_dims(a_hat_input,axis=0)])
      a_recon=self.decoder(z)
      a_true_flat=tf.reshape(a_true,[-1])
      a_recon_flat=tf.reshape(a_recon,[-1])
      reconstruction_loss=tf.reduce_mean(
          keras.losses.binary_crossentropy(a_true_flat,a_recon_flat)
      )
      kl_loss=-0.5*tf.reduce_sum(1+log_var-tf.square(mu)-tf.exp(log_var),axis=1)
      kl_loss=tf.reduce_mean(kl_loss)
      total_loss=reconstruction_loss+kl_loss
    grads=tape.gradient(total_loss,self.trainable_weights)
    self.optimizer.apply_gradients(zip(grads,self.trainable_weights))
    self.total_loss_tracker.update_state(total_loss)
    self.reconstruction_loss_tracker.update_state(reconstruction_loss)
    self.kl_loss_tracker.update_state(kl_loss)
    return{
        'loss':self.total_loss_tracker.result(),
        'reconstruction_losss':self.reconstruction_loss_tracker.result(),
        'kl_loss':self.kl_loss_tracker.result()
    }
  def call(self,inputs):
    x_input, _, a_hat_input = inputs
    mu, log_var, z=self.encoder([tf.expand_dims(x_input,axis=0),tf.expand_dims(a_hat_input,axis=0)])
    a_recon=self.decoder(z)
    return a_recon