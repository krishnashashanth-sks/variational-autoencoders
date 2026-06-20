import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

class GCNLayer(layers.Layer):
  def __init__(self,output_dim,activation=None,**kwargs):
    super().__init__(**kwargs)
    self.output_dim=output_dim
    self.activation=keras.activations.get(activation)
  def build(self,input_shape):
    self.kernel=self.add_weight(
        shape=(input_shape[0][-1],self.output_dim),
        initializer='glorot_uniform',
        name='kernel'
    )
    super().build(input_shape)
  def call(self,inputs):
    node_features,adj_matrix_hat=inputs
    output=tf.matmul(adj_matrix_hat,tf.matmul(node_features,self.kernel))
    if self.activation is not None:
      output=self.activation(output)
    return output

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_encoder(num_nodes,node_features_dim,latent_dim):
    encoder_input_features=keras.Input(shape=(num_nodes,node_features_dim),name='node_features_input')
    encoder_input_adj_hat=keras.Input(shape=(num_nodes,num_nodes),name='adj_hat_input')
    hidden=GCNLayer(64,activation='relu',name='gcn_hidden_1')([encoder_input_features,encoder_input_adj_hat])
    mu=GCNLayer(latent_dim,activation=None,name='latent_mu')([hidden,encoder_input_adj_hat])
    log_var=GCNLayer(latent_dim,activation=None,name='latent_log_var')([hidden,encoder_input_adj_hat])
    z=Sampling(name='z_sampling')([mu,log_var])
    return keras.Model(
        [encoder_input_features,encoder_input_adj_hat],
        [mu,log_var,z],
        name='encoder'
    )

def build_decoder(num_nodes,latent_dim):
    decoder_input_z=keras.Input(shape=(num_nodes,latent_dim),name='latent_z_input')
    adj_recon = layers.Lambda(lambda x: tf.sigmoid(tf.matmul(x, tf.transpose(x, perm=[0, 2, 1]))), name='adj_reconstruction')(decoder_input_z)
    return keras.Model(decoder_input_z,adj_recon,name='decoder')