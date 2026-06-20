from layers import *
from model import HVAE
import tensorflow as tf
import matplotlib.pyplot as plt

IMAGE_SIZE=(28,28,1)
LATENT_DIM_1=32
LATENT_DIM_2=16
BATCH_SIZE=128
EPOCHS=10

encoder=build_encoder(IMAGE_SIZE,LATENT_DIM_1,LATENT_DIM_2)

decoder=build_decoder(LATENT_DIM_1,LATENT_DIM_2)

# Instantiate and Compile the HVAE
hvae = HVAE(encoder, decoder)
hvae.compile(optimizer=keras.optimizers.Adam())

print("Generating new images from random latent vectors using HVAE...")

# Generate random latent vectors for z1 and z2
num_generated_images = 10
z_sample_z1 = tf.random.normal(shape=(num_generated_images, LATENT_DIM_1))
z_sample_z2 = tf.random.normal(shape=(num_generated_images, LATENT_DIM_2))

# Use the HVAE decoder to create images from these latent vectors
generated_images_hvae = hvae.decoder.predict([z_sample_z1, z_sample_z2])

# Visualize the generated images
plt.figure(figsize=(10, 2))
for i in range(num_generated_images):
    ax = plt.subplot(1, num_generated_images, i + 1)
    plt.imshow(generated_images_hvae[i].squeeze(), cmap='gray')
    plt.axis('off')
plt.show()

print("Inference complete. Generated HVAE images displayed above.")