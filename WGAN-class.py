import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import pandas as pd
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt
import numpy as np


# Setting Initial parameters
batch_size = 100
n_labels=5
latent_size = 125-n_labels
latent_dim = 1
   
# Importing X(ECGS) and Y(labels) data
Y_real = pd.read_csv('Y_10s_superclass.csv')
Y_real=np.array(Y_real)
Y_real=Y_real[:3000,:]
Y_unique=np.unique(Y_real, axis=0)


X_real = np.loadtxt('X_10s_all.csv')
X_real = X_real.reshape(int(X_real.shape[0]/1000),1000,12)[:3000,:,0].reshape(3000,1000,1)
   
# Converting these to form suitable for TF
dataset = tf.data.Dataset.from_tensor_slices((X_real,Y_real)).shuffle(buffer_size=1024).batch(batch_size)

# Lists to append Generator/Discriminator losses to 
gloss=[]
dloss=[]


# Basic Generator consisting of Convolutional Transpose layers
# Input shape (noise+label) = (125,1)
# Output shape = (1000, 1)

def define_generator():
    noise_label = tf.keras.Input(shape=((125,1,)))

    x = layers.Bidirectional(layers.LSTM(125, return_sequences=True))(noise_label)

    x = layers.Conv1DTranspose(filters=128, kernel_size=16, strides=2, padding='same')(x)
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)
  
    x = layers.Conv1DTranspose(filters=64, kernel_size=16, strides=2, padding='same')(x)
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)

    
    x = layers.Conv1DTranspose(filters=32, kernel_size=16, strides=2, padding='same')(x)
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv1DTranspose(filters=16, kernel_size=16, strides=1, padding='same')(x)
    x = layers.ReLU()(x)

    
    x = layers.Conv1D(filters=1, kernel_size=16, strides=1, padding='same', activation='sigmoid')(x)
    

    generator = keras.Model(noise_label , x, name = 'Generator')
    return generator


# The discriminator takes in an ECG and has 2 outputs: 
# predictions for validity and for label


def define_discriminator():
    
    image = tf.keras.Input(shape=((1000,1,)))
    
    x = layers.Conv1D(filters=32, kernel_size=16, strides=1, padding='same')(image)
    x = layers.LeakyReLU()(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Conv1D(filters=64, kernel_size=16, strides=1, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.MaxPool1D(pool_size=2)(x)

    x = layers.Conv1D(filters=128, kernel_size=16, strides=1, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Dropout(0.4)(x)

    x = layers.Conv1D(filters=256, kernel_size=16, strides=1, padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.MaxPool1D(pool_size=2)(x)

    x = layers.Flatten()(x)
    
    outputs = layers.Dense(1)(x)
    label = layers.Dense(64)(x)
    label = layers.Dense(n_labels)(label)
    label = layers.Activation('sigmoid', name = 'label')(label)
    
    outputs = [outputs, label]
    discriminator = keras.Model(image, outputs, name = 'Discriminator')
    return discriminator





class WGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile(run_eagerly=True)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        
    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)[0]

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
    


    def train_step(self, data):
        '''
        We override the models train step to implement a custom loss function.
        The ACGAN is trained as follows:
        1). A batch of real ECGs (images) and their labels are selected
        2). A random batch of labels are chosen, combined with noise and inputted
            into the generator to synthesize fake ECGs
        3). The real/fake images are fed through the discriminator and the various 
            losses are calculated. 
        4). The generator and discriminator weights are updated via GradientTape
            using the losses calculated from the batch. 
        '''
        
        
        real_images, real_labels = data
        rand_index=np.random.randint(Y_real.shape[0], size = (100))
        fake_labels = Y_real[rand_index,:]
    
        noise = np.random.uniform(-1,1.0,size=[batch_size, latent_size]) 
        combined_noise_label = tf.concat([noise, fake_labels],axis=1)

       # Note that we train the disc/gen equal amounts - can change this with a loop
                    
        with tf.GradientTape() as tape:
            # Generate fake images from the latent vector
            fake_images = self.generator(combined_noise_label, training=True)
            fake_images = K.cast(fake_images, 'float32')
            real_images = K.cast(real_images, 'float32')
            # Get the logits for the fake images
            fake_logits, fake_labels_pred = self.discriminator(fake_images, training=True)
            # Get the logits for the real images
            real_logits, real_labels_pred = self.discriminator(real_images, training=True)
    
            # Calculate the discriminator loss using the fake and real image logits
            d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
            # Calculate the gradient penalty
            gp = self.gradient_penalty(batch_size, real_images, fake_images)
            # Calculate the label losses
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            label_losses=bce(fake_labels, fake_labels_pred)+bce(real_labels, real_labels_pred)
            
            # Add the gradient penalty to the original discriminator loss
            d_loss = d_cost + 10*gp + label_losses
    
        # Get the gradients w.r.t the discriminator loss
        d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
        # Update the weights of the discriminator using the discriminator optimizer
        self.d_optimizer.apply_gradients(
            zip(d_gradient, self.discriminator.trainable_variables)
        )
        
        
        # We now train the generator 
        
        rand_index=np.random.randint(Y_real.shape[0], size = (100))
        fake_labels = Y_real[rand_index,:]
    
        noise = np.random.uniform(-1,1.0,size=[batch_size, latent_size]) 
        combined_noise_label = tf.concat([noise, fake_labels],axis=1)
        combined_noise_label = K.cast(combined_noise_label,"float32")
        
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            fake_images = self.generator(combined_noise_label, training=True)
            # Get the discriminator logits for fake images
            fake_logits, fake_labels_pred = self.discriminator(fake_images, training=True)
            # Calculate label loss
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            label_losses=bce(fake_labels, fake_labels_pred)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(fake_logits)+label_losses

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        
        
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
            
        return {
          "g_loss": self.gen_loss_tracker.result(),
          "d_loss": self.disc_loss_tracker.result()
        }




# Custom callback class to save images every 25 epochs 
class Save_plot(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        
        # Updating loss arrays for plotting
        if epoch%1==0:
            g,d = self.model.metrics
            gloss.append(g.result().numpy())
            dloss.append(d.result().numpy())
            
            
        # Here we choose a random real ECG, generate a synthetic one with the same
        # label and plot them both as well as the ACGAN losses.
        if epoch%100==0:
            noise_plot = np.random.uniform(-1,1.0,size=[1, latent_size])
            images = X_real[:,:,0]
            random_no=np.random.randint(images.shape[0])
            plot_ex = images[random_no,:]
            plot_labels = Y_real[random_no,:]
            
            combined_noise_label = tf.concat([noise_plot, plot_labels.reshape(1,5)],axis=1)
            x = self.model.generator.predict(combined_noise_label)
            
            real, label_pred = (self.model.discriminator.predict(x))
            label_prediction = [0 if x<0.5 else 1 for x in label_pred[0]]
            print('Label given: ', plot_labels)
            print('Prediction of label: ', label_prediction, real[0])
       
        
            c = np.array(range(1000))
            c = c.reshape(1000, 1)
            x = x.reshape(1000,1)
            plot_ex = plot_ex.reshape(1000,1)
            X = np.hstack((c, x))
            Y = np.hstack((c,plot_ex))
            fig, axs = plt.subplots(2, figsize=(17, 7))
            axs[0].plot(X[:,0], X[:,1], color = 'blue', lw=1)
            axs[0].plot(Y[:,0], Y[:,1], color = 'red', lw=1)
            axs[0].grid()
            axs[0].set_title('Epoch Number: {}, Label = {}'.format(epoch, plot_labels))
            axs[0].set_xlabel('Time interval')
            axs[0].set_ylabel('Normalised data value')

            
            axs[1].plot(gloss, color = 'g', label = 'Gen loss')
            axs[1].plot(dloss, color = 'c', label = 'Disc loss')
            axs[1].grid()
            axs[1].legend(loc="upper left")
            axs[1].set_xlabel('Epoch number')
            axs[1].set_ylabel('Loss')
            
            fig.savefig("Images/Images for AC-WGAN/Image_{}".format(epoch))
            plt.close(fig)

        if epoch%1000==0:
            self.model.generator.save('Saved Models/AC-WGAN/AC-WGAN-{}.h5'.format(epoch))
            
# Initialising the callback
plotter = Save_plot()


# Define the Wasserstein loss function for the discriminator,
# We will add the gradient penalty later to this loss function.
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss function for the generator.
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)



# Instantiate the WGAN model.
wgan = WGAN(
    discriminator=define_discriminator(),
    generator=define_generator(),
    latent_dim=120,
)

define_discriminator().summary()
define_generator().summary()

wgan.compile(
    d_optimizer=tf.keras.optimizers.RMSprop(2e-4),
    g_optimizer=tf.keras.optimizers.RMSprop(2e-4),
    d_loss_fn=discriminator_loss,
    g_loss_fn =generator_loss)

    
wgan.fit(dataset, epochs=3000, callbacks = [plotter])

wgan.generator.save('Saved Models/AC-WGAN/AC-WGAN-Final.h5')
