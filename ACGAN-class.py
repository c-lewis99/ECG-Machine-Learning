import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import pandas as pd
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt
import numpy as np


# Setting Initial parameters
batch_size = 128
n_labels=5
latent_size = 125-n_labels
latent_dim = 1
   
# Importing X(ECGS) and Y(labels) data
Y_real = pd.read_csv('Y_10s_superclass.csv')
Y_real=np.array(Y_real)
Y_real=Y_real[:1000,:]
Y_unique=np.unique(Y_real, axis=0)

X_real = np.loadtxt('X_10s_1000.csv')
X_real = X_real.reshape(X_real.shape[0], 1000, 1)
   
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
    x = layers.ReLU()(x)
  
    x = layers.Conv1DTranspose(filters=64, kernel_size=16, strides=2, padding='same')(x)
    x = layers.ReLU()(x)

    
    x = layers.Conv1DTranspose(filters=32, kernel_size=16, strides=2, padding='same')(x)
    x = layers.ReLU()(x)
    
    # x = layers.Conv1DTrans(filters=16, kernel_size=16, strides=1, padding='same')(x)
    # x = layers.ReLU()(x)

    
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
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    label = layers.Dense(128)(x)
    label = layers.Dense(n_labels)(label)
    label = layers.Activation('sigmoid', name = 'label')(label)
    
    outputs = [outputs, label]
    discriminator = keras.Model(image, outputs, name = 'Discriminator')
    return discriminator





class ACGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(ACGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(ACGAN, self).compile(run_eagerly=True)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn


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
        
        
        real_data, real_labels = data
        fake_labels = np.eye(n_labels)[np.random.choice(n_labels, batch_size)]
    
        noise = np.random.uniform(-1.0,1.0,size=[batch_size, latent_size]) 
        combined_noise_label = tf.concat([noise, fake_labels],axis=1)
                    
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
             
            # Generating fake images using noise+labels
            generated_images = self.generator(combined_noise_label, training=True)
            
            # Real/fake images fed into discriminator 
            real_output, real_pred_labels = self.discriminator(real_data,training=True)
            fake_output, fake_pred_labels = self.discriminator(generated_images,training=True)
    
            # Losses all calculated using Binary CrossEntropy:
            
            # Validation loss on real images
            d_loss_real = self.loss_fn(tf.ones_like(real_output),real_output)
            # Validation loss on fake images
            d_loss_fake = self.loss_fn(tf.zeros_like(fake_output),fake_output)
            # Label loss on real+fake images 
            d_loss_label = self.loss_fn(real_labels,real_pred_labels) + self.loss_fn(fake_labels, fake_pred_labels)
            # Total discriminator loss
            disc_loss = d_loss_real + d_loss_fake + d_loss_label
            
            # Validation loss on fake images masquerading as real
            g_loss = self.loss_fn(tf.ones_like(fake_output),fake_output)
            # Label losses on these fake images 
            g_loss_label = self.loss_fn(fake_labels, fake_pred_labels)
            # Total generator loss
            gen_loss = g_loss+g_loss_label

            # Calculating gradients and updating weights using optimizers
            gen_grad = gen_tape.gradient(gen_loss,self.generator.trainable_variables)
            disc_grad = disc_tape.gradient(disc_loss,self.discriminator.trainable_variables)
            
            self.g_optimizer.apply_gradients(zip(gen_grad,self.generator.trainable_variables))
            self.d_optimizer.apply_gradients(zip(disc_grad,self.discriminator.trainable_variables))


        # Updating metrics
        self.gen_loss_tracker.update_state(gen_loss)
        self.disc_loss_tracker.update_state(disc_loss)

            
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
        if epoch%25==0:
            noise_plot = np.random.uniform(0,1.0,size=[1, latent_size])
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
            
            fig.savefig("Images/Images for ACGAN2/Image_{}".format(epoch))
            plt.close(fig)
            
# Initialising the callback
plotter = Save_plot()

# Defining the ACGAN class
ac_gan = ACGAN(
    discriminator=define_discriminator(), generator=define_generator(), latent_dim=latent_dim
)

ac_gan.compile(
    d_optimizer=tf.keras.optimizers.RMSprop(2e-4),
    g_optimizer=tf.keras.optimizers.RMSprop(2e-4),
    
    # Here we define the intermediate loss function used to calculate the 
    # ACGAN losses (note the from_logits = False since we have already 
    # used sigmoid in the discriminator)
    
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=False))

ac_gan.fit(dataset, epochs=1000, callbacks = [plotter])



