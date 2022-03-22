import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import pandas as pd
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, regularizers, constraints
import scipy
from scipy import signal
from scipy.signal import savgol_filter
import sklearn 
from sklearn import metrics

from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm


# Setting Initial parameters
batch_size = 50
latent_size = 125
latent_dim = 100
   
# Importing X(ECGS) and Y(labels) data
#Y_real = pd.read_csv('Y_10s_superclass.csv')
#Y_real=np.array(Y_real)
#Y_real=Y_real[:1000,:]
#Y_unique=np.unique(Y_real, axis=0)

def norm(x):
    return 2*(x-min(x))/(max(x)-min(x))-1


X_real_all = np.loadtxt('X_10s_NORM.csv')
X_real_all = X_real_all.reshape(X_real_all.shape[0],1000,1)
X_real = X_real_all[:1000,:,:]
#for i in range(1000):
   # X_real[i,:,0]=norm(X_real[i,:,0])
# X_real = X_real.reshape(int(X_real.shape[0]/1000),1000,12)[:3000,:,0].reshape(3000,1000,1)
   
# Converting these to form suitable for TF
dataset = tf.data.Dataset.from_tensor_slices(X_real).shuffle(buffer_size=1024).batch(batch_size)

# Lists to append Generator/Discriminator losses to 
gloss=[]
dloss=[]

f_sim=[]
r_sim=[]
fr_sim=[]

fake_beat_std=[]

fake_bpm = pd.DataFrame(columns=['bpm', 'beat_std'])
real_bpm = pd.DataFrame(columns=['bpm', 'beat_std'])

critic = keras.models.load_model("fid_critic.h5")
fid_min=100

def peakfinder(X):
    locations = []
    s=pd.Series(X)
    s=s.rolling(3).var()
    s=s**2
    maxval=s.max()
    locations_temp,_=scipy.signal.find_peaks(s, maxval/50, distance=30)

    for counter, value in enumerate(locations_temp):
        if value<10:
            arr=X[:value+1]
            locations_temp[counter]=np.argmax(arr)
        else:
            arr=X[value-10:value+10]
            locations_temp[counter]=value-(10-np.argmax(arr))
        locations_temp=np.array(locations_temp)
    locations.append(locations_temp)
    
    return locations[0]

def beat_metrics(X):
    peaks = peakfinder(X)
    diff = np.diff(peaks)
    if len(diff)==0:
        return [0,0]
    bpm = int(60/np.mean(diff)*100)
    std = np.std(diff) 
    return [bpm, std]

for i in range(100):
    real_bpm.loc[i]=beat_metrics(X_real[i,:,0])


def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid
    

class MinibatchDiscrimination(Layer):

    def __init__(self, nb_kernels, kernel_dim, init='glorot_uniform', weights=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None, input_dim=None, **kwargs):
        self.init = initializers.get(init)
        self.nb_kernels = nb_kernels
        self.kernel_dim = kernel_dim
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(MinibatchDiscrimination, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2

        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = self.add_weight(shape=(self.nb_kernels, input_dim, self.kernel_dim),
            initializer=self.init,
            name='kernel',
            regularizer=self.W_regularizer,
            trainable=True,
            constraint=self.W_constraint)

        # Set built to true.
        super(MinibatchDiscrimination, self).build(input_shape)

    def call(self, x, mask=None):
        activation = K.reshape(K.dot(x, self.W), (-1, self.nb_kernels, self.kernel_dim))
        diffs = K.expand_dims(activation, 3) - K.expand_dims(K.permute_dimensions(activation, [1, 2, 0]), 0)
        abs_diffs = K.sum(K.abs(diffs), axis=2)
        minibatch_features = K.sum(K.exp(-abs_diffs), axis=2)
        return K.concatenate([x, minibatch_features], 1)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], input_shape[1]+self.nb_kernels

    def get_config(self):
        config = {'nb_kernels': self.nb_kernels,
                  'kernel_dim': self.kernel_dim,
                  'init': self.init.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(MinibatchDiscrimination, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

# Basic Generator consisting of Convolutional Transpose layers
# Input shape (noise+label) = (125,1)
# Output shape = (1000, 1)


def define_generator():
    noise = tf.keras.Input(shape=((125,100,)))

    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(noise)

    x = layers.Conv1DTranspose(filters=128, kernel_size=40, strides=1, padding='same')(x)
    x = layers.LeakyReLU(0.1)(x)
    
    x = layers.Conv1DTranspose(filters=128, kernel_size=40, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.1)(x)
    
    x = layers.Conv1DTranspose(filters=128, kernel_size=40, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.1)(x)

    x = layers.Conv1DTranspose(filters=128, kernel_size=40, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.1)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    
    x = layers.Convolution1D(filters=1, kernel_size=40, strides=1, padding='same', activation='sigmoid')(x)

    
    generator = keras.Model(noise, x, name = 'Generator')
    return generator



# The discriminator takes in an ECG and has 2 outputs: 
# predictions for validity and for label


def define_discriminator():
    
    image = tf.keras.Input(shape=((1000,1,)))
    
    x = layers.Dense(200)(image)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv1D(filters=128, kernel_size=40, strides=1, padding='same')(x)
    x = layers.MaxPool1D(pool_size=2)(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv1D(filters=64, kernel_size=40, strides=1, padding='same')(x)
    x = layers.MaxPool1D(pool_size=2)(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv1D(filters=2, kernel_size=40, strides=1, padding='same')(x)
    x = layers.MaxPool1D(pool_size=2)(x)
    x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)

    #x = layers.Dense(64)(x)
    #x = layers.LeakyReLU()(x)
    #x = layers.Dropout(0.3)(x)
    
    z = MinibatchDiscrimination(nb_kernels=100, kernel_dim=40, name="mbd")(x)

    x = layers.Dense(128)(z)
    x = layers.LeakyReLU()(x)
    
    #x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(1)(x)
    
    discriminator = keras.Model(image, [x,z], name = 'Discriminator')
    return discriminator





class WGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        self.disc_loss_tracker_real = keras.metrics.Mean(name="discriminator_loss_real")
        self.disc_loss_tracker_fake = keras.metrics.Mean(name="discriminator_loss_fake")
        self.disc_loss_tracker_gp = keras.metrics.Mean(name="discriminator_loss_gp")
        

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker,self.disc_loss_tracker_real,self.disc_loss_tracker_fake,self.disc_loss_tracker_gp]

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
            pred,_ = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def fft (self, image):
        ft = np.fft.fft(image, axis=1, norm = 'ortho')
        ft = np.abs(ft[:,1:501,:])
        
        return tf.concat((image, ft), axis=1)

    def fft_norm(self, image):
        ft = np.fft.fft(image, axis=1, norm = 'ortho')
        ft = np.abs(ft[:,5:505])
        ft = savgol_filter(ft, 51, 3, axis=1)
        for i in range(tf.shape(ft)[0]):
            ft[i] = (ft[i]-min(ft[i]))/(max(ft[i])-min(ft[i]))
        return ft

    def get_fakes(self, n_samples):
        noise = np.random.uniform(-1,1.0,size=[n_samples, latent_size, latent_dim])
        fake = tf.reshape(self.generator(noise), (n_samples, 1000))
        return fake

    def mse(self, pred, true):
        n=pred.shape[0]
        se=0
        for i in range(n):
            for j in range(n):
                se+=sklearn.metrics.mean_squared_error(pred[i], true[j])
        return 100*se/n**2

    def similarity(self, n_samples, data):
        
        fake=self.get_fakes(n_samples)

        random_no=np.random.randint(X_real.shape[0], size=n_samples)
        real = X_real[random_no]

        fake_fft = self.fft_norm(fake)
        real_fft = self.fft_norm(real)

        if data=='fake':
            return self.mse(fake_fft,fake_fft)
        
        elif data=='real':
            return self.mse(real_fft,real_fft)

        else:
            return self.mse(fake_fft,real_fft)

    def fid(self, n_samples):
        fake = self.get_fakes(2*n_samples)
        random_no = np.random.randint(1000, X_real_all.shape[0], size=n_samples*2)
        real = X_real_all[random_no]
        
        wass_fake,fake = critic.predict(fake)
        wass_real,real = critic.predict(real)

        
        return [calculate_fid(fake[:n_samples], fake[n_samples:]), calculate_fid(real[:n_samples], real[n_samples:]), calculate_fid(real[:n_samples], fake[:n_samples])]

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

        for i in range(2):
        
            real_images = data

            noise = np.random.uniform(-1,1.0,size=[batch_size, latent_size, latent_dim]) 

           # Note that we train the disc/gen equal amounts - can change this with a loop
                        
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(noise, training=False)
                fake_images = K.cast(fake_images, 'float32')
                real_images = K.cast(real_images, 'float32')
                # Get the logits for the fake images
                fake_logits,_ = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits,_ = self.discriminator(real_images, training=True)
        
                # Calculate the discriminator loss using the fake and real image logits
                d_cost_real, d_cost_fake = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                d_cost = d_cost_fake-d_cost_real
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)*10

                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp 
            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )
        
        
        # We now train the generator 

        noise = np.random.uniform(-1,1.0,size=[batch_size, latent_size, latent_dim]) 
        
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            fake_images = self.generator(noise, training=True)
            # Get the discriminator logits for fake images
            fake_logits,_ = self.discriminator(fake_images, training=False)

            # Calculate the generator loss
            g_loss = self.g_loss_fn(fake_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        
        
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        self.disc_loss_tracker_real.update_state(d_cost_real)
        self.disc_loss_tracker_fake.update_state(d_cost_fake)
        self.disc_loss_tracker_gp.update_state(gp)
            
        return {
          "g_loss": self.gen_loss_tracker.result(),
          "d_loss": self.disc_loss_tracker.result(),
          "d_loss_real": self.disc_loss_tracker_real.result(),
          "d_loss_fake": self.disc_loss_tracker_fake.result(),
          "d_loss_gp": self.disc_loss_tracker_gp.result()
        }




# Custom callback class to save images every 25 epochs 
class Save_plot(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        global fid_min
        
        # Updating loss arrays for plotting
        if epoch%1==0:
            g,d,a,b,c = self.model.metrics
            gloss.append(g.result().numpy())
            dloss.append(d.result().numpy())
            
            
        # Here we choose a random real ECG, generate a synthetic one with the same
        # label and plot them both as well as the ACGAN losses.
        if epoch%10==0:

            #Similarity Metrics
            
            #Poincare
            fake_sim = self.model.similarity(100, data='fake')
            real_sim = self.model.similarity(100, data='real')
            fake_real_sim = self.model.similarity(100, data='real_fake')

            f_sim.append(fake_sim)
            r_sim.append(real_sim)
            fr_sim.append(fake_real_sim)

            epoch_list = np.linspace(0, epoch, int(epoch/10+1))

            fid_metrics = self.model.fid(100)
            print('FID: FF = {0:.4f}, RR = {1:.4f}, FR = {2:.4f}'.format(100*fid_metrics[0], 100*fid_metrics[1], 100*fid_metrics[2]))
            print('FID min = {0:.4f}'.format(fid_min))
            if 100*fid_metrics[2]<fid_min:
                fid_min = 100*fid_metrics[2]
                self.model.generator.save('Saved Models/WGAN-GP/WGAN-GP-BEST.h5')
                
                
                

            

            # Beat Metrics

            if epoch>20:

                fakes = self.model.get_fakes(100)
                for i in range(100):
                    fake_bpm.loc[i]=beat_metrics(fakes[i])
                    
                fake_beat_std.append(fake_bpm.mean(axis=0)[1])
                
                print('Current - Real BPM = {0:.3f}, Fake BPM = {1:.3f}, Real STD = {2:.3f}, Fake STD = {3:.3f}'.format(real_bpm.mean(axis=0)[0], fake_bpm.mean(axis=0)[0],
                                                                                                                        real_bpm.mean(axis=0)[1], fake_bpm.mean(axis=0)[1]))
                print('Average - Fake STD = {0:.3f}'.format(np.mean(fake_beat_std)))
                
                print('Current - Fake-Fake similarity = {0:.3f}, Real-Real similarity = {1:.3f}, Fake-Real similarity = {2:.3f}'.format(fake_sim,real_sim, fake_real_sim))
                print('Average - Fake-Fake similarity = {0:.3f}, Real-Real similarity = {1:.3f}, Fake-Real similarity = {2:.3f}'.format(np.mean(f_sim[3:]),np.mean(r_sim[3:]),np.mean(fr_sim[3:])))
                                                                                                                              
                

            
            noise_plot = np.random.uniform(-1,1.0,size=[2, latent_size, latent_dim])
            images = X_real
            random_no=np.random.randint(images.shape[0], size=2)
            plot_ex = images[random_no,:,:]
           # plot_labels = Y_real[random_no,:]
            
            x1 = self.model.generator.predict(noise_plot)
            loss, features = self.model.discriminator.predict(x1)

            lossr, features_r = self.model.discriminator.predict(plot_ex)

            fig, axs = plt.subplots(4, figsize=(20, 15))
            axs[0].plot(x1[0,:,0], color = 'red', lw=1)
            axs[0].plot(x1[1,:,0], color = 'blue', lw=1)
            axs[0].set_title('Epoch Number: {}'.format(epoch))
            axs[0].set_xlabel('Time interval')
            axs[0].set_ylabel('Normalised data value')

            axs[1].plot(features[0], color = 'green', lw=1)
            axs[1].plot(features_r[0], color = 'purple', lw=1)

            axs[2].plot(epoch_list, f_sim, color = 'red', lw=1, label='Fake')
            axs[2].plot(epoch_list, r_sim, color = 'blue', lw=1, label='Real')
            axs[2].plot(epoch_list, fr_sim, color = 'green', lw=1, label = 'Fake-real')
            axs[2].grid()
            axs[2].legend(loc="upper right")
            axs[2].set_ylabel('Similarity')
         
            axs[3].plot(gloss, color = 'g', label = 'Gen loss')
            axs[3].plot(dloss, color = 'c', label = 'Disc loss')
            axs[3].grid()
            axs[3].legend(loc="upper left")
            axs[3].set_ylabel('Loss')
            
            fig.savefig("Images/Images for WGAN-GP/Image_{}".format(epoch))
            plt.close(fig)

        if epoch%500==0 and epoch>10:
            self.model.generator.save('Saved Models/WGAN-GP/UWGAN-GP-{}.h5'.format(epoch))
            self.model.discriminator.save('Saved Models/WGAN-GP/Disc-UWGAN-GP-{}.h5'.format(epoch))
            
# Initialising the callback
plotter = Save_plot()


# Define the Wasserstein loss function for the discriminator,
# We will add the gradient penalty later to this loss function.
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return real_loss, fake_loss


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
    d_optimizer=keras.optimizers.Adam(
    learning_rate=0.0005, beta_1=0.5, beta_2=0.9),
    
    g_optimizer=keras.optimizers.Adam(
    learning_rate=0.0001, beta_1=0.5, beta_2=0.9),
    
    d_loss_fn=discriminator_loss,
    g_loss_fn =generator_loss)

    
wgan.fit(dataset, epochs=3005, callbacks = [plotter])

#wgan.generator.save('Saved Models/WGAN-GP/WGAN-GP-Final.h5')



