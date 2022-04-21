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
from sn import SpectralNormalization
from mb import MinibatchDiscrimination

# Setting Initial parameters
batch_size = 100
n_labels=4
latent_size = 125-n_labels
latent_dim = 1
'''
X_real_all = np.loadtxt('X_10s_1000.csv')
X_real = X_real_all.reshape(1000,1000,1)

'''
# Import data
X = np.loadtxt('Data 10/X_multiclass.csv')
Y = np.loadtxt('Data 10/Y_multiclass.csv')
n_ecgs = 13000
n_valid = X.shape[0]-13000

X_real = X[:13000].reshape(13000,1000,1)
Y_real = Y[:13000]

X_valid = X[13000:].reshape(n_valid,1000,1)
Y_valid = Y[13000:]

balanced = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
balanced_labels = np.tile(balanced, (25,1))

# Converting these to form suitable for TF
dataset = tf.data.Dataset.from_tensor_slices((X_real,Y_real)).shuffle(buffer_size=1024).batch(batch_size)

# Lists to append Generator/Discriminator losses to 
gloss=[]
dloss=[]

fake_beat_std=[]

fake_bpm = pd.DataFrame(columns=['bpm', 'beat_std'])
real_bpm = pd.DataFrame(columns=['bpm', 'beat_std'])

# Importing pretrained critic to calculated FID score
critic = keras.models.load_model("fid_critic.h5")
fid_min=1000
epoch_no=0

# Function to find the peaks of the ECGs
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

# Calculates BPM and stddev of intra-beat time
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
	return fid*10


# Adapted from https://gist.github.com/sthalles/507ce723226274db8097c24c5359d88a and https://arxiv.org/pdf/1805.08318.pdf
class SelfAttention(keras.Model):
    def __init__(self, n_filters):
        super(SelfAttention, self).__init__()

        self.f = layers.Convolution1D(filters=n_filters, kernel_size=1, strides=1, padding='same', name='f_x', activation=None)
        self.g = layers.Convolution1D(filters=n_filters, kernel_size=1, strides=1, padding='same', name='g_x', activation=None)
        self.h = layers.Convolution1D(filters=n_filters , kernel_size=1, strides=1, padding='same', name='h_x', activation=None)
        self.gamma = tf.Variable(0., trainable=True, name="gamma")

    def call(self, x):
        # Shape(x) = [Batch, Length, Channels], eg ECG input is [50, 1000, 1]
        
        f = self.f(x)
        g = self.g(x)
        h = self.h(x)

        s = tf.matmul(g,f, transpose_b=True) # [B,N,C] * [B, N, C] = [B, N, N]
        b = tf.nn.softmax(s, axis=-1)
        o = tf.matmul(b, h)
        y = self.gamma * tf.reshape(o, tf.shape(x)) + x

        return y
        
        


# Generator consisting of Convolutional Transpose layers
# Input shape (noise+label+codes) = (125,1)
# Output shape = (1000, 1)

   
def define_generator():
    noise = tf.keras.Input(shape=((125,latent_dim,)))

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(noise)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    x = layers.Conv1DTranspose(filters=128, kernel_size=10, strides=1, padding='same')(x)
    x = layers.LeakyReLU(0.1)(x)
    
    x = layers.Conv1DTranspose(filters=128, kernel_size=10, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.1)(x)
    
    x = layers.Conv1DTranspose(filters=128, kernel_size=10, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.1)(x)

    x = layers.Conv1DTranspose(filters=64, kernel_size=10, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.1)(x)

    x = layers.Convolution1D(filters=1, kernel_size=10, strides=1, padding='same', activation='sigmoid')(x)

    generator = keras.Model(noise, x, name = 'Generator')
    return generator

    
def define_discriminator():
    
    image = tf.keras.Input(shape=((1000,1,)))
    
   # x = layers.Dense(400)(image)
    #x = layers.LeakyReLU()(x)
    
    #x = SpectralNormalization(layers.Conv1D(filters=256, kernel_size=10, strides=2, padding='same'))(image)
    x = layers.Conv1D(filters=256, kernel_size=10, strides=2, padding='same')(image)
    x = layers.LeakyReLU(0.1)(x)
    
    #x = SpectralNormalization(layers.Conv1D(filters=128, kernel_size=10, strides=2, padding='same'))(x)
    x = layers.Conv1D(filters=128, kernel_size=10, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.1)(x)
    
    #x = SpectralNormalization(layers.Conv1D(filters=64, kernel_size=10, strides=2, padding='same'))(x)
    x = layers.Conv1D(filters=64, kernel_size=10, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.1)(x)
    
    #x = SpectralNormalization(layers.Conv1D(filters=8, kernel_size=10, strides=2, padding='same'))(x)
    x = layers.Conv1D(filters=8, kernel_size=10, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.1)(x)

    flat = layers.Flatten()(x)
    
    mbd = MinibatchDiscrimination(nb_kernels=100, kernel_dim=10, name="mbd")(flat)

    disc_output = layers.Dense(256)(mbd)
    disc_output = layers.LeakyReLU(0.1)(disc_output)
    disc_output = layers.Dense(1)(disc_output)
    
    outputs = [disc_output, mbd]
    discriminator = keras.Model(image, outputs, name = 'Discriminator')
    return discriminator

def define_classifier():
    
    image = tf.keras.Input(shape=((1000,1,)))
    
    x = layers.Conv1D(filters=256, kernel_size=10, strides=1, padding='same')(image)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.MaxPool1D(strides=1, pool_size=2)(x)

    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv1D(filters=128, kernel_size=10, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.MaxPool1D(strides=1, pool_size=2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv1D(filters=128, kernel_size=10, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.MaxPool1D(strides=1, pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(filters=128, kernel_size=10, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.MaxPool1D(strides=1, pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(filters=128, kernel_size=10, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.MaxPool1D(strides=1, pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(4, activation='softmax')(x)
    
    classifier = keras.Model(image, x , name = 'Discriminator')
    return classifier





class CARDIGAN(keras.Model):
    def __init__(self, discriminator, generator, classifier, latent_dim):
        super(CARDIGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.classifier = classifier
        
        self.latent_dim = latent_dim
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        self.disc_loss_tracker_real = keras.metrics.Mean(name="discriminator_loss_real")
        self.disc_loss_tracker_fake = keras.metrics.Mean(name="discriminator_loss_fake")
        self.disc_loss_tracker_gp = keras.metrics.Mean(name="discriminator_loss_gp")
        self.class_loss_tracker_label_real = keras.metrics.Mean(name="classifier_loss_label_real")
        self.class_loss_tracker_label_fake = keras.metrics.Mean(name="classifier_loss_label_fake")



    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker,self.disc_loss_tracker_real,self.disc_loss_tracker_fake,
                self.disc_loss_tracker_gp,self.class_loss_tracker_label_real, self.class_loss_tracker_label_fake]

    def compile(self, d_optimizer, g_optimizer, c_optimizer, d_loss_fn, g_loss_fn):
        super(CARDIGAN, self).compile(run_eagerly=True)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.c_optimizer = c_optimizer
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


    def get_fakes(self, n_samples):

        random_no=np.random.randint(Y_real.shape[0], size=n_samples)
        labels = Y_real[random_no,:]
        noise = np.random.uniform(-1,1.0,size=[n_samples, latent_size, latent_dim])
        noise = tf.concat([noise, labels.reshape(n_samples, n_labels, 1)],axis=1)
        fake = tf.reshape(self.generator(noise), (n_samples, 1000))
        return fake, labels


    def fid(self, n_samples):
        fake,_ = self.get_fakes(2*n_samples)
        random_no = np.random.randint(0, X_real.shape[0], size=n_samples*2)
        real = X_real[random_no]
        
        wass_fake,fake = critic.predict(fake)
        wass_real,real = critic.predict(real)

        
        return [calculate_fid(fake[:n_samples], fake[n_samples:]), calculate_fid(real[:n_samples], real[n_samples:]), calculate_fid(real[:n_samples], fake[:n_samples])]

    def classifier_test(self):
        pred = self.classifier.predict(X_valid)
        Y_valid_pred=np.where(pred>0.5, 1,0)
        acc = tf.keras.metrics.CategoricalAccuracy()
        acc.update_state(Y_valid, Y_valid_pred)
        return acc.result().numpy()
        

    def train_step(self, data):
        '''
        We override the models train step to implement a custom loss function.
        The CARDIGAN is trained as follows:
        1). A batch of real ECGs (images) and their labels are selected
        2). A random batch of labels are chosen, combined with noise and inputted
            into the generator to synthesize fake ECGs
        3). The real/fake images are fed through the discriminator and the various 
            losses are calculated. 
        4). The generator and discriminator weights are updated via GradientTape
            using the losses calculated from the batch. 
        '''
        
        
        real_data, real_labels = data
        batch_size = tf.shape(real_data)[0]
        random_no=np.random.randint(Y_real.shape[0], size=batch_size)
        fake_labels = balanced_labels
        labels = fake_labels.reshape(batch_size, n_labels, 1)
        labels = np.repeat(labels, latent_dim, axis=2)
        noise = np.random.uniform(-1,1.0,size=[batch_size, latent_size, latent_dim])
        combined_noise_label = tf.concat([noise, labels],axis=1)

       # Note that we train the disc/gen equal amounts - can change this with a loop
                    
        with tf.GradientTape() as tape:
            # Generate fake images from the latent vector
            fake_images = self.generator(combined_noise_label, training=False)
            fake_images = K.cast(fake_images, 'float32')
            real_images = K.cast(real_data, 'float32')
            # Get the logits for the fake images
            fake_logits, _ = self.discriminator(fake_images, training=True)
            # Get the logits for the real images
            real_logits, _ = self.discriminator(real_images, training=True)
    
            # Calculate the discriminator loss using the fake and real image logits
            d_loss_fake, d_loss_real = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
            d_cost = d_loss_fake-d_loss_real
            
            # Calculate the gradient penalty
            gp = 10*self.gradient_penalty(batch_size, real_images, fake_images)
            
            # Add the gradient penalty to the original discriminator loss
            d_loss = d_cost + gp 
    
        # Get the gradients w.r.t the discriminator loss
        d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
        # Update the weights of the discriminator using the discriminator optimizer
        self.d_optimizer.apply_gradients(
            zip(d_gradient, self.discriminator.trainable_variables)
        )
        
        
        # We now train the generator 
        
        random_no=np.random.randint(Y_real.shape[0], size=batch_size)
        fake_labels = balanced_labels
        labels = fake_labels.reshape(batch_size, n_labels, 1)
        labels = np.repeat(labels, latent_dim, axis=2)
        noise = np.random.uniform(-1,1.0,size=[batch_size, latent_size, latent_dim])
        combined_noise_label = tf.concat([noise, labels],axis=1)
        
        combined_noise_label = K.cast(combined_noise_label,"float32")
        
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            fake_images = self.generator(combined_noise_label, training=True)
            # Get the discriminator logits for fake images
            fake_logits, _ = self.discriminator(fake_images, training=False)
            # Calculate label loss
            fake_labels_pred = self.classifier(fake_images, training=False)
            cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
            label_losses_g=cce(fake_labels, fake_labels_pred)
            # Calculate the generator loss
            g = self.g_loss_fn(fake_logits)
            g_loss = g+label_losses_g

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        
        #Now train the classifer
        #I will update the formula for this alpha value!!!
        #But essentially we only want the classifier to learn from the fake data when it is of very good quality
        
        if self.class_loss_tracker_label_fake.result()<0.2:
            alpha = 0.2
        else:
            alpha=0
            
        with tf.GradientTape() as tape:
            
            fake_labels_pred = self.classifier(fake_images, training=True)
            real_labels_pred = self.classifier(real_images, training=True)
            cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
            label_loss_real=cce(real_labels, real_labels_pred)
            label_loss_fake=cce(fake_labels, fake_labels_pred)

            label_loss = label_loss_real+alpha*label_loss_fake

        # Get the gradients w.r.t the classifier loss
        class_gradient = tape.gradient(label_loss, self.classifier.trainable_variables)
        # Update the weights of the classifier using the optimizer
        self.c_optimizer.apply_gradients(
            zip(class_gradient, self.classifier.trainable_variables)
        )



        
        self.gen_loss_tracker.update_state(g)
        self.disc_loss_tracker.update_state(d_loss)
        self.disc_loss_tracker_real.update_state(d_loss_real)
        self.disc_loss_tracker_fake.update_state(d_loss_fake)
        self.disc_loss_tracker_gp.update_state(gp)
        self.class_loss_tracker_label_real.update_state(label_loss_real)
        self.class_loss_tracker_label_fake.update_state(label_loss_fake)
            
        return {
          "g_loss": self.gen_loss_tracker.result(),
          "d_loss": self.disc_loss_tracker.result(),
          "D(r-f)": self.disc_loss_tracker_real.result()-self.disc_loss_tracker_fake.result(),
          "d_loss_gp": self.disc_loss_tracker_gp.result(),
          "label_real": self.class_loss_tracker_label_real.result(),
          "label_fake": self.class_loss_tracker_label_fake.result()
        }



# Custom callback class to save images every 25 epochs 
class Save_plot(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        global fid_min
        global epoch_no
        
        # Updating loss arrays for plotting
        if epoch%1==0:
            losses = self.model.metrics
            gloss.append(losses[0].result().numpy())
            dloss.append(losses[1].result().numpy())
            epoch_no+=1
            
            
        # Here we choose a random real ECG, generate a synthetic one with the same
        # label and plot them both as well as the AC-WGAN losses.
        if epoch%10==0:


            #FID
            epoch_list = np.linspace(0, epoch, int(epoch/10+1))
            ff=[]
            rr=[]
            fr=[]
            
            for i in range(10):
                fid_metrics = self.model.fid(1000)
                ff.append(fid_metrics[0])
                rr.append(fid_metrics[1])
                fr.append(fid_metrics[2])
                
            ff_mean, rr_mean, fr_mean = np.mean(ff), np.mean(rr), np.mean(fr)
            ff_std, rr_std, fr_std = np.std(ff), np.std(rr), np.std(fr)

            #We save the generator with the lowest (FID(fake, real)-FID(fake,fake)/2): we want high quality + diverse ECGs, hence the second term
            if (fr_mean-ff_mean/2)<fid_min:
                fid_min = fr_mean-ff_mean/2
                self.model.generator.save('Saved Models/CARDIGAN/CARDIGAN-BALANCED-BEST-GEN.h5')
                #self.model.discriminator.save('Saved Models/AC-WGAN3/AC-WGAN3-BEST-DISC-NMB.h5')
                
            print('FID: FF = {0:.2f}({1:.2f}), RR = {2:.2f}({3:.2f}), FR = {4:.2f}({5:.2f})'.format(ff_mean, ff_std, rr_mean, rr_std, fr_mean, fr_std))
            
            print('FID min = {0:.2f}'.format(fid_min))
            


            #BPM
            fakes,labels = self.model.get_fakes(100)
            for i in range(100):
                fake_bpm.loc[i]=beat_metrics(fakes[i])
                
            fake_beat_std.append(fake_bpm.mean(axis=0)[1])
            
            print('Current - Real BPM = {0:.3f}, Fake BPM = {1:.3f}, Real STD = {2:.3f}, Fake STD = {3:.3f}'.format(real_bpm.mean(axis=0)[0], fake_bpm.mean(axis=0)[0],
                                                                                                                    real_bpm.mean(axis=0)[1], fake_bpm.mean(axis=0)[1]))


            #Classifier test

            accuracy = self.model.classifier_test()
            print('Classifier accuracy = {0:.4f}'.format(accuracy))
            
            #Label examples
        
            images = X_real[:,:,0]
            random_no = np.random.randint(X_real.shape[0], size=5)
            plot_ex = images[random_no,:]
            
            x = fakes[:5,:]
            _,features_r = self.model.discriminator.predict(plot_ex)
            
            _,z = self.model.discriminator.predict(x)
   
        

            fig, axs = plt.subplots(5, figsize=(25, 17))
            axs[0].plot(x[0], color = 'red', lw=1)
            axs[0].set_title('Epoch Number: {}'.format(epoch))
            axs[0].set_xlabel('Time interval')
            axs[0].set_ylabel('Normalised data value')

            axs[1].plot(z[0,:], color = 'red', lw=1)

            axs[2].plot(x[1], color = 'blue', lw=1)

            axs[3].plot(z[1,:], color = 'blue', lw=1)
         
            axs[4].plot(gloss, color = 'g', label = 'Gen loss')
            axs[4].plot(dloss, color = 'c', label = 'Disc loss')
            axs[4].grid()
            axs[4].legend(loc="upper left")
            axs[4].set_xlabel('Epoch number')
            axs[4].set_ylabel('Loss')
            
            fig.savefig("Images/Images for CARDIGAN/Image_{}".format(epoch))
            plt.close(fig)

        if epoch%200==0:
            self.model.generator.save('Saved Models/CARDIGAN/CARDIGAN-BALANCED-GEN-{}.h5'.format(epoch))
            self.model.classifier.save('Saved Models/CARDIGAN/CARDIGAN-BALANCED-CLASS-{}.h5'.format(epoch))
            
# Initialising the callback
plotter = Save_plot()


# Define the Wasserstein loss function for the discriminator,
# We will add the gradient penalty later to this loss function.
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return [fake_loss, real_loss]


# Define the loss function for the generator.
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)



# Instantiate the WGAN model.
cardigan = CARDIGAN(
    discriminator=define_discriminator(),
    generator=define_generator(),
    classifier = define_classifier(),
    latent_dim=1,
)

define_discriminator().summary()
define_generator().summary()
define_classifier().summary()

cardigan.compile(
    d_optimizer=keras.optimizers.Adam(
    learning_rate=0.0005, beta_1=0.5, beta_2=0.9, decay = 1e-5),
    
    g_optimizer=keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9, decay = 1e-5),

    c_optimizer=keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.5, beta_2=0.9, decay = 1e-5),
    
     
    d_loss_fn=discriminator_loss,
    g_loss_fn =generator_loss)

    
cardigan.fit(dataset, epochs=2005, callbacks = [plotter])

