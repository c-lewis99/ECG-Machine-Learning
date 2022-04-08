# ECG-Machine-Learning
Using CNNs and GANs to classify/generate ECG signals

'Preprocessing' involves preparing the data for inputting into CNNs. This involves deleting erroneous data, 
normalizing, removing the isolectric lines, finding the peaks of the ECGs and splitting the data (in the case of 1 beat classification). 

In 'CNN' the data is split into various classes and a number of CNNs are tested and compared. 

The other files involve the generation of synthetic ECGs using Bi-LSTM GANs (more specifically, WGAN-GP and ACGANs). 
The WGAN-GP contains a number of additional features such as Minibatch Discrimination (to protect against mode collapse) and 
a Frechet Inception Score (FID) critic to evaluate the quality of the synthesized data. 

The INFOGAN is currently our best performing model. It contains an implementation of Infogan - https://arxiv.org/pdf/1606.03657.pdf but using WGAN-GP loss function and Bi-LSTM generator. 

It should be noted that this repo is a work in progress and will be updated accordingly. 
