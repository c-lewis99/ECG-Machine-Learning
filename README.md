# ECG-Machine-Learning
Using CNNs and GANs to classify/generate ECG signals

'Preprocessing' involves preparing the data for inputting into CNNs. This involves deleting erroneous data, 
normalizing, removing the isolectric lines, finding the peaks of the ECGs and splitting the data (in the case of 1 beat classification). 

In 'CNN' the data is split into various classes and a number of CNNs are tested and compared. 

The other files involve the generation of synthetic ECGs using Bi-LSTM GANs (more specifically, WGAN-GP and ACGANs). 
The WGAN-GP contains a number of additional features such as Minibatch Discrimination (to protect against mode collapse) and 
a Frechet Inception Score (FID) critic to evaluate the quality of the synthesized data. 

The INFOGAN - It contains an implementation of Infogan - https://arxiv.org/pdf/1606.03657.pdf but using WGAN-GP loss function and Bi-LSTM generator. 

CARDIGAN (pardon the pun!) is a  novel way of simultaneously generating new ECGs and training an auxillary classifier at the same time. It consists of three networks (generator/discriminator/classifier). Initially the classifier only learns from the real data; then, when the generator is consistently producing coonvincing samples, a label loss function for the fake data is activated, and the classifier begins to learn from these new samples.

It should be noted that this repo is a work in progress and will be updated accordingly. 
