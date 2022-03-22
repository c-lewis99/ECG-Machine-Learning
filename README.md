# ECG-Machine-Learning
Using CNNs and GANs to classify/generate ECG signals

'Preprocessing' involves preparing the data for inputting into CNNs. This involves deleting erroneous data, 
normalizing, removing the isolectric lines, finding the peaks of the ECGs and splitting the data (in the case of 1 beat classification). 

In 'CNN' the data is split into various classes and a number of CNNs are tested and compared. 

The other files involve the generation of synthetic ECGs using GANs (more specifically, WGAN-GP and ACGANs). This repo is a work in progress
and will be updated accordingly. 
