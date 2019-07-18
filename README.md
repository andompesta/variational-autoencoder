# variational-autoencoder
PyTorch Implementation of a Variational Autoencoder

Both the CNN and the normal feedforward autoencder are trained on MNIST dataset.
Note that, only a PyTorch implementation is provided. Moreover, the encoder produce the logit, which are then used by a Bernoulli distribution to sample black and white images.
If the problem is continuous, a Gaussian distribution is more appropriate.

##Theory
It is a generative model rooted in Bayesian inference theory and Variational inference. \
The idea is to generate a data-points from a given latent variable that encode the type of data we want to generate. 
For example, say, we want to generate an animal. First, we imagine the animal: it must have four legs, and it must be able to swim. 
Having those criteria, we could then actually generate the animal by sampling from the animal kingdom. 

Let use define some notation:
![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20x_i): data point
: latent variable
: probability distribution of the data   
: probability of the latent variable indicating the type of data we generate
: distribution of the generating data given latent variable. E.g. turning imagination into real animal 
