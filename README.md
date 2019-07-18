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

![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20z): latent variable

![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20p%28x_i%29): probability distribution of the data

![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20p%28z%29): probability of the latent variable indicating the type of data we generate

![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20p%28x_i%7Cz%29): distribution of the generating data given latent variable. E.g. turning imagination into real animal 


In variational Inference we resort to some information theory concepts: 
1. **Information** associated to an event is quantified as: ![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20I%28x_i%29%20%3D%20-%5Clog%20p%28x_i%29)
2. **Entropy** is known as the average information, or the expectation of the information: ![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20H%28X%29%20%3D%20%5Csum_%7Bx_i%20%5Cin%20X%7D%20p%28x_i%29%20%5Clog%20p%28x_i%29%20%7E%5Ctext%7B%20or%20%7D%7E%20%5Cint_%7Bx_i%20%5Cin%20X%7D%20p%28x_i%29%20%5Clog%20p%28x_i%29%20%5Cmathrm%7Bd%7Dx_i)
3. **KL divergence** is a measure of dissimilarity of one distribution w.r.t. an other distribution: ![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20KL%28p%28X%29%7C%7Cq%28X%29%29%20%3D%20%5Csum_%7Bx_i%20%5Cin%20X%7D%20p%28x_i%29%20%5Clog%20%5Cfrac%7Bq%28x_i%29%7D%7Bp%28x_i%29%7D%20%7E%5Ctext%7B%20or%20%7D%7E%20%5Cint_%7Bx_i%20%5Cin%20X%7D%20p%28x_i%29%20%5Clog%20%5Cfrac%7Bq%28x_i%29%7D%7Bp%28x_i%29%7D%20%5Cmathrm%7Bd%7Dx_i)
Note that: ![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20KL%28%5Ccdot%7C%7C%5Ccdot%29%20%5Cge%200) and ![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20KL%28p%28X%29%7C%7Cq%28X%29%29%20%5Cne%20KL%28q%28X%29%7C%7Cp%28X%29%29)


In Variational Autoencoder we are interested in generating new data; thus we are interested in the posterior distribution ![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20p%28z%7Cx_i%29). 
Assuming to know the such posterior, we can infer our latent variable distribution ![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20p%28z%29) by marginalisation over our dataset. 
his make a lot of sense if we think about it: we want to make our latent variable likely under our data so to generate  plausible data. 

According to Bayesian theory: ![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20p%28z%7Cx_i%29%20%3D%20%5Cfrac%7Bp%28x_i%7Cz%29%20p%28z%29%7D%7Bp%28x_i%29%7D%20%3D%20%5Cfrac%7Bp%28x_i%2C%20z%29%7D%7Bp%28x_i%29%7D).
However, computing ![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20p%28x_i%29) is complicated, since: ![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20p%28x_i%29%20%3D%20%5Cint%20p%28x_i%7Cz%29%20p%28z%29%20%5Cmathrm%7Bd%7Dz) is a marginal distribution which is intractable. If z is high dimensional,  we have to marginalise on all latent variables: ![equation](https://latex.codecogs.com/gif.latex?p%28x_i%29%20%3D%20%5Cint%20...%5Cint%20%5Cint%20p%28x_i%7Cz%29%20p%28z%29%20%5Cmathrm%7Bd%7Dz).


In variational inference we **approximate ![equation](https://latex.codecogs.com/gif.latex?p%28z%7Cx_i%29) by an other distribution ![equation](https://latex.codecogs.com/gif.latex?q%28z%7Cx_i%29)**. 
The intuition is to give at ![equation](https://latex.codecogs.com/gif.latex?q%28z%7Cx_i%29) a tractable form (Gaussian exponential) and "play" with the parameters of 
 so to match ![equation](https://latex.codecogs.com/gif.latex?p%28z%7Cx_i%29). 

Formally, we can rewrite our goal as: ![equation](https://latex.codecogs.com/gif.latex?%5Cmin%20KL%28q%28z%7Cx_i%29%7C%7Cp%28z%7Cx_i%29%29%20%3D%20-%20%5Csum_%7Bx_i%20%5Cin%20X%7D%20q%28z%7Cx_i%29%20%5Clog%20%5Cfrac%7Bp%28z%7Cx_i%29%7D%7Bq%28z%7Cx_i%29%7D)

![equation](https://latex.codecogs.com/gif.latex?%3D%20-%20%5Csum_%7Bx_i%20%5Cin%20X%7D%20q%28z%7Cx_i%29%20%5Clog%20%5Cfrac%7Bp%28x_i%2Cz%29%7D%7Bq%28z%7Cx_i%29%7D%20%5Cfrac%7B1%7D%7Bp%28x_i%29%7D)

![equation](https://latex.codecogs.com/gif.latex?%3D%20-%20%5Csum_%7Bx_i%20%5Cin%20X%7D%20q%28z%7Cx_i%29%20%5CBig%28%20%5Clog%20%5Cfrac%7Bp%28x_i%2Cz%29%7D%7Bq%28z%7Cx_i%29%7D%20-%20%5Clog%20p%28x_i%29%20%5CBig%29)

![equation](https://latex.codecogs.com/gif.latex?%3D%20-%20%5Csum_%7Bx_i%20%5Cin%20X%7D%20q%28z%7Cx_i%29%20%5Clog%20%5Cfrac%7Bp%28x_i%2Cz%29%7D%7Bq%28z%7Cx_i%29%7D%20&plus;%20%5Csum_%7Bx_i%20%5Cin%20X%7D%20q%28z%7Cx_i%29%20%5Clog%20p%28x_i%29). 

Note that ![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20p%28x_i%29) has no relation with the latent variable z. Similarly, ![equation](https://latex.codecogs.com/gif.latex?%5Csum_%7Bx_i%20%5Cin%20X%7D%20q%28z%7Cx_i%29%20%3D%201) because ![equation](https://latex.codecogs.com/gif.latex?x_i) is given.

![equation](https://latex.codecogs.com/gif.latex?%3D%20%5Clog%20p%28x_i%29%20-%20%5Csum_%7Bz%7D%20q%28z%7Cx_i%29%20%5Clog%20%5Cfrac%7Bp%28x_i%2C%20z%29%7D%7Bq%28z%7Cx_i%29%7D)

![equation](https://latex.codecogs.com/gif.latex?%5Clog%20p%28x_i%29%20%3D%20KL%28q%28z%7Cx_i%29%7C%7Cp%28z%7Cx_i%29%29%20&plus;%20%5Csum_%7Bz%7D%20q%28z%7Cx_i%29%20%5Clog%20%5Cfrac%7Bp%28x_i%2C%20z%29%7D%7Bq%28z%7Cx_i%29%7D)

Note that ![equation](https://latex.codecogs.com/gif.latex?%5Clog%20p%28x_i%29) is a constant since ![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20x_i) is given. Also, note that ![equation](https://latex.codecogs.com/gif.latex?KL%28q%28z%7Cx_i%29%7C%7Cp%28z%7Cx_i%29%29) was the quantity I wanted to minimise. Thus, **minimising the KL divergence is equal to maximise ![equation](https://latex.codecogs.com/gif.latex?%5Csum_z%20q%28z%7Cx_i%29%20%5Clog%20%5Cfrac%7Bp%28x_i%2Cz%29%7D%7Bq%28z%7Cx_i%29%7D) (Variational lower bound) up to a constant value.** 
The advantage is the variational lower bound  is tractable.

![equation](https://latex.codecogs.com/gif.latex?%5Csum_z%20q%28z%7Cx_i%29%20%5Clog%20%5Cfrac%7Bp%28x_i%2Cz%29%7D%7Bq%28z%7Cx_i%29%7D%20%3D%20%5Csum_z%20q%28z%7Cx_i%29%20%5Clog%20%5Cfrac%7Bp%28x_i%7Cz%29%20p%28z%29%7D%7Bq%28z%7Cx_i%29%7D)

![equation](https://latex.codecogs.com/gif.latex?%3D%20%5Csum_z%20q%28z%7Cx_i%29%20%5CBig%28%20%5Clog%20p%28x_i%7Cz%29%20&plus;%20%5Clog%20%5Cfrac%7Bp%28z%29%7D%7Bq%28z%7Cx_i%29%7D%5CBig%29%20%3D%20%5Csum_z%20q%28z%7Cx_i%29%20%5Clog%20p%28x_i%7Cz%29%20&plus;%20%5Csum_z%20q%28z%7Cx_i%29%20%5Clog%20%5Cfrac%7Bp%28z%29%7D%7Bq%28z%7Cx_i%29%7D)

![equation](https://latex.codecogs.com/gif.latex?%3D%20%5Cmathbb%7BE%7D_%7Bz%20%5Csim%20q%28z%7Cx_i%29%7D%20%5Cbig%5B%20%5Clog%20p%28x_i%7Cz%29%20%5Cbig%5D%20-%20KL%20%5Cbig%28q%28z%7Cx_i%29%7C%7Cp%28z%7Cx_i%29%5Cbig%29)

In Variational Autoencoder we define ![equation](https://latex.codecogs.com/gif.latex?q%28z%7Cx_i%29) and ![equation](https://latex.codecogs.com/gif.latex?p%28z%7Cx_i%29) as neural networks; moreover, we assume ![equation](https://latex.codecogs.com/gif.latex?p%28z%29) to be a normal Gaussian distribution with unit variance and zero mean.

## Results
Based on the code, we have trained a CNN-based Variational Autoencoder on the MNIST dataset.
[Fig.1](loss) report the training loss, while Fig. 2 shows us some generated example.
As it is possible to see, there are still some artifact. Maybe a better activation function would provide better results.

<p align="center">
  <img src="figures/cnn/loss.png" width="600" title="loss">
  
  <a name=loss>Fig.1 - Training loss </a>
</p>

<p align="center">
  <img src="figures/cnn/cnn_variational_autoencoder_pred.png" width="650" title="new-examples">
  
  <a name=new-example>Fig.2 - Generated examples</a>
</p>
