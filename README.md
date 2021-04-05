# CompSciCom
- Student Name: Ryan Ng
- Student #: 1001533860

# What is an Energy Based Model?
Energy Based Model (EBM) is a form of generative model which represents a probability distribution over the data by associating a scalar energy which measures the compatibility between the values of the variables. Small energy values represent very compatible configurations of the variables while large energy values represent incompatible configurations of the variables. In general, the density of an EBM over a single dependent variable <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;x" title="x" /></a> is:

<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;p_\theta(x)=&space;\dfrac{exp(-E_\theta(x))}{Z_\theta}" target="_blank"><img          src="https://latex.codecogs.com/gif.latex?\inline&space;p_\theta(x)=&space;\dfrac{exp(-E_\theta(x))}{Z_\theta}" title="p_\theta(x)= \dfrac{exp(-E_\theta(x))}{Z_\theta}" /></a>
</p>

where <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\theta" title="\theta" /></a> are our parameters, <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;x" title="x" /></a> is the input data, <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;E_\theta(x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;E_\theta(x)" title="E_\theta(x)" /></a> is the energy function which maps each point to a scalar and <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Z_\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Z_\theta" title="Z_\theta" /></a> is the normalizing constant.

<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Z_\theta&space;=&space;\int&space;exp(-E_\theta(x))&space;dx" target="_blank"><img    src="https://latex.codecogs.com/gif.latex?\inline&space;Z_\theta&space;=&space;\int&space;exp(-E_\theta(x))&space;dx" title="Z_\theta = \int exp(-E_\theta(x)) dx" /></a>
</p>

The benefit of this is that we can choose <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;E_\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;E_\theta" title="E_\theta" /></a> in whatever way we like, without any constraints. But the downside of this is that computing for <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Z_\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Z_\theta" title="Z_\theta" /></a> is often intractable.

# How do you train an EBM?
The standard method for training such a model is to differentiate the log likelihood and perform gradient ascent to maximize it. For an EBM we would be computing the following:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\nabla_\theta&space;\log&space;p_\theta(x)&space;=&space;-&space;\nabla_\theta&space;E_\theta(x)&space;-&space;\nabla_\theta&space;\log&space;Z_\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\nabla_\theta&space;\log&space;p_\theta(x)&space;=&space;-&space;\nabla_\theta&space;E_\theta(x)&space;-&space;\nabla_\theta&space;\log&space;Z_\theta" title="\nabla_\theta \log p_\theta(x) = - \nabla_\theta E_\theta(x) - \nabla_\theta \log Z_\theta" /></a>
</p>

However, as we mentioned before <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Z_\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Z_\theta" title="Z_\theta" /></a> is often intractable and therefore cannot easily compute the log likelihood. However we can rewrite <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\nabla_\theta&space;\log&space;Z_\theta&space;=&space;\mathbb{E}_{x&space;\sim&space;p_\theta(x)}[-\nabla_\theta&space;E_\theta(x)]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\nabla_\theta&space;\log&space;Z_\theta&space;=&space;\mathbb{E}_{x&space;\sim&space;p_\theta(x)}[-\nabla_\theta&space;E_\theta(x)]" title="\nabla_\theta \log Z_\theta = \mathbb{E}_{x \sim p_\theta(x)}[-\nabla_\theta E_\theta(x)]" /></a> (derivation is [here](https://arxiv.org/pdf/2101.03288.pdf)). Now <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Z_\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Z_\theta" title="Z_\theta" /></a> can be approximated using one sample Monte-Carlo.

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\nabla_\theta&space;log&space;Z_\theta&space;\simeq&space;-\nabla_\theta&space;E_\theta(\tilde{x})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\nabla_\theta&space;log&space;Z_\theta&space;\simeq&space;-\nabla_\theta&space;E_\theta(\tilde{x})" title="\nabla_\theta log Z_\theta \simeq -\nabla_\theta E_\theta(\tilde{x})" /></a>
</p>

where <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\tilde{x}&space;\sim&space;p_\theta(x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\tilde{x}&space;\sim&space;p_\theta(x)" title="\tilde{x} \sim p_\theta(x)" /></a>. To draw the sample, we can use a Markov Chain Monte Carlo using Langevin Dynamics. The idea is to start from a random point and slowly move toward the direction with high probability by using the gradients of <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;E_\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;E_\theta" title="E_\theta" /></a>. The algorithm is as follows: 

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\tilde{x}^{k}&space;\leftarrow&space;\tilde{x}^{k-1}&space;&plus;&space;\eta\nabla_xE_\theta(\tilde{x}^{k-1})&space;&plus;&space;\omega" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\tilde{x}^{k}&space;\leftarrow&space;\tilde{x}^{k-1}&space;&plus;&space;\eta\nabla_xE_\theta(\tilde{x}^{k-1})&space;&plus;&space;\omega" title="\tilde{x}^{k} \leftarrow \tilde{x}^{k-1} + \eta\nabla_xE_\theta(\tilde{x}^{k-1}) + \omega" /></a>
</p>

where <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\omega&space;\sim&space;N(0,&space;\sigma)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\omega&space;\sim&space;N(0,&space;\sigma)" title="\omega \sim N(0, \sigma)" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\tilde{x}^0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\tilde{x}^0" title="\tilde{x}^0" /></a> is typically a sample from a Uniform distribution. The idea of this algorithm is from a random starting point, we slowly move towards a point of higher probability using the gradients of <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;E_\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;E_\theta" title="E_\theta" /></a>. Ideally you would want to run this step many times until <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x^k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;x^k" title="x^k" /></a> converges but doing this is computational expensive and is usually limitted to <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;K" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;K" title="K" /></a> steps where <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;K" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;K" title="K" /></a> is a hyperparameter.

There are other training methods for EBMs such as Score Matching and Noise Contrastive Estimation which can be found [here](https://arxiv.org/pdf/2101.03288.pdf).

# Why use EBMs?
As mentioned before, an EBM is a form  of generative model which can learn about the distribution of the dataset. If the EBM is trained correctly, it has the power to produce other datasets that are similar to the distribution of the original dataset. An example of its application in the real world is image generation, where the EBM is capable of generating high quality images and auto-complete corrupted images. EBM can also be used for the typical classification models where we need to find good compatibility between our variales (low energy values). The research paper "Your classifier is secretly an energy based model" ([link](https://arxiv.org/pdf/1912.03263.pdf)), introduces a joint energy model (JEM) as a reinterpretation of the standard discriminative classifier. In the paper they mention that the performance of generative models as a classifier is way below that of a dicriminative model and the benefit of the generative model is outweighed by its performance in this situation. But with the JEM (hybrid model), it was able to retain its strong performance of a discriminative model and have the benefits of a generative model. Hence there is huge amount of potential with EBM in the future!

# References
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html

https://arxiv.org/pdf/1912.03263.pdf

https://arxiv.org/pdf/2101.03288.pdf
