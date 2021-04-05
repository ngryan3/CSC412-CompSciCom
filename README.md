# CompSciCom
- Student Name: Ryan Ng
- Student #: 1001533860

# What is an Energy Based Model?
Energy Based Model (EBM) is a form of generative model which represents a probability distribution over the data by associating a scalar energy which measures the compatibility between the values of the variables. Small energy values represent very compatible configurations of the variables while large energy values represent incompatible configurations of the variables. In general, the density of an EBM over a single dependent variable $x$ is:

<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;p_\theta(x)=&space;\dfrac{exp(-E_\theta(x))}{Z_\theta}" target="_blank"><img          src="https://latex.codecogs.com/gif.latex?\inline&space;p_\theta(x)=&space;\dfrac{exp(-E_\theta(x))}{Z_\theta}" title="p_\theta(x)= \dfrac{exp(-E_\theta(x))}{Z_\theta}" /></a>
</p>

where $\theta$ are our parameters, $x$ is the input data, $E_\theta(x)$ is the energy function which maps each point to a scalar and $Z_\theta$ is the normalizing constant.

<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Z_\theta&space;=&space;\int&space;exp(-E_\theta(x))&space;dx" target="_blank"><img    src="https://latex.codecogs.com/gif.latex?\inline&space;Z_\theta&space;=&space;\int&space;exp(-E_\theta(x))&space;dx" title="Z_\theta = \int exp(-E_\theta(x)) dx" /></a>
</p>

The benefit of this is that we can choose $E_\theta$ in whatever way we like, without any constraints. But the downside of this is that computing for $Z_\theta$ is often intractable.

# How do you train an EBM?
The standard method for training such a model is to differentiate the log likelihood and perform gradient ascent to maximize it. For an EBM we would be computing the following:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\nabla_\theta&space;\log&space;p_\theta(x)&space;=&space;-&space;\nabla_\theta&space;E_\theta(x)&space;-&space;\nabla_\theta&space;\log&space;Z_\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\nabla_\theta&space;\log&space;p_\theta(x)&space;=&space;-&space;\nabla_\theta&space;E_\theta(x)&space;-&space;\nabla_\theta&space;\log&space;Z_\theta" title="\nabla_\theta \log p_\theta(x) = - \nabla_\theta E_\theta(x) - \nabla_\theta \log Z_\theta" /></a>
</p>

However, as we mentioned before $Z_\theta$ is often intractable and therefore cannot easily compute the log likelihood. However we can rewrite $\nabla_\theta \log Z_\theta = \mathbb{E}_{x \sim p_\theta(x)}[-\nabla_\theta E_\theta(x)]$ (derivation is [here](https://arxiv.org/pdf/2101.03288.pdf)). Now $Z_\theta$ can be approximated using one sample Monte-Carlo.

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\nabla_\theta&space;log&space;Z_\theta&space;\simeq&space;-\nabla_\theta&space;E_\theta(\tilde{x})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\nabla_\theta&space;log&space;Z_\theta&space;\simeq&space;-\nabla_\theta&space;E_\theta(\tilde{x})" title="\nabla_\theta log Z_\theta \simeq -\nabla_\theta E_\theta(\tilde{x})" /></a>
</p>

where $\tilde{x} \sim p_\theta(x)$. To draw the sample, we can use a Markov Chain Monte Carlo using Langevin Dynamics. The idea is to start from a random point and slowly move toward the direction with high probability by using the gradients of $E_\theta$. The algorithm is as follows: 

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\tilde{x}^{k}&space;\leftarrow&space;\tilde{x}^{k-1}&space;&plus;&space;\eta\nabla_xE_\theta(\tilde{x}^{k-1})&space;&plus;&space;\omega" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\tilde{x}^{k}&space;\leftarrow&space;\tilde{x}^{k-1}&space;&plus;&space;\eta\nabla_xE_\theta(\tilde{x}^{k-1})&space;&plus;&space;\omega" title="\tilde{x}^{k} \leftarrow \tilde{x}^{k-1} + \eta\nabla_xE_\theta(\tilde{x}^{k-1}) + \omega" /></a>
</p>

where $\omega \sim N(0, \sigma)$ and $\tilde{x}^0$ is typically a sample from a Uniform distribution. The idea of this algorithm is from a random starting point, we slowly move towards a point of higher probability using the gradients of $E_\theta$. Ideally you would want to run this step many times until $x^k$ converges but doing this is computational expensive and is usually limitted to $K$ steps where $K$ is a hyperparameter.

# Why use EBMs?
