# CompSciCom
- Student Name: Ryan Ng
- Student #: 1001533860

# What is an Energy Based Model?
Energy Based Model (EBM) is a form of generative model which represents a probability distribution over the data by associating a scalar energy which measures the compatibility between the values of the variables. Small energy values represent very compatible configurations of the variables while large energy values represent incompatible configurations of the variables. In general, the density of an EBM over a single dependent variable $x$ is:

<img src="https://bit.ly/3fGyqGp" align="center" border="0" alt="p_\theta(x)= \dfrac{exp(-E_\theta(x))}{Z_\theta}" width="353" height="92" />

where $\theta$ are our parameters, $x$ is the input data, $E_\theta(x)$ is the energy function which maps each point to a scalar and $Z_\theta$ is the normalizing constant.

<img src="https://bit.ly/3dCAyMG" align="center" border="0" alt="Z_\theta = \int exp(-E_\theta(x)) dx" width="383" height="92" />

The benefit of this is that we can choose $E_\theta$ in whatever way we like, without any constraints. But the downside of this is that computing for $Z_\theta$ is often intractable.

# How do you train an EBM?
The standard method for training such a model is to differentiate the log likelihood and perform gradient ascent to maximize it. For an EBM we would be computing the following:
<img src="https://bit.ly/2Orcrs4" align="center" border="0" alt="\nabla_\theta \log p_\theta(x) = - \nabla_\theta E_\theta(x) - \nabla_\theta \log Z_\theta" width="586" height="39" />

However, as we mentioned before $Z_\theta$ is often intractable and therefore cannot easily compute the log likelihood. However we can rewrite $\nabla_\theta \log Z_\theta = \mathbb{E}_{x \sim p_\theta(x)}[-\nabla_\theta E_\theta(x)]$ (derivation is [here](https://arxiv.org/pdf/2101.03288.pdf)). Now $Z_\theta$ can be approximated using one sample Monte-Carlo.
<img src="https://bit.ly/3upN2Ou" align="center" border="0" alt="\nabla_\theta log Z_\theta \simeq -\nabla_\theta E_\theta(\tilde{x})" width="339" height="39" />

where $\tilde{x} \sim p_\theta(x)$. To draw the sample, we can use a Markov Chain Monte Carlo using Langevin Dynamics. The idea is to start from a random point and slowly move toward the direction with high probability by using the gradients of $E_\theta$. The algorithm is as follows: 
<img src="https://bit.ly/3cN7XFh" align="center" border="0" alt="\tilde{x}^{k} \leftarrow \tilde{x}^{k-1} + \eta\nabla_xE_\theta(\tilde{x}^{k-1}) + \omega" width="489" height="47" />

where $\omega \sim N(0, \sigma)$ and $\tilde{x}^0$ is typically a sample from a Uniform distribution. The idea of this algorithm is from a random starting point, we slowly move towards a point of higher probability using the gradients of $E_\theta$. Ideally you would want to run this step many times until $x^k$ converges but doing this is computational expensive and is usually limitted to $K$ steps where $K$ is a hyperparameter.

# Why use EBMs?
