#A Hebbian Spiking Model of Working Memory

**Gabriel L. Debastiani¹**, **Laurent Dragoni²**, **Vitor V. Cuziol³**

¹ Departamento de Física, Universidade Federal do Rio Grande do Sul, Porto Alegre, Rio Grande do Sul, Brazil
² Laboratoire J.A. Dieudonné, Université Nice Côte d'Azur, Nice, Provence-Alpes-Côte d'Azur, France
³ Departamento de Computação e Matemática, Faculdade de Filosofia, Ciências e Letras de Ribeirão Preto/Universidade de São Paulo, Ribeirão Preto, São Paulo, Brazil

## Abstract
Our model will be a spiking neuron network modeling the neocortex, based on a recent paper (Fiebig and Lansner, 2017), with the main objective of testing the hypothesis that Hebbian short-term potentiation is a possible mechanism of the encoding of working memory (WM). To do this, we will feed the network with a simple data vector and see how the "memory" reactivates after stimulation with a Poisson generator. We are motivated by the current revision of the neural mechanisms behind working memory, since previous theories of persistent activity were disputed by recent results (Lundqvist et al., 2016). One of the hypothesis that has gained attention is the short-term potentiation (STP), which is a type of Hebbian synaptic plasticity (Gerstner et al., 2018). The model, to be built in the NEST simulator with the Python language, will have synapses using the BCPNN learning rule (Lansner et al., 1989) and the depression model of the Tsodyks-Markram formalism. We expect that the memory reactivation in the model will occur in discrete oscillatory bursts rather than in persistent activity, trying to reproduce the main result of the original paper.

##References
[1] Fiebig, F., Lansner, A. A Spiking Working Memory Model Based on Hebbian Short-Term Potentiation. Journal of Neuroscience 4 January 2017, 37 (1) 83-96; DOI: https://doi.org/10.1523/JNEUROSCI.1989-16.2016

[2] Lundqvist, M., Rose, J., et al. Gamma and Beta Bursts Underlie Working Memory. Neuron. 2016 Apr 6;90(1):152-164. doi: 10.1016/j.neuron.2016.02.028. Epub 2016 Mar 17.

[3] Gerstner, W., Kistler, W.M., et al. Neuronal Dynamics. Chapter 19: Synaptic Plasticity and Learning. Accessed in 19 January 2018. [Link](http://neuronaldynamics.epfl.ch/online/Ch19.html)

[4] Lansner, A., Ekeberg, Ö. A One-Layer Feedback Artificial Neural Network With A Bayesian Learning Rule. Int. J. Neur. Syst. 01, 77 (1989). https://doi.org/10.1142/S0129065789000499