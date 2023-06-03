# Ensemble-Distribution-Distilation
This repo contains some of code files used for my dissertation on Uncertainty Estimation in Medical Imaging {Dissertation.pdf}. The Dataset is a multi-class image dataset of chest X-rays. The main contribution of this porject was extending the application of ENsemble Distribution Distillation to a multiclass classification problem where numerous positive labels are possible for an individual exemplar.

This is done by training a single CNN to paramaterise a mixture of Beta Distributions, which required the derivation of a custum loss function.
The purpose of this is to preserve both the superior accuracy and uncertainty estimation provided by ensembling multiple CNN's while reducing the computioanl requirements by only using one model for inference.
## Dataset
CheXpert Dataset : https://stanfordmlgroup.github.io/competitions/chexpert/
{example_input.jpg}

## Code
This repo contains the code files to:
1) Train an ensenmble of CNNs with weights randomly initialised 
2) Train an ensemble of CNNs with weights initialised from MNIST task
3) Distil an ensemble into an a single model using Ensemble Distribution Distiltion