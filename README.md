##Convolutional Variational Auto-encoder

This implementation of a CVAE is built upon the code from 
- https://github.com/y0ast/VAE-Torch and 
- https://github.com/soumith/dcgan.torch

and the great papers by Kingma and Welling [Stochastic Gradient VB and the Variational Auto-Encoder](http://arxiv.org/abs/1312.6114)
as well as Radford, Metz and Chintala [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434).

See the github [dcgan.torch project](https://github.com/soumith/dcgan.torch) for instructions on how to run it. If you have your data ready a simple call:
`DATA_ROOT=<data/folder> th main_cvae.lua`
should suffice.

The code is MIT licensed.

