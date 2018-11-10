# Fine-grained-Recognition-Model-Compression

![build](https://travis-ci.org/miguelvr/dropblock.png?branch=master)


Implementation of the CVPR2018 [Learning a Discriminative Filter Bank within a CNN for Fine-grained Recognition, and Model Compression with Attention Transfer](https://arxiv.org/abs/1611.09932) based on MobileNet, and distill the knowledge with [Attention Trabsfer]( https://arxiv.org/abs/1612.03928), [KD](https://arxiv.org/pdf/1503.02531.pdf), and [FitNet](https://arxiv.org/pdf/1412.6550.pdf) for model compresssion in Tensorflow.

## Usage

In this work ,I use MobileNet(v1/v2) as the the basemodel,differing from the paper with VggNet.

* First, train the teacher networks, Let *FLAGS.mimic=False*
  >*python t_s.py*
  
* Then train student networks with attention trasnfer&KD&feature maps mimic, Let *FLAGS.mimic=True*
  >*python t_s.py*
 
## Reference
Learning a Discriminative Filter Bank within a CNN for Fine-grained Recognition

Paying More Attention To Attention:ImprovingG The Performence Of Convolutional NeuralL Networks Via ATttwntion Transfer



## TODO

- [ ] auto pruning student with KD
