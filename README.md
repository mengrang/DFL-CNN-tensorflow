# Knowledge-Distillation-for-Fine-grained-Recognition



**Implementation of the CVPR2018 [Learning a Discriminative Filter Bank within a CNN for Fine-grained Recognition](https://arxiv.org/abs/1611.09932);
Model Compression with with [KD](https://arxiv.org/pdf/1503.02531),  [Attention Trabsfer]( https://arxiv.org/abs/1612.03928),   [Noisy teacher](https://arxiv.org/pdf/1610.09650), and [FitNet](https://arxiv.org/pdf/1412.6550), [Cheap Convolutions](https://arxiv.org/abs/1711.02613) for model compresssion in Tensorflow.**

## Requirements
First install tensorflow, then install other Python packages:
  >pip install -r requirements.txt
  
## Usage

In this work ,I use MobileNet(v1/v2) as the the basemodel,differing from the paper with VggNet.

* First, train the teacher networks, Let *FLAGS.mimic=False*
  >python dfb_train.py
  
* Then train student networks with attention trasnfer&KD&feature maps mimic, Let *FLAGS.mimic=True*
  >python dfb_train.py
 
## Reference

- [Learning a Discriminative Filter Bank within a CNN for Fine-grained Recognition](https://arxiv.org/abs/1611.09932)
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531)
- [Deep Model Compression: Distilling Knowledge from Noisy Teachers](https://arxiv.org/pdf/1610.09650)
- [Paying More Attention To Attention:ImprovingG The Performence Of Convolutional NeuralL Networks Via ATttwntion Transfer]( https://arxiv.org/abs/1612.03928)
- [Moonshine: Distilling with Cheap Convolutions](https://arxiv.org/abs/1711.02613)



## TODO

- [ ] auto pruning student with KD
