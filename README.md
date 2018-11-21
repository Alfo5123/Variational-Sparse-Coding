# Variational-Sparse-Coding

We aim to replicate the experiments described in the paper ["Variational Sparse Coding"](https://openreview.net/forum?id=SkeJ6iR9Km) from the ICLR 2019 submissions, as part of our participation in the  [ICLR Reproducibility Challenge 2019](https://reproducibility-challenge.github.io/iclr_2019/).


## Table of content
- [Description](#description)
- [Authors](#authors)
- [Results](#results)
- [Usage](#usage)
- [Observations](#observations)
- [References](#references)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Description 

We replicate the results of the recent paper *Variational Sparse Coding* and extend the results to new experiments.

## Authors

 - [Alfredo de la Fuente](https://alfo5123.github.io/)
 - [Robert Aduviri](https://github.com/Robert-Alonso)

## Results

**Playing with low-dimensional latent space variables**

<div align="center">
 <img src="results/images/latent8alpha001.gif" height="400px">
</div>

**Reconstruction results by modying encoding in 200-dimensional latent space**


<div align="center">
 <img src="results/images/latent200_alpha001.ex2.png" height="500px">
</div>

<div align="center">
 <img src="results/images/latent200_alpha001.ex4.png" height="500px">
</div>

**Varying latent space dimensionality**

MNIST               |  Fashion-MNIST               | 
:-------------------------:|:-------------------------:
![](results/images/latent_mnist_example.png)  |  ![](results/images/latent_fashion_example.png) 

## Usage

### Set up

>Requires Python 3.6 or higher.

The following lines will clone the repository and install all the required dependencies.

```
$ https://github.com/Alfo5123/Variational-Sparse-Coding.git
$ cd Variational-Sparse-Coding
$ pip install -r requirements.txt
```

### Datasets

In order to download datasets used in the paper experiments we use
```
$ python setup.py [datasets...]
```

with options `mnist`, `fashion` and `celeba`. For example, if case you want to replicate *all* the experiments in the paper, we must run the following line:

```
$ python setup.py mnist fashion celeba
```

It will download and store the datasets locally in the **data** folder. 

### Pretrained Models

Aiming to simplify the reproducibility research process, we store the checkpoints of the trained models in the following [link](https://drive.google.com/open?id=1rW02-rpQxAk9yLco8OTMFzFI28o-qOQI). In order to run the scripts & notebooks using pretrained models, you must download the checkpoints and put them in **models** within the **src** folder.

### Train Models 

```
$ cd src
$ python [model] [args] 
```

For example

```
$ python vae-mnist.py --dataset mnist --epochs 500 --report-interval 50 --lr 0.01 
```

```
$ python vsc-mnist.py --dataset fashion --epochs 1000 --report-interval 100 --lr 0.001 --alpha 0.01
```

To visualize training results in TensorBoard, we can use the following command from a new terminal within **src** folder. 

```
$ tensorboard --logdir='./logs' --port=6006
```


## Observations
- [ ] Typo ? Difference between formula 6 and 9 / 21 signs
- [ ] How gradually show we increase c? Linearly for 20k iterations?
- [ ] Equation 10 : Recognition function,  numerical inestability -ReLU( -Vout ) = 0 -> spike = 1 

## References

Papers:
- **[Variational Sparse Coding](https://openreview.net/pdf?id=SkeJ6iR9Km)**
- [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf)
- [Large-Scale Feature Learning With Spike-and-Slab Sparse Coding](https://arxiv.org/pdf/1206.6407.pdf)
- [Stick-Breaking Variational Autoencoders](https://arxiv.org/pdf/1605.06197.pdf)
- [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/pdf?id=Sy2fzU9gl)
- [Disentangling by Factorising](https://arxiv.org/pdf/1802.05983.pdf)
- [Neural Discrete Representation Learning](https://papers.nips.cc/paper/7210-neural-discrete-representation-learning.pdf)

## Acknowledgements 

Special thanks to [Emilien Dupont](https://github.com/EmilienDupont) for clarifying distinct aspects on variational autoencoders' implementation. 

## License
[MIT License](https://github.com/Alfo5123/Variational-Sparse-Coding/blob/master/LICENSE)

