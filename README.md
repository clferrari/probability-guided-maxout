## README

Official Pytorch implementation of the paper "Probability Guided Maxout", appeared at IEEE 25th International Conference on Pattern Recognition (ICPR) 2020.

![alt text](https://github.com/clferrari/probability-guided-maxout/blob/master/conf/method.png)

### Requirements

Tested with Python 3.8, PyTorch 1.6.

### Usage

To train the network with default parameters just run python train.py

If you want to train different architectures, the forward method needs to be modified. See models/resnet.py


### Citation

If you find the work useful, please cite us!

```
@inproceedings{ferrari2021probability,
  title={Probability Guided Maxout},
  author={Ferrari, Claudio and Berretti, Stefano and Del Bimbo, Alberto},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
  pages={6517--6523},
  year={2021},
  organization={IEEE}
}

```

### License

The software is provided under the MIT license (see LICENSE).


