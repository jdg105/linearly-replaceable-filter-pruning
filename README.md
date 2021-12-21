# linearly-replaceable-filter-pruning

Pytorch Implementation of the following paper:

"Linearly Replaceable Filters for Deep Network Channel Pruning", AAAI 2021

\


<img src="https://user-images.githubusercontent.com/38177577/101866129-612a0900-3bbb-11eb-8050-4e9203dee1b4.PNG"  width="630" height="350">

## Abstract

Convolutional neural networks (CNNs) have achieved remarkable results; 
however, despite the development of deep learning, 
practical user applications are fairly limited because heavy networks can be used solely with the latest hardware and software supports. 
Therefore, network pruning is gaining attention for general applications in various fields. 
This paper proposes a novel channel pruning method, Linearly Replaceable Filter (LRF), 
which suggests that a filter that can be approximated by the linear combination of other filters is replaceable. 
Moreover, an additional method called Weights Compensation is proposed to support the LRF method. 
This is a technique that effectively reduces the output difference caused by removing filters via direct weight modification. 
Through various experiments, we have confirmed that our method achieves state-of-the-art performance in several benchmarks. 
In particular, on ImageNet, LRF-60 reduces approximately 56% of FLOPs on ResNet-50 without top-5 accuracy drop. 
Further, through extensive analyses, we proved the effectiveness of our approaches.

## Training

For the pruning, run the following example code.

```
python main_pruning.py
```
## Citation


```
@inproceedings{joo2021linearly,
  title={Linearly Replaceable Filters for Deep Network Channel Pruning},
  author={Joo, Donggyu and Yi, Eojindl and Baek, Sunghyun and Kim, Junmo},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={9},
  pages={8021--8029},
  year={2021}
}
```
