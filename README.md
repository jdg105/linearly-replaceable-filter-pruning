# linearly-replaceable-filter-pruning

Pytorch Implementation of the following paper:

"Linearly Replaceable Filters for Deep Network Channel Pruning", AAAI

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
