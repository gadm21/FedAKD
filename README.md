# FedAKD
![federated learning][intro]

Federated Learning via Augmented knowledge distillation for Heterogenous Deep Human Activity Recognition systems

## This repository provides scripts for Federated Learning via Augmented Knowledge Distillation (FedAKD) applied to two Human Activity Recognition (HAR) datsets: 

1) Human Activity Recognition using Smartphone [(HARS)][hars_dataset] 
2) Human Activity Recognition using fitness Band (HARB) (self-collected from Mi Band 4)

## Knowledge Distillation 
![Knowledge Distillation][KD]

Knowledge Distillation (KD) is a technique to transfer knowledge from a trained model to a to-be-trained model. Unlike standard Federated Learning (FL) algorithms (FedAvg) which communicate model-dependent data (gradients or weights), KD can be used in the context of Federated Learning (FL) to distill knowledge among heterogeneous clients by communicating soft labels calculated using an un-labeled shared dataset. 

> Knowledge Distillation-based Federated Learning enables clients to independenlty design their learning models.


## Augmented Knowledge Distillation 

We push KD one step further by using an augmentation algorithm based on server-controlled permutation and mixup augmentation [1] to distill knowledge more efficiently. 












## References 

[1] Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2017). mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412.



[intro]: https://github.com/gadm21/FedAKD/assets/intro.png
[KD]: https://github.com/gadm21/FedAKD/assets/KD_overview
[hars_dataset]: https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones
