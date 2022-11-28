# FedAKD

### This repository provides scripts for Federated Learning via Augmented Knowledge Distillation (FedAKD) applied to two Human Activity Recognition (HAR) datsets: 

1) Human Activity Recognition using Smartphone [(HARS)][hars_dataset] 
2) Human Activity Recognition using fitness Band (HARB) (self-collected from Mi Band 4)
![federated learning][intro]


## Knowledge Distillation 
![Knowledge Distillation][KD]

Knowledge Distillation (KD) is a technique to transfer knowledge from a trained model to a to-be-trained model. Unlike standard Federated Learning (FL) algorithms (FedAvg) which communicate model-dependent data (gradients or weights), KD can be used in the context of Federated Learning (FL) to distill knowledge among heterogeneous clients by communicating soft labels calculated using an un-labeled shared dataset. 

> Knowledge Distillation-based Federated Learning enables clients to independenlty design their learning models.


## Augmented Knowledge Distillation 

We push KD one step further by using an augmentation algorithm based on a server-controlled permutation and mixup augmentation [1] to distill knowledge more efficiently. 


## Results 

We evaluate FedAKD on the two previously mentioned HAR datasets against a recent KD-based FL algorithm: FedMD [2]  


<table id="tabular">
<tbody>
<tr style="border-top: none !important; border-bottom: none !important;">
<td style="text-align: left; border-left-style: solid !important; border-left-width: 1px !important; border-right-style: solid !important; border-right-width: 1px !important; border-bottom: none !important; border-top: none !important; border-top-style: solid !important; border-top-width: 1px !important; border-bottom-style: solid !important; border-bottom-width: 1px !important; " colspan="6">Average accuracy gains of   Federated Learning experiments (%)</td>
</tr>
<tr style="border-top: none !important; border-bottom: none !important;">
<td style="text-align: left; border-left-style: solid !important; border-left-width: 1px !important; border-right-style: solid !important; border-right-width: 1px !important; border-bottom: none !important; border-top: none !important; border-bottom-style: solid !important; border-bottom-width: 1px !important; " colspan="2">Dataset</td>
<td style="text-align: left; border-left: none !important; border-right-style: solid !important; border-right-width: 1px !important; border-bottom: none !important; border-top: none !important; border-bottom-style: solid !important; border-bottom-width: 1px !important; " colspan="2">HARS</td>
<td style="text-align: left; border-left: none !important; border-right-style: solid !important; border-right-width: 1px !important; border-bottom: none !important; border-top: none !important; border-bottom-style: solid !important; border-bottom-width: 1px !important; " colspan="2">HARB</td>
</tr>
<tr style="border-top: none !important; border-bottom: none !important;">
<td style="text-align: left; border-left-style: solid !important; border-left-width: 1px !important; border-right-style: solid !important; border-right-width: 1px !important; border-bottom: none !important; border-top: none !important; border-bottom-style: solid !important; border-bottom-width: 1px !important; " colspan="2">Data distribution</td>
<td style="text-align: left; border-left: none !important; border-right-style: solid !important; border-right-width: 1px !important; border-bottom: none !important; border-top: none !important; border-bottom-style: solid !important; border-bottom-width: 1px !important; " colspan="1">i.i.d</td>
<td style="text-align: left; border-left: none !important; border-right-style: solid !important; border-right-width: 1px !important; border-bottom: none !important; border-top: none !important; border-bottom-style: solid !important; border-bottom-width: 1px !important; " colspan="1">Non-i.i.d</td>
<td style="text-align: left; border-left: none !important; border-right-style: solid !important; border-right-width: 1px !important; border-bottom: none !important; border-top: none !important; border-bottom-style: solid !important; border-bottom-width: 1px !important; " colspan="1">i.i.d</td>
<td style="text-align: left; border-right-style: solid !important; border-right-width: 1px !important; border-bottom-style: solid !important; border-bottom-width: 1px !important; border-top: none !important; width: auto; vertical-align: middle; ">Non-i.i.d</td>
</tr>
<tr style="border-top: none !important; border-bottom: none !important;">
<td style="text-align: left; border-left-style: solid !important; border-left-width: 1px !important; border-right-style: solid !important; border-right-width: 1px !important; border-bottom: none !important; border-top: none !important; width: auto; border-bottom-style: solid !important; border-bottom-width: 1px !important; border-bottom-style: solid !important; border-bottom-width: 1px !important; " colspan="1" rowspan="2">Method</td>
<td style="text-align: left; border-left: none !important; border-right-style: solid !important; border-right-width: 1px !important; border-bottom: none !important; border-top: none !important; border-bottom-style: solid !important; border-bottom-width: 1px !important; " colspan="1">FedMD</td>
<td style="text-align: left; border-left: none !important; border-right-style: solid !important; border-right-width: 1px !important; border-bottom: none !important; border-top: none !important; border-bottom-style: solid !important; border-bottom-width: 1px !important; " colspan="1">24.5</td>
<td style="text-align: left; border-left: none !important; border-right-style: solid !important; border-right-width: 1px !important; border-bottom: none !important; border-top: none !important; border-bottom-style: solid !important; border-bottom-width: 1px !important; " colspan="1">7.2</td>
<td style="text-align: left; border-left: none !important; border-right-style: solid !important; border-right-width: 1px !important; border-bottom: none !important; border-top: none !important; border-bottom-style: solid !important; border-bottom-width: 1px !important; " colspan="1">11.5</td>
<td style="text-align: left; border-right-style: solid !important; border-right-width: 1px !important; border-bottom-style: solid !important; border-bottom-width: 1px !important; border-top: none !important; width: auto; vertical-align: middle; ">-2.7</td>
</tr>
<tr style="border-top: none !important; border-bottom: none !important;">
<td style="text-align: left; border-left: none !important; border-right-style: solid !important; border-right-width: 1px !important; border-bottom: none !important; border-top: none !important; border-bottom-style: solid !important; border-bottom-width: 1px !important; " colspan="1">FedAKD   (ours)</td>
<td style="text-align: left; border-left: none !important; border-right-style: solid !important; border-right-width: 1px !important; border-bottom: none !important; border-top: none !important; border-bottom-style: solid !important; border-bottom-width: 1px !important; " colspan="1">25.4</td>
<td style="text-align: left; border-left: none !important; border-right-style: solid !important; border-right-width: 1px !important; border-bottom: none !important; border-top: none !important; border-bottom-style: solid !important; border-bottom-width: 1px !important; " colspan="1">27.5</td>
<td style="text-align: left; border-left: none !important; border-right-style: solid !important; border-right-width: 1px !important; border-bottom: none !important; border-top: none !important; border-bottom-style: solid !important; border-bottom-width: 1px !important; " colspan="1">12.7</td>
<td style="text-align: left; border-right-style: solid !important; border-right-width: 1px !important; border-bottom-style: solid !important; border-bottom-width: 1px !important; border-top: none !important; width: auto; vertical-align: middle; ">0.4</td>
</tr>
</tbody>
</table>





## References 

[1] Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2017). mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412.
[2] Li, D., & Wang, J. (2019). Fedmd: Heterogenous federated learning via model distillation. arXiv preprint arXiv:1910.03581.


[intro]: https://github.com/gadm21/FedAKD/blob/main/assets/intro.png
[KD]: https://github.com/gadm21/FedAKD/blob/main/assets/KD_overview.png
[hars_dataset]: https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones
