# VSCNet
## Overview
This project includes source codes and examples of VSCNet. 

## Environment
* Pytorch >= 1.10.2
* Python >= 3.6.5
* torchvision >= 0.2.1

## Training
We provide training codes for reproduction. Please refer to [option](./opts.py) to see more parameters.
### Pre-leaning hierarchical knowledge base
####
Training **FLA model** for associating visual-semantic pairs and creating **Visual-Semantic Knowledge base (VSK)**.

    python train_offline.py --result_path result/ --hierarchy_path [PATH_TO_SAVE_VSK] --art_epoch [EPOCH_OF_CLUSTER] --net_v [BACKBONE]
### Training Classification model
Training the classification using **CIM**.

    python train_oneline.py --result_path result/ --hierarchy_path [PATH_TO_SAVE_HIERARCHY] --art_epoch [EPOCH_OF_CLUSTER] --net_v [BACKBONE]
## Testing
    python test.py --hierarchy_path [PATH_TO_SAVE_HIERARCHY] --art_epoch [EPOCH_OF_CLUSTER] --net_v [BACKBONE]
