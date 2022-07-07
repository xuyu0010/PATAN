# Partial Adversarial Temporal Attentive Network (PATAN)
This repository is the demo code for the ICCV 2021 paper (oral) "[Partial Video Domain Adaptation with Partial Adversarial Temporal Attentive Network](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_Partial_Video_Domain_Adaptation_With_Partial_Adversarial_Temporal_Attentive_Network_ICCV_2021_paper.pdf)". This repository is built based on the [MFNet](https://github.com/cypw/PyTorch-MFNet) repository. We thank the authors of MFNet for their excellent work.

![alt text](./figures/figure-2-structure-3.png "Structure of PATAN")

## Prerequisites
This repository is built with PyTorch, with the following packages necessary for training and testing:
```
PyTorch (1.2.0 recommended)
opencv-python (pip)
easydl (pip)
```

## Project Detail and Dataset Download
Please visit our [project page](https://xuyu0010.github.io/pvda.html) to find out details of this work and to download the dataset.

## Training and Testing
To train the dataset, simply run:
```python
python train_da.py
```
Alternatively, you may test with the simple DANN with:
```python
python train_da.py --da-method DANN
```
To test the dataset, change directory to the '/test' folder and run:
```python
python evaluate_da.py
```

### Notes on training and testing
- The pretrained model where we start our training from is now uploaded to [Gdrive](https://drive.google.com/file/d/1DlBLrG-skHiwJkqD0wGrQkvXnN_dNXnN/view?usp=sharing).
- Notes on the '/exps' folder can be found in the README file in that folder.
- We provide a demo weight [here](https://drive.google.com/file/d/1rfBFAiBQjGDvFC7iJLw4oui_IlPJqW36/view?usp=sharing), you should locate it in the '/exps' folder.

__If you find our paper useful, please cite our paper:__
```
@inproceedings{xu2021partial,
  title={Partial video domain adaptation with partial adversarial temporal attentive network},
  author={Xu, Yuecong and Yang, Jianfei and Cao, Haozhi and Chen, Zhenghua and Li, Qi and Mao, Kezhi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9332--9341},
  year={2021}
}
```
