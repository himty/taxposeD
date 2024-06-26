# TAXPoseD
## Learning Distributional Demonstration Spaces for Task-Specific Cross-Pose Estimation

[[Paper](https://arxiv.org/abs/2405.04609)] [[Website](https://sites.google.com/view/tax-posed/home)]

[TAX-PoseD](https://sites.google.com/view/tax-posed), a method for learning relative placement prediction tasks, learns a spatially-grounded latent distribution over demonstrations without human annotations, using an architecture with geometric inductive biases.

<p align="center">
<img src="./doc/3rack_spin.gif" alt="drawing" height="250">
<img src="./doc/3rack_place.gif" alt="drawing" height="250">
</p>

## Repository structure

- `multimodal`- Stable latest branch
- `multimodal_dev`- Latest branch
- `multimodal_icra2024`- ICRA 2024 paper's model configurations **(you are here)**

## Installation 

To install dependencies like pytorch, ndf_robot and other libraries, please follow the instructions in the [TAX-Pose Github repo](https://github.com/r-pad/taxpose/tree/main?tab=readme-ov-file#installation). 

Then, install this repository with:

```
pip install -e .
```

## Datasets

In our paper, we use the same 1-rack NDF training dataset as TAX-Pose, as described [here](https://github.com/r-pad/taxpose/tree/main?tab=readme-ov-file#download-the-data).

We have also experimented with environments from the [RPDiff](https://github.com/anthonysimeonov/rpdiff?tab=readme-ov-file#download-assets) paper.


## Training a TAX-PoseD model

To train a 2-rack mug-hanging model:

```
python train_residual_flow_multimodal.py --config-path=../configs/icra2024 --config-name=train_2rackvariety_densez_learned_prior dataset_root=TODO test_dataset_root=TODO log_dir=TODO rpdiff_descriptions_path=TODO
```

Other ICRA submission model configurations can be found in `configs/icra2024/*`


## Cite

```
@article{wang2024taxposed,
  title={Learning Distributional Demonstration Spaces for Task-Specific Cross-Pose Estimation},
  author={Wang, Jenny and Donca, Octavian and Held, David},
  journal={IEEE International Conference on Robotics and Automation (ICRA), 2024},
  year={2024}
}
```
