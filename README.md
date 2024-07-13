# Semi-Supervised Denoising Diffusion Model for All-in-One Adverse Weather Removal
![Framework image](images/framework.png)
## Introduction
This is the official repository for our recently submitted paper "Semi-Supervised Denoising Diffusion Model for All-in-One Adverse Weather Removal", where more implementation details are presented.

## Abstract
Adverse weather removal aims to restore clear vision in adverse weather conditions. Despite the recent remarkable progress, existing methods are mostly tailored for specific weather types and rely heavily on large amounts of pairwise labeled data. Unlike previous arts, in this paper, we make the first attempt to propose a semi-supervised learning framework named SemiDDM-Weather, ingeniously integrating an accelerated Denoising Diffusion Model (DDM) into a standard teacher-student network, which achieves consistent visually highquality all-in-one adverse weather removal with limited labeled data. Specifically, to improve the accuracy of pseudo-labels for
semi-supervised learning, we construct a reliable bank to store the more scientifically defined “best-ever” outputs from the teacher network. Additionally, to harness the demonstrated generative strengths of diffusion models for inverse imaging tasks and to accelerate model inference—making diffusion models more suitable for our task—we adopt the fast Wavelet Diffusion model as the backbone, which requires only four, rather than hundreds, of sampling steps. More importantly, owing to the introduction of wavelet in our model, perceptually clearer vision can be restored, even surpassing the Ground Truth (GT). Experimental results validate that our approach, the first semi-supervised all-inone framework, outperforms fully supervised ones in restoration capability for various adverse weather scenarios.

## Dependencies
- Ubuntu==18.04
- Pytorch==1.13.1
- CUDA==11.7

Other dependencies are listed in `requirements.txt`

## Usage

### 1. Prepare the dataset
We perform experiments for image deraining on [Snow100K](https://sites.google.com/view/yunfuliu/desnownet), combined image deraining and dehazing on [Outdoor-Rain](https://github.com/liruoteng/HeavyRainRemoval), and raindrop removal on
the [RainDrop](https://github.com/rui1996/DeRaindrop) datasets. 

### 2. Train the first stage
````bash
python train.py --num_channels 12 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 196 --num_epoch 500 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --ch_mult 1 2 2 2 --current_resolution 32 --attn_resolutions 16 --num_disc_layers 4 --rec_loss --net_type wavelet --use_pytorch_wavelet
````

### 3. Train the second stage
````bash
python train.py --num_channels 12 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --batch_size 64 --num_epoch 650 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --ch_mult 1 2 2 2 --current_resolution 32 --attn_resolutions 16 --num_disc_layers 4 --rec_loss --net_type wavelet --use_pytorch_wavelet
````

### 4. Test
````bash
python test.py --num_channels 12 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 2  --image_size 64 --current_resolution 32 --attn_resolutions 16 --net_type wavelet --use_pytorch_wavelet
````

## Acknowledgement
* The training code architecture is based on the [Semi-UIR](https://github.com/Huang-ShiRui/Semi-UIR) and [WaveDiff](https://github.com/VinAIResearch/WaveDiff)and thanks for their work.
* We also thank for the following repositories: [IQA-Pytorch](https://github.com/chaofengc/IQA-PyTorch)
* Thanks for their nice contribution.
