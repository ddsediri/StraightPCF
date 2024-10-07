# StraightPCF: Straight Point Cloud Filtering (CVPR 2024)
Official code implementation for the paper "StraightPCF: Straight Point Cloud Filtering".

To run, please install the required dependencies. The code has been tested on and NVIDIA RTX 3090 GPU with the following settings:

```
Python 3.9
Ubuntu 22.04
CUDA 11.8
PyTorch 2.0.1
PyTorch3D 0.7.4
PyG 2.3.1
```

# Installation requirements

Run the following pip and conda install commands to set up the environment:
```
conda create -n myenv python=3.9
conda activate myenv
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
conda install pyg -c pyg
pip install point-cloud-utils==0.29.6
pip install plyfile
pip install pandas
pip install tensorboard
pip install torchsummary
conda install pytorch-cluster -c pyg
```

# Data
Our data is the same as ``Score-Based Point Cloud Denoising`` by Shitong Luo and Wei Hu. Kudos to them for their excellent implementation and resources. Please check their GitHub repo [here](https://github.com/luost26/score-denoise). We will also make the data available as a zip file, for ease of use. Please download the code and place it within ```./data```.

# How to run

## Inference only
Please run the following commands to test on the PUNet and PCNet:
```
python test_straightpcf.py --niters=1 --seed_k=6 --seed_k_alpha=1 --dataset='PUNet' --resolution='10000_poisson' --noise='0.01';
python test_straightpcf.py --niters=1 --seed_k=6 --seed_k_alpha=1 --dataset='PUNet' --resolution='50000_poisson' --noise='0.01';
python test_straightpcf.py --niters=2 --seed_k=6 --seed_k_alpha=1 --dataset='PUNet' --resolution='10000_poisson' --noise='0.02';
python test_straightpcf.py --niters=2 --seed_k=6 --seed_k_alpha=1 --dataset='PUNet' --resolution='50000_poisson' --noise='0.02';
python test_straightpcf.py --niters=3 --seed_k=6 --seed_k_alpha=1 --dataset='PUNet' --resolution='10000_poisson' --noise='0.03';
python test_straightpcf.py --niters=3 --seed_k=6 --seed_k_alpha=1 --dataset='PUNet' --resolution='50000_poisson' --noise='0.03';
python test_straightpcf.py --niters=1 --seed_k=6 --seed_k_alpha=1 --dataset='PCNet' --resolution='10000_poisson' --noise='0.01';
python test_straightpcf.py --niters=1 --seed_k=6 --seed_k_alpha=1 --dataset='PCNet' --resolution='50000_poisson' --noise='0.01';
python test_straightpcf.py --niters=2 --seed_k=6 --seed_k_alpha=1 --dataset='PCNet' --resolution='10000_poisson' --noise='0.02';
python test_straightpcf.py --niters=2 --seed_k=6 --seed_k_alpha=1 --dataset='PCNet' --resolution='50000_poisson' --noise='0.02';
python test_straightpcf.py --niters=3 --seed_k=6 --seed_k_alpha=1 --dataset='PCNet' --resolution='10000_poisson' --noise='0.03';
python test_straightpcf.py --niters=3 --seed_k=6 --seed_k_alpha=1 --dataset='PCNet' --resolution='50000_poisson' --noise='0.03';
```

Please run the following commands to test on the Kinect data:
```
python test_straightpcf.py --niters=2 --tot_its=1 --seed_k=6 --seed_k_alpha=1 --dataset='Kinect_v1' --resolution='unknown_res' --noise='unknown_noise';
```

Please run the following commands to test on the RueMadame data:
```
python test_straightpcf.py --niters=2 --seed_k=6 --seed_k_alpha=10 --dataset='RueMadame' --resolution='unknown_res' --noise='unknown_noise';
```

You should get the results on the terminal. The evaluation code is within ```./utils/valuate.py```. The output from the network is stored at ```./data/results```.

## Train the network
Training the full network is a 3 step process:

First train a single velocitymodule using:
```
python train_vm.py --val_freq=5000 --train_cvm_network=False --feat_embedding_dim=256 --decoder_hidden_dim=64
```

Thereafter, using that checkpoint, train the coupled VM stack with the following:
```
python train_cvm.py --val_freq=5000 --train_cvm_network=True
```

Given the coupled VM stack checkpoint, the full network can be trained using:
```
python train_straightpcf.py --val_freq=2000 --train_cvm_network=True --feat_embedding_dim=128 --decoder_hidden_dim=64
```

The folder for each training run is placed within ```./logs``` and you can access the necessary checkpoints there. 

## Acknowledgement and citation
Our code is partially based on ``Score-Based Point Cloud Denoising`` by Shitong Luo and Wei Hu. Kudos to them for their excellent implementation and resources. Please check their GitHub repo [here](https://github.com/luost26/score-denoise).

If you find our paper interesting and our code useful, please cite our paper with the following BibTex citation:
```
@InProceedings{de_Silva_Edirimuni_2024_CVPR,
    author    = {de Silva Edirimuni, Dasith and Lu, Xuequan and Li, Gang and Wei, Lei and Robles-Kelly, Antonio and Li, Hongdong},
    title     = {StraightPCF: Straight Point Cloud Filtering},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {20721-20730}
}
```
Please also check cite our related work, which is the basis for part of the code implementation: 
```
@InProceedings{de_Silva_Edirimuni_2023_CVPR,
    author    = {de Silva Edirimuni, Dasith and Lu, Xuequan and Shao, Zhiwen and Li, Gang and Robles-Kelly, Antonio and He, Ying},
    title     = {IterativePFN: True Iterative Point Cloud Filtering},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {13530-13539}
}
```