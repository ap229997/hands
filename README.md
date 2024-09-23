# Demo

This repo contains demo for hand pose estimation using [HaMeR](https://arxiv.org/abs/2312.05251) and [WildHands](https://ap229997.github.io/projects/hands/assets/paper.pdf) models. Note that hand crops are obtained from [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) predictions which are not always accurate and currently the code does not handle the erroneous cases.

## Installation
Clone the repo:
```
git clone -b demo --single-branch --recursive https://github.com/ap229997/hands.git
```

Setup conda environment:
```bash
conda create --name demo python=3.10
conda activate demo
```

Install dependencies:
```bash
conda install pytorch=1.13.1 torchvision=0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d

pip install -e .[all]
pip install -v -e third-party/ViTPose
pip install easydict
```

Download the trained models:
```bash
bash fetch_models.sh
```

Besides these files, you also need to download the MANO model. Please visit the [MANO website](https://mano.is.tue.mpg.de) and register to get access to the downloads section. HaMeR requires only the right hand model `MANO_RIGHT.pkl` (put it under the `downloads/_DATA/data/mano` folder). WildHands requires both `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` (put them under the `downloads/wildhands` folder).

Set the required paths:
```bash
export CACHE_DIR_HAMER=downloads/_DATA
export HAMER_MANO_DIR=downloads/_DATA/data
export WILDHANDS_MANO_DIR=downloads/wildhands
export INTRX_PATH=downloads/wildhands/intrx.pkl
```

## Usage

Some example images are provided in the `downloads/example_data` folder. The code also requires the camera focal length to get 3D predictions. The default value is set to 1000 which works for the provided example images.

HaMeR assumes a focal length of 5000 in its predictions. It needs to be changed to the focal length of the camera used to capture the images to get accurate 3D predictions.
```bash
CUDA_VISIBLE_DEVICES=0 python demo.py --img_folder downloads/example_data --out_folder out --hamer_ckpt downloads/_DATA/hamer_ckpts/checkpoints/hamer.ckpt
```

WildHands requires the focal length as input to the network. This model is trained on egocentric data only.
```
CUDA_VISIBLE_DEVICES=0 python demo.py --img_folder downloads/example_data --out_folder out --focal_length 1000 --wildhands_ckpt downloads/wildhands/wildhands.ckpt
```

## Acknowledgements
Check out these amazing repos as well which form the basis of this codebase:
- [HaMeR](https://github.com/geopavlakos/hamer)
- [ARCTIC](https://github.com/zc-alexfan/arctic)

## Citing
If you find this code useful, please consider citing:

```bibtex
@inproceedings{Prakash2024Hands,
    author = {Prakash, Aditya and Tu, Ruisen and Chang, Matthew and Gupta, Saurabh},
    title = {3D Hand Pose Estimation in Everyday Egocentric Images},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2024}
}

@inproceedings{pavlakos2024reconstructing,
    title={Reconstructing Hands in 3{D} with Transformers},
    author={Pavlakos, Georgios and Shan, Dandan and Radosavovic, Ilija and Kanazawa, Angjoo and Fouhey, David and Malik, Jitendra},
    booktitle={CVPR},
    year={2024}
}
```