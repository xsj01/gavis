# GAVIS: Uncertainty-driven 3D Gaussian Splatting Active Mapping via Anisotropic Visibility Field

CVPR 2026

### [Website](https://gatech-rl2.github.io/GAVIS/) | [Paper](https://arxiv.org/abs/2605.30342)

[Shangjie Xue](https://xsj01.github.io/), [Jesse Dill](https://www.linkedin.com/in/jesse-n-dill/), [Dhruv Ahuja](https://www.linkedin.com/in/dhruvahuja8/), [Frank Dellaert](https://dellaert.github.io/), [Panagiotis Tsiotras](https://ae.gatech.edu/directory/person/panagiotis-tsiotras), [Danfei Xu](https://faculty.cc.gatech.edu/~danfei/)


## Install instructions
```bash
conda create -y -n gavis python=3.10
python -m pip install --upgrade pip
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

pip install ninja
conda install -y -c "nvidia/label/cuda-11.7.1" cuda-nvcc
conda install -y -c "nvidia/label/cuda-11.7.1" cuda-toolkit

python -m pip install --no-build-isolation git+https://github.com/YixunLiang/simple-knn.git
python -m pip install --no-build-isolation submodules/vanilla_3dgs_rasterizer
python -m pip install --no-build-isolation git+https://https://github.com/xsj01/gavis_rasterizer.git

conda install tqdm
pip install bpy==3.6.0 --extra-index-url https://download.blender.org/pypi/
pip install opencv-python
pip install bpycv
pip install gtsam
pip install trimesh
pip install omegaconf
pip install gym
pip install plyfile
pip install jaxtyping
pip install numpy==1.25.0
pip install open3d
pip install pytorch3d -f  https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu117_pyt201/download.html
pip install tensorboard
pip install zmq
python -m pip install numpy numpy-quaternion
pip install nerfacc
pip install einops
pip install gdown  # for downloading datasets (scripts/download_data.sh)

# for image generation
sudo snap install blender --classic
```

## Dataset download

```
bash scripts/download_data.sh
```

## Dev instructions for the rasterizer

Custom rasterizer source code is linked [here](https://github.com/xsj01/gavis_rasterizer).

Command to install:
```
pip install git+https://github.com/xsj01/gavis_rasterizer.git
```

## Run the example

To run the provided example script, from the repository root execute:

```bash
python scripts/example.py
```
