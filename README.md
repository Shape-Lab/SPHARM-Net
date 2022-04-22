# SPHARM-Net: Spherical Harmonics-based Convolutional Neural Network

## Description
SPHARM-Net is a spherical harmonics-based convolutional neural network for vertex-wise inference (spherical data segmentation).

A rotation-equivariant convolutional filter can avoid rigid data augmentation, and rotational equivariance can be achieved in spectral convolution without a specific neighborhood definition. Nevertheless, the limited resources of a modern machine enable only band-limited spectral components that might lose geometric details.

SPHARM-Net seeks (1) a constrained spherical convolutional filter that supports an infinite set of spectral components and (2) an end-to-end framework without rigid data augmentation. The proposed filter encodes all the spectral components without the full harmonic expansion to capture geometric details. Thanks to rotational equivariance, the training time can be drastically reduced while improving segmentation accuracy. The proposed convolution is fully composed of matrix transformations, which offers efficient, fast spectral processing. Although SPHARM-Net was tested on brain data, it can be extended to any spherical data segmentation.

![](https://user-images.githubusercontent.com/9325798/158522422-9aa64f68-f1ab-4595-b5a6-79ed285b691a.png)
<p align="center"><img src="https://user-images.githubusercontent.com/9325798/149944654-0c7a9d32-b72a-4f44-9846-81155b7a81d4.png" width="50%"/></p>

## Package Dependency
### Minimum Version
- Python (3.7)
- PyTorch (1.1.0)
- NumPy (1.11.3)
- SciPy (1.2.1)
- Joblib (0.14.1)
- tqdm (any)
### Experiments in Publication
- Python (3.7)
- PyTorch (1.9.0)
- NumPy (1.19.2)
- SciPy (1.5.2)
- Joblib (1.0.1)
- tqdm (4.61.1)
## Step 0. Environment setup
```
git clone https://github.com/Shape-Lab/SPHARM-Net.git
```
The proposed convolution is independent of spherical tessellation. However, it tends to achieve better performance as the triangle area becomes uniform. We recommend an icosahedral mesh for SPHARM-Net. An icosahedral sphere at level 6 (40,962 vertices) can be obtained by the following command line:
```
mkdir sphere && wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=16JPochuVpaUqnumLW34uh7pRxUWrcEF_' -O ./sphere/ico6.vtk
```

Depending on the NVIDIA GPU architectures, the package version numbers vary. For the Ampere architecture, run:
```
conda create --name spharm-net python=3.7
conda activate spharm-net
conda install pytorch==1.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```
For the Turing or earlier architectures, run:
```
conda create --name spharm-net python=3.7
conda activate spharm-net
conda install pytorch==1.1.0 cudatoolkit=10.0 -c pytorch
```
To install the required packages, run:
```
pip install -e .
```

Two public parcellation datasets were used in this work (thank the authors for sharing their datasets!):

- [Mindboggle-101](https://mindboggle.info)
- [NAMIC](http://people.csail.mit.edu/ythomas/code_release/AnonBuckner39.tar.gz) (open "save as" if link click doesn't work)

## Step 1. Data preparation
We assume a standard [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/)'s directory structure with manual label files (`*.annot`).
```
$SUBJECTS_DIR
├── subj_1
│   ├── surf
│   │   ├── lh.curv
│   │   ├── lh.inflated.H
│   │   ├── lh.sulc
│   │   └── lh.sphere
│   └── label
│       └── lh.manual.annot
.
.
└── subj_N
    ├── surf
    │   ├── lh.curv
    │   ├── lh.inflated.H
    │   ├── lh.sulc
    │   └── lh.sphere
    └── label
        └── lh.manual.annot
```

For training, spherical data need to be re-tessellated by a reference sphere (e.g., `./sphere/ico6.vtk` in Step 0). The training samples follow a binary format (features: `double`, labels: `int16`). The naming convention should follow the predefined pattern: `subjectID.?h.aug0.*.dat` if the default loader is used. To generate re-tessellated mesh files, run the following command line:
```
python ./scripts/prep.py --data-dir $SUBJECTS_DIR --hemi lh --annot manual
```
By default, the training samples will be generated in `./dataset/` using `./sphere/ico6.vtk`, and `curv`, `sulc`, `inflated.H` are re-tessellated. Use `--in-ch` to include/exclude input channels. You can further accelerate the file creation with multi-threading (use `--threads`). The script can also generate the left and right hemispheres together (use `--hemi lh rh`). If registered spheres are preferred, use `--native-sphere` to specify the registered sphere (e.g., `sphere.reg`). Once generated, you will see something like:
```
./dataset
├── features
│   ├── subj_1.lh.aug0.curv.dat
│   ├── subj_1.lh.aug0.inflated.H.dat
│   ├── subj_1.lh.aug0.sulc.dat
.   .
.   .
│   ├── subj_N.lh.aug0.curv.dat
│   ├── subj_N.lh.aug0.inflated.H.dat
│   └── subj_N.lh.aug0.sulc.dat
├── labels
│   ├── subj_1.lh.aug0.label.dat
.   .
.   .
│   └── subj_N.lh.aug0.label.dat
└── label_csv
    ├── subj_1.lh.csv
    .
    .
    └── subj_N.lh.csv
```
> You may want to verify how the `annot` files are converted in `label_csv`.

## Step 2. Training
### Arguments
To train SPHARM-Net, you will need to supply flags properly. Here are some popular flags:
- `--sphere` : path to the sphere mesh file used for the re-tessellation in Step 0
- `--hemi` : hemisphere (`lh` or `rh` or `lh rh`)
- `--n-splits` : k-fold cross-validation (default: 5)
- `--fold` : cross-validation fold (`$fold`=1, ..., k)
- `--classes` : Labels of interest. It does not necessarily contain all the labels; the excluded label IDs become zero. Check out the generated label IDs in `label_csv` (see Step 1) . In our experiments, we used
	- Mindboggle-101: `0 2 3 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 34 35`
	- NAMIC: `0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34`
- `--ckpt-dir` : path to checkpoint (`pth`) at the best accuracy (default: `./logs/best_model_fold$fold.py`)
- `--resume` : path to checkpoint to resume (if paused in past training), e.g., `./logs/best_model_fold$fold.pth`.

See also more options in `./scripts/train.py`.

### Training
Overall, you can train the model using the following command line (Mindboggle-101 example):
```
fold=1
python ./scripts/train.py \
--data-dir ./dataset \
--classes 0 2 3 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 34 35 \
--fold $fold \
--hemi lh \
--sphere ./sphere/ico6.vtk
```
If successful, you will see
```
Loading data...
Loading model...
Down 1	| C:3 -> 128	| L:80
Down 2	| C:128 -> 256	| L:40
Down 3	| C:256 -> 512	| L:20
Bottom	| C:512 -> 512	| L:10
Up 1	| C:1024 -> 256	| L:20
Up 2	| C:512 -> 128	| L:40
Up 3	| C:256 -> 128	| L:80
Final	| C:128 -> 32	| L:80
Num of params 4311616
100%|████████████████████████████████████████████████████████████| 60/60 [00:07<00:00,  8.44it/s]
Saving checkpoint...
100%|████████████████████████████████████████████████████████████| 60/60 [00:06<00:00,  9.53it/s]
```
> The GPU memory consumption may vary depending on CUDA kernels.

## Step 3. Inference
FreeSurfer's `surf` folder is an input to the inference in SPHARM-Net. In this step, we use the saved model `./logs/best_model_fold$fold.pth` in Step 2. To label an unseen subject, run the following command line:
```
python ./scripts/parc.py --subj_dir $SUBJECTS_DIR/subj_1/surf --hemi lh
```
By default, the vertex-wise inference is written in `./output/subj_1.lh.label.txt` (label per vertex per line) using `lh.sphere` in the above example. Like Step 1, use `--native-sphere` to specify the registered sphere (e.g., `sphere.reg`) as needed. Also, use `--surface` to overlay the inference onto a mesh file. For example, adding `--surface white` to the above command line will generate `./output/subj_1.lh.label.vtk` that overlays `./output/subj_1.lh.label.txt` onto `$SUBJECTS_DIR/subj_1/surf/lh.white` (gray/white surface). You can then display it with visualization tools such as [ParaView](https://www.paraview.org/).

Since the proposed convolution learns harmonic coefficients, any spherical tessellation can be used for inference. In general, a reference sphere is not mandatory for inference, unlike spatial spherical convolution approaches. A re-tessellation approach can introduce sampling artifacts (jagged boundaries) while achieving fast inference with high memory efficiency. The figure below shows a slight difference along the parcellation boundaries with (left) and without (right) spherical re-tessellation (click to zoom-in the figure).

![](https://user-images.githubusercontent.com/9325798/159154720-d6b8ddc4-0abe-4702-96fe-782742ecf06a.png)

> Use `--sphere` to set a reference mesh for spherical re-tessellation.

## Citation
We work hard to share our code for better reproducible research. Please cite the following paper if you use (part of) our code in your research, including data preparation, neural networks, inference, utilities, etc.:
```
@article{Ha2022:SPHARMNet,
  title     = {SPHARM-Net: Spherical Harmonics-based Convolution for Cortical Parcellation},
  author    = {Ha, Seungbo and Lyu, Ilwoo},
  journal   = {IEEE Transactions on Medical Imaging},
  year      = {2022},
  publisher = {IEEE}
}
```
