

# A New Approach to 3D ICP Covariance Estimation [[paper](https://arxiv.org/pdf/1909.05722.pdf)]

## Overview


In mobile robotics, scan matching of point clouds using Iterative Closest Point (ICP) allows estimating   sensor displacements. It may prove important to assess the associated uncertainty about the obtained rigid transformation, especially for sensor fusion purposes. We propose a novel approach to 3D ICP covariance computation that accounts for all the sources of errors as listed in Censi's pioneering work, namely wrong convergence, underconstrained situations, and sensor noise. Our approach builds on two facts. First, ICP is not a standard sensor: owing to wrong convergence the concept of  ICP covariance _per se_ is actually meaningless, as the dispersion in the ICP outputs may largely depend  on the accuracy of the initialization, and is thus inherently related to the prior uncertainty on the displacement. We capture this using the unscented transform, which also reflects correlations between initial and final uncertainties. Then, assuming white sensor noise leads to overoptimism: ICP is biased, owing to e.g. calibration biases, which we account for. Our solution is tested on  publicly available real data ranging from structured to unstructured environments, where our algorithm predicts consistent results with actual uncertainty, and compares very favorably to previous methods. We finally demonstrate the benefits of our method for pose-graph localization, where our approach improves accuracy and robustness   of the   estimates.

## Code
This repo contains the code for reproducing the results of this [paper](https://arxiv.org/pdf/1909.05722.pdf). The code is based on Python and has been tested under Python 3.5 on a Ubuntu 16.04 machine. ICP algorithm is called throught our modified version of the libpointmatcher library.

 
### Installation & Prerequies

1.  Clone this repo
```
git clone https://github.com/CAOR-MINES-ParisTech/3d-icp-cov.git
mkdir data
mkdir results
```

2.  Install the following required Python packages, `matplotlib`, `numpy`, `scipy`, `alphashape`, e.g. with the pip command
```
pip install matplotlib numpy scipy alphashape
```
3. Clone our fork of the [`libpointmatcher`](https://github.com/CAOR-MINES-ParisTech/libpointmatcher)  library and build them the  library
```
cd 3d-icp-cov
git clone https://github.com/CAOR-MINES-ParisTech/libpointmatcher.git
cd libpointmatcher
mkdir build && cd build
cmake ..
make
sudo make install
cd ../..
```

4. Download the [_Challenging data sets for point cloud registration algorithms_](https://projects.asl.ethz.ch/datasets/doku.php?id=laserregistration:laserregistration). Extract the zip file of each sequence corresponding to the point clouds in base frame in the `data` folder 

### Get started
1. Modify paths and parameters in the class Param at the end of `python/utils.py`

2. Launch the main file
	```
	cd python
	python3 main.py
	```
	 Be patient, it will reproduce the results of the paper.


## Paper
The paper _A New Approach to 3D ICP Covariance Estimation for Mobile Robotics, M. Brossard, S. Bonnabel and A. Barrau. 2019_, relative to this repo is available at this [url](https://arxiv.org/pdf/1909.05722.pdf).


### Citation

If you use this code in your research, please cite:

```
@article{brossard2020anew,
  author = {Martin Brossard and Silv\`ere Bonnabel and Axel Barrau},
  title = {{A New Approach to 3D ICP Covariance Estimation}},
  year = {2020},
  journal={IEEE Robotics and Automation Letters},
  publisher={IEEE},
}
```

If you use the original [`libpointmatcher`](https://github.com/ethz-asl/libpointmatcher)  library  in your research, please cite the original paper:

```
@article{Pomerleau12comp,
	author = {Pomerleau, Fran{\c c}ois and Colas, Francis and Siegwart, Roland and Magnenat, St{\'e}phane},
	title = {{Comparing ICP Variants on Real-World Data Sets}},
	journal = {Autonomous Robots},
	year = {2013},
	volume = {34},
	number = {3},
	pages = {133--148},
	month = feb
}
```

### Authors
Martin Brossard^, Silvère Bonnabel^ and Axel Barrau°

^MINES ParisTech, PSL Research University, Centre for Robotics, 60 Boulevard Saint-Michel, 75006 Paris, France

°Safran Tech, Groupe Safran, Rue des Jeunes Bois-Châteaufort, 78772, Magny Les Hameaux Cedex, France

