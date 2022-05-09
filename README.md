# TransFetch
This repo contains code accompaning the manuscript "Fine-Grained Address Segmentation for Attention-Based Variable-Degree Prefetching"
```
@inproceedings{10.1145/3528416.3530236,
author = {Zhang, Pengmiao and Srivastava, Ajitesh and Nori, Anant V. and Kannan, Rajgopal and Prasanna, Viktor K.},
title = {Fine-Grained Address Segmentation for Attention-Based Variable-Degree Prefetching},
year = {2022},
isbn = {9781450393386},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3528416.3530236},
doi = {10.1145/3528416.3530236},
booktitle = {Proceedings of the 19th ACM International Conference on Computing Frontiers},
pages = {103–112},
numpages = {10},
keywords = {address segmentation, prefetching, machine learning, attention},
location = {Turin, Italy},
series = {CF '22}
}
```

## Dependencies

- python: 3.x
- Pytorch: 0.4+
- NVIDIA GPU



## Simulator

The simulator is a modification of ChampSim (https://github.com/Quangmire/ChampSim).

## Traces

The traces can be found at [this link](https://utexas.box.com/s/2k54kp8zvrqdfaa8cdhfquvcxwh7yn85). Alternatively, the `download.sh` file can be used to download all of the files to avoid bulk download restrictions from Box. 

`cd ./ChampSim`

`./download.sh`

## Run Model Training

We provide a sample script to run the model training for one application. After downloaded the traces, simply do:

* cd to root directory

* `./run_train.sh`

The script will generate reports of training and a prefetching file will be generated under directory `./res/train` for simulation.

## Run Simulation

After generating the prefetching file,  simulation can be done through ChampSim through:

* `./run_sim.sh`

The simulation reports and evaluation results will be generated at directory `./res/sim`.



