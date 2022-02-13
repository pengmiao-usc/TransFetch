# TransFetch
This repo contains code accompaning the manuscript "Fine-Grained Address Segmentation for Attention-Based Variable-Degree Prefetching"

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



