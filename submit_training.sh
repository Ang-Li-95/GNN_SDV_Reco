#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=8             # This needs to match Trainer(num_nodes=...)
#SBATCH --ntasks-per-node=16   # This needs to match Trainer(devices=...)
#SBATCH --time=10-00:00:00
#SBATCH --qos=long
#SBATCH --mem-per-cpu=8G   # memory per cpu-core

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# run script from above
srun python train_GNN.py
#srun python testEmb.py
