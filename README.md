# GNN Vertex Reconstruction

## Overview

The architechture is composed of two parts:
- Embedding network: takes tracks as input, output the distance between each pair of tracks in the latent space, graphs will be constructed using tracks as nodes and connections between two tracks as edges. The latent distances between tracks are used as features of edges.
- GNN: takes the graphs generated in the Embedding network and output the scores for each edge.

Overall, the network constructs one graph for each event, using tracks are nodes and connections between two tracks as edges. The embedding distance and the GNN score can be used to select the "good edges". Depending on whether a group of tracks are connected by good edges, clusters of tracks can de determined, which will be further used as the input of vertex reconstruction.

## Prepare the data
The GNN uses `.pt` files as input. The following script take the root files as input, split the events into `train` and `test` dataset, and convert the data into `.pt` format.

```
python make_data.py --input_dir /eos/vbc/experiments/cms/store/user/lian/CustomNanoAOD_v3_MLTraining_new/ --output /eos/vbc/group/cms/ang.li/GNN_VtxReco/EMBInput
```

## Set training parameters
The hyper-parameters are set by `yaml` config files. One can adjust the input/output path, training parameters in `emb_config.yaml` for the embedding network  and `gnn_config.yaml` for the GNN.

## Embedding network
- To train locally:
```
python train_EMB.py
```
- To train on slurm (the script below needs to be modified for different training process):
```
sbatch submit_training.sh
```
- After training, run inference to get the graph:
```
python Inference_EMB.py --input_dir /eos/vbc/group/cms/ang.li/GNN_VtxReco/EMBInput/train/ --output_dir /eos/vbc/group/cms/ang.li/GNN_VtxReco/EMBOutput_dcut0p1415/train --model EMB_0603_3/model-epoch=99-val_loss=0.00290.ckpt
```
or
```
python Inference_EMB_per_file.py --input_file /eos/vbc/group/cms/ang.li/MLInput_new/zjetstonunuht0200_2018/output_1-5.pt --output_dir ./  --model /users/ang.li/public/SoftDV/ML/GNN_SDV_Reco/EMB_out_0502_2/model-epoch=57-val_loss=0.00285.ckpt;
```
- Filter the graphs to remove invalid ones (this is to make all events valid for training GNN)
```
python filter_data.py --input_dir /eos/vbc/group/cms/ang.li/GNN_VtxReco/EMBOutput_dcut0p1415/train/ --output_dir /eos/vbc/group/cms/ang.li/GNN_VtxReco/EMBOutput_dcut0p1415_filtered/train/
```

## GNN 
- To train locally:
```
python train_GNN.py
```
- To train on slurm (the script below needs to be modified for different training process):
```
sbatch submit_training.sh
```
- Inference
```
python Inference_GNN_per_file.py --input_file /eos/vbc/group/cms/ang.li/MLEMB_output_new_2/stop_M1000_980_ct2_2018/out_NANOAODSIMoutput_0.pt --output_dir /eos/vbc/group/cms/ang.li/MLGNN_0517_1/stop_M1000_980_ct2_2018 --model /users/ang.li/public/SoftDV/ML/GNN_SDV_Reco/GNN_0517_1/model-epoch=121-val_loss=0.71373.ckpt;
```

## Export to ONNX
Currently the script was written as notebook: https://github.com/Ang-Li-95/GNN_SDV_Reco/blob/master/export_onnx.ipynb
