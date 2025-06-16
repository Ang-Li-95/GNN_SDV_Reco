#!/usr/bin/env python
# coding: utf-8
import sys
import os
import yaml
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

from lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from Networks.GNN.InteractionNetwork import InteractionNetwork

# Load config file
with open('gnn_config.yaml') as c:
    cl = yaml.load(c, Loader=yaml.FullLoader)
    config = cl['gnn_learning_configs']

model = InteractionNetwork(config)

save_directory = os.path.join(config['output_dir'])
os.makedirs(save_directory, exist_ok=True)

# saves top-K checkpoints based on "val_loss" metric
checkpoint_callback = ModelCheckpoint(
    save_top_k=10,
    monitor="val_loss",
    mode="min",
    dirpath=save_directory,
    filename="model-{epoch:02d}-{val_loss:.5f}",
)

logger = CSVLogger(save_directory, name='training')
trainer = Trainer(
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=16,
    num_nodes=8,
    strategy="ddp",
    max_epochs=config["max_epochs"],
    logger=logger,
    callbacks=[checkpoint_callback],
    #detect_anomaly=True
)

trainer.fit(model)
print(checkpoint_callback.best_model_path)
print(checkpoint_callback.best_model_score)
trainer.save_checkpoint(os.path.join(save_directory, "model.ckpt"))

