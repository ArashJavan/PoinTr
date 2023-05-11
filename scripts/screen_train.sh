#!/usr/bin/env bash

set -x
GPUS=0
NAME="tree_adapointr_3"

ARGS="--config cfgs/treePC_models/AdaPoinTr.yaml
      --start_ckpts ckpts/AdaPoinTr_PCN.pth 
      --exp_name ${NAME}
      --num_workers 16
      --deterministic"
        
screen -S ${NAME} -L -Logfile ${NAME}.log -dm scripts/train.sh $GPUS $ARGS
