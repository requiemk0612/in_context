#!/bin/bash

python run_experiment.py \
  --command manifest \
  --data-root /data/lky/data/rs_seg \
  --manifest manifests/isaid_fold0_mvp.jsonl \
  --fold 0 --shots 1 --num-episodes 50 \
  --window-crop 512 --window-stride 256 --seed 45