#!/bin/bash

python run_experiment.py \
 --command run \
 --insid3-root /data2/cld/in_context/INSID3-main \
 --data-root /data/lky/data/rs_seg \
 --manifest manifests/isaid_fold0_mvp.jsonl \
 --output-dir outputs/test \
 --methods B0,B1,B2,B3,A0,A7 \
 --device cuda \
 --window-batch-size 1