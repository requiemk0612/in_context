#!/bin/bash

python run_experiment.py \
  --command run \
  --manifest manifests/isaid_fold0_mvp.jsonl \
  --output-dir outputs/isaid_fold0_factorial \
  --methods A0,A1,A2,A3,A4,A5,A6,A7 \
  --replays '' \
  --token-bank duplicate