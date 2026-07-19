nohup python inference_segmentation.py   \
--exp-name xxxxx  \
--fold x  \
> output/xxxx.log 2>&1 &

#customize!

nohup python inference_segmentation.py   \
--exp-name sliding_isaid_gla_fold0  \
--fold 0 \
-sw  -kve  -dn  -pa \
> output/sliding_isaid_gla_fold0.txt 2>&1 &