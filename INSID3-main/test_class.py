from collections import Counter
from datasets.isaid_new import DatasetISAIDNew

ds = DatasetISAIDNew(
    "/data/lky/data/rs_seg/iSAID",
    shot=1,
    num_test=2000,
    fold=1,
)
print(Counter(class_id for _, class_id in ds.img_metadata))