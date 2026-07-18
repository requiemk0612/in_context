import sys
sys.path.insert(0, '.')
from collections import Counter
from datasets.isaid_new import DatasetISAIDNew

ds = DatasetISAIDNew(
    datapath='/data/lky/data/rs_seg/iSAID',
    shot=1,
    num_test=1000,
)

counter = Counter()
for i in range(len(ds)):
    sample = ds[i]
    class_id = sample['class_id'].item()
    counter[class_id] += 1

print("Total episodes:", len(ds))
print("Class distribution:")
for cid in sorted(counter.keys()):
    print(f"  class {cid:2d} ({ds.CATEGORIES[cid]:20s}): {counter[cid]:4d}")
print("Unique classes sampled:", len(counter))
print("Expected per class if uniform:", len(ds) // len(ds.class_ids))
