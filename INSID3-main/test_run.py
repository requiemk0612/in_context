
import sys
sys.path.insert(0, '.')
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from models import build_insid3
from datasets.isaid_new import DatasetISAIDNew
from utils.metrics import Evaluator

model = build_insid3(
    model_size='large',
    image_size=1024,
    mask_refiner='bilinear',
    device='cuda:1',
)
model.eval()

ds = DatasetISAIDNew(
    datapath='/data/lky/data/rs_seg/iSAID',
    shot=1,
    num_test=10000,
)
# 选择测试类别
target_class_id = 0

# 查找第一个属于该类别的样本索引
sample_idx = None
for i in range(len(ds)):
    if ds.img_metadata[i][1] == target_class_id:
        sample_idx = i
        break

if sample_idx is None:
    print(f"Error: Class {target_class_id} not found in dataset!")
    exit()

sample = ds[sample_idx]  # 使用找到的索引

class_id = sample['class_id'].item()
class_name = ds.CATEGORIES[class_id]

ref_img = sample['ref_imgs'][0]
ref_mask = sample['ref_masks'][0]
tgt_img = sample['tgt_img']
tgt_mask = sample['tgt_mask']

# 强制让目标等于参考
#tgt_img = ref_img
#tgt_mask = ref_mask

print("ref_mask dtype:", ref_mask.dtype)
print("ref_mask shape:", ref_mask.shape)
print("ref_mask unique values:", torch.unique(ref_mask))
print("ref_mask min/max:", ref_mask.min().item(), ref_mask.max().item())

model.set_reference(ref_img, ref_mask)
model.set_target(tgt_img)
pred_mask = model.segment()

# 计算 IoU：取 index 1（foreground）
area_inter, area_union = Evaluator.classify_prediction(pred_mask, tgt_mask > 0.5)
inter_fg = area_inter[1, 0].item()
union_fg = area_union[1, 0].item()
iou_fg = inter_fg / (union_fg + 1e-10)
print(f"class: {class_name} (id={class_id})")
print(f"foreground intersection: {inter_fg}, union: {union_fg}")
print(f"foreground IoU: {iou_fg:.4f}")

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

axes[0, 0].imshow(np.array(ref_img))
axes[0, 0].set_title('Reference Image')

axes[0, 1].imshow(np.array(ref_mask), cmap='gray', vmin=0, vmax=1)
axes[0, 1].set_title('Reference Mask')

axes[1, 0].imshow(np.array(tgt_img))
axes[1, 0].set_title('Target Image')

axes[1, 1].imshow(np.array(tgt_mask), cmap='gray', vmin=0, vmax=1)
axes[1, 1].set_title(f'GT Mask (sum={tgt_mask.sum():.0f})')

axes[1, 2].imshow(np.array(pred_mask.cpu()), cmap='gray', vmin=0, vmax=1)
axes[1, 2].set_title(f'Pred Mask (IoU={iou_fg:.3f})')

# 叠加图
overlay = np.array(tgt_img).copy().astype(float) / 255.0
red = np.zeros_like(overlay)
red[..., 0] = pred_mask.cpu().numpy() * 0.5
overlay = np.clip(overlay + red, 0, 1)
axes[0, 2].imshow(overlay)
axes[0, 2].set_title('Target + Pred overlay')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.savefig('debug_iou.png', dpi=150)
print('saved debug_iou.png')
print("area_inter shape:", area_inter.shape)
print("area_union shape:", area_union.shape)
