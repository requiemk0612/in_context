# 修复后推荐运行顺序（v2）

旧版 A0–A7 结果使用了一个实现偏差：`dn_cutoff` 在默认
`token-bank=duplicate` 下没有真正 mask attention logits。v2 已修复，因此
**不要对旧输出目录使用 `RESUME=1`**。

先验证 reference imbalance、forward saturation 和 KVE+DN 退化：

```bash
# 1. 本地接口测试
bash /data2/cld/in_context/GLA-CLIP/scripts/unittest.sh

# 2. 单独生成诊断 manifest：reference >=200 tokens 且占比 >=5%
bash /data2/cld/in_context/GLA-CLIP/scripts/manifest_reference_diagnostic.sh

# 3. 先只改变 reference 条件，forward gate 仍保持 faithful sim>0
bash /data2/cld/in_context/GLA-CLIP/scripts/GLA_reference_smoke.sh

# 4. 快速检查空预测、forward saturation、FG/BG NN margin 和 attention collapse
python /data2/cld/in_context/GLA-CLIP/scripts/extract_diagnostics.py \
  /data2/cld/in_context/GLA-CLIP/gla_insid3_experiments/outputs/A0_A7_refdiag_v2_zero_smoke/metrics.jsonl \
  --methods all --only-problems

# 5. 仅当第 4 步仍显示 forward 空/饱和时，单独运行 adaptive gate 变体
bash /data2/cld/in_context/GLA-CLIP/scripts/GLA_diagnostic_smoke.sh
```

只有 smoke 不再大面积空预测后，才运行 DN 小范围扫描：

```bash
bash /data2/cld/in_context/GLA-CLIP/scripts/GLA_dn_sweep.sh
```

上述 `ref>=200 / ratio>=5% / adaptive gate` 是**故障验证设置**，不能替代正式
小目标评测。正式 factorial 仍使用预注册的 ref20 manifest 和 faithful
`sim>0` gate：

```bash
bash /data2/cld/in_context/GLA-CLIP/scripts/manifest.sh
bash /data2/cld/in_context/GLA-CLIP/scripts/GLA_ablation.sh
```

第一次运行不设置 `RESUME`；只有同一份 v2 输出被中断后才使用
`RESUME=1 OUTPUT_DIR=...`。

窗口特征提取默认使用 `WINDOW_BATCH_SIZE=2`。若显存不足可回退：

```bash
WINDOW_BATCH_SIZE=1 bash /data2/cld/in_context/GLA-CLIP/scripts/GLA_ablation.sh
```

该参数只加速共享的 DINO window extraction，不改变结果口径，也不会显著
加速后续每个方法的 dense KVE/matching。超参数 sweep 建议关闭昂贵诊断：
`MATCHING_DIAGNOSTICS=0 SAVE_CHECKPOINTS=0`。显存充足时可额外尝试
`QUERY_CHUNK=256`；它比继续增大 window batch 更直接作用于 attention 速度。

# 原 baseline 运行顺序

所有脚本都会自动切换到 `gla_insid3_experiments/`，因此既可以从 `GLA-CLIP/`，也可以从任意工作目录调用。

```bash
# 1. 不加载真实 DINO 权重的单元/烟雾测试
bash /data2/cld/in_context/GLA-CLIP/scripts/unittest.sh

# 2. 固化 50 个 episode（reference 筛选规则变化后必须重新生成）
bash /data2/cld/in_context/GLA-CLIP/scripts/manifest.sh

# 3. 用 1 个真实 episode 检查三条 baseline 能否跑通
bash /data2/cld/in_context/GLA-CLIP/scripts/SW_smoke.sh

# 4. 运行 manifest 中全部 episode
bash /data2/cld/in_context/GLA-CLIP/scripts/SW_diagnostic.sh
```

快速提取 B0/B1/B3 的 reference、forward/backward/candidate token 和核心指标：

```bash
python /data2/cld/in_context/GLA-CLIP/scripts/extract_diagnostics.py \
  /data2/cld/in_context/GLA-CLIP/gla_insid3_experiments/outputs/SW_smoke_ref20/metrics.jsonl

# 只显示 backward 或 candidate 全零的问题行
python /data2/cld/in_context/GLA-CLIP/scripts/extract_diagnostics.py \
  /path/to/metrics.jsonl --only-problems
```

运行 faithful A0–A7 全因子消融（额外保存 B1 原始 late-SW 对照）：

```bash
bash /data2/cld/in_context/GLA-CLIP/scripts/GLA_ablation.sh

# 先跑 5 个 episode
EPISODE_LIMIT=5 OUTPUT_DIR=outputs/A0_A7_v2_smoke \
  bash /data2/cld/in_context/GLA-CLIP/scripts/GLA_ablation.sh

# 中断后续跑
RESUME=1 OUTPUT_DIR=outputs/isaid_fold0_A0_A7_v2_ref20 \
  bash /data2/cld/in_context/GLA-CLIP/scripts/GLA_ablation.sh
```

可通过环境变量限制 episode 数或指定新输出目录：

```bash
EPISODE_LIMIT=5 OUTPUT_DIR=outputs/SW_diagnostic_5 \
  bash /data2/cld/in_context/GLA-CLIP/scripts/SW_diagnostic.sh
```

输出目录已经存在 `metrics.jsonl` 时程序会拒绝重复追加；请指定新 `OUTPUT_DIR`，或直接在命令行使用 `--resume`。

`manifest.sh` 默认固定 `--min-reference-tokens 20 --min-reference-ratio 0`；
两个阈值都可用同名大写环境变量覆盖。`SW_smoke.sh` 和
`SW_diagnostic.sh` 固定 faithful ref20 设置。token 数按 INSID3 实际的
`原 mask → 1024×1024 → 64×64 feature mask` 路径计算。旧 manifest
没有这项筛选，必须重新运行 `manifest.sh`；运行阶段也会再次计算并校验，
若 reference 少于 20 tokens 会直接报出 image id 和实际 token 数。

## Baseline 与分辨率

- `B0 / No-SW`：整图统一压缩到 INSID3 的 `1024×1024` 输入，DINOv3 patch-16 输出 `64×64` feature，再做一次全图 reasoning。
- `B1 / Late-SW`：每个 `512×512` crop 分别放大到 `1024×1024`，各自产生 `64×64` feature 并独立 reasoning，最后融合连续分数。
- `B3 / Early-SW`：各窗只提取 feature，先映射到全图 token canvas；iSAID 896×896、crop 512 时通常得到约 `112×112`，再限制到最多 `4096` tokens（通常 `64×64`）后统一 reasoning，以符合 INSID3 层次聚类的基准尺度。

## 中间结果

`SW_smoke.sh` 和 `SW_diagnostic.sh` 默认启用 `--save-checkpoints`：

```text
outputs/<run>/checkpoints/<episode>/
  reference.pt           # reference mask/raw/debiased feature/prototype/token count
  window_extraction.pt   # crop 坐标、raw/debiased feature、全图 token 坐标
  B0.pt                   # No-SW reasoning 全阶段与最终拼接结果
  B1.pt                   # Late-SW 每窗 matching/candidate/cluster/score
  B3.pt                   # Early-SW 融合尺度、统一 reasoning 与最终结果
```

每个 method checkpoint 都记录 `source_window_specs`、`reasoning_window_specs` 和 `resolution`，可直接核对输入尺度、feature 尺度及 token 数。Tensor 会在保存前转到 CPU，便于离线读取。
