# 当前阶段运行顺序

所有脚本都会自动切换到 `gla_insid3_experiments/`，因此既可以从 `GLA-CLIP/`，也可以从任意工作目录调用。

```bash
# 1. 不加载真实 DINO 权重的单元/烟雾测试
bash /data2/cld/in_context/GLA-CLIP/scripts/unittest.sh

# 2. 首次运行时固化 50 个 episode
bash /data2/cld/in_context/GLA-CLIP/scripts/manifest.sh

# 3. 用 1 个真实 episode 检查三条 baseline 能否跑通
bash /data2/cld/in_context/GLA-CLIP/scripts/SW_smoke.sh

# 4. 运行 manifest 中全部 episode
bash /data2/cld/in_context/GLA-CLIP/scripts/SW_diagnostic.sh
```

可通过环境变量限制 episode 数或指定新输出目录：

```bash
EPISODE_LIMIT=5 OUTPUT_DIR=outputs/SW_diagnostic_5 \
  bash /data2/cld/in_context/GLA-CLIP/scripts/SW_diagnostic.sh
```

输出目录已经存在 `metrics.jsonl` 时程序会拒绝重复追加；请指定新 `OUTPUT_DIR`，或直接在命令行使用 `--resume`。

## Baseline 与分辨率

- `B0 / No-SW`：整图统一压缩到 INSID3 的 `1024×1024` 输入，DINOv3 patch-16 输出 `64×64` feature，再做一次全图 reasoning。
- `B1 / Late-SW`：每个 `512×512` crop 分别放大到 `1024×1024`，各自产生 `64×64` feature 并独立 reasoning，最后融合连续分数。
- `B3 / Early-SW`：各窗只提取 feature，先映射到全图 token canvas；iSAID 896×896、crop 512 时通常得到约 `112×112`，再限制到最多 `4096` tokens（通常 `64×64`）后统一 reasoning，以符合 INSID3 层次聚类的基准尺度。

## 中间结果

`SW_smoke.sh` 和 `SW_diagnostic.sh` 默认启用 `--save-checkpoints`：

```text
outputs/<run>/checkpoints/<episode>/
  reference.pt           # reference mask/raw/debiased feature/prototype
  window_extraction.pt   # crop 坐标、raw/debiased feature、全图 token 坐标
  B0.pt                   # No-SW reasoning 全阶段与最终拼接结果
  B1.pt                   # Late-SW 每窗 matching/candidate/cluster/score
  B3.pt                   # Early-SW 融合尺度、统一 reasoning 与最终结果
```

每个 method checkpoint 都记录 `source_window_specs`、`reasoning_window_specs` 和 `resolution`，可直接核对输入尺度、feature 尺度及 token 数。Tensor 会在保存前转到 CPU，便于离线读取。

