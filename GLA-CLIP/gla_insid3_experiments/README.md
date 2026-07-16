# GLA-INSID3 实验脚本

该目录以**独立适配层**方式调用 `/data2/cld/in_context/INSID3-main`，不修改 INSID3 源码，也不会向 `/data/lky/data/rs_seg` 写入任何文件。manifest、日志、指标和 checkpoint 都被强制限制在 `GLA-CLIP/` 内。

实现内容对应《实验设计方案》的 MVP 与完整三模块 factorial：

- `B0`：整图 No-SW；
- `B1/B2`：SW-Late + uniform/Hann 连续分数融合；
- `B3`：SW-Early 特征融合后全局 reasoning；
- `B4/B5`：全图 CRF / 每窗 CRF 故障对照；
- `A0`–`A7`：KVE、Proxy Anchor、Dynamic Normalization 的 `2^3` 全因子消融；
- `R-D1`–`R-D5`：raw feature、debiased feature、candidate、cluster、score 的 checkpoint replay；
- OFC、CWSD、rank correlation、forward/backward/candidate flip、CCD、CWOD、BER-FG/BG、按 GT boundary distance 匹配的 SEE；
- duplicate control、implementation manifest、逐 episode JSONL 和 paired bootstrap 95% CI。

## 运行环境

直接使用 INSID3 已配置好的 Python 环境；脚本不会自动安装依赖或联网下载。DINOv3 由 INSID3 当前的 `models/__init__.py` 通过 `source="local"` 加载。默认 large 权重路径沿用其现有配置。

先在 SSH 上进入：

```bash
cd /data2/cld/in_context/GLA-CLIP/gla_insid3_experiments
```

## 1. 固化 episode

fold 0 的 50 个跨窗 episode：

```bash
python run_experiment.py \
  --command manifest \
  --data-root /data/lky/data/rs_seg \
  --manifest manifests/isaid_fold0_mvp.jsonl \
  --fold 0 --shots 1 --num-episodes 50 \
  --min-reference-tokens 20 \
  --min-reference-ratio 0 \
  --window-crop 512 --window-stride 256 --seed 0
```

`--manifest` 的相对路径以本目录为基准。即使传入绝对路径，脚本也会拒绝写到 `GLA-CLIP/` 之外。

## 2. 运行 MVP

```bash
python run_experiment.py \
  --command run \
  --insid3-root /data2/cld/in_context/INSID3-main \
  --data-root /data/lky/data/rs_seg \
  --manifest manifests/isaid_fold0_mvp.jsonl \
  --output-dir outputs/isaid_fold0_mvp \
  --methods B0,B1,B2,B3,A1,A2,A3,A7 \
  --replays D1,D3,D4,D5 \
  --min-reference-tokens 20 \
  --device cuda --window-batch-size 2
```

默认会先对首个 crop 做 duplicate control；差异超过 `1e-5` 时立即停止。显存不足时优先减小 `--window-batch-size` 和 `--query-chunk`。B3/D4 的全局层次聚类近似二次复杂度，分别受 `--early-max-tokens` 和 `--d4-max-tokens` 限制。

reference 会在 manifest 生成时按 INSID3 的实际 feature-mask 路径筛选，并在推理时二次校验。主脚本建议固定 `--min-reference-tokens 20`；若需降到 10，必须重新生成 manifest，并在所有方法中保持同一阈值。

`--min-reference-ratio` 可同时按 64×64 reference grid 的前景占比筛选。
`scripts/manifest_reference_diagnostic.sh` 使用 `>=200 tokens` 且 `>=5%`，只用于
验证 reference/background imbalance，不替代正式 small-object manifest。

forward gate 提供三个显式口径：`zero`（原始 INSID3 的 `sim>0`）、
`quantile`（固定 top fraction）与 `adaptive`（仅在 zero gate 为空或饱和时
回退）。正式主结果固定 `zero`；诊断 smoke 使用
`adaptive --forward-quantile 0.9 --forward-max-positive-ratio 0.95`。

只跑 manifest 前 N 个 episode 可使用 `--episode-limit N`；`0` 表示全部。三条基础 baseline 的真实单 episode 检查可直接运行 `../scripts/SW_smoke.sh`。

中断后续跑使用同一输出目录并加 `--resume`。不加 `--resume` 时若 `metrics.jsonl` 已存在，脚本会拒绝追加，避免重复 episode 污染统计。

## 3. 完整 factorial

```bash
python run_experiment.py \
  --command run \
  --manifest manifests/isaid_fold0_mvp.jsonl \
  --output-dir outputs/isaid_fold0_factorial \
  --methods B1,A0,A1,A2,A3,A4,A5,A6,A7 \
  --replays '' \
  --token-bank duplicate
```

当前 v2 已修复 DN/fixed normalization 的 cutoff：所有 token-bank 都会在
softmax 前把未通过的 logits 置为 `-inf`。修复前的 A0–A7 指标不可与 v2
混合或使用 `--resume` 继续追加。

`--matching-diagnostics` 会额外保存每个 target token 对 reference 前景/背景
的最大相似度与 margin，适合 smoke/checkpoint 调试；正式全量运行默认关闭，
以避免额外显存和时间开销。常规 JSONL 仍会记录 forward 分布、NN 命中率、
reference-index 集中度、空间特征 dispersion、attention entropy/top-1 mass。

遥感扩展 bank 可通过 `--token-bank deduplicated` 或 `--token-bank topk --topk 256` 单独运行。主 factorial 应固定同一 bank、窗口、episode、threshold 和 stitch 设置。

## 4. 汇总与 paired CI

```bash
python summarize_results.py \
  --metrics outputs/isaid_fold0_mvp/metrics.jsonl \
  --baseline B1 \
  --bootstrap-samples 10000
```

输出默认是同目录的 `summary.json`。paired delta 定义为 `method - B1`，每个指标同时标注 `higher_is_better` 或 `lower_is_better`。

## 输出结构

```text
outputs/<run>/
  implementation_manifest.json
  duplicate_control.json
  metrics.jsonl
  summary.json
  checkpoints/<episode>/reference.pt          # 仅 --save-checkpoints
  checkpoints/<episode>/window_extraction.pt  # 仅 --save-checkpoints
  checkpoints/<episode>/<method>.pt           # 仅 --save-checkpoints
```

checkpoint 暴露 raw/debiased feature、forward similarity、reference NN、backward membership、candidate、cluster、seed、cross/intra/combined score、continuous score 和 pre-CRF mask；同时保存输入/feature/reasoning 分辨率。大型 window feature 只在共享的 `window_extraction.pt` 保存一次，避免 B0/B1/B3 重复占用空间。

## 本地核心测试

测试不加载 DINO 权重，也不访问数据集：

```bash
python -m unittest discover -s tests -v
```

当前脚本针对实验方案的 iSAID-5i 主实验实现。Potsdam 的 tile split、颜色映射和 void 规则在设计中被明确要求先写入预注册 manifest；在这些信息未给定前，脚本不会猜测映射或改动数据集。
