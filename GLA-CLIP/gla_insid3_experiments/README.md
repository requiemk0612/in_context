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
  --device cuda --window-batch-size 2
```

默认会先对首个 crop 做 duplicate control；差异超过 `1e-5` 时立即停止。显存不足时优先减小 `--window-batch-size` 和 `--query-chunk`。B3/D4 的全局层次聚类近似二次复杂度，分别受 `--early-max-tokens` 和 `--d4-max-tokens` 限制。

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
  checkpoints/<episode>/<method>.pt   # 仅 --save-checkpoints
```

checkpoint 暴露 raw/debiased feature、forward similarity、reference NN、backward membership、candidate、cluster、seed、cross/intra/combined score、continuous score 和 pre-CRF mask。默认不保存大型 tensor。

## 本地核心测试

测试不加载 DINO 权重，也不访问数据集：

```bash
python -m unittest discover -s tests -v
```

当前脚本针对实验方案的 iSAID-5i 主实验实现。Potsdam 的 tile split、颜色映射和 void 规则在设计中被明确要求先写入预注册 manifest；在这些信息未给定前，脚本不会猜测映射或改动数据集。
