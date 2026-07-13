# GLA-INSID3：面向滑动窗口语义一致性的实验设计 Pipeline

> 文档性质：实验预注册与实施设计，不包含实验结果，不要求在本阶段运行实验。  
> 核心目标：先定位 INSID3 引入滑动窗口后，语义不一致首次出现并被放大的具体步骤；再将 GLA-CLIP 的 Key-Value Extension、Proxy Anchor、Dynamic Normalization 迁移到 INSID3，并通过完整消融判断各模块是否真正解决问题。  
> 主要数据：iSAID-5i；Potsdam 作为超高分辨率遥感压力测试。  
> 代码基线：仓库中的 `INSID3/` 与 `GLA-CLIP/` 当前版本。

---

## 1. 先给出实验主线与关键结论

本项目不应一开始就把 GLA-CLIP 三个模块全部塞入 INSID3。正确顺序是：

1. 建立不带滑窗的原始 INSID3 基线，并冻结 episode、参考图、随机种子与后处理设置。
2. 建立两个滑窗基线：
   - **SW-Late**：每个窗口独立跑完整 INSID3，最后融合窗口分数或 mask。它用于暴露最直观的窗口语义不一致。
   - **SW-Early**：每个窗口只提取特征，先把窗口特征融合到全图坐标，再统一运行 INSID3 后半段。它用于判断问题主要来自 encoder，还是来自“每窗独立聚类/seed/aggregation”。
3. 在滑窗 pipeline 的六个检查点逐层量化不一致，并用 checkpoint replay 做因果干预，找出第一个产生显著不一致的步骤。
4. 只在确认的故障点迁移 GLA 模块。首选迁移位置是 **DINOv3 特征提取之后、INSID3 候选定位之前的 target semantic branch**；原始特征分支先保留给 clustering 和 intra-image similarity，避免跨窗 attention 抹掉边界结构。
5. 在完全相同的窗口、episode 和后处理设置下，跑 `2^3=8` 组 KVE/Proxy/DN 全因子消融，并同时报告分割精度、窗口一致性和代价。

需要特别区分两个问题：

- **窗口观测不一致**：同一个原图位置在两个重叠窗口中得到不同的 feature、candidate score、cluster decision 或 mask。
- **窗口边界错误**：最终融合结果在窗口网格附近出现额外错误或断裂。

前者可以不依赖 GT 直接测量，是定位因果步骤的主指标；后者需要 GT，用于验证这种不一致是否真的伤害分割。

---

## 2. 从论文与当前代码得到的基线事实

### 2.1 当前 INSID3 的真实推理链

根据论文与当前实现 [`INSID3/models/insid3.py`](INSID3/models/insid3.py)，单个 episode 的实际流程为：

```text
reference image(s) + mask(s) ─┐
                              ├─ resize 到 1024×1024
target image ─────────────────┘
        ↓
DINOv3 最后一层 dense feature
        ├─ original branch：target clustering、intra-image similarity
        └─ debiased branch：reference-target matching、candidate、cross similarity
        ↓
forward candidate ∩ backward-NN majority candidate
        ↓
target agglomerative clustering
        ↓
reference prototype 与候选 cluster prototype 选 seed
        ↓
cross similarity × intra similarity × 当前代码中的 area weight
        ↓
阈值化 → 双线性上采样 → 可选 CRF
```

与本实验直接相关的代码事实：

- 当前 `build_transform()` 直接把长宽都缩放为 `image_size×image_size`，不保持宽高比。
- 当前 `_extract_features()` 没有滑窗接口；reference 与 target 被拼入同一 batch，但各图仍独立经过 encoder。
- 位置基底在初始化时只构建一次。当前代码实际使用归一化后的全零图，而论文文字写的是 Gaussian noise；这两个版本必须作为实现变体记录，不能混用后再归因于滑窗。
- 原始 feature 用于 target clustering 与 seed-to-cluster 的 intra similarity；去偏 feature 用于 reference-target correspondence 与 cross similarity。这种双空间设计应在首轮迁移中保留。
- 当前 candidate 不仅有论文中的 backward NN，还额外要求 forward similarity `> 0`，空时退化到 top 10% 分位数。
- 当前 aggregation 还乘以 candidate 在 cluster 中的面积占比 `area_weights`，这不在论文 Eq. (13) 的简单 `cross × intra` 表达中。
- 默认后处理是双线性上采样；只有打开参数后才运行 CRF。所有诊断必须分别保存 **pre-CRF** 和 **post-CRF**，否则 CRF 会掩盖或制造 seam。
- iSAID loader 会随机选择 reference；只设置 seed 仍不足以支持所有方法逐 episode 严格配对，因此必须先固化 episode manifest。
- 仓库目前没有 Potsdam dataset adapter，也没有 Potsdam 数据准备说明。

### 2.2 GLA-CLIP 可迁移的本质

GLA-CLIP 的三个模块并不是三个同层级的普通后处理：

1. **Key-Value Extension (KVE)** 改变 attention 的可见范围：当前窗口 query 从只看本窗 key/value，变为看全部窗口 token。
2. **Proxy Anchor** 改变 query 本身：把局部 query 替换为从全局高相似 token 迭代聚合出的语义中心，减弱 inner-window bias。
3. **Dynamic Normalization (DN)** 改变 attention 分布：窗口越多时更强地平移/抑制噪声；高置信 token 越少时更强地放大有效响应。

GLA-CLIP 当前代码还提供了两点工程依据：

- 所有窗口先组成一个 batch，再在最后一层 attention 进行全局 key/value 访问；最终把每窗 logits 双线性放回原坐标并除以 coverage count。
- `crop_indices` 记录每个 crop token 对应的原图位置，query smoothing 会利用本窗 8 邻域以及重叠窗口的同位置 token。

但不能原样复制 CLIP 实现，因为 INSID3 没有“DINO attention map × CLIP value token”的双 backbone 结构。迁移后应定义为 **DINO-only global feature propagation**，并明确 Q/K/V 来自哪个 INSID3 feature branch。

### 2.3 需要冻结的实现口径

正式实验前创建一个 `implementation_manifest.yaml`，至少记录：

- INSID3 与 GLA-CLIP 的代码版本或文件哈希；
- DINOv3 型号、权重标识、patch size、feature 层；
- `image_size`、SVD 输入类型（zero/noise）、`svd_components`；
- candidate 是否使用 forward gate；
- aggregation 是否使用 area weight；
- clustering 是否保持当前无 spatial-connectivity 的实现；
- bilinear/CRF 后处理选择；
- window crop、stride、edge anchoring、padding和融合权重；
- episode manifest 与数据 split。

第一轮应完全遵循当前代码；论文口径变体单独做 ablation，不能在主结果中悄悄替换。

---

## 3. 研究问题与假设

### RQ1：滑动窗口的不一致第一次出现在哪一步？

- **H1a（encoder-locality）**：同一原图 patch 处于不同窗口上下文或不同窗口内坐标时，DINOv3 raw feature 已明显不同。
- **H1b（positional reset）**：每个窗口都从 `(0,0)` 重新开始位置编码，INSID3 的位置去偏可能缓解，也可能因窗口尺度/插值不同而放大不一致。
- **H1c（discrete amplification）**：raw/debiased feature 差异不大，但 backward NN、candidate 二值化、agglomerative clustering、seed argmax 或 aggregation threshold 将小差异放大为 mask 分歧。
- **H1d（stitch/postprocess）**：主要问题不是语义，而是 token-to-pixel 插值、窗口边缘、硬 mask 融合或每窗 CRF 引起的几何 seam。

### RQ2：GLA 三个模块迁移到哪里最合理？

- **H2**：优先对齐 debiased semantic branch，改善 candidate/cross similarity；原始 structural branch 暂不改变，可最大程度保留 INSID3 聚类所需的局部结构。
- 如果诊断显示 raw feature clustering 已是主要故障点，再增加 dual-branch 或 structural-branch 对齐，而不是预先同时改两条分支。

### RQ3：哪个 GLA 模块带来真实收益？

- **H3a**：KVE 扩大语义上下文，但在高重叠遥感图中会引入大量重复 background token，单独使用可能收益有限。
- **H3b**：Proxy Anchor 可能是主要一致性来源，因为它直接把不同窗口观测拉向共同语义中心。
- **H3c**：DN 对小目标和窗口数较多的图更重要，但其收益应体现在“减少 false global matches”，而非只提高平均 mIoU。

### RQ4：一致性改善是否等价于分割改善？

- **H4**：不是。把所有窗口输出过度平滑也会让一致性指标变好，因此必须联合报告 mIoU、boundary quality、small-object recall 与一致性指标。

---

## 4. 总体实验架构

### 4.1 四个可调用接口

为满足逐步诊断与模块替换，滑窗版 INSID3 设计为四个明确接口；每个接口都可保存和重新载入中间量：

| 接口 | 输入 | 输出 | 用途 |
|---|---|---|---|
| `I1_extract_windows` | 原图、window spec | crop、原图坐标、raw/debiased features、token center | 测 encoder 与位置去偏的一致性 |
| `I2_align_features` | 每窗 features、全局 token bank | aligned semantic/structural features、attention diagnostics | 插拔 KVE/Proxy/DN |
| `I3_reason_per_window` | 每窗 feature、reference prototype | similarity、NN、candidate、cluster、seed、combined score | 找离散放大步骤 |
| `I4_stitch_and_refine` | 每窗连续 score/mask、坐标、coverage | 全图 score、mask、可选 CRF | 区分推理与拼接问题 |

禁止只返回最终 hard mask。至少要暴露：`raw_feat`、`debiased_feat`、`sim_fwd`、`nn_ref_index`、`candidate_mask`、`cluster_labels`、`cluster_prototypes`、`seed_id`、`cross_sim`、`intra_sim`、`combined_score`、`pre_crf_mask`。

### 4.2 两种滑窗 baseline

#### B0：原始 INSID3（No-SW）

整张 target 按当前代码缩放到 `1024×1024` 后推理。它是精度参考，不是严格的等尺度对照，因为滑窗保留了更多局部分辨率。

#### B1：SW-Late（窗口独立推理，晚融合）

每个 target window 独立完成：

```text
feature → debias → candidate → clustering → seed → aggregation → continuous score
```

然后将每窗 continuous score 映射到原图并融合，最后只阈值一次。若现有实现暂时只能输出 hard mask，第一版可融合 binary probability，但必须标为退化 baseline。

此设置最容易出现：同一物体在不同窗口被选为不同 seed、相同 cluster 被一窗接受另一窗拒绝、目标跨窗时各窗只看到局部部件。

#### B2：SW-Early（窗口特征早融合，全局后半段）

每窗只跑 DINOv3 和 debias；将重叠 token 映射到统一原图 token canvas，以加权均值融合为一张全局 feature map，再统一运行 candidate、clustering、seed 与 aggregation。

B1 与 B2 的差异用于判断：

- `B2 ≫ B1`：主要故障来自每窗独立的 candidate/clustering/seed/aggregation。
- `B2 ≈ B1` 且两者都差：问题已存在于窗口 feature，或者 feature stitch 本身失败。
- `B1` overlap feature 一致但最终 mask 不一致：重点检查离散决策与后处理。

注意：全局 agglomerative clustering 的复杂度对高分辨率 token 数近似二次增长。B2 是诊断上界；Potsdam 上可只在裁出的中型 ROI 运行，或先做 window-region clustering 再进行全局 region merge，不能假装它天然可扩展到整块 6000×6000 tile。

### 4.3 窗口生成与拼接规范

窗口生成遵循 GLA 的 edge anchoring 逻辑：

```text
y1 = grid_y × stride
y2 = min(y1 + crop, H)
y1 = max(y2 - crop, 0)
```

x 方向同理，确保右边缘和下边缘完整覆盖。每个窗口保存：

- `window_id`、`(x1,x2,y1,y2)`；
- resize 前后尺寸及缩放率；
- 每个 feature token center 对应的浮点原图坐标；
- token 距最近窗口边界的距离；
- coverage map 和同一原图位置的 observation group。

融合设置至少比较：

1. uniform average（与 GLA 当前代码一致）；
2. center-weighted/Hann blending（降低边缘 token 权重）；
3. winner-take-center（同一点只取距离窗口中心最近的观测，作为几何控制）。

主实验使用 continuous score average；**不得先在每窗阈值化再平均**。CRF 只在全图 stitch 完成后运行；每窗 CRF 仅作为故障对照。

---

## 5. Phase A：数据、episode 与分层

### 5.1 iSAID-5i 主实验

直接沿用 [`INSID3/datasets/isaid.py`](INSID3/datasets/isaid.py) 的 3-fold、15 类设置，但新增固定 episode manifest：

```text
episode_id, fold, class_id,
reference_image_id(s), reference_mask_id(s),
target_image_id, target_mask_id,
window_spec_id
```

建议流程：

- 先从 fold 0 固定少量 diagnostic-dev episodes，用于接口调试与超参数方向判断；
- 最终报告 3 folds 全部结果，并为所有方法复用完全相同的 manifest；
- 每类至少覆盖 small/medium/large 三个目标面积层级；
- 单独标注目标相对窗口的拓扑：`fully-contained`、`cross-one-boundary`、`cross-multiple-boundaries`、`touch-image-edge`；
- 原图若不足以产生多个窗口，不把重复单窗样本计入“跨窗一致性”平均。

iSAID 仓库数据通常已经是裁好的 patch，因此主窗口建议从 `crop=512, stride=256` 起步，并将每窗 resize 到 DINOv3 的 1024 输入。该设置会改变有效物理尺度，所以必须同时报告 resize ratio，并把“同一原图位置的跨窗对比”作为主要诊断，而不能只拿 No-SW 精度差直接归因于语义不一致。

### 5.2 Potsdam 压力测试

当前仓库没有 Potsdam adapter。设计上新增与 iSAID 相同字段的 ICS episode adapter：

- 用有标签的 tile 划分 reference pool、diagnostic-dev 与 held-out target tile，按 tile 隔离以避免同区域泄漏；
- 将语义标签转换成每类 binary reference/target mask；
- ignore void/boundary pixels，不把其计入 mIoU、BER 或 seam error；
- 对整块超高分辨率 tile 使用 `crop=1024, stride=512` 作为主设置；另取 `stride∈{256,768,1024}` 做 overlap sweep；
- 同时构造 2048×2048 或 3072×3072 ROI 子集供全局 clustering/B2 使用，整 tile 只跑可扩展的 window/region 版本。

Potsdam 的具体 tile 列表、颜色到 class 的映射和 split 必须写入 manifest 后再开始实验，不能根据结果改 split。

### 5.3 必须分层报告的样本属性

- foreground 面积：按 target GT patch/pixel 占比分位数分 small/medium/large；
- 跨越窗口数：目标覆盖 1、2、3+ 个窗口；
- overlap coverage：同一点被 1、2、4+ 个窗口观察；
- window-edge distance；
- reference-target 外观差异；
- 类别与数据集；
- 是否存在同类 distractor。

GLA 的 DN 以高置信 token 数近似物体尺度，因此 small/large 分层是检验其机制是否成立的必要条件，而不是附加可视化。

---

## 6. Phase B：精准定位语义不一致出现在哪一步

### 6.1 诊断检查点

| 编号 | 检查点 | 比较对象 | 核心指标 | 要回答的问题 |
|---|---|---|---|---|
| D0 | crop/坐标映射 | 同一原图点的 token center、插值位置 | coordinate error、coverage holes | 是否只是几何映射错误？ |
| D1 | raw DINOv3 feature | overlap 中同一原图位置的多窗 feature | cosine、L2、CKA、edge-distance curve | encoder 是否已产生窗口上下文偏差？ |
| D2 | debiased feature | D1 相同配对去偏前后 | `Δcosine`、position-conditioned discrepancy | debias 缓解还是放大了窗口坐标重置？ |
| D3 | cross-image matching | 多窗中的 sim/NN/candidate | score MAE、rank corr、NN-mask flip、candidate disagreement | correspondence 阶段是否首次发生语义翻转？ |
| D4 | clustering | overlap 的 cluster partition | co-association disagreement、boundary F1、prototype match | 小 feature 差异是否被离散聚类放大？ |
| D5 | seed/aggregation | seed 与 cluster accept/reject | seed consistency、score margin、acceptance flip | argmax/threshold 是否是主要放大器？ |
| D6 | stitch/upsample/CRF | pre/post stitch 与 pre/post CRF mask | CWOD、BER、seam excess error | seam 来自推理还是后处理？ |

### 6.2 无 GT 的窗口一致性指标

#### 6.2.1 Overlap Feature Consistency（OFC）

对同一原图 token 位置 (p) 在窗口 (a,b) 中的观测：

\[
\mathrm{OFC}=\frac{1}{|\mathcal O|}\sum_{(p,a,b)\in\mathcal O}
\cos(f_p^{(a)}, f_p^{(b)}).
\]

分别计算 raw、debiased、aligned feature；并画 OFC 随 window-edge distance 的曲线。只报告均值会掩盖“窗口中心一致、边缘崩坏”的典型模式。

#### 6.2.2 Cross-Window Score Discrepancy（CWSD）

同一位置的 reference similarity 分数差：

\[
\mathrm{CWSD}=\operatorname{mean}_{(p,a,b)}|s_p^{(a)}-s_p^{(b)}|.
\]

同时报告 Spearman rank correlation，避免只看绝对标度变化。

#### 6.2.3 Candidate Flip Rate（CFR）

同一点在两个窗口中的 forward gate、backward-NN mask membership 或最终 candidate 是否翻转。需拆成：

- `forward_flip`；
- `backward_NN_index_change`；
- `backward_mask_membership_flip`；
- `final_candidate_flip`。

这可以精确判断问题来自相似度轻微漂移，还是 NN 落到了 reference mask 边界两侧。

#### 6.2.4 Cluster Co-association Disagreement（CCD）

cluster id 在不同窗口中没有可比性，不能直接算 label accuracy。对 overlap 内采样 patch pair ((p,q))，比较两个窗口是否都把它们分在同一 cluster：

\[
\mathrm{CCD}=\Pr\left[
\mathbf 1(c_p^{a}=c_q^{a})\ne
\mathbf 1(c_p^{b}=c_q^{b})
\right].
\]

再用 overlap IoU/Hungarian matching 对 cluster prototype 配对，报告匹配 prototype cosine。

#### 6.2.5 Cross-Window Output Disagreement（CWOD）

在 stitch 前，对所有 overlap 位置比较每窗连续 score 和二值 decision：

- score variance；
- pairwise probability MAE；
- binary disagreement rate；
- foreground/background 分开统计。

### 6.3 有 GT 的 seam 指标

#### 6.3.1 Binary ICS 版 BER

沿用 GLA-CLIP 的思想。令 \(\mathcal B\) 为跨窗口网格边界的相邻像素对，只保留 GT 二值标签相同的 pair：

\[
\mathrm{BER}=\frac{
\sum_{(p,q)\in\mathcal B}\mathbf 1[y_p=y_q\land \hat y_p\ne\hat y_q]
}{
\sum_{(p,q)\in\mathcal B}\mathbf 1[y_p=y_q]
}\times100.
\]

binary ICS 中 foreground 与 background 数量严重不平衡，所以必须额外报告 `BER-FG` 与 `BER-BG`。

#### 6.3.2 Seam Excess Error（SEE）

窗口边界带内错误率减去与其 GT boundary-distance 匹配的非窗口边界带错误率：

```text
SEE = error(window-boundary band) - error(matched control band)
```

它可避免把真实物体边界刚好经过窗口线的困难错误全部误判为滑窗 artifact。

#### 6.3.3 标准任务指标

- foreground mIoU（与 INSID3 原评测一致）；
- foreground/background IoU；
- Dice/F1；
- boundary F-score；
- small/medium/large object recall；
- episode-wise win/tie/loss 与 bootstrap 95% CI。

### 6.4 三组必要控制实验

1. **Duplicate control**：完全相同 crop 复制两次进 batch，理论上所有 checkpoint 应一致。若不一致，先排查 nondeterminism、autocast、hook 或索引错误。
2. **Context-only control**：同一原图 overlap patch 在两个不同邻域窗口中，统一 resize ratio，只改变外围上下文。它最接近“语义上下文不一致”。
3. **Position-only control**：尽量保持 crop 内容相同，通过 padding/平移让目标 patch 位于不同窗口内坐标，测位置编码与 debias 的影响。

另做 `center-only stitch` 与 `uniform stitch` 对照；若只换融合权重就消除大部分 BER，而 D1-D5 一致，则问题主要是窗口边缘质量，不应宣称 GLA 解决了语义问题。

### 6.5 用 checkpoint replay 做因果定位

仅观察相关性不足以确定哪一步“导致”最终不一致。对每个阶段 (D_s) 做 canonicalization intervention：

1. 保存每窗中间状态；
2. 对 overlap 同一原图位置，在阶段 (D_s) 把多窗观测替换为其 canonical mean/consensus；
3. 其余下游步骤原样重放；
4. 比较最终 CWOD、BER、SEE 和 mIoU 的变化。

示例：

- canonicalize raw feature 后不一致大幅下降：D1 是主因；
- raw/debiased feature 对齐后仍不下降，但统一 candidate 后下降：D3 的 NN/gate 是主放大点；
- candidate 一致，独立 clustering 后又分裂：D4 是首个离散故障点；
- pre-CRF 一致、post-CRF 不一致：D6 后处理是主因。

“首个主要故障点”的预注册判据：该阶段相较上一阶段造成显著的 disagreement 增量，且对该阶段做 canonicalization 后最终 CWOD/BER 的 paired bootstrap 95% CI 明确改善。不要用肉眼挑一个最好解释的 case。

---

## 7. Phase C：把 GLA 三个模块迁移到 INSID3

### 7.1 首选插入位置：semantic branch feature aligner

对第 (l) 个 target window 的 DINOv3 token，定义：

```text
F_raw^(l)  = normalize(DINO(window_l))
F_deb^(l)  = normalize(P_perp F_raw^(l))
```

首轮迁移采用：

- `Q,K,V` 均来自 target 的 **debiased DINOv3 feature**；
- 输出 `F_sem_aligned` 只替换 INSID3 中的 `feat_tgt_deb`，用于 forward/backward matching、candidate 与 cross similarity；
- `F_raw` 保持不变，用于 clustering 与 intra similarity；
- reference feature 不做跨 target 窗 attention，仍按原 INSID3 构造 debiased reference prototype。

这样能最小化改动并回答：“GLA 式全局目标图语义传播，能否稳定 INSID3 的跨图匹配？”

第二阶段才比较插入位置：

| 变体 | semantic branch | structural branch | 目的 |
|---|---|---|---|
| S-align | aligned | raw | 主方案，最小干预 |
| R-align | debiased | aligned raw | 检查 clustering 是否需要跨窗对齐 |
| Dual-align | aligned | aligned | 最大对齐，检查是否过平滑 |
| Post-candidate | 原始 | 原始 | 只对 candidate/cluster prototype 做 region-level 对齐，低成本探索 |

### 7.2 Key-Value Extension 的 DINO-only 定义

对当前窗口 query (Q_l\in\mathbb R^{N\times D})，收集全部 (L) 个窗口：

\[
K_g=[F^{(1)};\ldots;F^{(L)}],\qquad
V_g=[F^{(1)};\ldots;F^{(L)}].
\]

计算：

\[
S_l=Q_lK_g^\top,\qquad
\hat F_l=\operatorname{Normalize}
\left(\operatorname{Softmax}(A_l)V_g\right).
\]

其中 (A_l) 是经过 fixed 或 dynamic normalization 与 mask 后的 attention score。KVE 关闭时，(K,V) 只来自当前窗口。

需要同时测两个 token-bank 版本：

- **faithful-duplicate bank**：与 GLA 一致，重叠区域在每个窗口中各保留一份；
- **coordinate-deduplicated bank**：同一原图 token 位置先融合成一份，防止高重叠遥感图中背景被重复计数。

主论文迁移结论以 faithful 版本为准；deduplicated 是遥感场景的扩展消融。

### 7.3 Proxy Anchor

初始 query (q_i^{(0)}) 先做 L2 normalize。高置信集合：

\[
\mathcal P_i^{(t)}=\{j\mid
\cos(q_i^{(t)},k_j)>\rho\}.
\]

迭代更新：

\[
q_i^{(t+1)}=operatorname{Normalize}
\left(\frac{1}{|\mathcal P_i^{(t)}|}
\sum_{j\in\mathcal P_i^{(t)}}k_j\right).
\]

然后用最终 proxy 代替原 query。空集合时强制保留自身 token；并记录每个 query 的 `|P_i|`、inner/outer-window positive 比例及 proxy drift。

初始主设置采用论文实现说明的 `ρ=0.6, T=2`。仓库代码默认更接近 `ρ=0.55, T=5`，放入 sensitivity，不与主设置混为一谈。

Query smoothing 做独立开关：

- 本窗口 8-neighbor；
- 共享原图坐标的 overlapping-window token；
- 二者联合。

如果 DINOv3 不存在 CLIP 代码中描述的 high-norm anomaly，smoothing 可能无益，因此它不是 Proxy 的默认组成部分，必须单独报告。

### 7.4 Dynamic Normalization

按照 GLA 论文，对第 (i) 个 query：

\[
u=1+\lambda_1\log(1+L),\qquad
w_i=1+\frac{\lambda_2}{|\mathcal P_i|}.
\]

\[
A_i=w_i\left(S_i-
u\cdot\operatorname{mean}(S_i)\right).
\]

随后将低于阈值的 token mask 为 (-\infty)，再 softmax。主设置从论文的 `λ1=0.3, λ2=30` 起步，但必须在 diagnostic-dev 上先检查 DINOv3 cosine 分布；CLIP/DINOv1 的数值范围不能被假定与 debiased DINOv3 相同。

必须记录：

- 每 query 的 `u,w_i,|P_i|`；
- 被 mask 的 token 比例；
- attention entropy；
- attention mass 的 inner/outer-window 比例；
- attention 落在 GT foreground/background 的比例（仅分析，不参与推理）；
- small/medium/large 分层结果。

若 Proxy 关闭，DN 所需的 `|P_i|` 仍由原 query 的阈值邻域计算，但不更新 query。这样 DN-only 才有明确可运行定义。

### 7.5 低成本 region-level 迁移

由于 dense KVE 的 attention 复杂度约为 (O(L^2N^2D))，Potsdam 可能不可承受。预留一个不与主方法混淆的扩展：

1. 每窗先聚类；
2. 用 cluster prototype 作为 global K/V；
3. query 为 token 或 candidate cluster prototype；
4. Proxy/DN 在 region bank 上运行；
5. 将对齐后的 region score 回写 token。

比较 dense-global、top-k-global、coordinate-deduplicated 与 region-global 四种 bank，报告效果—显存—时延 Pareto，而不是只追求最高 mIoU。

---

## 8. Phase D：完整对照与消融矩阵

### 8.1 滑窗机制主对照

| ID | 方法 | 目的 |
|---|---|---|
| B0 | No-SW INSID3 | 原始精度参考 |
| B1 | SW-Late + uniform score stitch | 暴露完整窗口独立推理不一致 |
| B2 | SW-Late + center-weighted stitch | 判断边缘质量能解释多少问题 |
| B3 | SW-Early feature stitch + global reasoning | 判断后半段独立决策的贡献 |
| B4 | SW-Late + global CRF after stitch | 标准后处理 |
| B5 | SW-Late + per-window CRF | 故障对照，检验 CRF 是否制造 seam |

### 8.2 GLA 三模块 `2^3` 全因子消融

所有组固定相同的 SW-Late/S-align、window spec、reference episodes、stitch 与后处理：

| ID | KVE | Proxy | DN | 解释 |
|---|:---:|:---:|:---:|---|
| A0 | × | × | × | local attention/alignment baseline |
| A1 | ✓ | × | × | KVE 独立贡献 |
| A2 | × | ✓ | × | local-domain proxy 独立贡献 |
| A3 | × | × | ✓ | local-domain DN 独立贡献 |
| A4 | ✓ | ✓ | × | global context + anchor |
| A5 | ✓ | × | ✓ | global context + scale normalization |
| A6 | × | ✓ | ✓ | 无 global K/V 时两种稳定化是否有效 |
| A7 | ✓ | ✓ | ✓ | Full GLA-INSID3 |

为保持模块定义正交：KVE 关闭时，Proxy 的候选域和 DN 的 attention 域都限制在当前窗口；KVE 打开时才使用 global token bank。另加一组 `global-proxy/local-value` 作为低成本探索，但不放进主 factorial。

### 8.3 INSID3 注入位置消融

在 A7 基础上比较：

- S-align；
- R-align；
- Dual-align；
- 只对 candidate token；
- 只对 cluster prototype。

判断依据不只看 mIoU：如果 Dual-align 提高 OFC 但显著降低 boundary F-score 或把小物体并入大区域，说明 structural branch 被过度平滑。

### 8.4 Window 与超参数敏感性

至少考察：

- overlap ratio：0%、25%、50%、75%；
- window crop：iSAID `{384,512,768}`，Potsdam `{768,1024,1536}`，均记录送入 encoder 的 resize ratio；
- `ρ∈{0.5,0.55,0.6,0.65,0.7}`；
- `T∈{0,1,2,5}`；
- `λ1∈{0,0.1,0.3,0.5}`；
- `λ2∈{0,10,30,50}`；
- token bank：duplicate/deduplicated/top-k/region；
- debias rank 与 window-local/global positional basis；
- zero-image basis 与 Gaussian-noise basis。

超参数只在 diagnostic-dev 确定一次，最终 iSAID folds 与 Potsdam held-out 不能逐数据集重新挑最优值。可以同时报告“shared setting”和“oracle per-dataset best”，但二者必须明确分开。

---

## 9. 实验执行顺序与停止条件

### Stage 0：基线复现检查

产物：固定 episode manifest、B0 mIoU、代码/权重/配置哈希。  
通过条件：同一 manifest 重跑逐 episode 结果一致；duplicate control 各 checkpoint 误差接近数值精度。

### Stage 1：只实现滑窗，不加入 GLA

先完成 B1/B2，输出 D0-D6 所有检查点。  
通过条件：coverage 无空洞；token-to-original 坐标误差受控；连续 score stitch 与 hard-mask stitch 被明确区分。

### Stage 2：定位首个故障点

在 iSAID diagnostic-dev 和少量 Potsdam ROI 上运行 D0-D6、三个控制实验与 checkpoint replay。  
产物：每阶段 disagreement 表、edge-distance 曲线、至少 10 个固定可视化 case。  
停止条件：已能用干预而非仅相关性说明主要故障点；若 D0 就失败，先修坐标/插值，不继续做 GLA。

### Stage 3：最小迁移

只实现 S-align 的 A0、A1、A2、A3、A7，快速判断模块方向。  
如果 A7 连 OFC/CFR/CWOD 都不改善，先检查 Q/K/V 定义与 cosine 分布，不立即扩大超参搜索。

### Stage 4：完整 factorial 与位置消融

补齐 A0-A7、注入位置和 bank ablation；固定 shared hyperparameters。

### Stage 5：最终评测

在 iSAID 3 folds 与 Potsdam held-out 上一次性评测，报告：

- overall 与分层 mIoU/Dice/boundary F；
- OFC/CWSD/CFR/CCD/CWOD；
- BER-FG/BER-BG/SEE；
- latency、peak memory、global token 数、attention 计算量；
- paired bootstrap 95% CI 与 per-episode scatter。

---

## 10. 结果表模板

### 10.1 故障定位表

| Method | raw OFC ↑ | deb OFC ↑ | CWSD ↓ | CFR ↓ | CCD ↓ | Seed flip ↓ | pre-CRF CWOD ↓ | BER-FG ↓ | mIoU ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | | | | | | | | | |
| Canonicalize@D1 | | | | | | | | | |
| Canonicalize@D2 | | | | | | | | | |
| Canonicalize@D3 | | | | | | | | | |
| Canonicalize@D4 | | | | | | | | | |
| Canonicalize@D5 | | | | | | | | | |

### 10.2 GLA factorial 表

| ID | KVE | Proxy | DN | mIoU ↑ | BER-FG ↓ | CWOD ↓ | small recall ↑ | boundary F ↑ | time | memory |
|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | × | × | × | | | | | | | |
| A1 | ✓ | × | × | | | | | | | |
| A2 | × | ✓ | × | | | | | | | |
| A3 | × | × | ✓ | | | | | | | |
| A4 | ✓ | ✓ | × | | | | | | | |
| A5 | ✓ | × | ✓ | | | | | | | |
| A6 | × | ✓ | ✓ | | | | | | | |
| A7 | ✓ | ✓ | ✓ | | | | | | | |

### 10.3 分层机制检验表

| Scale / topology | `|P_i|` | mask ratio | attention entropy | outer positive mass | ΔCWOD | ΔmIoU |
|---|---:|---:|---:|---:|---:|---:|
| small, contained | | | | | | |
| small, cross-boundary | | | | | | |
| large, 2 windows | | | | | | |
| large, 3+ windows | | | | | | |

---

## 11. 成功判据与结果解释

### 11.1 Full 方法成功

Full GLA-INSID3 只有同时满足以下条件才算解决问题：

1. 相对固定的 naive SW baseline，CWOD、BER-FG 或 SEE 显著下降；
2. mIoU 不下降，最好有 paired CI 支持的提升；
3. small-object recall 不因 global smoothing 明显下降；
4. 改善在 cross-boundary 与 multi-window 目标上强于 fully-contained 目标，符合机制预期；
5. attention diagnostics 显示 outer-window semantic positive mass 上升，而非所有 token 被平均化；
6. 额外显存与时延有完整报告。

### 11.2 一致性变好但 mIoU 下降

说明方法可能发生 semantic collapse 或过平滑。检查：

- Proxy 的 `ρ` 是否过低、T 是否过多；
- DN 是否 over-mask；
- structural branch 是否不应对齐；
- 重复 overlap token 是否让大面积背景主导；
- small-object recall 和 boundary F 是否下降。

### 11.3 mIoU 提高但 BER/CWOD 不变

说明收益可能来自更好的全局语义匹配，而不是消除窗口不一致。可以作为性能提升报告，但不能得出“解决 seam”的结论。

### 11.4 KVE 无效、Proxy 有效

这与 GLA 原论文消融中 KVE 独立贡献难以干净分离的疑问相符。优先继续 region/global-proxy 低成本方案，不必为 dense KVE 的 (L^2N^2) 代价强行辩护。

### 11.5 debias 后窗口一致性反而下降

分别检查：

- basis 是按 encoder input 的窗口坐标构建，还是按全图坐标构建；
- zero-image 与 Gaussian-noise basis；
- 同一原图位置在不同窗口 local coordinate 上的投影响应；
- reference 使用去偏而 structural target 使用 raw 的双空间差异。

可增加 `window-local basis`、`shared basis`、`no debias` 三联对照，但不应直接移除 INSID3 debias 后把变化归因于 GLA。

---

## 12. 建议的日志与可视化产物

每个 episode 保存一个轻量 metadata JSON；大型 tensor 按阶段和方法单独存储并允许关闭。至少输出：

- 原图、reference mask、GT、窗口网格和 coverage heatmap；
- 同一 overlap 位置在不同窗口的 raw/debiased/aligned cosine；
- reference similarity map、backward NN membership、candidate map；
- clustering 边界、seed cluster、每 cluster cross/intra/combined score；
- 每窗 pre-stitch score、variance map、最终 pre/post-CRF mask；
- Proxy high-confidence token、proxy drift；
- DN mask ratio、attention entropy、inner/outer attention mass；
- 错误按 window edge distance 的曲线。

固定挑选：最好、最差、改善最大、退化最大、small cross-boundary、large multi-window 六类 case，避免只展示成功样例。

---

## 13. 预期代码改动边界（后续实施时）

本阶段不写代码；后续实现建议保持以下边界：

```text
INSID3/
  models/
    insid3.py                 # 只增加 checkpoint 暴露与 aligner 调用
    window_aligner.py         # KVE / Proxy / DN
  utils/
    sliding_window.py         # crop、坐标、stitch、coverage
    consistency_metrics.py    # OFC/CWSD/CFR/CCD/CWOD/BER/SEE
  datasets/
    potsdam.py                # 新 adapter
  inference_segmentation.py   # 增加 --window-* 与 --align-* 参数
  analysis_window_stages.py   # checkpoint replay 与可视化
```

所有开关都应能独立启用，并将完整配置写入结果目录。建议关键参数：

```text
--window-mode {none,late,early}
--window-crop --window-stride
--stitch-mode {uniform,hann,center}
--align-stage {none,semantic,structural,dual,region}
--kve --proxy-anchor --dynamic-norm
--proxy-rho --proxy-iters
--dn-lambda1 --dn-lambda2
--token-bank {duplicate,deduplicated,topk,region}
--save-checkpoints
--episode-manifest
```

---

## 14. 最小可行实验（MVP）

若资源或时间有限，最少做以下 12 项也能形成完整结论链：

1. iSAID 固定 50 个跨窗 diagnostic episodes；
2. B0 No-SW；
3. B1 SW-Late uniform；
4. B2 SW-Late Hann；
5. B3 SW-Early；
6. D1-D6 逐层指标；
7. canonicalize@D1；
8. canonicalize@D3；
9. canonicalize@D4/D5；
10. A1 KVE-only；
11. A2 Proxy-only 与 A3 DN-only；
12. A7 Full。

即使 MVP 中 Full 方法没有提升，只要 checkpoint replay 能准确指出语义不一致首次出现与主要放大位置，实验仍然回答了最关键的研究问题。

---

## 15. 最终应形成的结论格式

最终报告不应只写“GLA 有效/无效”，而应按以下句式给出可证伪结论：

> 在固定的 iSAID/Potsdam ICS episodes 和窗口设置下，同一原图位置的语义不一致首次在 **[D1/D2/…]** 达到显著水平，并主要在 **[candidate/clustering/seed/stitch]** 被放大。对该阶段做 canonicalization 后，CWOD/BER 改善 **[数值与 CI]**，说明其具有因果贡献。GLA-INSID3 中 **[KVE/Proxy/DN]** 是主要有效模块，它在 **[目标尺度/跨窗拓扑]** 上降低 **[一致性指标]**，同时令 mIoU **[变化]**；其代价为 **[显存/时延]**。因此推荐的最终 pipeline 是 **[具体模块、插入分支和 token bank]**，而不是无条件采用完整 dense KVE。

该格式能同时回答“问题发生在哪里”“为什么发生”“哪个模块解决了它”“是否值得付出计算代价”四个问题。

---

## 16. 本设计的本地材料依据

本方案基于仓库内以下材料交叉核对，而不是只依据论文摘要：

- [`MinerU_markdown_INSID3_2076363961508925440.md`](MinerU_markdown_INSID3_2076363961508925440.md)：INSID3 主文、补充材料、公式、消融、实现细节与计算开销；
- [`MinerU_markdown_Looking_Beyond_the_Window_Global-Local_Aligned_CLIP_2076363575528103936.md`](MinerU_markdown_Looking_Beyond_the_Window_Global-Local_Aligned_CLIP_2076363575528103936.md)：GLA-CLIP 主文与补充材料，包括 BER、三模块公式、遥感实验及复杂度；
- [`week19上下文分割.md`](week19上下文分割.md)：已有的 INSID3、GLA-CLIP、SPAR 调研与待验证想法；
- [`INSID3/`](INSID3/)：模型、dataset、metric、refinement、inference 与参数实现；
- [`GLA-CLIP/`](GLA-CLIP/)：窗口生成/拼接、KV extension、proxy、dynamic normalization、query smoothing 与配置实现。

GLA-CLIP 的遥感结果属于 open-vocabulary semantic segmentation，不能直接当作迁移到 in-context segmentation 后必然有效的证据；它只支持“该机制值得在遥感滑窗场景检验”。本实验的结论必须由固定 ICS episodes 上的 factorial 与阶段诊断给出。
