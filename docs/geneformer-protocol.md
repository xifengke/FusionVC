# 实验 04：冻结 Geneformer 特征迁移（geneformer-v1）

日期：2026-10-09。使用已下载的 Geneformer V1-10M、Genecorpus-30M 词典和固定官方源码
`1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5`，不更换为默认 V2。

## 固定比较与监督边界

主组沿用 baseline-v1 的 unpaired pseudobulk MSE。附加 Energy MMD、MSE + 0.1 × Energy MMD，
每组种子 42/43/44，共九次正式训练和完整官方验证。既有 MLP、Transformer、scGPT 的
27 次损失比较作为固定参考；原 checkpoint 和结果保留。测试集继续保留到八组完成后统一评估。

保持 150 个训练靶点、50 个验证靶点、18,080 个输出基因和官方完整细胞数。
固定同一组 256 个候选输入基因、同一组 2,048 个训练 non-targeting 对照细胞。
AdamW lr=0.001、weight_decay=0.0001、batch=16、32 个训练控制细胞、32 个 MMD 真实细胞、
120 epochs 上限、patience=20、gradient_clip=1、bf16 预测头训练。选择规则仍为
固定 256 个训练对照细胞的验证 pseudobulk MAE；不按本轮评分调输入、损失系数或结构。
MMD 与附加 64+64 细胞分布评价规则见 [mmd-v1](mmd-protocol.md)。

## 基因 ID、输入排序和输出覆盖

使用原始训练文件 `var/gene_id` 的 Ensembl ID，并核对其基因顺序与 baseline cache 相同。
不用外部 gene_info 的当前 symbol 反查旧名称，也不猜测别名。通过固定 30M canonical-ID
字典映射，只有同时存在于 token 和 median 词典的输入基因参与编码。
映射重复时先合并原始 count；缺失输入明确记录并跳过，所有输出保留。

仅读取训练 non-targeting 的原始 count 行，控制索引与前几组一致。
对每个细胞的非零输入 count 除以对应 Genecorpus-30M 非零表达中位数，
使用官方 `np.argsort` 降序规则排序，不追加 CLS/EOS，不保留零表达 token，右侧 pad=0。
官方库的逐细胞 library-size 缩放为同一正数，不改变排序，故在固定候选输入上省略它。
最大长度 2,048，本轮输入最多 256，实际不触发截断。这是受限输入的特征迁移实验，
并非对完整转录组进行原论文的标准零样本扰动分析。

每个输出靶点用冻结 256 维 pretrained word embedding、原训练对照拟合的 34 维描述和
coverage 标记，共 291 维。缺失靶点回退为零 pretrained embedding + 训练描述 + coverage=0。
不会给未见验证靶点创建新的可训练 ID embedding。

## 编码器与预测头

Transformers 4.57.1 的 BertModel，256 维、6 层、4 heads、FFN 512、ReLU、绝对位置编码。
使用 eager attention、float32、永久 eval/no_grad。逐 tensor 严格加载全部 BERT 编码器参数；
旧 checkpoint 的 `position_ids` buffer 必须等于 `arange(2048)` 才允许适配为现代非持久 buffer。
忽略原 MLM 预测头；缺层、shape 不符、异常额外参数均失败，不随机初始化缺失 encoder 层。

最后一层输出按非 pad token 均值池化（官方 EmbExtractor 的 emb_layer=0），
再对 2,048 个训练对照细胞均值，形成固定 256 维 context。
context `256 → 64`、target `291 → 64`，拼接 `128 → 256 → 64`，最后 `64 → 18,080`
零初始化位移解码器，与 scGPT 使用同一个 FrozenFeaturePredictor。
本轮可训练参数为 1,259,808，冻结参数为 10,196,992。
共同条件位移作用于独立训练对照细胞，保持非负表达和 library-size 归一化。
没有同一细胞扰动前后的配对，也没有新增细胞状态依赖响应。

相比 scGPT，预训练语料、编码器维度、秩输入、可识别基因数和预测头参数量不同。
这能比较当前冻结特征迁移方案，不能完全隔离纯预训练的因果效应。

## 缓存、来源与检查

缓存 `data/processed/geneformer-frozen-v1` 保存控制/原始行索引、逐细胞 token 与长度、
基因到 Ensembl/token 的映射、独立控制 embedding、均值 context 与 target features。
manifest 记录已下载资产 SHA256、源码 SHA256、baseline 输入 SHA256、原始训练文件
大小与时间戳、Transformers 版本、数组 SHA256、覆盖和提取成本，最后写完成标记。
训练和预测重新校验来源；checkpoint 与 feature manifest 必须一致。

编码器检查直接与官方 BertForMaskedLM.from_pretrained 的最后一层对照；
排序和非 padding pooling 对照下载源码中未修改的函数。三种损失 seed 42 重训后检查
每个 tensor 和选择 epoch；验证产物检查全部基因、细胞、靶点、聚合成绩与共同控制抽样。
冻结权重与原三架构 checkpoint 同时检查未变。

```powershell
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe scripts/check_geneformer_parity.py
./.venv/Scripts/python.exe scripts/run_geneformer.py --phase smoke
./.venv/Scripts/python.exe scripts/run_geneformer.py --phase all
./.venv/Scripts/python.exe scripts/check_geneformer_reproducibility.py
./.venv/Scripts/python.exe scripts/verify_geneformer_artifacts.py
```

环境安装见 [requirements](../requirements/README.md)。结果写入独立 `results/geneformer`，
这些是 H1 CRISPRi 的验证成绩，不是测试成绩或历史排行榜归一化分数。

参考：[官方模型卡](https://huggingface.co/ctheodoris/Geneformer)、
[官方 tokenizer](https://huggingface.co/ctheodoris/Geneformer/blob/1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5/geneformer/tokenizer.py)、
[官方 embedding 接口](https://geneformer.readthedocs.io/en/latest/geneformer.emb_extractor.html)。

## 本轮完成情况

9/9 次正式训练及完整官方验证完成，四种架构 × 三种损失共 36 次比较。
21 项测试通过；真实编码器、官方排序、非 pad pooling 和 16 个缓存控制细胞的
原始 count/token/embedding 对照通过，数值差异为 0。输入覆盖 254/256，全部训练和验证靶点覆盖。
三种损失 seed 42 的逐 tensor/最佳 epoch/特征来源复现通过；原三架构 33 个 checkpoint 未变。

均值组 MAE=0.028047、纯 MMD=0.027828、混合=0.028013。纯 MMD 较自身均值组
MAE 降低约 0.78%，但 DES/PDS 下降，附加分布指标也未同步改善。
三组均未超过无扰动 MAE 对照，结论仅适用于本轮冻结特征、受限输入、共同条件位移与固定选择规则。
完整成绩和配对区间见 [结果报告](../results/geneformer/README.md)。
