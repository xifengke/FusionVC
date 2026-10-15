# 三种冻结表示融合实验（fusion-v1）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run（用户授权实现并执行三种融合）
- Origin Date: 2026-10-15
- Verification Status: VERIFIED（产物、原生token身份/计数、当前环境逐tensor复现通过；不代表测试集或生物结论已验证）
- Version Label: fusion-v1

## 在正式运行前固定的设计

问题：在当前 H1 hESC 单基因 CRISPRi、未见扰动验证划分上，联合冻结 scGPT 和
Geneformer 的表示能否改善预测？本轮依次运行 Concat、Gated、双向 gene-token
Cross-Attention。每种架构仅采用原主组的均值 MSE 损失、种子 42/43/44，不根据
首个种子的结果改变其余模型/种子配置，不追加损失或超参数搜索。

150 个训练靶点、50 个验证靶点、18,080 个输出基因，训练控制抽样、表达变换、
官方评价版本和完整 98,927 个验证细胞与 baseline-v1 一致。两编码器永久冻结。
沿用 batch=16、32 个控制细胞、AdamW lr=0.001 / weight_decay=0.0001、最多120轮、
patience=20、梯度裁剪1、bf16预测头；固定256个训练对照上的验证 pseudobulk MAE
选 checkpoint。附加 Energy MMD 评价与旧组相同，不用于选择。正式测试集不读取。

## 两路输入和融合头

完全复用已有 scgpt-frozen-v1、geneformer-frozen-v1 的来源校验与特征。
两路各有固定训练对照 context 和靶点特征（预训练基因 embedding、34维训练描述、
coverage 标记）。控制索引与基因列表必须相同。缺失靶点沿用各编码器既定回退规则。

每路分别计算 `u = GELU(P_context(LayerNorm(context)) + P_target(target))`，维度64。

- Concat：`concat(u_sc, u_gf) -> Linear(128,64) -> GELU`。
- Gated：`alpha = sigmoid(Linear(concat(u_sc,u_gf),1))`，
  `fused = alpha*u_sc + (1-alpha)*u_gf`。门控随靶点条件变化；context固定，
  没有逐细胞状态响应。权重不是因果贡献。
- Cross-Attention：将两路 contextualized gene-token prototypes 分别投影为64维，
  加对应的条件向量，执行一层双向4头 attention、残差与LayerNorm，按有效token
  均值池化，再用与Concat相同的 `128 -> 64` 压缩。

三组共同接 `64 -> 256 -> 64 -> 18,080`，GELU/dropout=0.1，最终解码器零初始化。
预测仍是共同条件位移，沿用 clipping、library normalization 与独立对照细胞异质性。
没有学习完整单细胞分布。训练头参数量不完全相等，正式报告记录差异；若有收益，
不能单凭本轮把它归因于融合机制、互补信息或预训练的独立因果效应。

## Cross-Attention token 的身份与缓存

使用与单模型相同的2,048个训练 non-targeting 对照、固定256个候选输入。
scGPT按原分箱顺序和同一局部RNG重提最后层token；Geneformer使用原缓存的真实
rank序列、长度、padding mask，绝不在送入编码器前重排。模型分开加载，float32、
eval/no_grad。提取后非padding池化应与既有逐细胞embedding最大误差不超过1e-5。

仅在编码完成后按真实基因ID聚合：scGPT每个可识别输入基因在2048个对照上的token
均值；Geneformer按非padding且该基因有表达的对照token均值，并保存观察次数。
canonical token别名合并到第一个候选基因索引。保存每路output gene index、symbol、
有效mask和模型来源。两路允许不同基因集合和顺序；attention不逐位置配对，padding
在输入、key/value和pooling中隔离。不存在“两个已池化向量的attention”冒充基因交互。

这是**按基因汇总的训练对照token原型**，不是逐细胞cross-attention；会丢失
细胞之间的协变结构。它额外保留了单模型池化时丢弃的信息，且模型容量不同，故
Cross-Attention与Concat/Gated也不能作为严格隔离attention机制的消融比较。

独立缓存 `data/processed/fusion-tokens-v1` 仅保存小型原型、ID和mask，不存完整
细胞×基因token矩阵。manifest绑定原资产/单模型cache/提取脚本与数组SHA256。
旧cache、checkpoint、成绩及无扰动对照全部保留。

## 运行、复现与比较

```powershell
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe scripts/run_fusion.py --phase smoke
./.venv/Scripts/python.exe scripts/check_fusion_features.py
./.venv/Scripts/python.exe scripts/run_fusion.py --phase all
./.venv/Scripts/python.exe scripts/check_fusion_reproducibility.py
./.venv/Scripts/python.exe scripts/verify_fusion_artifacts.py
```

可分组运行：`scripts/run_fusion.py --phase all --models concat`，随后把模型名换成
`gated`、`cross_attention`；九次正式结果齐全后才生成完整比较报告。

训练/评价串行运行，每个子进程30分钟硬超时，保存日志并监控进程；失败先诊断，
不隐瞒错误或更换正式配置重试。smoke独立命名，不进入正式比较。
每组seed42在独立目录重训，比较全部state_dict tensor、最佳epoch和特征来源。
正式九次运行均检查全部基因/细胞/靶点、聚合分数、相同控制抽样、临时预测清理、
checkpoint哈希，并核验旧42个checkpoint未变。

结果位于 `results/fusion`。主表与四个单模型均值损失组三种子及固定无扰动对照比较。
MAE/PDS/DES分别解读。逐靶点配对区间使用三种子均值后对50个靶点做5000次bootstrap，
只是对当前拟合模型、固定控制抽样和该验证靶点集合的条件性区间；非独立生物重复，
不充当多重比较校正后的显著性检验，不外推至其它细胞类型或测试成绩。

## 实际完成

2026-10-15：9/9次正式训练及完整官方验证完成；30项测试、Ruff检查通过。
三个模型seed42重训的所有state_dict张量、最佳epoch和特征来源完全一致。
9个新增运行、21个主比较运行和42个旧checkpoint核验通过；临时预测矩阵已清理。
原生token身份、观察次数和86,363个Geneformer padding槽独立核对通过。
详细结果与条件性解释见[融合报告](../results/fusion/README.md)。
