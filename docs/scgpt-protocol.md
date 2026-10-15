# 实验 03：冻结 scGPT 特征迁移（scgpt-v1）

日期：2026-10-08。使用原八组设计中的第三组 Frozen scGPT。
共享 baseline-v1 的监督数据、共同条件位移输出和选择规则；附加两种 MMD 损失。

## 预先固定比较

主比较：MLP、随机初始化 Transformer、Frozen scGPT，均使用 unpaired pseudobulk MSE。
附加比较：同一个 scGPT 预测头分别使用 Energy MMD、MSE + 0.1 × Energy MMD。
种子 42/43/44，scGPT 新增 9 次正式实验，复用已完成的 18 次 MLP/Transformer 损失对照。
不会覆盖旧实验和报告，不按本轮结果重新选择输入或调整损失权重。

150 个官方训练靶点，50 个官方验证靶点，测试集留到后续模型全部完成。
全部 18,080 个输出基因、每靶点官方完整预测细胞数及官方 MAE/PDS/DES 保持一致。
AdamW lr=0.001、weight_decay=0.0001、batch=16、32 个训练控制细胞、MMD 真实集合 32 细胞、
最多 120 epochs、patience=20、gradient_clip=1、bf16 预测头训练均沿用原预算。
所有组按固定 256 个训练对照的验证 pseudobulk MAE 选最佳 checkpoint。
MMD 真实细胞从全训练细胞独立抽样，不存在同一细胞扰动前后的配对。
损失公式、附加 64+64 细胞 Energy MMD 与随机数规则见 [mmd-v1](mmd-protocol.md)。

## 权重与官方实现适配

使用已下载 `models/scgpt/whole-human/{best_model.pt,args.json,vocab.json}`，
每次准备/使用缓存核对下载记录中的 SHA256，不下载或替换成其他 checkpoint。
官方 [scGPT 仓库](https://github.com/bowang-lab/scGPT) 将 Flash/PyTorch 作为可选运行后端。
本项目对照固定论文版本 `74603ca5d8a2adf2a7420936c73784a1fef01597` 的本地源码。

适配器保留 512 维、12 层、8 heads、FFN 512、ReLU、post-norm、gene embedding +
连续值编码，没有位置编码。对 Flash 权重按官方方式 `Wqkv. → in_proj_` 逐 tensor 转换。
153 个 encoder/value_encoder/transformer_encoder 张量严格加载；缺层或 shape 不符立即失败。
未使用原始重构解码器、MVC decoder 和不参与该 `_encode` 前向的 flag_encoder。
整个编码器参数禁止梯度，永久 eval，特征提取使用 float32，不启用 dropout。

`scripts/check_scgpt_parity.py` 对真实权重运行未修改的官方 `TransformerModel._encode`，
并对相同随机数比较官方表达分箱。结果保存在 `results/scgpt/implementation-parity.json`。
不安装不参与计算的旧版 torchtext/FlashAttention，也不使用伪造随机权重作为回退。

## 输入、覆盖和缓存

仍由训练对照方差选取原固定 256 个输入基因；其中 242 个在 checkpoint 词表中。
缺失 14 个的名称写入特征 manifest；仅编码可识别输入，不猜测别名或静默改词表。
逐细胞对这 242 个基因的非零连续 log-expression 按官方分位数规则分 51 bins：
0 留给零表达，非零为 1–50，重复分位点用固定本地 RNG seed=2025 均匀分摊。
这里固定保留零表达 gene token，不追加 CLS；checkpoint 本身记录 `no_cls=true`。
归一化沿用原训练 scale，分箱为逐细胞秩/分位操作，输出始终保持连续表达而不是 bin 编号。

从训练 non-targeting 控制矩阵以 seed=2025 无放回抽 2,048 个细胞，与基线描述拟合的控制抽样一致。
每个细胞对最终全部可识别 gene token 均值池化，再对 2,048 个细胞均值，得到固定 512 维 context。
这是实际完整 scGPT encoder 前向提取的控制细胞表示，不仅是静态词表查表。
缓存同时保存独立细胞 embedding 和索引，方便后续模型使用，但本轮监督头只用均值 context。
模型输入只由训练对照和预训练资源产生，不读取验证表达拟合输入，不读取测试文件。

每个靶点输入为冻结 GeneEncoder（embedding + LayerNorm）的 512 维基因表示、
原训练对照拟合的 34 维描述和一个词表覆盖标记，共 547 维。
全部 150 训练靶点及 50 验证靶点有预训练 gene embedding；未见验证靶点无需未训练的 ID 参数。
缺失靶点的统一回退为零预训练 embedding + 原训练描述 + coverage=0；本轮未触发。
整个输出基因列表仍有 240 个不被词表识别，它们由监督解码器预测，不删除。

context `512 → 64` 与 target `547 → 64` 各经可训练投影，拼接后 `128 → 256 → 64`，
GELU/dropout，最后共用 `64 → 18,080` 的零初始化位移解码器。
按原 clipping/library renormalization 将同一条件位移作用于独立训练控制集合。
可训练参数略不同，报告逐次参数量；不声称参数预算完全相同或完全隔离预训练的因果效果。
输出仍为共同条件位移，没有引入状态依赖响应或单细胞配对监督。

特征缓存 `data/processed/scgpt-frozen-v1` 包含权重/config/vocab SHA256、输入来源、代码 hash、
控制矩阵大小与时间戳、特征数组 hash、覆盖和成本记录；完成标记最后写入。
训练 checkpoint 引用该来源记录，评价前重新核对，来源变化或数组损坏立即失败。
编码器仅运行一次，特征提取时间/显存与九次预测头优化成本分开报告。

## 运行与交付

```powershell
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe scripts/check_scgpt_parity.py
./.venv/Scripts/python.exe scripts/run_scgpt.py --phase smoke
./.venv/Scripts/python.exe scripts/run_scgpt.py --phase all
./.venv/Scripts/python.exe scripts/check_scgpt_reproducibility.py
./.venv/Scripts/python.exe scripts/verify_scgpt_artifacts.py
```

可单独选择 `prepare/train/evaluate/report`；完成实验核对配置、特征和 checkpoint 后复用。
小规模训练输出独立写入 `outputs/smoke_03_scgpt*`，不计入正式成绩。
结果见 [scGPT 比较报告](../results/scgpt/README.md)，旧报告继续保留。
这些是当前 H1 CRISPRi 验证结果，不能称为测试成绩、历史排行榜总分或通用虚拟细胞能力。

## 本轮完成情况

9/9 次 scGPT 训练与完整验证已完成，三架构 × 三损失共 27 次正式比较。
18 项测试通过；官方完整 encoder 和分箱一致性检查、三种损失 seed 42 的逐 tensor 复现通过。
九次评分的控制抽样与原基线一致，全部小型评分产物与 checkpoint 校验通过，原 24 个 checkpoint 未变。
主组 MAE=0.028064，较 MLP 降低约 0.30%，但按验证靶点重采样的 MAE 差值区间包含 0。
纯 MMD 组 MAE=0.027938，较自身均值组降低约 0.45%，DES/PDS 未同步改善。
三组仍未超过无扰动 MAE 对照；这些结论仅适用于本轮固定的冻结表示、共同条件位移和选择规则。
