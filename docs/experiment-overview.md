# FusionVC 详细实验总览

本文保留完整实验配置和成绩。项目定位与阅读入口见[首页](../README.md)，资源说明见[运行条件与数据](resources-and-data.md)。

面向 VCC2025 未见基因扰动预测的实验项目。八组模型已完成实现、训练和完整官方验证；
七架构 × 三损失 × 三种子共63组对照，另有3个蒸馏 Student。
在访问测试表达前冻结清单后，27个 checkpoint 和一次无扰动对照已完成完整官方测试。

研究限定于 H1 hESC 单基因 CRISPRi。所有成绩来自本机实际运行；模型学习共同条件位移，
不宣称通用虚拟细胞能力。当前目录尚未发布 GitHub，远端 CI 尚未运行。

## 已完成实验

| 模型 | 训练损失 | 正式验证运行数 | 验证选择的最终损失 |
| --- | --- | ---: | --- |
| MLP | mean / energy / hybrid | 9 | energy |
| 随机 Transformer | mean / energy / hybrid | 9 | hybrid |
| 冻结 scGPT | mean / energy / hybrid | 9 | energy |
| 冻结 Geneformer | mean / energy / hybrid | 9 | energy |
| Concat | mean / energy / hybrid | 9 | energy |
| Gated Fusion | mean / energy / hybrid | 9 | energy |
| Cross-Attention | mean / energy / hybrid | 9 | hybrid |
| 蒸馏 MLP | mean MSE + 1.0 × KD | 3 | mean_kd（预先固定） |

mean 比较无配对细胞集合均值；energy 使用负欧氏距离核的 Energy MMD；
hybrid 为均值 MSE + 0.1 × Energy MMD。三种子固定42/43/44。
同一架构的损失对照保持结构、训练预算和选择规则一致。最早六个基线 checkpoint 另行保留，
不当作额外独立重复。

输入使用训练对照拟合的34维基因描述、256候选基因，输出保留18,080个基因。
scGPT 和 Geneformer 完全冻结，两路表示使用相同2,048个训练对照。
Concat/Gated 使用池化表示；Cross-Attention 使用绑定真实基因ID的242/254个基因原型。
融合头容量和信息粒度不同，因此融合比较不能独立隔离机制的作用。

## 蒸馏结果

Teacher 按九个融合组的三种子完整验证 MAE 均值选择，使用胜出组固定seed42，
实际为 **Concat + Energy MMD**。只对150个训练扰动生成软标签，
Student 保持原 MLP 的1,266,144参数及相同训练预算，不使用验证/测试软标签训练。

完整验证集98,927个细胞、50个未见扰动、18,080个基因：

| 方法 | MAE ↓ | PDS ↑ | DES ↑ |
| --- | ---: | ---: | ---: |
| 无扰动对照 | 0.027547 | 0.514400 | 0.000187 |
| 未蒸馏 MLP | 0.028150 ± 0.000330 | 0.515467 ± 0.001514 | 0.110852 ± 0.012764 |
| 蒸馏 MLP | 0.027756 ± 0.000114 | 0.515733 ± 0.001007 | 0.108059 ± 0.001013 |
| 实际 Teacher（固定seed42） | 0.027640 | 0.516800 | 0.111431 |

蒸馏相对同架构纯监督 MLP 的验证 MAE 改善约1.40%，PDS/DES没有显示同步改善。
Student 与 Teacher 的 MAE 仍未低于无扰动对照。不能将 MAE 改善扩大为所有指标或完整
分布预测能力改善；验证集还用于早停与Teacher选择，最终泛化以独立测试报告为准。
详见[蒸馏报告](../results/distillation/README.md)。

## 最终保留测试

实际测试规模为 **170,846个细胞、18,080个基因、100个未见扰动**，其中对照38,176个。
每架构按三种子验证MAE均值选择损失，保留整组三种子，不挑选最佳种子。
三Student、原MLP mean三种子蒸馏对照及一次无扰动对照均完整评分。
测试没有用于训练、损失选择、Teacher选择或超参数调整。

| 模型 | 验证选择的训练损失 | MAE ↓ | PDS ↑ | DES ↑ |
| --- | --- | ---: | ---: | ---: |
| 无扰动对照 | 固定一次抽样 | 0.026342 | 0.509600 | 0.000189 |
| MLP | energy | 0.026853 ± 0.000278 | 0.511533 ± 0.002065 | 0.112435 ± 0.023945 |
| Transformer | hybrid | 0.031355 ± 0.000293 | 0.510033 ± 0.002892 | 0.169593 ± 0.003904 |
| Frozen scGPT | energy | 0.026664 ± 0.000273 | 0.514233 ± 0.004933 | 0.117656 ± 0.018280 |
| Frozen Geneformer | energy | 0.026587 ± 0.000169 | 0.512000 ± 0.002100 | 0.109469 ± 0.010449 |
| Concat | energy | 0.026487 ± 0.000096 | 0.510200 ± 0.000361 | 0.109815 ± 0.007405 |
| Gated Fusion | energy | 0.026492 ± 0.000119 | 0.511700 ± 0.001752 | 0.110926 ± 0.015619 |
| Cross-Attention | hybrid | 0.026463 ± 0.000091 | 0.512367 ± 0.003062 | 0.113918 ± 0.012364 |
| Distilled MLP | mean_kd | 0.026509 ± 0.000061 | 0.511933 ± 0.000764 | 0.114757 ± 0.001295 |

以下是蒸馏的同架构对照，不能用主表中MLP energy替代未蒸馏MLP mean：

| 方法 | MAE ↓ | PDS ↑ | DES ↑ |
| --- | ---: | ---: | ---: |
| 未蒸馏MLP（mean） | 0.026957 ± 0.000330 | 0.511867 ± 0.001940 | 0.121207 ± 0.017274 |
| 蒸馏MLP | 0.026509 ± 0.000061 | 0.511933 ± 0.000764 | 0.114757 ± 0.001295 |
| 实际Teacher（固定seed42，非三种子均值） | 0.026381 | 0.509900 | 0.114143 |

Student相对未蒸馏MLP的测试MAE改善约**1.66%**，100个靶点中87个的三种子平均MAE更低。
对应差值为−0.000448，条件性95%靶点区间[−0.000531, −0.000366]。
PDS基本持平；DES从0.121207降至0.114757，差值区间[−0.012403, −0.000482]。
这表明均值误差改善伴随关键变化基因恢复的取舍，不能表述为全面提升或无损蒸馏。

八组神经模型的MAE均值均高于无扰动对照；Cross-Attention的MAE均值在神经模型中最低，
Transformer的DES均值最高。Frozen scGPT的PDS均值最高；单项排序不构成全面优胜结论。
所有模型DES均高于无扰动对照。配对区间仅对当前拟合模型与固定控制抽样成立，
未做多重比较校正，训练种子不是生物重复。

完整表格、四项逐靶点指标、配对区间及来源核验见[最终测试报告](../results/final-test/README.md)。
[冻结选择](../docs/final-model-selection.json)、[评价协议](../docs/final-evaluation-protocol.md)、
[全项目验收](../docs/project-completion-verification.json)保留完整来源链。

## 独立 Student 推理

已导出并验收约26.3MB的独立包，默认位于 `outputs/deployment/student-v1`。
推理只需包内五个文件与FusionVC运行依赖，不调用Teacher或两个基础编码器。

```powershell
./.venv/Scripts/python.exe scripts/predict_student.py --export --target ACLY --cells 32
```

默认生成 `outputs/demo/student_prediction.npz`，含32×18,080的连续log1p表达、
基因名和尺度元数据。演示使用256个固定训练对照，抽样口径与正式完整评分不同。
用法及输出读取见[推理说明](../docs/student-inference.md)，实际验收见
[推理包校验](../docs/student-demo-verification.json)。

## 复现与核验

实际环境：Windows、Python3.12、PyTorch2.8.0+cu128、RTX4060 Laptop 8GB。
固定cell-eval0.6.6 / pdex0.1.27，报告官方原始MAE、PDS、DES，附加Energy MMD单列，
不拼造历史排行榜总分。评分使用全部细胞和输出基因，DE采用已对照原生实现的
Mann–Whitney流程及全基因×靶点BH；各组共用固定控制抽样。

```powershell
./scripts/setup_training.ps1 -Python 'E:/Python/python.exe'
./.venv/Scripts/python.exe -m pytest -q --basetemp tmp/pytest-local
./.venv/Scripts/python.exe -m ruff check src scripts tests
```

完整下载、训练、复现、冻结、评分与报告入口见[复现说明](../docs/reproduce.md)。
本地69项测试及Ruff通过；新增六个融合MMD组的seed42、KD=0公平性及正式KD seed42
均精确复现。蒸馏前69个checkpoint和567个历史结果文件保持原SHA256。
最终28次评分的完整细胞、基因、靶点、聚合、来源与控制抽样核验通过；临时预测矩阵均已清理。
[最终代码检查](../docs/final-validation-checks.json)与[全项目验收](../docs/project-completion-verification.json)绑定当前代码和产物。
核验范围与资源成本见各阶段报告，不把冻结编码器参数比当作实际推理加速。

## 结果与文件

| 内容 | 入口 |
| --- | --- |
| 八组独立测试、蒸馏同架构对照 | [最终测试报告](../results/final-test/README.md) |
| 七架构、三损失、三种子完整验证矩阵 | [融合与MMD报告](../results/fusion-mmd/README.md) |
| Teacher、Student、未蒸馏对照及实测成本 | [蒸馏报告](../results/distillation/README.md) |
| 原均值融合、门控及原生token身份 | [融合报告](../results/fusion/README.md) |
| 原四单模型与无扰动完整对照 | [对照报告](../results/controls/README.md) |
| 各阶段状态 | [项目记录](../docs/project-status.md) |

`configs/` 固定实验与划分；`src/fusionvc/` 包含模型、训练、评分和报告；
`scripts/` 为下载及运行入口；`tests/` 检查损失、梯度、表达空间、原生实现一致性与来源边界。
`results/` 保存小型成绩、逐靶点结果、图和核验收据；大型 `data/`、`models/`、
`vendor/`、`outputs/` 按忽略规则保留在本机，不混入源码产物。

## 来源

- [VCC2025](https://doi.org/10.1016/j.cell.2025.06.008)
- [scGPT](https://doi.org/10.1038/s41592-024-02201-0)
- [Geneformer](https://doi.org/10.1038/s41586-023-06139-9)
- [官方cell-eval](https://github.com/ArcInstitute/cell-eval)

下载校验与源码版本见 `docs/download-status.json`、`docs/source-provenance.json`。
第三方数据、源码和预训练权重遵循各自条款，本项目不对它们重新许可。
