# FusionVC 项目状态

更新：2026-10-15。原定八组模型的训练、完整官方验证、知识蒸馏、独立测试与推理包均已完成。

| 阶段 | 状态 | 规模及证据 |
| --- | --- | --- |
| 数据与环境 | 完成 | 固定版本、SHA256/CRC、独立GPU环境、来源记录 |
| MLP、随机Transformer | 完成 | 每架构三损失×三种子；原六个基线另行保留 |
| 冻结scGPT、Geneformer | 完成 | 每架构三损失×三种子；官方实现/词表/永久冻结核验 |
| Concat、Gated、Cross-Attention | 完成 | 每架构三损失×三种子；原生geneID、padding、三均值与六MMD组精确复现 |
| 知识蒸馏 | 完成 | 三Student完整验证；KD=0精确复现原MLP；KD seed42精确重训；成本实测 |
| Student独立推理 | 完成 | 固定seed42、约26.3MB、CPU演示与原模型输出精确一致 |
| 最终保留测试 | 完成 | 冻结27个checkpoint+一次无扰动对照，170,846细胞×18,080基因、100个测试扰动 |
| 最终汇总与完整性验收 | 完成 | 全部评分、来源、逐靶点聚合、共享抽样、历史快照及独立推理核验通过 |

七架构×三损失×三种子共63个验证比较，另有3个正式蒸馏Student。
融合MMD阶段18次新增正式运行与六次seed42重跑已完成。
Teacher依九个融合组的三种子完整验证MAE均值选择，固定使用Concat energy seed42。
Student的软标签仅来自150个训练靶点，KD系数固定1.0，没有测试回流训练或模型选择。

验证集上，蒸馏MLP的MAE=0.027756±0.000114，相对未蒸馏MLP改善约1.40%；
PDS=0.515733±0.001007，DES=0.108059±0.001013，并未观察到同步改善。
实际单Teacher的MAE=0.027640；无扰动对照MAE=0.027547。
以上为验证成绩。独立测试已另行完成：Student MAE=0.026509±0.000061，
相对原MLP mean的0.026957±0.000330改善约1.66%；PDS基本持平、DES下降。
Cross-Attention的测试MAE均值最低（0.026463），但八组神经模型均值都未低于
无扰动MAE=0.026342；Transformer的DES最高（0.169593）。完整取舍与区间见最终报告。

本地69项测试与Ruff通过。融合MMD与蒸馏的Material Passport均为VERIFIED，
仅指来源、完整产物和复现核验，不代表生物学结论已获验证。
蒸馏前69个checkpoint及567个历史结果文件保持原SHA256，早期阶段报告保留原路径。

最终选择在访问测试表达前已冻结，保留七架构各自验证选择的整组三种子，
三Student、原MLP mean三种子蒸馏对照和一次无扰动对照；不按测试成绩挑选模型。
最终test产物单独保存并标记test_used=true，历史验证报告的test_used=false仍指当时边界。

结果入口：[最终独立测试](../results/final-test/README.md)、[全项目验收](project-completion-verification.json)、
[完整验证矩阵](../results/fusion-mmd/README.md)、[蒸馏与成本](../results/distillation/README.md)、
[冻结选择](final-model-selection.json)、[推理说明](student-inference.md)、[复现步骤](reproduce.md)。

代码仍为共同条件位移结构，异质性主要来自训练对照；并未实现逐细胞状态依赖响应。
本项目限定于H1 hESC单基因CRISPRi，不声明通用虚拟细胞能力。
GitHub尚未发布，远端Actions尚未运行；本地检查不冒充远端CI。
