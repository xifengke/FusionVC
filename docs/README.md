# 文档导航

## 第一次阅读

| 文档 | 用途 |
| --- | --- |
| [项目首页](../README.md) / [English](../README.en.md) | 项目定位、方法流程、核心结果 |
| [定位与局限](project-scope.md) | 学习项目的范围、不能支持的结论、统计解释 |
| [资源与数据](resources-and-data.md) | 显存、RAM与磁盘的区别，完整测试集实际使用情况 |
| [复现说明](reproduce.md) | 源码检查、推理、完整实验和本机验收的不同前提 |
| [Student推理](student-inference.md) | 模型包、输出格式和使用限制 |
| [发布说明](publication.md) | 公开内容、未包含的大文件、作者和许可事项 |

## 完整成绩

- [最终测试](../results/final-test/README.md)：八组主比较、蒸馏同架构对照、实际Teacher与区间。
- [详细实验总览](experiment-overview.md)：完整配置、验证成绩与测试汇总。
- [融合与MMD](../results/fusion-mmd/README.md)：七架构×三损失×三种子的验证矩阵。
- [知识蒸馏](../results/distillation/README.md)：训练规则、KD=0公平性、重跑和成本。
- [均值融合](../results/fusion/README.md)：融合定义、门控、原生token身份与来源。
- [无扰动及原四模型](../results/controls/README.md)：基础对照。

## 阶段协议与来源记录

按实现顺序为 [baseline](baseline-protocol.md)、[MMD](mmd-protocol.md)、
[scGPT](scgpt-protocol.md)、[Geneformer](geneformer-protocol.md)、[fusion](fusion-protocol.md)、
[fusion MMD](fusion-mmd-protocol.md)、[distillation](distillation-protocol.md)、[final test](final-evaluation-protocol.md)。
阶段协议中的预先固定规则和当时状态，与后续执行完成状态区分阅读。

[初始资源校验](asset-validation.json)、[训练/验证资源来源](download-status.json)、
[源码来源](source-provenance.json)、[测试来源](test-download-status.json)、
[最终本机核验](project-completion-verification.json)保留历史出处。
其中绝对路径只记录原运行环境，不要求读者在新机器创建相同路径。
`VERIFIED`是相应内部核验状态，不是科学结论或官方赛事认证。

## 历史材料

[可行性评估](feasibility.md)、[最初下载计划](download-plan.md)、
[前期下载完成记录](download-summary.md)、[环境搭建记录](environment-notes.md)
保留早期计划及阶段状态。过往“尚未训练”“测试集待下载”等句子不描述最终完成状态。
最终事实以[项目状态](project-status.md)和[最终测试报告](../results/final-test/README.md)为准。
