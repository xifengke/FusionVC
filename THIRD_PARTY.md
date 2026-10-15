# 第三方来源与许可边界

本项目实现学习与方法对照流程，不声称拥有下列数据、模型、论文或上游软件。
源码包默认不再分发原始矩阵、预训练权重、上游源码快照、论文PDF或训练对照表达子集。
使用或另行分发这些资源时，应查阅所用固定版本的实际条款和来源说明。

| 资源 | 来源 | 用途与记录 |
| --- | --- | --- |
| VCC2025 / Virtual Cell Atlas | [官方数据仓库](https://github.com/ArcInstitute/arc-virtual-cell-atlas) | H1 hESC训练、验证和测试；[前期记录](docs/download-status.json)、[测试记录](docs/test-download-status.json) |
| scGPT | [官方仓库](https://github.com/bowang-lab/scGPT) | 冻结whole-human表示；[适配协议](docs/scgpt-protocol.md) |
| Geneformer | [官方模型与源码](https://huggingface.co/ctheodoris/Geneformer) | V1-10M及30M词典；[适配协议](docs/geneformer-protocol.md) |
| cell-eval / pdex | [官方评分仓库](https://github.com/ArcInstitute/cell-eval) | 固定cell-eval0.6.6 / pdex0.1.27及历史源码核对；[源码来源](docs/source-provenance.json) |
| Python依赖 | [项目配置](pyproject.toml)、[环境说明](requirements/README.md) | 各依赖遵循各自许可证，环境freeze不等于重新许可 |

## 源码、权重与数据分别核对

本地取得的scGPT源码LICENSE标注MIT，cell-eval历史源码LICENSE标注MIT，Geneformer所取README标注Apache-2.0。
这些是所取材料的记录，不把源码许可证自动扩大为所有数据、权重、字典和派生产物的统一许可。
最终以具体资源及固定修订的条款为准；本页不替代这些条款。

FusionVC自有代码由作者选择采用根目录的 [MIT 许可证](LICENSE)，版权署名为 `Copyright (c) 2026 xifengke`。
这一授权不改变第三方代码、数据、模型权重、字典和其他资源的条款；涉及上游材料时仍须保留其原有声明并遵守对应条款。
[GitHub许可说明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)解释了公开仓库与明确许可的区别。

## 学术来源

- [Virtual Cell Challenge](https://doi.org/10.1016/j.cell.2025.06.008)
- [scGPT](https://doi.org/10.1038/s41592-024-02201-0)
- [Geneformer](https://doi.org/10.1038/s41586-023-06139-9)

本项目的冻结表示和表达解码设定不等于逐项复现上述论文全部实验。
历史阅读和来源检查见[文档导航](docs/README.md)；未取得的全文和未做的原论文实验不表述为已完成。
