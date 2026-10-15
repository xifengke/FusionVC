# FusionVC 前期下载完成记录

历史记录提示：本文件仅描述2026-10-03前期下载。后来已建立GPU环境、完成八组实验并使用完整测试集。
当时80–90GB规划不是全项目实际占用或最低配置保证；当前事实见[资源说明](resources-and-data.md)。

完成日期：2026-10-03。目录：`E:/各种项目/FusionVC/`。

**90GB 空闲空间足够本阶段。已完成前期数据、模型及源码下载，项目实际文件大小合计约 22.86GB（21.29GiB）。** 完成时 E 盘空余约 84.36GB（78.57GiB）。磁盘余量会随其他程序使用而变化。

GB 为十进制；GiB 为二进制，Windows 容量显示通常采用后者。本记录统计文件逻辑大小，不等同于逐簇磁盘占用。

## 已取得的资源

| 资源 | 大小 | 本地位置 |
| --- | ---: | --- |
| VCC 2025 训练矩阵 | 15.4825GB | `data/raw/vcc2025/train/adata_Training.h5ad` |
| VCC 2025 验证矩阵 | 6.9290GB | `data/raw/vcc2025/validation/adata_Validation.h5ad` |
| 官方基因列表、训练/验证扰动细胞数 CSV | 约 120KB | `data/raw/vcc2025/` |
| scGPT whole-human 权重、配置及词表 | 206.70MB | `models/scgpt/whole-human/` |
| Geneformer V1-10M 权重、配置及四个 V1 字典 | 44.61MB | `models/geneformer-v1/` |
| 官方源码快照及归档包 | 72.95MB | `vendor/` |
| 论文资料、基因映射 | 43.11MB | `references/` |
| 独立下载及校验工具环境 | 83.10MB | `.venv-download/` |

核心数据和模型清单共 14 个文件，合计 22,662,903,756 bytes，全部下载完成。其余空间用于源代码、资料、下载工具、日志等。没有未完成的 `.part` 或断点状态文件，也没有解压后重复保存的大型表达矩阵。

源码包括 scGPT 当前版本及 2024 年快照、Geneformer tokenizer/embedding 等必要文件、cell-eval 当前版本及 2025 比赛结束前快照。版本、原始网址和 SHA256 记录在 [source-provenance.json](source-provenance.json) 与 [download-status.json](download-status.json)。Geneformer V1 权重固定于官方修订 `1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5`。

论文资料实际情况：比赛 PDF 已从用户提供的文件复制；scGPT 作者预印本 PDF 已下载。Geneformer 公开 PDF 端点未能成功取得，保留了官方论文及作者稿网页链接，见 [references/README.md](../references/README.md)。因此目前不包含三篇完整 PDF。

## 校验结果

- 比赛矩阵及 CSV 与官方 CRC32C 一致；可用的官方 MD5 也已核对。Geneformer 提供的权重/字典 SHA256 已核对。
- 所有核心文件均核对了大小并生成本地 SHA256。scGPT 公开下载源未提供可核对的官方 checksum，已确认权重归档完整、CPU 可加载，配置和词表可解析。
- Geneformer safetensors 张量结构可读取。
- 训练集：221,273 个细胞 × 18,080 个基因，150 个扰动靶点；验证集：98,927 个细胞 × 18,080 个基因，50 个扰动靶点。细胞数包含各自的非靶向对照。
- 两份 h5ad 均可读取，CSR 稀疏矩阵，基因顺序与官方列表完全一致；所有配套扰动细胞数均吻合，训练与验证扰动靶点没有交叉。
- 表达值抽样检查通过：有限、非负、整数值。此检查为抽样，不代表逐元素扫描。
- scGPT 词表覆盖官方 18,080 个输出基因中的 17,840 个，缺失 240 个已记录。后续需处理输入词表覆盖，输出仍应保留完整比赛基因列表。

详细结果：[asset-validation.json](asset-validation.json)、[scgpt-weight-structure.json](scgpt-weight-structure.json)、[disk-usage.json](disk-usage.json)。

## 后续空间与环境

测试矩阵约 11.95GB，留到最终评估前下载。FASTQ、预训练细胞语料和其他大模型权重不在本阶段下载范围。

本次创建的环境只用于下载和校验。**GPU 训练环境尚未配置，八组实验尚未训练。** 本机现有系统 PyTorch 为 CPU 版，后续需单独配置兼容 RTX 4060 的 GPU 环境，参考 [environment-notes.md](environment-notes.md)。

后续环境预留约 8–12GB，另为安装缓存保留 2–4GB；这是估算，尚未锁定 GPU 依赖。加上测试集、模型输出和受控缓存，完整项目按约 80–90GB 总空间预算管理；不得全量缓存所有细胞的 gene-level embedding。当前余量足以继续前期实现，训练时需限制缓存和 checkpoint 保留数量。

数据、权重、源码快照、环境及缓存已加入 `.gitignore`，后续 GitHub 仓库应提交项目代码、下载脚本、版本清单和成绩报告，避免提交大型资源。
