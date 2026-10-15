# FusionVC 下载清单与磁盘预算

历史规划提示：以下是2026-10-03的下载前预算，不是当前状态。后续八组实验、蒸馏和完整测试均已完成，见[资源与数据](resources-and-data.md)。

核查日期：2026-10-03。范围：VCC 2025、scGPT whole-human、Geneformer V1-10M，以及运行八组实验的必要代码与环境。本文保留下载前的完整项目预算；用户随后授权的前期下载已经完成，实际占用约 22.86GB，详情见 [download-summary.md](download-summary.md)。本阶段未下载测试集，未配置 GPU 训练环境。下文的下载状态表述为预算核查时的记录。

GB/MB 均为十进制：1 GB = 1,000,000,000 bytes。Windows 常用的容量显示相当于 GiB：1 GiB = 1,073,741,824 bytes。

## 必须取得的文件

| 内容 | 文件/范围 | 已核实大小 | 用途 |
| --- | --- | ---: | --- |
| VCC 2025 训练数据 | train/adata_Training.h5ad | 15.482497461 GB | 150 个训练靶点，拟合所有模型 |
| VCC 2025 验证数据 | validation/adata_Validation.h5ad | 6.928967541 GB | 50 个验证靶点，只做调参和模型选择 |
| VCC 2025 测试数据 | test/adata_Test.h5ad | 11.950739168 GB | 100 个测试靶点，最终评估 |
| VCC 2025 配套 CSV | gene_names.csv、三个 pert_counts CSV | 121,719 bytes | 固定输出基因、细胞数与靶点 |
| scGPT whole-human | best_model.pt、args.json、vocab.json | 206.704197 MB | 冻结 scGPT 编码器 |
| Geneformer V1-10M | config.json、model.safetensors、四个 V1 字典 | 44.614732 MB | 冻结原论文对应的小模型与 tokenizer |

比赛三份矩阵与配套 CSV 合计 **34.362325889 GB**。

比赛文件加两模型所需文件，已核实部分合计 **34.613644818 GB（约 32.24 GiB）**。这是已有文件大小的合计，不含环境、论文、源代码或后续实验输出。

三个 h5ad 已通过公开地址的 HEAD 检查，返回 HTTP 200，并核对 Content-Length。因此本次已确认验证和测试数据可公开访问。测试标签虽可下载，仍隔离为最终评估资料，不作为训练输入或标签。

h5ad 直接作为数据文件使用，不需要再解压成一份同大小的副本。内部压缩、预处理重写和密集化属于不同操作，可能额外占空间。

## 来源与选取规则

- VCC：[官方年度数据说明](https://github.com/ArcInstitute/arc-virtual-cell-atlas/blob/main/virtual-cell-challenge/README.md)，公开桶前缀 `https://storage.googleapis.com/arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/`。文件大小来自 Google Cloud Storage JSON 元数据 API 并以 HEAD 复核。
- scGPT：[官方仓库](https://github.com/bowang-lab/scGPT)链接的 [whole-human 文件夹](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y)。大小由公开文件夹文件元数据取得。下载前要再次检查配额与文件版本；本次不代表已经成功传输权重。
- Geneformer：[V1 模型文件](https://huggingface.co/ctheodoris/Geneformer/tree/main/Geneformer-V1-10M)和 [V1 字典目录](https://huggingface.co/ctheodoris/Geneformer/tree/main/geneformer/gene_dictionaries_30m)。选择 safetensors 权重，不再重复下载内容相同的 pytorch_model.bin。
- Geneformer 四个字典分别为 ensembl_mapping_dict_gc30M.pkl、gene_median_dictionary_gc30M.pkl、gene_name_id_dict_gc30M.pkl、token_dictionary_gc30M.pkl，总计 3,430,631 bytes。
- 小型机器可读清单：[download-metadata.json](download-metadata.json)，包含对象名、文件大小、可用校验元数据及 scGPT 文件 ID。真正下载时固定可获得的版本，做断点续传和完整性验证。

另外需要 scGPT、Geneformer 的必要源码、官方 cell-eval 评价代码、基因映射资料以及两篇模型论文的可合法取得版本和补充材料。比赛 PDF 已有，不需要重复下载。这部分按 **不超过约 0.2 GB** 预留，属于预算而非逐文件精确统计；论文全文的合法公开副本是否可取得仍需在下载阶段确认。

## 环境预算：估计，尚未安装核验

建立独立项目环境，通常包括 Python、支持 NVIDIA GPU 的 PyTorch、适配的 transformers、scGPT/Geneformer 运行所需库、anndata、scanpy、h5py、numpy、scipy、pandas、评价与绘图库。版本需按两模型的兼容性锁定，不能直接把所有包升级到最新。

- 安装包下载量：预留 **3–6 GB**。
- 安装后的独立环境：预留 **8–12 GB**。
- 下载/安装缓存及临时文件：额外预留 **2–4 GB**。
- 初始网络下载总量，包括完整比赛数据、模型、小文件及环境：约 **38–41 GB**。
- 初始准备完成的磁盘占用：约 **45–52 GB**，含环境及上述缓存余量。

这些环境数字是保守工程估算，不是已经锁定依赖后得到的安装报告。若包兼容性要求多个独立环境或使用 WSL，还会增加空间；应先验证最小环境，再决定是否需要。当前预算不包含新建 WSL。

## 完成八组实验的额外空间

| 项目 | 预算范围 | 控制方式 |
| --- | ---: | --- |
| token、embedding 及预处理缓存 | 8–15 GB | 缓存可复用对照，分片保存，不全量展开所有细胞的 gene embedding |
| 各模型预测、评价结果 | 5–10 GB | 分模型评估，保存汇总结果，限制同时保留的完整预测矩阵 |
| checkpoint 与日志 | 2–5 GB | 每组保留最佳及必要恢复点，限制逐 epoch 快照 |
| 处理临时文件 | 3–5 GB | 分块读写，避免复制整份数据 |

初始占用加后续工作空间，预计约 **63–87 GB**。建议统一按 **80–90 GB** 预留，且训练时持续检查磁盘余量。这是采用上述缓存与输出保留策略的预算，不是任意保存全部中间结果的上限。

参考：150,000 cells × 1,024 tokens × 768 dimensions × FP16，需要约 235.93 GB；因此不实施对全部训练细胞的此类基因表示全量缓存。网络下载小，不代表所有缓存都小。

## 本机当前容量与建议

核查时 E 盘空余约 **102.70 GB / 95.65 GiB**，C 盘空余约 **54.85 GB / 51.08 GiB**。

E 盘足以先下载全部必要资产并准备环境；按 80–90 GB 总预算进行八组实验，仍有约 12.7–22.7 GB 余量，但需要控制缓存和预测保留量。模型缓存、安装包缓存、项目环境和临时目录尽量放在项目下，避免默认缓存占用 C 盘。

建议下载目标：`E:/各种项目/FusionVC/` 下的 data/raw/vcc2025、models/scgpt、models/geneformer-v1、references、vendor；环境与缓存也尽可能放在该项目内。

不需要取得：官方桶约 **5.810 TB 的 FASTQ 原始测序文件**，scGPT/Geneformer 的千万级预训练语料，Geneformer V2 其他大权重，以及无关的器官模型。原始 FASTQ 大小来自本次桶清单汇总。

下载前用户要求先统计；随后已授权下载前期资源并完成。实际范围及校验结果以 [download-summary.md](download-summary.md) 为准，不递归下载整个数据桶或完整模型仓库。
