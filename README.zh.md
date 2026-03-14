# MGTEval

![MGTEval workflow](./workflow.png)

<p align="center">
  <a href="./README.md"><strong>English</strong></a> |
  <a href="./README.zh.md"><strong>中文</strong></a>
</p>

**宣传视频**: [https://www.youtube.com/watch?v=1CVoGQFW4KU](https://www.youtube.com/watch?v=1CVoGQFW4KU)  
**项目主页**: [http://uncoverai.cn](http://uncoverai.cn)

这里是我们 `ACL 2026 Demo` 投稿 `MGTEval: An Interactive Platform for Systematic Evaluation of Machine-Generated Text Detectors` 与 `ICLR 2026` 论文 [Learning From Dictionary: Enhancing Robustness of Machine-Generated Text Detection in Zero-Shot Language via Adversarial Training](https://openreview.net/forum?id=bTcFHJo1Zk) 的官方代码仓库。  
如需查看 ICLR 2026 方法 **TASTE** 的实现，请导航到 [`src/detectors/finetuned/TASTE/README.md`](src/detectors/finetuned/TASTE/README.md)。  
本仓库介绍了 MGTEval：一个用于机器生成文本检测系统化评测的交互式平台，覆盖数据构建、数据攻击、检测器训练与性能评估的一体化工作流。  
该工作主要由西安交通大学（XJTU）的 `Yuanfan Li` 与 `Qi Zhou` 在 `Xiaoming Liu` 教授指导下完成，两位作者的贡献比例约为 70% 与 30%。

---

## 目录

- [Features / 特性](#特性)
- [Introduction / 引言](#引言)
- [Supported detectors / 支持的检测器](#支持的检测器)
- [Quick Start / 快速开始](#快速开始)
- [Workflow and examples / 工作流与示例](#工作流与示例)
- [License / 许可证](#许可证)
- [Citation / 引用](#引用)

---

## 特性

- **一体化流程**：将数据构建、数据攻击、检测器训练与性能评估整合到同一工作流中。  
- **丰富检测器生态**：支持 25+ 指标型与模型型检测器，包括 Binoculars、DetectGPT、GLTR、Fast-DetectGPT、DNA-GPT、DeTeCtive、MPU、PECOLA、TASTE 等。  
- **全面攻击能力**：内置 12+ 文本攻击类型，包括 span 扰动、释义改写、拼写扰动、同义替换、回译与 humanize。  
- **完整评测指标**：支持 Accuracy、F1、AUROC、AUPR、ECE、Brier、TPR@FPR、risk-coverage、bootstrap CI 与 ASR 鲁棒性分析。  
- **细粒度分析**：支持按语言、域来源、模型来源、文本长度分组统计，并输出曲线图、图表与样本级预测结果。  
- **灵活执行形态**：同时提供 CLI 与 Web UI，并通过 WebSocket 实时显示 Build/Attack/Train/Detect/Demo 日志。  
- **实用模型访问**：支持 Hugging Face、本地 checkpoint、镜像端点与 ModelScope 缓存。  
- **可复现实验产物**：自动保存 run config、manifest、summary、checkpoint、curve、plot 与结构化 JSON 结果。

## 引言

我们提出了 MGTEval，一个可扩展的机器生成文本检测系统化评测平台。现有 MGT 检测评测通常分散在数据集、预处理、攻击方式与评测指标上，难以进行公平对比与稳定复现。MGTEval 将流程统一为四个模块：**数据构建（Dataset Building）**、**数据攻击（Dataset Attack）**、**检测器训练（Detector Training）**、**性能评估（Performance Evaluation）**。平台支持通过可配置 LLM 构建自定义基准、对测试集施加多种文本攻击、在统一接口下训练检测器，并输出有效性、鲁棒性与效率相关结果。

---

## 支持的检测器

### 指标型检测器

| 检测器 | Key | 论文 | 会议 | 链接 | 简介 |
|---|---|---|---|---|---|
| Binoculars | binoculars | Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text | ICML 2024 | [link](https://arxiv.org/abs/2401.12070) | 通过比较两个语言模型（观察者与执行者）的交叉困惑度之比来零样本识别机器生成文本。 |
| DetectGPT | detectgpt | DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature | ICML 2023 Oral | [link](https://arxiv.org/abs/2301.11305) | 对输入文本进行扰动并测量概率曲率——机器生成的文本倾向于占据对数概率空间的局部极大值。 |
| DNA-DetectLLM | dnadetectllm | DNA-DetectLLM: Unveiling AI-Generated Text via a DNA-Inspired Mutation-Repair Paradigm | NIPS 2025 Spotlight | [link](https://openreview.net/forum?id=yQoHUijSHx) | 采用DNA启发的突变-修复范式：对文本词元进行突变，观察语言模型的修复能力来区分人类与AI文本。 |
| DNA-GPT | dnagpt | DNA-GPT: Divergent N-Gram Analysis for Training-Free Detection of GPT-Generated Text | ICLR 2024 | [link](https://arxiv.org/abs/2305.17359) | 通过分歧N元组分析检测GPT生成文本——比较原始文本与重新生成版本之间的N-gram分歧模式。 |
| Entropy | entropy | N/A | N/A | N/A | 测量每个词元位置的预测熵均值——机器生成文本通常呈现更低的熵分布模式。 |
| Fast-DetectGPT | fastdetectgpt | Fast-DetectGPT: Efficient Detection of Machine-Generated Text via Sampling Discrepancy | ICLR 2024 | [link](https://arxiv.org/abs/2310.05130) | 通过条件概率曲率估计替代昂贵的扰动采样，大幅加速DetectGPT的检测流程。 |
| GLTR | gltr | GLTR: Statistical Detection and Visualization of Generated Text | ACL 2019 | [link](https://arxiv.org/abs/1906.04043) | 通过语言模型逐词排名统计（top-k分桶计数）的可视化与聚合来区分人类与机器文本。 |
| LASTDE | lastde | Training-free LLM-generated Text Detection by Mining Token Probability Sequences | ICLR 2025 | [link](https://openreview.net/forum?id=vo4AHjowKi) | 通过挖掘词元概率序列中的统计模式来免训练检测LLM生成文本。 |
| LASTDE++ | lastdepp | Training-free LLM-generated Text Detection by Mining Token Probability Sequences | ICLR 2025 | [link](https://openreview.net/forum?id=vo4AHjowKi) | LASTDE的增强版本，采用改进的概率序列挖掘方法和额外的统计特征以提升检测能力。 |
| Likelihood | likelihood | N/A | N/A | N/A | 计算参考语言模型下每个词元的平均对数概率，作为基线检测信号。 |
| LogRank | logrank | N/A | N/A | N/A | 对逐词排名取对数后求平均，提供比原始排名更稳健的基线检测指标。 |
| LRR | lrr | DetectLLM: Leveraging Log Rank Information for Zero-Shot Detection of Machine-Generated Text | EMNLP 2023 Findings | [link](https://arxiv.org/abs/2306.05540) | 结合对数似然与对数排名信息进行零样本检测，利用两种互补的统计信号。 |
| NPR | npr | DetectLLM: Leveraging Log Rank Information for Zero-Shot Detection of Machine-Generated Text | EMNLP 2023 Findings | [link](https://arxiv.org/abs/2306.05540) | 通过在嵌套上下文中归一化预测概率比来零样本检测机器生成文本。 |
| RAIDAR | raidar | Raidar: geneRative AI Detection viA Rewriting | ICLR 2024 | [link](https://arxiv.org/abs/2401.12970) | 通过改写输入文本并比较语义相似度来检测AI生成文本——人类文本被改写后变化更大。 |
| Rank | rank | N/A | N/A | N/A | 利用每个词元的平均预测排名作为检测统计量——机器文本通常具有更低的平均排名。 |
| TOCSIN | tocsin | Zero-Shot Detection of LLM-Generated Text using Token Cohesiveness | EMNLP 2024 | [link](https://arxiv.org/abs/2409.16914) | 通过测量词元聚合性（token-level预测的一致性）来零样本识别LLM生成文本。 |

### 模型型检测器

| 检测器 | Key | 论文 | 会议 | 链接 | 简介 |
|---|---|---|---|---|---|
| CoCo | coco | CoCo: Coherence-Enhanced Machine-Generated Text Detection Under Low Resource With Contrastive Learning | EMNLP 2023 | [link](https://aclanthology.org/2023.emnlp-main.1005/) | 在低资源场景下，将文本连贯性信号融入对比学习框架以增强检测效果。 |
| DeTeCtive | detective | DeTeCtive: Detecting AI-generated Text via Multi-Level Contrastive Learning | NIPS 2024 | [link](https://arxiv.org/abs/2410.13964) | 采用多层次对比学习（词、句、篇章级别）学习细粒度表征，用于AI生成文本检测。 |
| Finetuned Detector | finetuned | N/A | N/A | N/A | 加载本地微调的分类检查点作为检测器——支持任何HuggingFace兼容模型。 |
| GREATER | greater | Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training | ACL 2025 | [link](https://arxiv.org/abs/2502.12734) | 通过对抗训练增强检测器对文本扰动攻击的鲁棒性——以铁磨铁策略提升防御能力。 |
| ImBD | imbd | Aligning Machine Stylistic Preference for Machine-Revised Text Detection | AAAI 2025 Oral | N/A | 通过对齐机器文体偏好来处理检测数据中的类不平衡问题——先模仿，后检测。 |
| Longformer | longformer | Longformer: Long-Document Transformer | N/A | N/A | 利用Longformer的全局注意力架构对长文档进行人类/机器生成文本分类。 |
| MPU | mpu | Multiscale Positive-Unlabeled Detection of AI-Generated Texts | ICLR 2024 Spotlight | [link](https://arxiv.org/abs/2305.18149) | 通过多尺度正样本-无标签学习处理短文本检测任务，即使没有标注负样本也能有效工作。 |
| PECOLA | pecola | Does DETECTGPT Fully Utilize Perturbation? Bridging Selective Perturbation to Fine-tuned Contrastive Learning Detector would be Better | ACL 2024 | [link](https://arxiv.org/abs/2402.00263) | 将选择性扰动与微调对比学习相结合，改进了DetectGPT对扰动信息的利用方式。 |
| RADAR | radar | RADAR: Robust AI-Text Detection via Adversarial Learning | NIPS 2023 | [link](https://arxiv.org/abs/2307.03838) | 联合训练改写检测器与改写生成器，构建对常见规避攻击的鲁棒检测能力。 |
| TASTE | taste | Learning From Dictionary: Enhancing Robustness of Machine-Generated Text Detection in Zero-Shot Language via Adversarial Training | ICLR 2026 | N/A | 通过词典驱动的对抗训练增强跨语言机器生成文本检测的鲁棒性。训练仅使用英文数据；多语言数据集用于评测。 |

---

## 快速开始

### 使用 conda

步骤 1：创建环境
```bash
conda create -n mgteval python=3.12 -y
```

步骤 2：激活环境
```bash
conda activate mgteval
```

步骤 3：升级 pip
```bash
pip install -U pip
```

步骤 4：可编辑模式安装 MGTEval
```bash
pip install -e .
```

步骤 5：验证安装
```bash
mgteval-cli --help
mgteval-cli list
```

步骤 6：启动服务
```bash
./start_dev.sh
```

### 使用 venv

步骤 1：创建虚拟环境
```bash
python -m venv .venv
```

步骤 2：激活虚拟环境
```bash
source .venv/bin/activate
```

步骤 3：升级 pip
```bash
pip install -U pip
```

步骤 4：可编辑模式安装 MGTEval
```bash
pip install -e .
```

步骤 5：验证安装
```bash
mgteval-cli --help
mgteval-cli list
```

---

## 工作流与示例

本节按完整流程介绍如何从数据构建到最终评测。请直接使用 `examples` 目录中的 YAML 配置文件。

### 第一步：从人类文本构建数据集

示例文件：
- `examples/build/build_dataset.yaml`

命令：
```bash
mgteval-cli build examples/build/build_dataset.yaml
```

该步骤会：
- 读取输入数据集中的人类文本；
- 基于标签构建提示词；
- 使用配置的后端生成机器文本；
- 输出成对的人类/机器数据。

### 第二步：对数据集施加攻击

示例文件：
- `examples/attack/build_attack_dataset.yaml`

命令：
```bash
mgteval-cli attack examples/attack/build_attack_dataset.yaml
```

该步骤会：
- 读取第一步输出数据或任意已有数据；
- 按配置施加文本攻击；
- 生成用于鲁棒性评估的攻击数据集。

### 第三步：训练检测器

#### 分支 A：指标型训练

示例文件：
- `examples/train/binoculars.yaml`

命令：
```bash
mgteval-cli train examples/train/binoculars.yaml
```

该步骤会：
- 在训练集上运行检测器；
- 拟合校准器，将分数映射到概率；
- 保存校准结果用于后续评估。

#### 分支 B：模型型训练

示例文件：
- `examples/train/coco.yaml`

命令：
```bash
mgteval-cli train examples/train/coco.yaml
```

该步骤会：
- 在训练集上微调检测器；
- 保存 checkpoint 与训练摘要。

### 第四步：执行检测评估

#### 分支 A：指标型评估

示例文件：
- `examples/detect/binoculars.yaml`

命令：
```bash
mgteval-cli detect examples/detect/binoculars.yaml
```

该步骤会：
- 在测试集上运行评估；
- 输出指标与曲线；
- 可选评估攻击数据集上的鲁棒性。

#### 分支 B：模型型评估

示例文件：
- `examples/detect/coco.yaml`

命令：
```bash
mgteval-cli detect examples/detect/coco.yaml
```

该步骤会：
- 加载训练好的 checkpoint；
- 在测试集上评估；
- 输出指标与曲线。

---

## 许可证

本仓库采用 [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0) 开源协议。

## 引用

如果你觉得我们的 **TASTE** 工作有帮助，欢迎引用：

```text
@inproceedings{
    li2026learning,
    title={Learning From Dictionary: Enhancing Robustness of Machine-Generated Text Detection in Zero-Shot Language via Adversarial Training},
    author={Yuanfan Li and Qi Zhou and Zexuan Xie},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=bTcFHJo1Zk}
}
```
