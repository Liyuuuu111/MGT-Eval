# MGTEval: 一个统一的MGT检测器评估框架

欢迎使用MGTEval的0.0.1测试版本！本测试版本旨在测试所有基于逻辑的检测器（见[支持的检测器](#支持的检测器)部分）的逻辑、功能与跨平台迁移能力，以及包体能否在不同的电脑环境下安装。要完成测试，请详细阅读[检测器与接口](#检测器与接口)部分以调用合法的接口完成测试。

## 支持的检测器

目前主要测试对15种基于逻辑的检测器的支持情况。所有检测器如下：

| 检测器                   | 仓库/链接                                                                                                                   | 来源会议 / 年份              | 复现完成时间 |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------- | ---------------------- | ------ |
| GLTR                  | [HendrikStrobelt/detecting-fake-text](https://github.com/HendrikStrobelt/detecting-fake-text)                           | ACL 2019               |        |
| Entropy               | [YichenZW/Robust-Det](https://github.com/YichenZW/Robust-Det)                                                           | ACL 2019                      |        |
| Likelihood               | [YichenZW/Robust-Det](https://github.com/YichenZW/Robust-Det)                                                           | ACL 2019                      |        |
| DetectGPT             | [eric-mitchell/detect-gpt](https://github.com/eric-mitchell/detect-gpt)                                                 | ICML 2023              |        |
| Rank                  | [YichenZW/Robust-Det](https://github.com/YichenZW/Robust-Det)                                                           | Findings of EMNLP 2023 |        |
| LogRank               | [YichenZW/Robust-Det](https://github.com/YichenZW/Robust-Det)                                                           | Findings of EMNLP 2023 |        |
| DetectLLM-NPR         | [mbzuai-nlp/DetectLLM](https://github.com/mbzuai-nlp/DetectLLM)                                                         | Findings of EMNLP 2023 |        |
| DetectLLM-LRR         | [mbzuai-nlp/DetectLLM](https://github.com/mbzuai-nlp/DetectLLM)                                                         | Findings of EMNLP 2023 |        |
| Fast-DetectGPT        | [baoguangsheng/fast-detect-gpt](https://github.com/baoguangsheng/fast-detect-gpt)                                       | ICLR 2024              |        |
| Binoculars            | [kelvin-hawk/Binoculars](https://github.com/kelvin-hawk/Binoculars)                                                     | ICML 2024              |        |
| RAIDAR                | [cvlab-columbia/RaidarLLMDetect](https://github.com/cvlab-columbia/RaidarLLMDetect)                                     | ICLR 2024              |        |
| DNAGPT                | [Xianjun-Yang/DNA-GPT](https://github.com/Xianjun-Yang/DNA-GPT)                                                         | ICLR 2024              |        |
| TOCSIN   | [TOCSIN](https://github.com/Shixuan-Ma/TOCSIN)                                     | EMNLP 2024               |        |
| Lastde       | [TrustMedia-zju/Lastde_Detector](https://github.com/TrustMedia-zju/Lastde_Detector)                                     | ICLR 2025              |        |
| Lastde++       | [TrustMedia-zju/Lastde_Detector](https://github.com/TrustMedia-zju/Lastde_Detector)                                     | ICLR 2025              |        |


## 安装

### 从wheel安装

```bash
conda create -n test python=3.10 -y         # 创建虚拟环境，您可以将名称换成您喜欢的名字
conda activate test         # 激活虚拟环境，您可以将名称换成您喜欢的名字
python -m pip install mgt_eval-0.0.1-py3-none-any.whl   # 安装
```

### 从pip安装

>测试阶段完成后此安装方式将开放

这会自动下载所有所需的依赖。如果存在依赖冲突，请将该测试结果报告给管理员并自行下载所需包。

### 验证安装

```bash
mgteval list
(或mgt_eval list)
```

看到可用检测器列表即安装成功。

---

## 数据格式与基本约定

* **文件格式**：支持 HC3 风格 `json/jsonl`，或统一的行级 JSONL：
  `{"text": "...", "label": 0|1, ...}`，或Fast DetectGPT测试风格的JSON文件。主流的Benchmark的格式几乎都支持。
  其中 **label: 0=人类文本，1=机器文本**（评测/校准均遵循该约定）。
* **可选分组列**：`lang, source, model, sub_source` 等存在则自动识别，无需手工指定。
* **抽样**：所有示例均提供 `--sample_k`，便于快速冒烟测试；设为 `<=0` 表示全量。

---

## 检测器与接口

> 说明：
>
> * “**所需模型映射**”一列标注了 CLI 统一参数 `--model1 / --model2` 在该检测器中的含义；若无 `model2` 则表示单模型。
> * “**必需参数**”除数据与模型外的**必填**项；“**可选参数**”为常用可调项（均可通过 `--detector_kwargs` JSON 传入，或部分在 CLI 顶层有直达参数，若不传入可选参数，将使用原始论文的默认值）。
> * 记号示例：`max_len`≈`max_token_observed`，同义映射由内部完成；布尔开关若无值可写为 `true/false`。

| 检测器（规范名）          | 所需模型映射                                                        | 必需参数                                                          | 可选参数（常用）                                                                                                                                                         | 备注                                      |
| ----------------- | ------------------------------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| **binoculars**    | `model1=observer`, `model2=performer`                         | —                                                             | `mode`(`low-fpr`/`accuracy`), `max_len`, `prob_slope`, `device`, `use_bfloat16`                                             | -              |
| **detectgpt**     | `model1=score_model`, `model2=mask_model`                     | —                                                             | `pct_words_masked`, `span_length`, `n_perturbations`, `buffer_size`, `mask_top_p`, `max_len`, `prob_slope`, `use_zscore`, `chunk_size`, `device`, `use_bfloat16` | 典型掩码扰动法                                 |
| **fastdetectgpt** | `model1=scoring_model_name`, `model2=sampling_model_name`     | —                                                             | `tokenizer_name`(默认=采样模型), `max_length`, `fp16/bfloat16`, `device`                                                                                               | 采样/评分双模型                                |
| **gltr**          | `model1=score_model`                                          | —                                                             | `max_len`, `device`, `use_bfloat16`                                                                                                          | 统计特征（rank/green-yellow-red）             |
| **lastde**        | `model1=score_model`                                          | —                                                             | `max_len`, `prob_slope`, `embed_size`, `epsilon_mult`, `tau_prime`, `device`, `use_bfloat16`                                                                     | 单模型 DE 统计                               |
| **lastdepp**      | `model1=scoring_model`, `model2=reference_model(可省略=同model1)` | —                                                             | `n_samples`, `embed_size`, `epsilon_mult`, `tau_prime`, `max_len`, `prob_slope`, `device`                                                                        | “++”版含参考采样                              |
| **likelihood**    | `model1=score_model`                                          | —                                                             | `max_len`, `device`, `use_bfloat16`, `threshold`                                                                                                                 | 似然基线                                    |
| **rank**          | `model1=score_model`                                          | —                                                             | `max_len`, `device`, `use_bfloat16`                                                                                                                              | token rank 基线                           |
| **logrank**       | `model1=score_model`                                          | —                                                             | `max_len`, `device`, `use_bfloat16`                                                                                                                              | log-rank 基线                             |
| **lrr**           | `model1=score_model`                                          | —                                                             | `max_len`, `device`, `use_bfloat16`, `k_runs`, `calibrator_path`                                                                                                 | Log-Rank Ratio（支持外部校准器）                 |
| **entropy**       | `model1=score_model`                                          | —                                                             | `use_analytic`, `distrib_params`, `max_length`, `fp16`, `group_cols`                                                                                             | 信息熵基线                                   |
| **dnagpt**        | `model1=score_model`                                          | —                                                          | `dataset_name`(如`squad`/`pubmed`)、`truncate_ratio`,`regen_number`,`max_len`,`do_top_k/p`,`top_k`,`top_p`,`temperature`,`device`,`use_bfloat16`                   | 若复现 PubMed 流程需正确 `dataset_name` 与长度下限   |
| **npr**           | `model1=score_model`, `model2=mask_model`                     | —                                                             | `pct_words_masked`, `span_length`, `n_perturbation`, `chunk_size`, `buffer_size`, `mask_top_p`, `max_len`, `device`, `use_bfloat16`                              | 归一化 log-rank 扰动                         |
| **raidar**        | `model1=rewrite_model`                                        | —                                                             | `calibrate_k`, `calibrate_seed`, `rewrite_input_max_tokens`, `max_new_tokens_factor`, `use_openai`, `openai_model`, `device`                                     | 含小样本自标定与改写器                             |
| **tocsin**        | `model1=score_model`, `model2=reference_model`                | `basemodel`(`Fast`/`lrr`/`likelihood`/`logrank`/`standalone`) , `bart_ckpt`(`path/to/your/bart_model`)| `mask_pct`, `perturb_per_text`, `dataset_file`, `max_len`, `device`, `use_bfloat16`                                                                 | `--basemodel`/`--bart_ckpt`为 CLI 顶层直达参数 |

> **如何传参**
>
> * 统一 CLI：**模型**用 `--model1/--model2`，其他入参统一走 `--detector_kwargs`（JSON 字符串），例如：
>   `--detector_kwargs '{"max_len":512,"mode":"low-fpr"}'`
> * 个别检测器在 CLI 还提供了直达参数（如 TOCSIN 的 `--basemodel`、`--bart_ckpt`），等价于在 JSON 中传递同名键。

---

## 校准（Calibrate）工作流

**动机**：基于逻辑的检测器输出的只有原始分数，这可以自然的求AUROC，但不能自然的进行分类，因为分类需要概率和阈值。我们参考MGTBench中的做法，利用一个小数据集（这里是HC3）对分数（或者输出的多维指标）进行逻辑回归，从而使用逻辑回归自然的输出概率。

**目标**：学习一个一维打分 → 概率的校准映射（缺省为 Platt LR），输出 JSON，后续可在运行时加载以获得可比较的概率。注意，我们已经提供了一个在HC3数据集进行校准的结果，模型会自动加载。您可以在您的数据集上再次运行校准。

### 通用命令模板

```bash
mgteval calibrate \
  --detector <det_name> \
  --data /path/to/train.jsonl \
  --model1 <path-or-hf-id> \
  [--model2 <path-or-hf-id>] \
  [--sample_k 10000] \
  [--batch_size 32] \
  [--l2 1e-2 --max_iter 200 --tol 1e-6] \
  [--out_dir calibration_results] \
  [--detector_kwargs '<JSON for detector extras>'] \
  [--device cuda:0] [--bf16]
```

[ ]包裹的部分为可选字段，如果不写，则使用默认值。
### 典型示例

注意：我们强烈建议您使用本地的模型路径与数据集路径，并使用默认的超参数进行测试。

* **Binoculars**：

```bash
mgteval calibrate \
  --detector binoculars \
  --data /data/hc3/all.jsonl \
  --model1 falcon-7b \
  --model2 falcon-7b-instruct \
  --sample_k 2000 \
```

* **DetectGPT**（掩码扰动）：

```bash
mgteval calibrate \
  --detector detectgpt \
  --data /data/hc3/all.jsonl \
  --model1 gpt2-medium \
  --model2 t5-large \
  --sample_k 2000 \
  --detector_kwargs '{"pct_words_masked":0.3,"span_length":2,"n_perturbations":100,"max_len":256,"use_zscore":true}'
```

* **Fast-DetectGPT**：

```bash
mgteval calibrate \
  --detector fastdetectgpt \
  --data /data/hc3/all.jsonl \
  --model1 falcon-7b \
  --model2 falcon-7b-instruct \
  --sample_k 2000 \
```

* **LRR / Rank / LogRank / Likelihood / Entropy**（单模型基线）：

```bash
mgteval calibrate \
  --detector lrr \
  --data /data/hc3/all.jsonl \
  --model1 /models/gpt2 \
  --sample_k 2000 \
  --detector_kwargs '{"max_len":512}'
```

* **DNAGPT**（指定数据域）：

```bash
mgteval calibrate \
  --detector dnagpt \
  --data /data/hc3 \
  --model1 gpt2 \
  --sample_k 2000 \
```

* **Lastde / Lastde++**：

```bash
# Lastde
mgteval calibrate \
  --detector lastde \
  --data /data/hc3/all.jsonl \
  --model1 /models/gpt-neo-2.7B \
  --sample_k 2000 \
  --detector_kwargs '{"max_len":512,"embed_size":3,"epsilon_mult":10,"tau_prime":5,"prob_slope":-6.0}'

# Lastde++
mgteval calibrate \
  --detector lastdepp \
  --data /data/hc3/all.jsonl \
  --model1 /models/gpt-neo-2.7B \
  --model2 /models/gpt-neo-2.7B \
  --sample_k 2000 \
  --detector_kwargs '{"n_samples":100,"embed_size":4,"epsilon_mult":8.0,"tau_prime":15,"max_len":512}'
```

* **NPR**：

```bash
mgteval calibrate \
  --detector npr \
  --data /data/hc3/all.jsonl \
  --model1 gpt2 \
  --model2 t5-large \
  --sample_k 2000 \
  --detector_kwargs '{"pct_words_masked":0.3,"span_length":2,"n_perturbation":100,"max_len":400}'
```

* **RAIDAR**：

```bash
mgteval calibrate \
  --detector raidar \
  --data /data/hc3/all.jsonl \
  --model1 /models/Llama-3-8B-Instruct \
  --sample_k 2000 \
  --detector_kwargs '{"calibrate_k":32,"calibrate_seed":42,"rewrite_input_max_tokens":400,"max_new_tokens_factor":0.1}'
```

* **TOCSIN**（注意直达参数）：

```bash
mgteval calibrate \
  --detector tocsin \
  --data /data/hc3/all.jsonl \
  --model1 /models/gpt-neo-2.7B \
  --model2 /models/gpt-neo-2.7B \
  --basemodel Fast \
  --bart_ckpt facebook/bart-base \
  --sample_k 2000 \
  --detector_kwargs '{"mask_pct":0.015,"perturb_per_text":10,"max_len":512}'
```

> 产物：`--out_dir` 下生成的 `*.json` 校准文件。不同检测器的加载键可能是 `calibrator` 或 `calibrator_path`，见下文“运行”。

---

## 运行（Run）工作流

**目标**：对测试集执行检测，输出评测统计与（可选）曲线；如已训练校准器，可加载实现可比概率。

### 通用命令模板

```bash
mgteval run \
  --detector <det_name> \
  --data /path/to/test.jsonl \
  --model1 <path-or-hf-id> \
  [--model2 <path-or-hf-id>] \
  --sample_k 1000 \
  --batch_size 8 \
  --threshold 0.5 \
  --out runs/<det_name> \
  --detector_kwargs '<JSON>' \
  [--device cuda:0] [--bf16 true] [--k_runs 1]
```

### 快速示例

* **基线（LogRank）**

```bash
mgteval run \
  --detector logrank \
  --model1 /models/gpt2 \
  --data /data/hc3/all.jsonl \
  --sample_k 2000
```

* **Binoculars**

```bash
mgteval run \
  --detector binoculars \
  --data /data/hc3/all.jsonl \
  --model1 /models/observer-neo-2.7B \
  --model2 /models/performer-neo-2.7B \
  --sample_k 2000
```

* **DetectGPT**

```bash
mgteval run \
  --detector detectgpt \
  --data /data/hc3/all.jsonl \
  --model1 gpt2-medium \
  --model2 t5-large \
  --sample_k 1000 \
  --detector_kwargs '{"pct_words_masked":0.3,"span_length":2,"n_perturbations":5,"max_len":256,"use_zscore":true}'
```

* **Fast-DetectGPT**

```bash
mgteval run \
  --detector fastdetectgpt \
  --data /data/hc3/all.jsonl \
  --model1 falcon-7b \
  --model2 falcon-7b-instruct \
  --sample_k 1000 \
```

* **GLTR**

```bash
mgteval run \
  --detector gltr \
  --data /data/hc3/all.jsonl \
  --model1 /models/gpt2 \
  --sample_k 2000 \
  --detector_kwargs '{"max_len":512}'
```

* **Lastde / Lastde++**

```bash
# Lastde
mgteval run \
  --detector lastde \
  --data /data/hc3/all.jsonl \
  --model1 /models/gpt-neo-2.7B \
  --sample_k 1000 \
  --detector_kwargs '{"max_len":512,"embed_size":3,"epsilon_mult":10,"tau_prime":5}'

# Lastde++
mgteval run \
  --detector lastdepp \
  --data /data/hc3/all.jsonl \
  --model1 /models/gpt-neo-2.7B \
  --model2 /models/gpt-neo-2.7B \
  --sample_k 1000 \
  --detector_kwargs '{"n_samples":100,"embed_size":4,"epsilon_mult":8.0,"tau_prime":15,"max_len":512}'
```

* **NPR**

```bash
mgteval run \
  --detector npr \
  --data /data/hc3/all.jsonl \
  --model1 gpt2 \
  --model2 t5-large \
  --sample_k 1000 \
  --detector_kwargs '{"pct_words_masked":0.3,"span_length":2,"n_perturbation":100,"max_len":400}'
```

* **RAIDAR**

```bash
mgteval run \
  --detector raidar \
  --data /data/hc3/all.jsonl \
  --model1 /models/Llama-3-8B-Instruct \
  --sample_k 500 \
  --detector_kwargs '{"rewrite_input_max_tokens":400,"max_new_tokens_factor":0.1,"calibrate_k":32,"calibrate_seed":42}'
```

* **TOCSIN**

```bash
mgteval run \
  --detector tocsin \
  --data /data/hc3/all.jsonl \
  --model1 /models/gpt-neo-2.7B \
  --model2 /models/gpt-neo-2.7B \
  --basemodel Fast \
  --bart_ckpt facebook/bart-base \
  --sample_k 1000 \
  --detector_kwargs '{"mask_pct":0.015,"perturb_per_text":10,"max_len":512}'
```

---

## 备用：专用脚本入口（Quickstarts & 校准脚本）

除统一 CLI 外，你也可以直接运行仓库内提供的脚本（构造即评测），例如：

* `python quickstart_binoculars.py ...`
* `python quickstart_detectgpt.py ...`
* `python quickstart_fastdetectgpt.py ...`
* `python quickstart_gltr_named.py ...`
* `python quickstart_lastde.py ...` / `python quickstart_lastdepp.py ...`
* `python quickstart_dna_gpt.py ...`
* `python quickstart_tocsin.py ...`
* `python quickstart_raidar_named.py ...`
* 统一校准脚本：`python call_calibrate.py --detector <det> --model1 ... [--model2 ...] --data ... [--detector_args_json ...]`
  该脚本支持把**未知参数**自动解析为 `detector_kwargs`（例如 `--tau_prime 5` 变成 `{"tau_prime":5}`）。

---

## 常见问题（FAQ）

1. **`--model1/--model2` 到底映射到哪个入参？**
   见上表“所需模型映射”。内部构造器会将统一接口映射到各检测器的实际 `__init__` 参数（比如 `observer/performer`、`score_model/mask_model` 等）。

2. **样本抽样与全量评测**
   `--sample_k <= 0` 即全量评测。建议先用小样本做冒烟，再切全量出正式曲线。

3. **半精度/混合精度**
   大多数检测器支持 `--bf16`（统一开关）或在 `--detector_kwargs` 传 `use_bfloat16/fp16`。显存吃紧时可适当降低 `batch_size` 与长度上限。

4. **数据标签方向**
   均假定 `label=0` 为**人类**、`label=1` 为**机器**。如数据源相反，请先转换（或在你的加载器中交换标签）。

---

## 最小工作示例（从零到有）

1. **列出检测器**

```bash
mgteval list
```

2. **对 HC3 子集做校准（LogRank）**

```bash
mgteval calibrate \
  --detector logrank \
  --data /data/hc3/all.jsonl \
  --model1 path/to/your/model \
  --sample_k 2000 \
```

这会自动对HC3数据集进行打分、校准，结果会覆盖到/.local/share/mgt_eval的文件夹下。这一步可以省略，如果不进行校准，我们的校准器会自动将预先校准好的结果拷贝到这个目录。校准结果可以被覆盖，只需再次运行校准命令即可。

3. **加载校准器运行（LogRank 举例）**

```bash
mgteval run \
  --detector logrank \
  --data /data/hc3 \
  --model1 gpt2 \
  --sample_k 2000
```

校准器会自动搜索校准路径，所以在绝大多数的情况下，您无需手动指定校准器的位置。

---

## 您的任务

首先，我谨代表本项目的所有作者，向参加该项目的测试者表达真诚的感谢，您的参与不仅可以大大改善该项目的易用性、功能性，而且也对可信人工智能的发展做出了重要贡献。

您需要完成如下的几项任务：

* 详细阅读项目文档（如上的md内容）
* 使用wheel安装整个项目文件（见[安装](#安装)）部分。您可能可以不使用`conda`进行安装，可以使用`venv`或者尝试在自己的本地安装。
* 项目的基本功能测试：
  * i) 项目能否正常安装、正常使用`mgteval list`调用所有支持模型？
  * ii) 检测器能否通过命令行方式调用与校准？
  * iii) 检测器是否能够通过代码方式调用与校准？（代码调用与校准文件参考zip文件，应该可以直接使用）
  * iv) 