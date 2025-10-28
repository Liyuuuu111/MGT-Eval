# mgt_eval/detectors/pretrained/pretrained.py
"""
Pretrained detector：
  - 自动适配三类模型：
      * SequenceClassification：直接输出“AI 类别”的概率
      * MaskedLM（BERT/Roberta 等）：用伪似然 (PLL) 近似平均 NLL
      * CausalLM（GPT/LLaMA/Qwen 等）：按自回归平均 NLL
  - 当 model_path 指向 .pt/.pth/.bin/.ckpt/.safetensors 等**权重文件**时：
      * 不再从该路径读取 config；
      * 而是用 tokenizer_path（或 ckpt_base）构建 SeqCls 模型（num_labels=ckpt_num_labels），
        再把 checkpoint 的 state_dict 加载（strict=False，自动清理前缀）。
  - 新增：LoRA 适配器目录自动识别与加载（AutoPeftModel* 优先；回退 PeftModel.from_pretrained），
          安全加载包装（优先 safetensors；若 torch<2.6 且仅 .bin 则报可操作错误），
          因果族识别仅对明确的 CausalLM 模型尝试因果头，
          对 MLM 的 PLL 评分过程增加 tqdm 进度条。
"""

from typing import List, Optional, Dict, Any
import os
import torch
import numpy as np
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
)
from packaging.version import parse as _V
from tqdm.auto import tqdm

from ..base import DetectorBase
from ..registry import register

# --- PRDetect shim (optional) ---
try:
    # 需要你已经放好了 mgt_eval/detectors/finetuned/prdetect.py（或我之前给你的单文件）
    from mgt_eval.detectors.finetuned.prdetect import PRDetectDetector as _PRDShim
except Exception:
    _PRDShim = None

# --- PEFT (LoRA) optional imports (with explicit error capture) ---
_PEFT_IMPORT_ERROR = None
try:
    from peft import PeftModel  # 最小依赖：即便 AutoPeft* 导入失败也尽量保住它
    try:
        from peft import (
            AutoPeftModelForCausalLM,
            AutoPeftModelForMaskedLM,
            AutoPeftModelForSequenceClassification,
        )
    except Exception as _e_auto:
        AutoPeftModelForCausalLM = None
        AutoPeftModelForMaskedLM = None
        AutoPeftModelForSequenceClassification = None
        _PEFT_IMPORT_ERROR = _e_auto  # 记录 AutoPeft* 的具体异常
except Exception as _e_peft:
    PeftModel = None
    AutoPeftModelForCausalLM = None
    AutoPeftModelForMaskedLM = None
    AutoPeftModelForSequenceClassification = None
    _PEFT_IMPORT_ERROR = _e_peft

def _is_prdetect_dir(path: str) -> bool:
    if not isinstance(path, str) or not os.path.isdir(path):
        return False
    cands = [
        os.path.join(path, "prdetect_gcn.pt"),
        os.path.join(path, "best", "prdetect_gcn.pt"),
        os.path.join(path, "last", "prdetect_gcn.pt"),
    ]
    return any(os.path.isfile(p) for p in cands)


def _is_ckpt_file(path: str) -> bool:
    if not isinstance(path, str):
        return False
    if not os.path.isfile(path):
        return False
    ext = os.path.splitext(path)[-1].lower()
    return ext in {".pt", ".pth", ".bin", ".ckpt", ".safetensors"}


def _is_lora_dir(path: str) -> bool:
    """简单判定 LoRA 适配器目录：含 adapter_config.json 或 adapter_model.(safetensors|bin) 即认定"""
    if not isinstance(path, str) or not os.path.isdir(path):
        return False
    return any(
        os.path.isfile(os.path.join(path, fn))
        for fn in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")
    )


def _load_state_dict(ckpt_path: str, device: str = "cpu"):
    # 支持 torch 与 safetensors
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file as load_sft
        sd = load_sft(ckpt_path, device=device)
    else:
        # 注意：这条路径会使用 torch.load（非 transformers 的安全检查）
        sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    # 清理常见前缀
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        new_sd[k] = v
    return new_sd


def _basename_like_id(p: str) -> str:
    """将本地路径或 HF id 统一取最后一段作为“名称”"""
    if not p:
        return "unknown"
    p = p.strip("/").strip("\\")
    base = os.path.basename(p)
    if "/" in p and base == "":
        base = p.split("/")[-1]
    return base or p


def _path_has_safetensors(model_path: str) -> bool:
    if not model_path:
        return False
    if os.path.isdir(model_path):
        try:
            for fn in os.listdir(model_path):
                if fn.endswith(".safetensors"):
                    return True
        except Exception:
            pass
        return False
    return str(model_path).endswith(".safetensors")


def _torch_too_old_for_bin() -> bool:
    try:
        ver = _V(torch.__version__.split("+")[0])
        return ver < _V("2.6")
    except Exception:
        return False


def _from_pretrained_safe(factory, model_id_or_path: str, **kwargs):
    """
    统一安全加载：
      - 若本地目录不存在 *.safetensors 且 torch>=2.6，则直接允许加载 .bin（use_safetensors=False）
      - 否则先尝试 use_safetensors=True；如果因为缺少 safetensors 报 OSError，再自动回退到 .bin
      - 若 torch<2.6 且只能 .bin，则给出清晰可操作错误
    """
    kw = dict(kwargs or {})

    # 情况 1：本地目录，且没有 safetensors
    local_dir = os.path.isdir(model_id_or_path)
    has_sft = _path_has_safetensors(model_id_or_path) if local_dir else False

    if local_dir and (not has_sft):
        if _torch_too_old_for_bin():
            # torch<2.6 不能安全加载 .bin
            raise RuntimeError(
                "[pretrained] 检测到 PyTorch 版本 < 2.6 且当前本地目录无 *.safetensors，"
                "Transformers 出于安全原因（CVE-2025-32434）禁止加载 .bin。\n"
                "解决方案：升级 PyTorch 至 2.6+，或将权重转换为 safetensors。"
            )
        # torch>=2.6：允许 .bin 直接加载
        kw.setdefault("use_safetensors", False)
        return factory.from_pretrained(model_id_or_path, **kw)

    # 情况 2：未知/远端 或 本地存在 safetensors
    kw_try_sft = dict(kw)
    kw_try_sft.setdefault("use_safetensors", True)
    try:
        return factory.from_pretrained(model_id_or_path, **kw_try_sft)
    except OSError as e:
        # 没有 safetensors 文件，尝试回退 .bin（仅 torch>=2.6）
        msg = str(e).lower()
        missing_sft = ("no file named" in msg and "safetensors" in msg) or ("safetensors" in msg and "not found" in msg)
        if missing_sft:
            if _torch_too_old_for_bin():
                raise RuntimeError(
                    "[pretrained] 未找到 safetensors 且当前 PyTorch < 2.6，无法安全回退到 .bin。\n"
                    "请升级 PyTorch 至 2.6+ 或提供 safetensors 权重。"
                ) from e
            kw_try_bin = dict(kw)
            kw_try_bin["use_safetensors"] = False
            return factory.from_pretrained(model_id_or_path, **kw_try_bin)
        raise
    except ValueError as e:
        # transformers 的安全检查：要求 torch>=2.6（即使 weights_only=True）
        msg = str(e)
        needs_new_torch = ("upgrade torch to at least v2.6" in msg) or ("require users to upgrade torch to at least v2.6" in msg)
        if needs_new_torch:
            has_sft_any = _path_has_safetensors(model_id_or_path) if os.path.isdir(model_id_or_path) else False
            if _torch_too_old_for_bin() and not has_sft_any:
                raise RuntimeError(
                    "[pretrained] 仅 .bin 权重且 PyTorch < 2.6。出于安全原因被拒绝加载。\n"
                    "解决：升级 PyTorch 至 2.6+ 或将权重转换为 safetensors。"
                ) from e
        raise


@register("pretrained")
class PretrainedDetector(DetectorBase):
    """
    Args:
        model_path: HF 模型 id/目录，或权重文件 (.pt/.pth/.bin/.ckpt/.safetensors) 或 LoRA 目录
        tokenizer_path: HF tokenizer id/目录（默认同 model_path；LoRA 时建议指向基座）
        name: 预训练/基座模型名（用于元数据与输出文件名）
        device: "cuda" 或 "cpu"（默认自动）
        max_length: 截断/填充长度
        fp16: 在 CUDA 上使用半精度
        pll_stride: MLM 伪似然的并行掩码步长
        ai_label_id: 序列分类时“AI 类别”的 id（默认 1）
        ckpt_num_labels: 当以 checkpoint 文件加载 SeqCls 模型时使用的类别数（默认 2）
        ckpt_base: 当以 checkpoint 文件加载或 LoRA 加载时用于构建 Config/基座的模型 id/目录（默认用 tokenizer_path）
        show_progress: 在 MLM PLL 评分过程中显示进度条（默认 True）
        detector_type: 标注类型（默认 "Model-based"）
    """

    def __init__(
        self,
        model_path: str = "gpt2",
        tokenizer_path: Optional[str] = None,
        name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 512,
        fp16: bool = True,
        pll_stride: int = 64,
        ai_label_id: int = 1,
        ckpt_num_labels: int = 2,
        ckpt_base: Optional[str] = None,
        show_progress: bool = True,
        detector_type: Optional[str] = "Model-based",
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            name=name,
            device=device,
            max_length=max_length,
            fp16=fp16,
            pll_stride=pll_stride,
            ai_label_id=ai_label_id,
            ckpt_num_labels=ckpt_num_labels,
            ckpt_base=ckpt_base,
            show_progress=show_progress,
            **kwargs,
        )
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.ckpt_base = ckpt_base or self.tokenizer_path
        self._prdetect_shim = None
        self.detector_type = detector_type or "Model-based"
        self._is_lora = False  # 是否 LoRA 适配器目录

        # 名称推断
        if name:
            inferred = name
        elif _is_ckpt_file(model_path):
            base = _basename_like_id(self.ckpt_base)
            if base and base.lower() != _basename_like_id(model_path).lower():
                inferred = base
            else:
                inferred = os.path.splitext(os.path.basename(model_path))[0]
        else:
            inferred = _basename_like_id(model_path)

        self.name = inferred
        self.DETECTOR_NAME = self.name

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = int(max_length)
        self.fp16 = bool(fp16)
        self.pll_stride = max(1, int(pll_stride))
        self.ai_label_id = int(ai_label_id)
        self.ckpt_num_labels = int(ckpt_num_labels)
        self.show_progress = bool(show_progress)
        self.detector_type = detector_type or "Model-based"

        self._config = None
        self._tokenizer = None
        self._model = None
        self._kind = None  # "causal" | "mlm" | "seqcls" | "prdetect"

    # --------------------- Load ---------------------
    def load(self):
        # ---------- NEW: PRDetect shim 入口 ----------
        if _PRDShim is not None and _is_prdetect_dir(self.model_path):
            # 将 tokenizer_path 复用为 roberta_path；若没给则再由 prdetector 自己去 summary 里找或回退 roberta-base
            roberta_path = self.tokenizer_path or self.model_path
            self._prdetect_shim = _PRDShim(
                model_path=self.model_path,
                roberta_path=roberta_path,
                max_length=self.max_length,
                device=self.device,
            )
            self._prdetect_shim.load()
            self._kind = "prdetect"
            self.detector_type = "Graph-based (PRDetect via Pretrained shim)"
            self.is_loaded = True
            return

        # 先加载 tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True)

        # 强制确保有 pad_token & pad_token_id，并固定右侧 padding（Qwen 系列更稳）
        if self._tokenizer.pad_token_id is None:
            if getattr(self._tokenizer, "eos_token", None) is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
                self._tokenizer.pad_token_id = self._tokenizer.convert_tokens_to_ids(self._tokenizer.pad_token)
            else:
                # 实在没有 eos，就动态新增一个 [PAD]
                self._tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self._tokenizer.pad_token_id = self._tokenizer.convert_tokens_to_ids(self._tokenizer.pad_token)

        # Qwen/causal 家族通常右侧 padding 更安全
        try:
            self._tokenizer.padding_side = "right"
        except Exception:
            pass

        # ---------- NEW: LoRA 适配器目录 ----------
        if _is_lora_dir(self.model_path):
            if (AutoPeftModelForCausalLM is None) and (PeftModel is None):
                raise RuntimeError(
                    "[pretrained] 检测到 LoRA 目录，但导入 peft 失败。\n"
                    f"具体异常：{repr(_PEFT_IMPORT_ERROR)}\n"
                    "请确认 `pip show peft` 在当前环境可见，并尝试升级：\n"
                    "  pip install -U peft transformers accelerate\n"
                )

            # 基座用 ckpt_base（若未显式给出，则退回 tokenizer_path）
            base_id = self.ckpt_base or self.tokenizer_path
            if not base_id:
                raise RuntimeError(
                    "[pretrained] LoRA 加载需要指定 ckpt_base 或 tokenizer_path 作为基座模型。"
                )

            # 读取基座 config 决定采用哪类头
            try:
                base_cfg = AutoConfig.from_pretrained(base_id)
            except Exception as e:
                raise RuntimeError(
                    f"[pretrained] 无法从基座 '{base_id}' 读取 config，请确认其为有效的 HF 目录/ID（含 config.json）。"
                ) from e

            def _has_arch_suffix_base(suffix: str) -> bool:
                archs = set(getattr(base_cfg, "architectures", []) or [])
                return any(a.endswith(suffix) for a in archs)

            model_type = (getattr(base_cfg, "model_type", "") or "").lower()
            CAUSAL_FAMILIES = {
                "gpt2", "gptj", "gpt_neo", "gpt-neox", "mpt", "llama", "falcon",
                "qwen", "qwen2", "phi", "mistral", "mixtral", "gemma", "xlnet"
            }

            self._is_lora = True

            # 优先尝试 AutoPeftModel* 直接从 LoRA 目录加载（更稳，能自动匹配任务类型）
            def _autopeft_try():
                if AutoPeftModelForSequenceClassification and (
                    _has_arch_suffix_base("ForSequenceClassification") or getattr(base_cfg, "num_labels", None)
                ):
                    return AutoPeftModelForSequenceClassification.from_pretrained(
                        self.model_path, base_model_name_or_path=base_id
                    ), "seqcls"
                if AutoPeftModelForMaskedLM and model_type in {
                    "bert", "roberta", "xlm-roberta", "deberta", "deberta-v2",
                    "electra", "albert", "camembert", "distilbert", "flaubert",
                    "bart", "t5"
                }:
                    return AutoPeftModelForMaskedLM.from_pretrained(
                        self.model_path, base_model_name_or_path=base_id
                    ), "mlm"
                if AutoPeftModelForCausalLM and model_type in CAUSAL_FAMILIES:
                    return AutoPeftModelForCausalLM.from_pretrained(
                        self.model_path, base_model_name_or_path=base_id
                    ), "causal"
                return None, None

            model, kind = _autopeft_try()
            if model is None:
                # 回退：用 _from_pretrained_safe 先把基座按类别加载，再套 PeftModel.from_pretrained(lora_dir)
                if _has_arch_suffix_base("ForSequenceClassification") or getattr(base_cfg, "num_labels", None):
                    base_model = _from_pretrained_safe(AutoModelForSequenceClassification, base_id)
                    kind = "seqcls"
                elif model_type in {
                    "bert", "roberta", "xlm-roberta", "deberta", "deberta-v2",
                    "electra", "albert", "camembert", "distilbert", "flaubert",
                    "bart", "t5"
                }:
                    base_model = _from_pretrained_safe(AutoModelForMaskedLM, base_id)
                    kind = "mlm"
                elif model_type in CAUSAL_FAMILIES:
                    base_model = _from_pretrained_safe(AutoModelForCausalLM, base_id)
                    kind = "causal"
                else:
                    # 兜底：再尝试 SeqCls → MLM
                    try:
                        base_model = _from_pretrained_safe(AutoModelForSequenceClassification, base_id)
                        kind = "seqcls"
                    except Exception:
                        base_model = _from_pretrained_safe(AutoModelForMaskedLM, base_id)
                        kind = "mlm"

                if PeftModel is None:
                    raise RuntimeError(
                        "[pretrained] 需要 peft.PeftModel 来装载 LoRA 适配器，请 `pip install peft`。"
                    )
                model = PeftModel.from_pretrained(base_model, self.model_path)

            self._config = base_cfg
            self._model = model
            self._kind = kind

            # ★ 在这里插入（同步 pad_token_id）
            try:
                if getattr(self._model.config, "pad_token_id", None) is None:
                    self._model.config.pad_token_id = self._tokenizer.pad_token_id
            except Exception:
                pass

            # 精度与设备
            if self.fp16 and self.device.startswith("cuda"):
                try:
                    self._model.half()
                except Exception:
                    pass
            self._model.to(self.device)
            self._model.eval()
            self.is_loaded = True
            return

        # ---------- 分支 1：若是 checkpoint 权重文件（.pt/.bin/...） ----------
        if _is_ckpt_file(self.model_path):
            # 用 ckpt_base 读取 Config 并构建 SeqCls 模型
            try:
                self._config = AutoConfig.from_pretrained(self.ckpt_base)
            except Exception as e:
                raise RuntimeError(
                    f"无法从 ckpt_base='{self.ckpt_base}' 读取 Config。"
                    f" 请将 ckpt_base 指向一个包含 config.json 的 HF 目录或模型 id，"
                    f"通常可与 tokenizer_path 一致。"
                ) from e

            # 确保 num_labels
            self._config.num_labels = getattr(self._config, "num_labels", self.ckpt_num_labels)
            if self._config.num_labels != self.ckpt_num_labels:
                self._config.num_labels = self.ckpt_num_labels

            # 构建空白 SeqCls 模型（不访问网络）
            self._model = AutoModelForSequenceClassification.from_config(self._config)

            # 加载 checkpoint 权重
            sd = _load_state_dict(self.model_path, device="cpu")
            self._model.load_state_dict(sd, strict=False)

            self._kind = "seqcls"

        # ---------- 分支 2：常规 HF 模型目录/id ----------
        else:
            # 根据 config 决定采用哪类模型头
            self._config = AutoConfig.from_pretrained(self.model_path)

            def _has_arch_suffix(suffix: str) -> bool:
                archs = set(getattr(self._config, "architectures", []) or [])
                return any(a.endswith(suffix) for a in archs)

            model_type = (getattr(self._config, "model_type", "") or "").lower()

            # 仅对明确的因果族尝试 CausalLM
            CAUSAL_FAMILIES = {
                "gpt2", "gptj", "gpt_neo", "gpt-neox", "mpt", "llama", "falcon",
                "qwen", "qwen2", "phi", "mistral", "mixtral", "gemma", "xlnet"
            }

            # 1) 优先 SeqCls
            if _has_arch_suffix("ForSequenceClassification") or getattr(self._config, "num_labels", None):
                try:
                    self._model = _from_pretrained_safe(AutoModelForSequenceClassification, self.model_path)
                    self._kind = "seqcls"
                except Exception:
                    self._model = None

            # 2) 其次 MaskedLM（BERT/Roberta/XLM-R/DeBERTa/DistilBERT/BART/T5 等）
            if self._model is None and model_type in {
                "bert", "roberta", "xlm-roberta", "deberta", "deberta-v2",
                "electra", "albert", "camembert", "distilbert", "flaubert",
                "bart", "t5"
            }:
                try:
                    self._model = _from_pretrained_safe(AutoModelForMaskedLM, self.model_path)
                    self._kind = "mlm"
                except Exception:
                    self._model = None

            # 3) 最后 CausalLM（仅明确因果族）
            if self._model is None and model_type in CAUSAL_FAMILIES:
                try:
                    self._model = _from_pretrained_safe(AutoModelForCausalLM, self.model_path)
                    self._kind = "causal"
                except Exception:
                    self._model = None

            # 4) 兜底：再尝试 SeqCls → MLM
            if self._model is None:
                try:
                    self._model = _from_pretrained_safe(AutoModelForSequenceClassification, self.model_path)
                    self._kind = "seqcls"
                except Exception:
                    self._model = _from_pretrained_safe(AutoModelForMaskedLM, self.model_path)
                    self._kind = "mlm"

        # ★ 在这里插入（同步 pad_token_id）
        try:
            if getattr(self._model.config, "pad_token_id", None) is None:
                self._model.config.pad_token_id = self._tokenizer.pad_token_id
        except Exception:
            pass
        # 精度与设备
        if self.fp16 and self.device.startswith("cuda"):
            try:
                self._model.half()
            except Exception:
                pass
        self._model.to(self.device)
        self._model.eval()
        self.is_loaded = True

    # --------------------- Scoring Dispatcher ---------------------
    @torch.no_grad()
    def score_batch(self, texts: List[str]) -> np.ndarray:
        if getattr(self, "_kind", None) == "prdetect" and self._prdetect_shim is not None:
            return self._prdetect_shim.score_batch(texts)
        if self._kind == "seqcls":
            return self._score_seqcls(texts)
        elif self._kind == "mlm":
            return self._score_mlm_pll(texts)
        else:
            return self._score_causallm(texts)

    # --------------------- SequenceClassification ---------------------
    def _score_seqcls(self, texts: List[str]) -> np.ndarray:
        toks = self._tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)

        outputs = self._model(**toks)
        logits = outputs.logits  # (B, C)
        num_labels = logits.shape[-1]
        if num_labels == 1:
            probs_ai = torch.sigmoid(logits.squeeze(-1))
        else:
            probs = torch.softmax(logits, dim=-1)
            ai_id = self.ai_label_id
            # 如果 config 的 id2label 有提示，则尝试自动识别
            id2label = getattr(self._config, "id2label", None)
            if isinstance(id2label, dict):
                for k, v in id2label.items():
                    if str(v).lower() in {"ai", "machine", "fake", "generated", "mgt", "llm"}:
                        ai_id = int(k)
                        break
            ai_id = max(0, min(ai_id, num_labels - 1))
            probs_ai = probs[:, ai_id]

        return probs_ai.detach().float().cpu().numpy()

    # --------------------- CausalLM: mean NLL ---------------------
    @torch.no_grad()
    def _score_causallm(self, texts: List[str]) -> np.ndarray:
        toks = self._tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)

        input_ids = toks["input_ids"]
        attention_mask = toks["attention_mask"]

        # labels 与 input_ids 对齐；pad 位置用 -100 忽略
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, T, V)

        # 下移对齐（预测下一个 token）
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_attn = (shift_labels != -100)

        # 对 -100 的位置用 0 占位索引，随后用 mask 清零
        safe_labels = shift_labels.masked_fill(~shift_attn, 0)

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_logp = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
        token_logp = token_logp * shift_attn.float()

        lengths = shift_attn.float().sum(dim=1).clamp_min(1.0)
        mean_nll = -(token_logp.sum(dim=1) / lengths)

        # NLL 越低 => 分数越高（更像 AI）
        scores = 1.0 / (1.0 + mean_nll)
        return scores.detach().float().cpu().numpy()

    # --------------------- MaskedLM: pseudo log-likelihood (PLL) ---------------------
    @torch.no_grad()
    def _score_mlm_pll(self, texts: List[str]) -> np.ndarray:
        assert self._tokenizer.mask_token_id is not None, "MaskedLM 需要 tokenizer.mask_token_id"
        toks = self._tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)

        input_ids = toks["input_ids"]
        attention_mask = toks["attention_mask"]
        B, T = input_ids.size()
        mask_id = self._tokenizer.mask_token_id

        # 排除特殊 token（如 [CLS],[SEP],[PAD]）的位置
        special_ids = set(
            tid for tid in [
                getattr(self._tokenizer, "cls_token_id", None),
                getattr(self._tokenizer, "sep_token_id", None),
                getattr(self._tokenizer, "pad_token_id", None),
                getattr(self._tokenizer, "bos_token_id", None),
                getattr(self._tokenizer, "eos_token_id", None),
            ] if tid is not None
        )

        scores: List[float] = []
        # 加推理进度条（按样本）
        iterator = range(B)
        if self.show_progress and B > 1:
            iterator = tqdm(iterator, desc="PLL", leave=False, dynamic_ncols=True)

        for b in iterator:
            ids = input_ids[b]
            attn = attention_mask[b]
            pos = torch.nonzero(attn == 1, as_tuple=False).squeeze(-1)
            if pos.numel() == 0:
                scores.append(float(0.5))
                continue

            if special_ids:
                mask_non_special = torch.ones_like(ids, dtype=torch.bool)
                for sid in special_ids:
                    mask_non_special &= (ids != sid)
                pos = pos[mask_non_special[pos]]
                if pos.numel() == 0:
                    scores.append(float(0.5))
                    continue

            total_logp = 0.0
            count = 0

            # 分块掩码，避免超显存；这里不再嵌套进度条，避免过多刷新
            for start in range(0, pos.numel(), self.pll_stride):
                chunk_pos = pos[start:start + self.pll_stride]
                csz = chunk_pos.numel()
                ids_exp = ids.unsqueeze(0).repeat(csz, 1)
                attn_exp = attn.unsqueeze(0).repeat(csz, 1)

                arange_c = torch.arange(csz, device=ids.device)
                ids_exp[arange_c, chunk_pos] = mask_id

                outputs = self._model(input_ids=ids_exp, attention_mask=attn_exp)
                logits = outputs.logits
                log_probs = torch.log_softmax(logits, dim=-1)

                orig_tokens = ids[chunk_pos]
                logp = log_probs[arange_c, chunk_pos, orig_tokens]
                total_logp += float(logp.sum().detach().cpu())
                count += int(csz)

            mean_nll = - (total_logp / max(count, 1))
            score = 1.0 / (1.0 + mean_nll)
            scores.append(float(score))

        return np.array(scores, dtype=np.float32)
