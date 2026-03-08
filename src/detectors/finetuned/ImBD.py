# mgt_eval/detectors/finetuned/ImBD.py
from __future__ import annotations

import os, json, time, random, platform, shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable

import numpy as np
from tqdm.auto import tqdm
from loguru import logger
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# NEW:
from calibration.runner import (
    Calibrate as _Calibrate,
    _auto_out_path as _calib_auto_out,
    _apply_platt_1d as _apply_platt,
)

# NEW for registry & base
from ..base import DetectorBase
from ..registry import register

from eval.evaluator import evaluate_detector as _evaluate_detector

# ---- （） ----
W_EPOCH = 8
W_MEM   = 8
W_NUM   = 8
W_STEP  = 8
SEP     = " "

# /TF32
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- LoRA（PEFT）----
_PEFT_AVAILABLE = True
try:
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
except Exception:
    _PEFT_AVAILABLE = False

# ：
from data_utils.load import (
    load_dataset_unified_pairs,   # -> (List[Tuple[str,str]], meta)
    load_dataset_unified,         # -> (List[Dict{text,label?...}], group_cols)
)

try:
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            torch.set_float32_matmul_precision('medium')
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

# ---- mgt_eval （//） ----
from train.registry import register_train
from train.train import (
    _reset_and_mark_cuda_peaks,
    _collect_cuda_peaks,
    _save_loss_plot,
    _build_data_info,
)

# ============== （/） ==============
DETECTOR_NAME    = "ImBD-SPO"
detector_type    = "Model-based"
CITATION_AUTHORS = "AAAI 2025 Oral: Imitate Before Detect"
CITATION_TITLE   = "Aligning Machine Stylistic Preference for Machine-Revised Text Detection"
CITATION_LINK    = "N/A"

# ==============  ==============
def _seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---- （ + ）----
import sys

def _get_user_cache_dir(*sub: str) -> str:
    """
    返回跨平台的用户缓存目录，优先使用 platformdirs。
    Linux:  $XDG_CACHE_HOME 或 ~/.cache
    macOS:  ~/Library/Caches
    Windows:%LOCALAPPDATA%
    最后拼接 'mgt_eval' 和可选子目录 *sub。
    """
    base = None
    # 1) （）
    env_override = os.environ.get("MGT_EVAL_CACHE_DIR", None)
    if env_override:
        base = os.path.expanduser(env_override)

    if base is None:
        try:
            from platformdirs import user_cache_dir as _user_cache_dir
            base = _user_cache_dir(appname="mgt_eval", appauthor=False)
        except Exception:
            # platformdirs：
            if sys.platform == "darwin":
                base = os.path.join(os.path.expanduser("~/Library/Caches"), "mgt_eval")
            elif os.name == "nt":
                base = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~\\AppData\\Local")), "mgt_eval")
            else:
                base = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "mgt_eval")

    path = os.path.join(base, *sub) if sub else base
    os.makedirs(path, exist_ok=True)
    return path

def _default_calib_root() -> str:
    """
    校准文件默认根目录：
    1) MGT_EVAL_CALIB_DIR 环境变量（若设置）
    2) 用户缓存目录下 mgt_eval/calibration_results
    """
    env_root = os.environ.get("MGT_EVAL_CALIB_DIR", None)
    if env_root:
        root = os.path.expanduser(env_root)
        os.makedirs(root, exist_ok=True)
        return root
    return _get_user_cache_dir("calibration_results")


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _timestamp() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def _dump_jsonl(path: str, examples: List[Dict[str, Any]]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return path

# imports ：
# from pathlib import Path

# import
import sys
from pathlib import Path

def _user_cache_root() -> Path:
    """跨平台用户缓存根目录，优先环境变量 MGT_EVAL_CALIB_DIR。"""
    env = os.getenv("MGT_EVAL_CALIB_DIR")
    if env:
        return Path(env)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "mgt_eval" / "calibration_results"
    if os.name == "nt":
        base = os.getenv("LOCALAPPDATA", str(Path.home()))
        return Path(base) / "mgt_eval" / "calibration_results"
    base = os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    return Path(base) / "mgt_eval" / "calibration_results"

def _safe_calib_filename(
    detector_name: str,
    checkpoint_dir: str,
    calib_data_path: str,
    sample_k: Optional[int] = None,
    seed: int = 114514,
) -> str:
    ck = Path(checkpoint_dir).name or "model"
    dj = Path(calib_data_path).stem or "data"
    sk = "all"
    if sample_k is not None:
        try:
            sk = str(int(sample_k)) if int(sample_k) > 0 else "all"
        except Exception:
            sk = str(sample_k)
    sd = int(seed) if seed is not None else 114514
    parts = [detector_name, ck, dj, sk, str(sd)]
    return "_".join(parts) + ".json"

# _ensure_calibrator
def _ensure_calibrator(
    checkpoint_dir: str,
    calib_data_path: str,
    detector_name: str = "imbd",
    preferred_root: Optional[str] = None,
) -> str:
    """
    始终生成“文件路径”并传给 Calibrate；若官方失败，回退到本地 Platt，并写入兼容键 (A/B + beta1/beta0)。
    完全避免硬编码绝对路径，落在用户缓存目录。
    """
    root = Path(preferred_root) if preferred_root else _user_cache_root()
    root.mkdir(parents=True, exist_ok=True)

    # ；，
    try:
        target_path = _calib_auto_out(detector_name, checkpoint_dir, None, calib_data_path, None, 114514, str(root))
    except Exception:
        target_path = str(root)

    target = Path(target_path)
    if target.is_dir():
        target = root / _safe_calib_filename(detector_name, checkpoint_dir, calib_data_path)

    # ，
    if target.is_file():
        return str(target)

    # 1)  Calibrate —— “”
    try:
        res = _Calibrate(
            detector=detector_name,
            model1=checkpoint_dir,
            data=calib_data_path,
            out=str(target),
        )
        outp = res.get("path") if isinstance(res, dict) else None
        final_path = Path(outp) if (isinstance(outp, str) and outp) else target
        return str(final_path)
    except Exception as e:
        logger.warning(f"[calib] official calibrator failed; fallback to local Platt. err={e}")

    # 2) ： Platt，
    try:
        # pairs  single（）
        with open(calib_data_path, "r", encoding="utf-8") as f:
            lines = [json.loads(x) for x in f if x.strip()]
        scores, labels = [], []
        # ： ComputeScore.from_bundle
        mdl = ComputeScore.from_bundle(checkpoint_dir, device=str(_device()))
        mdl.eval()
        for ex in lines:
            text = str(ex.get("text", "") or "")
            if not text:
                continue
            sc = mdl.score_single(text)["score"]
            scores.append(float(sc))
            y = ex.get("label")
            labels.append(int(y) if y is not None else 0)

        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int32)
        # Platt： sigmoid(s) ~ y
        # ，L2
        X = np.vstack([scores, np.ones_like(scores)]).T  # [s, 1]
        y = labels.astype(np.float64)
        lam = 1e-6
        A, B = np.linalg.solve(X.T @ X + lam * np.eye(2), X.T @ y)  # y ≈ A*s + B
        payload = {
            "detector": detector_name,
            "checkpoint": checkpoint_dir,
            "data": calib_data_path,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "calibrator": {
                "type": "platt",
                "A": float(A), "B": float(B),
                "beta1": float(A), "beta0": float(B)
            },
            "stats": {
                "n": int(scores.size),
                "pos": int(labels.sum()),
                "neg": int(scores.size - labels.sum())
            }
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return str(target)
    except Exception as e2:
        # ：“”，
        logger.warning(f"[calib] local Platt fallback failed: {e2}")
        return str(target)

def _require_peft():
    if not _PEFT_AVAILABLE:
        raise RuntimeError(
            "LoRA requested but 'peft' is not installed. Please `pip install peft`."
        )

# ==============  ==============
class PairDataset(Dataset):
    def __init__(self, data_json_path: str):
        self.path = data_json_path
        with open(data_json_path, "r", encoding="utf-8") as f:
            data_json = json.load(f)
        self.data = self._process(data_json)

    def _process(self, data_json: Dict[str, Any]) -> Dict[str, List[str]]:
        if "pubmed" in self.path:
            orig = [qa.split("Answer:")[1].strip() for qa in data_json["original"]]
            rew  = [qa.split("Answer:")[1].strip() for qa in data_json["rewritten"]]
        else:
            orig = data_json["original"]
            rew  = data_json["rewritten"]
        assert len(orig) == len(rew), "original 与 rewritten 数量不一致"
        return {"original": orig, "rewritten": rew}

    def __len__(self):
        return len(self.data["original"])

    def __getitem__(self, idx: int):
        return self.data["original"][idx], self.data["rewritten"][idx]

# ============== SPO analytic  &  ==============
def get_sampling_discrepancy_analytic(logits_ref: torch.Tensor,
                                      logits_score: torch.Tensor,
                                      labels: torch.Tensor):
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref   = logits_ref[:, :, :vocab]
        logits_score = logits_score[:, :, :vocab]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref    = torch.softmax(logits_ref,   dim=-1)

    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)  # [B,T]
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref  = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)

    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / (var_ref.sum(dim=-1).sqrt() + 1e-8)
    return discrepancy, log_likelihood.sum(dim=-1)

def calculate_SPO_loss(model_prefered_logprob, model_disprefered_logprob,
                       ref_prefered_logprob, ref_disprefered_logprob,
                       beta=0.5):
    pref_rel = model_prefered_logprob    - ref_prefered_logprob
    disp_rel = model_disprefered_logprob - ref_disprefered_logprob
    reward_accuracies = (pref_rel > disp_rel).float().mean(dim=-1)
    reward_margins    = (pref_rel - disp_rel).mean(dim=-1)
    loss = -F.logsigmoid(beta * (pref_rel - disp_rel)).mean(dim=-1)
    return loss, pref_rel.mean(dim=-1), disp_rel.mean(dim=-1), reward_accuracies, reward_margins

# ：HF  dtype （ deprecated ）
def _hf_from_pretrained(model_dir: str, *, dtype: Optional[torch.dtype] = None):
    kw = {}
    if dtype is not None:
        try:
            kw["dtype"] = dtype
            return AutoModelForCausalLM.from_pretrained(model_dir, **kw)
        except TypeError:
            kw.pop("dtype", None)
            kw["torch_dtype"] = dtype
            return AutoModelForCausalLM.from_pretrained(model_dir, **kw)
    return AutoModelForCausalLM.from_pretrained(model_dir)

# ============== （ --lora ） ==============
class ComputeScore(nn.Module):
    """
    训练期：用 base_model / reference_model 构造；若 --lora 则仅给 scoring_model 注入 LoRA。
    推理/评测期：用 from_bundle(dir) 直接从本地目录加载（model/ + 可选 lora/），
    绝不联网，不使用任何占位符名。
    """
    def __init__(self,
                 scoring_model_name: str,
                 reference_model_name: str,
                 SPO_beta: float = 0.5,
                 device: Optional[str] = None,
                 cache_dir: str = "./models",
                 # LoRA（）
                 use_lora: bool = False,
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.05,
                 lora_target_modules: Optional[str] = None,
                 lora_bias: str = "none",
                 # NEW
                 reference_device: Optional[str] = None,
                 max_length: int = 512,
                 ):
        super().__init__()
        self.device = torch.device(device) if device else _device()
        if reference_device is not None:
            self.ref_device = torch.device(reference_device)
        else:
            self.ref_device = torch.device("cuda:1") if (torch.cuda.is_available() and torch.cuda.device_count() >= 2) else self.device
        self.max_length = int(max_length)

        self.scoring_model_name   = scoring_model_name
        self.reference_model_name = reference_model_name
        self.beta = float(SPO_beta)
        self.cache_dir = cache_dir
        self.use_lora = bool(use_lora)

        # ----  ----
        def _load_tok_and_model(name: str, dev: torch.device):
            tok = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token if getattr(tok, "eos_token", None) is not None else tok.unk_token
            mdl = AutoModelForCausalLM.from_pretrained(name, cache_dir=cache_dir)
            try:
                mdl.config.use_cache = False
            except Exception:
                pass
            mdl.to(dev)
            return tok, mdl

        logger.info(f"Loading scoring_model: {self.scoring_model_name}")
        self.scoring_tokenizer,  self.scoring_model  = _load_tok_and_model(self.scoring_model_name, self.device)
        logger.info(f"Loading reference_model: {self.reference_model_name}")
        self.reference_tokenizer, self.reference_model = _load_tok_and_model(self.reference_model_name, self.ref_device)
        for p in self.reference_model.parameters():
            p.requires_grad = False
        self.reference_model.eval()

        # ---- LoRA（ use_lora ；reference ）----
        self.lora_info: Optional[Dict[str, Any]] = None
        if self.use_lora:
            _require_peft()
            target_modules = self._parse_target_modules_from_str(lora_target_modules, self.scoring_model)
            lcfg = LoraConfig(
                r=int(lora_r), lora_alpha=int(lora_alpha), lora_dropout=float(lora_dropout),
                bias=lora_bias, task_type=TaskType.CAUSAL_LM, target_modules=target_modules
            )
            self.scoring_model = get_peft_model(self.scoring_model, lcfg)
            try:
                self.scoring_model.print_trainable_parameters()
            except Exception:
                pass
            self.lora_info = {
                "enabled": True, "r": int(lora_r), "alpha": int(lora_alpha),
                "dropout": float(lora_dropout), "bias": str(lora_bias),
                "task_type": "CAUSAL_LM", "target_modules": target_modules,
            }

        # NEW: （）
        try:
            if hasattr(self.scoring_model, "gradient_checkpointing_enable"):
                self.scoring_model.gradient_checkpointing_enable()
                if hasattr(self.scoring_model, "enable_input_require_grads"):
                    self.scoring_model.enable_input_require_grads()
        except Exception:
            pass

        self.criterion_fn = get_sampling_discrepancy_analytic
        self.forward = self.forward_SPO
        self._report_params()

    # class ComputeScore ，__init__
    def to(self, device: Optional[str] = None, *, dtype: Optional[torch.dtype] = None, non_blocking: bool = False):
        """
        只把 scoring_model 与本模块的轻量 buffer 移到目标 device；
        reference_model 始终保留在 self.ref_device，避免被外部的 model.to(device) 误搬走。
        """
        if device is None:
            return self
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        self.device = dev

        if dtype is not None:
            self.scoring_model.to(self.device, dtype=dtype, non_blocking=non_blocking)
        else:
            self.scoring_model.to(self.device, non_blocking=non_blocking)

        if not hasattr(self, "_dummy_buf"):
            self.register_buffer("_dummy_buf", torch.empty(0, device=self.device), persistent=False)
        else:
            self._dummy_buf = self._dummy_buf.to(self.device, non_blocking=True)
        return self

    # ---------- LoRA ----------
    @staticmethod
    def _parse_target_modules_from_str(s: Optional[str], model: nn.Module) -> List[str]:
        if s:
            s = s.strip()
        if not s or s.lower() in {"", "auto"}:
            common = {"q_proj","k_proj","v_proj","o_proj","out_proj",
                      "c_attn","c_proj","c_fc","fc_in","fc_out",
                      "dense_h_to_4h","dense_4h_to_h"}
            found = set()
            for n, m in model.named_modules():
                if isinstance(m, nn.Linear):
                    leaf = n.split(".")[-1]
                    if leaf in common and leaf != "lm_head":
                        found.add(leaf)
            if not found:
                for n, m in model.named_modules():
                    if isinstance(m, nn.Linear):
                        leaf = n.split(".")[-1]
                        if leaf != "lm_head":
                            found.add(leaf)
            return sorted(found)
        if s.lower() == "all-linear":
            found = set()
            for n, m in model.named_modules():
                if isinstance(m, nn.Linear):
                    leaf = n.split(".")[-1]
                    if leaf != "lm_head":
                        found.add(leaf)
            return sorted(found)
        return [x.strip() for x in s.split(",") if x.strip()]

    def _report_params(self):
        total, trainable = 0, 0
        for p in list(self.scoring_model.parameters()) + list(self.reference_model.parameters()):
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        logger.info(f"[params] total={total/1e6:.2f}M | trainable={trainable/1e6:.2f}M")
        if self.lora_info:
            logger.info(f"[LoRA] {self.lora_info}")

    # ---------- SPO  ----------
    def _get_SPO_input(self, tokenized=None, text: List[str]=[""], labels: Optional[torch.Tensor]=None, training_module: bool=False):
        if training_module:
            out = self.scoring_model(input_ids=tokenized.input_ids, attention_mask=tokenized.attention_mask)
            logits_score = out.logits[:, :-1, :]

            with torch.no_grad():
                tokenized_ref = self.reference_tokenizer(
                    text, return_tensors="pt", padding=True,
                    return_token_type_ids=False, add_special_tokens=True,
                    return_attention_mask=True, truncation=True, max_length=self.max_length
                ).to(self.ref_device)
                logits_ref = self.reference_model(
                    input_ids=tokenized_ref.input_ids,
                    attention_mask=tokenized_ref.attention_mask
                ).logits[:, :-1, :].to(self.device, non_blocking=True)

            labels = labels.to(logits_score.device, non_blocking=True)
            crit, SPO_input = self.criterion_fn(logits_ref, logits_score, labels)
        else:
            with torch.no_grad():
                tokenized = self.reference_tokenizer(
                    text, return_tensors="pt", padding=True,
                    return_token_type_ids=False, add_special_tokens=True,
                    return_attention_mask=True, truncation=True, max_length=self.max_length
                ).to(self.ref_device)
                out = self.reference_model(
                    input_ids=tokenized.input_ids,
                    attention_mask=tokenized.attention_mask
                )
                logits_score = out.logits[:, :-1, :].to(self.device, non_blocking=True)
                logits_ref   = logits_score

            labels = labels.to(logits_score.device, non_blocking=True)
            crit, SPO_input = self.criterion_fn(logits_ref, logits_score, labels)
        return crit, SPO_input, logits_score

    def forward_SPO(self, texts: Tuple[str, str]):
        original_text, sampled_text = texts

        # original
        tok = self.scoring_tokenizer(
            original_text, return_tensors="pt", padding=True,
            return_token_type_ids=False, truncation=True, max_length=self.max_length
        ).to(self.device)
        labels = tok.input_ids[:, 1:]
        ref_original_crit, ref_disprefered_logprob, _ = self._get_SPO_input(tok, [original_text], labels, training_module=False)
        train_original_crit, train_disprefered_logprob, _ = self._get_SPO_input(tok, [original_text], labels, training_module=True)

        # rewritten
        tok2 = self.scoring_tokenizer(
            sampled_text, return_tensors="pt", padding=True,
            return_token_type_ids=False, truncation=True, max_length=self.max_length
        ).to(self.device)
        labels2 = tok2.input_ids[:, 1:]
        ref_sampled_crit, ref_prefered_logprob, _ = self._get_SPO_input(tok2, [sampled_text], labels2, training_module=False)
        train_sampled_crit, train_prefered_logprob, _ = self._get_SPO_input(tok2, [sampled_text], labels2, training_module=True)

        SPOloss, _, _, _, _ = calculate_SPO_loss(
            train_prefered_logprob, train_disprefered_logprob,
            ref_prefered_logprob,   ref_disprefered_logprob,
            beta=self.beta
        )
        out = dict(
            crit=[ref_original_crit, train_original_crit, ref_sampled_crit, train_sampled_crit],
            loss=SPOloss
        )
        return out

    @torch.no_grad()
    def score_single(self, text: str) -> Dict[str, float]:
        tok = self.scoring_tokenizer(
            text, return_tensors="pt", padding=True,
            return_token_type_ids=False, truncation=True, max_length=self.max_length
        ).to(self.device)
        labels = tok.input_ids[:, 1:]
        crit, logprob, _ = self._get_SPO_input(tok, [text], labels, training_module=True)
        return {
            "score": float(crit.detach().cpu().view(-1)[0].item()),
            "loglik": float(logprob.detach().cpu().view(-1)[0].item()),
        }

    # ---------- /： +  lora/ ----------
    def save_pretrained(self, save_directory: str, *, save_base: bool = False):
        """
        默认仅保存 LoRA 适配器，并在 meta 中记录底模与参考模的“外部路径”，以显著减少保存时间与磁盘占用。
        如需兼容旧流程（将干净底模也落盘到 <dir>/model/），可传 save_base=True。

        修复：当 self.use_lora=False（全参微调）时，强制 save_base=True，并采用 layout='single'，
        保证评测期从 <dir>/model/ 载入“微调后的 scoring 权重”。
        """
        os.makedirs(save_directory, exist_ok=True)

        # ，
        if not self.use_lora:
            save_base = True

        meta = {
            "format": "imbd-spo-bundle",
            "bundle_layout": "single" if save_base else "ref-by-path",
            "version": 5,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "beta": float(self.beta),
            "device": str(self.device),
            "max_length": int(getattr(self, "max_length", 512)),
            "base_model_path": os.path.abspath(str(self.scoring_model_name)),     # （）
            "reference_model_path": os.path.abspath(str(self.reference_model_name)),
            "has_lora": isinstance(self.scoring_model, PeftModel),
            "lora_path": "lora" if isinstance(self.scoring_model, PeftModel) else None,
        }

        if save_base:
            mdl_dir = os.path.join(save_directory, "model")
            os.makedirs(mdl_dir, exist_ok=True)

            # tokenizer  scoring_tokenizer （ tokenizer ）
            tok_source = self.scoring_tokenizer
            try:
                tok_source.save_pretrained(mdl_dir)
            except Exception:
                # ：
                tok_clean = AutoTokenizer.from_pretrained(self.scoring_model_name)
                if tok_clean.pad_token is None:
                    tok_clean.pad_token = tok_clean.eos_token if tok_clean.eos_token is not None else tok_clean.unk_token
                tok_clean.save_pretrained(mdl_dir)

            # “”
            try:
                if isinstance(self.scoring_model, PeftModel):
                    # LoRA: （base ），single+LoRA ； LoRA  ref-by-path
                    self.scoring_model.base_model.save_pretrained(mdl_dir)
                else:
                    self.scoring_model.save_pretrained(mdl_dir)
            except Exception as e:
                raise RuntimeError(f"save_pretrained(model) failed: {e}")

            meta["paths"] = {"model": "model"}
        else:
            meta["paths"] = {"model": None}

        if meta["has_lora"]:
            lora_dir = os.path.join(save_directory, "lora")
            os.makedirs(lora_dir, exist_ok=True)
            self.scoring_model.save_pretrained(lora_dir)

        with open(os.path.join(save_directory, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_bundle(cls, load_directory: str, device: Optional[str] = None, cache_dir: str = "./models",
                    reference_device: Optional[str] = None, max_length: int = 512) -> "ComputeScore":
        """
        仅从本地 checkpoint 目录构造，不触发任何联网。
        支持两种布局：
          - single:  <dir>/model/ + 可选 <dir>/lora/ （scoring 从 model/ 载入微调权重；reference 从 meta.reference_model_path 载入干净基座）
          - ref-by-path: meta.json 中给出 base_model_path / reference_model_path（均为本地目录）
        """
        if not os.path.isdir(load_directory):
            raise ValueError(f"Directory {load_directory} does not exist.")
        meta_path = os.path.join(load_directory, "meta.json")
        if not os.path.isfile(meta_path):
            raise RuntimeError(f"Invalid bundle at {load_directory}: meta.json not found.")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        layout = meta.get("bundle_layout", "single")
        has_lora = bool(meta.get("has_lora", False))
        lora_dir = os.path.join(load_directory, meta.get("lora_path", "lora")) if has_lora else None

        # ， __init__
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.device = torch.device(device) if device else _device()
        if reference_device is not None:
            self.ref_device = torch.device(reference_device)
        else:
            self.ref_device = torch.device("cuda:1") if (torch.cuda.is_available() and torch.cuda.device_count() >= 2) else self.device

        self.beta = float(meta.get("beta", 0.05))
        self.cache_dir = cache_dir
        self.use_lora = has_lora
        self.max_length = int(meta.get("max_length", max_length))

        # （/）
        self.scoring_model_name   = meta.get("base_model_path") or meta.get("scoring_model_name")
        self.reference_model_name = meta.get("reference_model_path") or meta.get("reference_model_name") or self.scoring_model_name

        tok = None
        mdl_sc = None
        mdl_rf = None

        use_fp16 = (torch.cuda.is_available())
        dtype = torch.float16 if use_fp16 else None

        if layout == "single":
            mdl_dir = os.path.join(load_directory, (meta.get("paths", {}) or {}).get("model", "model"))
            if not os.path.isdir(mdl_dir):
                raise RuntimeError(f"Bundle layout is 'single' but model dir not found: {mdl_dir}")

            # scoring： model/
            tok = AutoTokenizer.from_pretrained(mdl_dir)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token if getattr(tok, "eos_token", None) is not None else tok.unk_token

            mdl_sc = _hf_from_pretrained(mdl_dir, dtype=dtype)
            try: mdl_sc.config.use_cache = False
            except Exception: pass
            mdl_sc.to(self.device).eval()

            # reference： meta.reference_model_path “”
            base_rf = self.reference_model_name
            if not (isinstance(base_rf, str) and os.path.isdir(base_rf)):
                raise RuntimeError(f"'single' bundle requires local reference base at reference_model_path: {base_rf}")
            mdl_rf = _hf_from_pretrained(base_rf, dtype=dtype)
            try: mdl_rf.config.use_cache = False
            except Exception: pass
            mdl_rf.to(self.ref_device).eval()

        else:
            # ref-by-path：（ LoRA， adapter）
            base_sc = self.scoring_model_name
            base_rf = self.reference_model_name
            if not (isinstance(base_sc, str) and os.path.isdir(base_sc)):
                raise RuntimeError(f"ref-by-path bundle requires a local base model dir at 'base_model_path', got: {base_sc}")
            if not (isinstance(base_rf, str) and os.path.isdir(base_rf)):
                raise RuntimeError(f"ref-by-path bundle requires a local reference model dir at 'reference_model_path', got: {base_rf}")

            tok = AutoTokenizer.from_pretrained(base_sc)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token if getattr(tok, "eos_token", None) is not None else tok.unk_token

            mdl_sc = _hf_from_pretrained(base_sc, dtype=dtype)
            try: mdl_sc.config.use_cache = False
            except Exception: pass
            mdl_sc.to(self.device).eval()

            mdl_rf = _hf_from_pretrained(base_rf, dtype=dtype)
            try: mdl_rf.config.use_cache = False
            except Exception: pass
            mdl_rf.to(self.ref_device).eval()

        # LoRA， scoring side
        self.lora_info = None
        if has_lora:
            _require_peft()
            if not os.path.isdir(lora_dir):
                raise RuntimeError(f"meta.json indicates LoRA but 'lora/' dir not found at {load_directory}")
            mdl_sc = PeftModel.from_pretrained(mdl_sc, lora_dir)
            self.lora_info = {"enabled": True, "path": "lora"}

        # & eval
        for p in mdl_sc.parameters():
            p.requires_grad = False
        for p in mdl_rf.parameters():
            p.requires_grad = False
        mdl_sc.eval(); mdl_rf.eval()

        self.scoring_tokenizer = tok
        self.reference_tokenizer = tok
        self.scoring_model = mdl_sc
        self.reference_model = mdl_rf

        self.criterion_fn = get_sampling_discrepancy_analytic
        self.forward = self.forward_SPO

        logger.info(
            f"[load] layout={layout} | has_lora={has_lora} | "
            f"score_base={'model/' if layout=='single' else meta.get('base_model_path')} | "
            f"ref_base={meta.get('reference_model_path')} | "
            f"dev(score)={self.device} | dev(ref)={self.ref_device}"
        )
        self._report_params()
        return self


# ============== （ from_bundle，） ==============
class _ImBDEvalAdapter:
    DETECTOR_NAME = DETECTOR_NAME
    detector_type = detector_type

    def __init__(self, checkpoint_dir: str, beta: float, device: str, cache_dir: str, calibrator_path: Optional[str] = None):
        self.name = DETECTOR_NAME
        self.checkpoint_dir = checkpoint_dir
        self.beta = float(beta)
        self.device = device
        self.cache_dir = cache_dir
        self.calibrator_path = calibrator_path
        self._model: Optional[ComputeScore] = None
        self._calib: Optional[Dict[str, Any]] = None

    def _lazy_load(self):
        if self._model is None:
            self._model = ComputeScore.from_bundle(
                self.checkpoint_dir,
                device=self.device,
                cache_dir=self.cache_dir,
                reference_device=self.device,   # ， GPU1
            )
            self._model.eval()
        with open(self.calibrator_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # ，
        cal = raw.get("calibrator", raw)
        # beta0/beta1， A/B
        if isinstance(cal, dict):
            if "beta0" not in cal and "B" in cal:
                cal["beta0"] = cal["B"]
            if "beta1" not in cal and "A" in cal:
                cal["beta1"] = cal["A"]

        # {"calibrator": {...}}
        self._calib = {"calibrator": cal}


    def evaluate(self, examples: List[Dict[str, Any]], batch_size: int = 8, threshold: float = 0.5, show_progress: bool = True):
        self._lazy_load()
        mdl = self._model
        scores: List[float] = []
        labels: List[int] = []
        from tqdm.auto import tqdm as _tqdm
        it = _tqdm(examples, desc="ImBD(eval) scoring", dynamic_ncols=True, disable=not show_progress, leave=False)
        for ex in it:
            text = str(ex.get("text", "") or "")
            if not text:
                continue
            s = mdl.score_single(text)
            scores.append(float(s["score"])); y = ex.get("label")
            labels.append(int(y) if y is not None else 0)

        probs: Optional[List[float]] = None
        preds: List[int] = []
        if self._calib is not None and "calibrator" in self._calib:
            probs_arr = _apply_platt(np.asarray(scores, dtype=np.float64), self._calib["calibrator"])
            probs = [float(x) for x in probs_arr.tolist()]
            preds = [1 if p >= 0.5 else 0 for p in probs]
        else:
            preds = [1 if s >= 0.0 else 0 for s in scores]

        class _SimpleEvalResult: pass
        res = _SimpleEvalResult()
        res.labels = labels
        res.preds = preds
        res.scores = scores
        res.probs = probs
        res.meta = {
            "checkpoint": self.checkpoint_dir,
            "beta": self.beta,
            "calibrator_path": self.calibrator_path,
        }
        return res

# ============== （ROC/PR） ==============
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def get_roc_metrics(real_preds: List[float], fake_preds: List[float]):
    y_true  = [0]*len(real_preds) + [1]*len(fake_preds)
    y_score = list(real_preds) + list(fake_preds)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr.tolist(), tpr.tolist(), float(auc(fpr, tpr))

def get_precision_recall_metrics(real_preds: List[float], fake_preds: List[float]):
    y_true  = [0]*len(real_preds) + [1]*len(fake_preds)
    y_score = list(real_preds) + list(fake_preds)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return precision.tolist(), recall.tolist(), float(auc(recall, precision))

# ---- ： evaluator  timestamp ， base_dir ----
def _collapse_eval_dir(out_root: str, base_name: str) -> str:
    """
    out_root 下若存在 base_name_YYYYMMDD-HHMMSS 这样的目录，把内容合并进 out_root/base_name，并删除时间戳目录。
    返回最终稳定目录路径。
    """
    base_dir = os.path.join(out_root, base_name)
    try:
        candidates = []
        for d in os.listdir(out_root):
            full = os.path.join(out_root, d)
            if os.path.isdir(full) and d.startswith(base_name + "_"):
                candidates.append(full)
        if candidates:
            # /
            latest = max(candidates, key=lambda p: os.path.getmtime(p))
            os.makedirs(base_dir, exist_ok=True)
            for name in os.listdir(latest):
                src = os.path.join(latest, name)
                dst = os.path.join(base_dir, name)
                if os.path.exists(dst):
                    if os.path.isdir(dst):
                        for root, _, files in os.walk(src):
                            rel = os.path.relpath(root, src)
                            tgt = os.path.join(dst, rel) if rel != "." else dst
                            os.makedirs(tgt, exist_ok=True)
                            for f in files:
                                shutil.move(os.path.join(root, f), os.path.join(tgt, f))
                    else:
                        os.remove(dst)
                        shutil.move(src, dst)
                else:
                    shutil.move(src, dst)
            shutil.rmtree(latest, ignore_errors=True)
    except Exception as e:
        logger.warning(f"[eval-dir] collapse failed: {e}")
    return base_dir

# ============== （） ==============
def _evaluate_model_SPO(
    model: "ComputeScore",
    pairs: List[Tuple[str, str]],
    device: torch.device,
    *,
    epoch: Optional[int] = None,
    epochs: Optional[int] = None,
) -> Dict[str, Any]:
    """
    验证/测试阶段：样式与训练对齐，但仅显示：
    Epoch | GPU_mem | L | avg | acc | acc_avg | | loss_avg
    """
    model.to(device)
    model.eval()

    real, fake = [], []
    total_loss = 0.0
    correct = 0
    t0 = time.time()
    loader = DataLoader(pairs, batch_size=1, shuffle=False)

    # --- （） ---
    logger.info(
        "\n" +
        f"{'Epoch':>{W_EPOCH}}{SEP}"
        f"{'GPU_mem':>{W_MEM}}{SEP}"
        f"{'L':>{W_NUM}}{SEP}"
        f"{'avg':>{W_NUM}}{SEP}"
        f"{'acc':>{W_NUM}}{SEP}"
        f"{'acc_avg':>{W_NUM}}{SEP}"
        f"{'loss_avg':>{W_NUM}}"
    )

    pbar = tqdm(loader, dynamic_ncols=True, leave=False)
    with torch.no_grad():
        for i, batch in enumerate(pbar, start=1):
            t_step0 = time.time()
            (o, r) = batch
            out = model((o[0], r[0]))
            l_now = float(out.get('loss', 0.0))
            total_loss += l_now
            avg_loss = total_loss / i

            # ：fake > real
            crit_ori = float(out['crit'][1].detach().cpu().view(-1)[0].item())
            crit_fake = float(out['crit'][3].detach().cpu().view(-1)[0].item())
            real.append(crit_ori); fake.append(crit_fake)
            hit = 1 if crit_fake > crit_ori else 0
            correct += hit
            acc_now = hit
            acc_avg = correct / i
            if torch.cuda.is_available():
                mem_all = torch.cuda.memory_allocated() / 1e9
                mem_res = torch.cuda.memory_reserved() / 1e9
                mem_txt = f"{mem_all:.2f}/{mem_res:.2f}G"
            else:
                mem_txt = "0.00/0.00G"

            # （；）
            desc = (
                f"{f'{epoch}/{epochs}' if (epoch and epochs) else 'val/test':>{W_EPOCH}}{SEP}"
                f"{mem_txt:>{W_MEM}}{SEP}"
                f"{l_now:>{W_NUM}.4f}{SEP}"
                f"{avg_loss:>{W_NUM}.4f}{SEP}"
                f"{acc_now:>{W_NUM}d}{SEP}"
                f"{acc_avg:>{W_NUM}.4f}"
                f"{avg_loss:>{W_NUM}.4f}"
            )
            pbar.set_description(desc)
    fpr, tpr, roc_auc = get_roc_metrics(real, fake)
    p, r, pr_auc      = get_precision_recall_metrics(real, fake)
    logger.info(
        f"[eval] ROC_AUC={roc_auc:.4f} | PR_AUC={pr_auc:.4f} | "
        f"Real μ/σ={np.mean(real):.2f}/{np.std(real):.2f} | "
        f"Fake μ/σ={np.mean(fake):.2f}/{np.std(fake):.2f} | "
        f"time={time.time()-t0:.2f}s"
    )

    return dict(
        ROC_AUC=float(roc_auc),
        PR_AUC=float(pr_auc),
        fpr=fpr, tpr=tpr,
        crit_real=real, crit_fake=fake,
        real_mean=float(np.mean(real)), real_std=float(np.std(real)),
        fake_mean=float(np.mean(fake)), fake_std=float(np.std(fake)),
        time_sec=float(time.time()-t0),
        pair_acc=float(correct / max(1, len(pairs))),
        avg_loss=float(total_loss / max(1, len(pairs))),
    )

# ============== （ AUROC  ep>=2） ==============
@dataclass
class TrainCfg:
    dataset_train: str
    dataset_val:   str
    dataset_test:  str

    train_limit: Optional[int] = None
    val_limit:   Optional[int] = None
    test_limit:  Optional[int] = None
    output_dir: str = "runs_imbd_spo"
    base_model:      str = "EleutherAI/gpt-neo-125m"
    reference_model: Optional[str] = None
    lr: float   = 1e-4
    beta: float = 0.05
    a: int      = 1
    epochs: int = 2
    val_freq: int = 1
    seed: int = 42
    cache_dir: str = "./models"
    task_name: str = "ai_detection"
    select_best_from_epoch: int = 2

    # =====  LoRA  =====
    lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[str] = None  # "auto"/"all-linear"/"q_proj,k_proj,..."
    lora_bias: str = "none"

def _run_imbd(cfg: TrainCfg, **kwargs) -> Dict[str, Any]:
    _seed_everything(cfg.seed)
    device = _device()
    torch.set_grad_enabled(True)

    env = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
    }

    out_root = f"{cfg.output_dir}_{_timestamp()}"
    os.makedirs(out_root, exist_ok=True)

    args_json = os.path.join(out_root, "train_args.json")
    with open(args_json, "w", encoding="utf-8") as f:
        json.dump({"args": {**cfg.__dict__}, "env": env,
                   "data": _build_data_info(cfg.dataset_train, cfg.dataset_val, cfg.dataset_test)}, f, ensure_ascii=False, indent=2)

    tr_pairs, _ = load_dataset_unified_pairs(cfg.dataset_train, pair_by=None, sample_k_pairs=cfg.train_limit)
    va_pairs, _ = load_dataset_unified_pairs(cfg.dataset_val,   pair_by=None, sample_k_pairs=cfg.val_limit)
    te_pairs, _ = load_dataset_unified_pairs(cfg.dataset_test,  pair_by=None, sample_k_pairs=cfg.test_limit)

    assert len(tr_pairs) > 0 and len(va_pairs) > 0 and len(te_pairs) > 0, "train/val/test 至少各有 1 对样本"

    logger.info(f"[mgt_eval] Using detector: {DETECTOR_NAME} (Type={detector_type})")
    logger.info(f"[mgt_eval] Paper: {CITATION_TITLE} | Link: {CITATION_LINK}")
    logger.info(f"[mgt_eval] Device: {device} | CUDA={torch.cuda.is_available()} | GPUs={torch.cuda.device_count()}")
    logger.info(f"[data] train_pairs={len(tr_pairs)} | val_pairs={len(va_pairs)} | test_pairs={len(te_pairs)}")
    logger.info(f"[args] {json.dumps(cfg.__dict__, ensure_ascii=False, indent=2)}")

    ref_model = cfg.reference_model or cfg.base_model
    ref_dev = "cuda:1" if (torch.cuda.is_available() and torch.cuda.device_count() >= 2) else str(device)
    model = ComputeScore(
        cfg.base_model, ref_model, SPO_beta=cfg.beta, device=str(device), cache_dir=cfg.cache_dir,
        use_lora=cfg.lora, lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        lora_target_modules=cfg.lora_target_modules, lora_bias=cfg.lora_bias,
        reference_device=ref_dev, max_length=512
    )

    train_loader = DataLoader(tr_pairs, batch_size=1, shuffle=True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, len(train_loader) * cfg.epochs), eta_min=0.0, last_epoch=-1
    )
    scaler = GradScaler()
    model.to(device)

    run_dir  = os.path.join(out_root, f"{cfg.task_name}_spo_lr_{cfg.lr}_beta_{cfg.beta}_a_{cfg.a}")
    run_best = os.path.join(run_dir, "best"); os.makedirs(run_best, exist_ok=True)
    run_last = os.path.join(run_dir, "last"); os.makedirs(run_last, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    mem_ctx = _reset_and_mark_cuda_peaks()

    global_step = 0
    step_idx, step_loss = [], []
    best_auc, best_ep = -1.0, -1
    val_history: List[Dict[str, Any]] = []

    a = max(1, int(cfg.a))  # grad accumulation
    loss_accum = torch.tensor(0.0, device=device)

    for ep in range(1, cfg.epochs + 1):
        logger.info("\n" + f"{'Epoch':>{W_EPOCH}}{SEP}{'GPU_mem':>{W_MEM}}{SEP}{'L':>{W_NUM}}{SEP}"
                f"{'avg':>{W_NUM}}{SEP}{'lr':>{W_NUM}}{SEP}{'step':>{W_STEP}}{SEP}"
                f"{'step_t(s)':>{W_NUM}}{SEP}{'pairs/s':>{W_NUM}}")
        model.train()
        ep_avg, cnt = 0.0, 0
        ep_t0 = time.time()
        pbar = tqdm(train_loader, desc=f"{ep}/{cfg.epochs}", dynamic_ncols=True, leave=False)

        for batch in pbar:
            t_step0 = time.time()
            (orig, rew) = batch
            texts = (orig[0], rew[0])

            with autocast():
                out = model(texts)
                loss_accum = loss_accum + (out["loss"].to(torch.float32)) / a

            if ((global_step + 1) % a) == 0:
                scaler.scale(loss_accum).backward()
                scaler.step(optimizer)
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                step_loss.append(float(loss_accum.item()))
                step_idx.append(global_step + 1)
                loss_accum = torch.tensor(0.0, device=device)
                scheduler.step()

            cnt += 1
            ep_avg = (ep_avg * (cnt - 1) + float(out["loss"].item())) / cnt
            cur_lr = optimizer.param_groups[0]["lr"]
            mem_res = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
            mem_all = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            step_t = time.time() - t_step0
            thr = 1.0 / step_t if step_t > 0 else 0.0

            desc = (f"{f'{ep}/{cfg.epochs}':>{W_EPOCH}}{SEP}"
                    f"{f'{mem_all:.2f}/{mem_res:.2f}G':>{W_MEM}}{SEP}"
                    f"{float(out['loss'].item()):>{W_NUM}.4f}{SEP}"
                    f"{float(ep_avg):>{W_NUM}.4f}{SEP}"
                    f"{float(cur_lr):>{W_NUM}.2e}{SEP}"
                    f"{int(global_step):>{W_STEP}d}{SEP}"
                    f"{float(step_t):>{W_NUM}.3f}{SEP}"
                    f"{float(thr):>{W_NUM}.2f}")
            if hasattr(pbar, "set_description"):
                pbar.set_description(desc)

            global_step += 1

        logger.info(f"[val] epoch={ep} evaluating on validation set...")
        val_result = _evaluate_model_SPO(
            model, va_pairs, device,
            epoch=ep, epochs=cfg.epochs,
        )
        val_result_epoch = {"epoch": ep, **val_result}
        val_history.append(val_result_epoch)

        # last（LoRA:  adapter；： model/）
        model.save_pretrained(run_last, save_base=not cfg.lora)

        # best
        if ep >= int(cfg.select_best_from_epoch):
            if val_result["ROC_AUC"] >= best_auc:
                best_auc, best_ep = float(val_result["ROC_AUC"]), ep
                model.save_pretrained(run_best, save_base=not cfg.lora)
                logger.info(f"[val] new best AUROC={best_auc:.4f} at epoch {best_ep} (saved to best/)")

        logger.info(f"[epoch {ep}] avg_loss={ep_avg:.4f} | time_ep={time.time()-ep_t0:.2f}s | "
                    f"best(AUROC)={best_auc:.4f}@{best_ep}")

    mem_stats = _collect_cuda_peaks(mem_ctx)
    loss_png  = _save_loss_plot(step_idx, step_loss, out_dir=out_root, filename="train_loss.png", smooth_window=0)

    val_json = os.path.join(out_root, "val_results.json")
    with open(val_json, "w", encoding="utf-8") as f:
        json.dump({"best_epoch": best_ep, "best_AUROC": best_auc, "history": val_history}, f, ensure_ascii=False, indent=2)
    logger.info(f"[val] all validation results saved to: {val_json}")

    # ---------- ： from_bundle， ----------
    logger.info("[test] evaluating best model on test set...")
    best_dir_to_use = run_best if best_ep != -1 else run_last
    model_test = ComputeScore.from_bundle(best_dir_to_use, device=str(device), cache_dir=cfg.cache_dir)
    test_result = _evaluate_model_SPO(
        model_test, te_pairs, device,
        epoch=best_ep if best_ep != -1 else cfg.epochs,
        epochs=cfg.epochs,
    )
    test_json = os.path.join(out_root, "test_results.json")
    with open(test_json, "w", encoding="utf-8") as f:
        json.dump(test_result, f, ensure_ascii=False, indent=2)
    logger.info(f"[test] results saved to: {test_json}")
    calibrator_path = None
    eval_dir = None
    # ----------  +  ----------
    try:
        calib_dir = os.path.join(out_root, "calib"); os.makedirs(calib_dir, exist_ok=True)
        calib_examples = []
        for o, r in va_pairs:
            calib_examples.append({"text": str(o), "label": 0})
            calib_examples.append({"text": str(r), "label": 1})
        calib_jsonl = _dump_jsonl(os.path.join(calib_dir, "dev_pairs_as_single.jsonl"), calib_examples)

        calibrator_path = _ensure_calibrator(best_dir_to_use, calib_jsonl, detector_name="imbd")
        logger.info(f"[calib] calibrator = {calibrator_path}")

        eval_base_name = "eval_result"
        eval_dir = os.path.join(out_root, eval_base_name); os.makedirs(eval_dir, exist_ok=True)
        test_examples_single = []
        for o, r in te_pairs:
            test_examples_single.append({"text": str(o), "label": 0})
            test_examples_single.append({"text": str(r), "label": 1})
        _dump_jsonl(os.path.join(eval_dir, "test_pairs_as_single.jsonl"), test_examples_single)

        adapter = _ImBDEvalAdapter(
            checkpoint_dir=best_dir_to_use,
            beta=cfg.beta,
            device=str(device),
            cache_dir=cfg.cache_dir,
            calibrator_path=calibrator_path,
        )
        _ = _evaluate_detector(
            detector=adapter,
            dataset=test_examples_single,
            batch_size=8,
            threshold=0.5,
            out_dir=eval_dir,
            save_curves=True,
            show_progress=True,
        )
        # ： evaluator  eval_result_YYYY... ， eval_result
        eval_dir = _collapse_eval_dir(out_root, eval_base_name)
        logger.info(f"[eval-unified] results saved under: {eval_dir}")
    except Exception as e:
        logger.warning(f"[post-train calibrate/eval] skipped due to error: {e}")
        calibrator_path = None
        eval_dir = None

    summary = {
        "best_dir": best_dir_to_use,
        "last_dir": run_last,
        "best_val_AUROC": best_auc,
        "best_epoch": best_ep,
        "memory": mem_stats,
        "artifacts": {
            "args_json": args_json,
            "val_results_json": val_json,
            "test_results_json": test_json,
            "loss_plot": loss_png,
        },
        "output_root": out_root,
        "calibrator_path": calibrator_path,
        "unified_eval_dir": eval_dir,
    }

    with open(os.path.join(out_root, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "train": {
            "model_dir": best_dir_to_use,
            "best_val_AUROC": best_auc,
            "best_epoch": best_ep,
            "artifacts": summary["artifacts"],
            "output_root": out_root,
        }
    }

# ============== （） ==============
@register_train("imbd")
def train_imbd(**kwargs) -> Dict[str, Any]:
    cfg = TrainCfg(
        dataset_train = kwargs.get("dataset_train"),
        dataset_val   = kwargs.get("dataset_val"),
        dataset_test  = kwargs.get("dataset_test"),
        train_limit   = kwargs.get("train_limit"),
        val_limit     = kwargs.get("val_limit"),
        test_limit    = kwargs.get("test_limit"),
        output_dir    = kwargs.get("output_dir", "runs_imbd_spo"),
        base_model    = kwargs.get("base_model", "EleutherAI/gpt-neo-125m"),
        reference_model = kwargs.get("reference_model", None),
        lr            = kwargs.get("lr", 1e-4),
        beta          = kwargs.get("beta", 0.05),
        a             = kwargs.get("a", 1),
        epochs        = kwargs.get("epochs", 2),
        val_freq      = kwargs.get("val_freq", 1),
        seed          = kwargs.get("seed", 42),
        cache_dir     = kwargs.get("cache_dir", "./models"),
        task_name     = kwargs.get("task_name", "ai_detection"),
        select_best_from_epoch = kwargs.get("select_best_from_epoch", 2),
        # LoRA
        lora = kwargs.get("lora", kwargs.get("lora_enable_scoring", False) or kwargs.get("lora_enable_reference", False)),
        lora_r = kwargs.get("lora_r", 16),
        lora_alpha = kwargs.get("lora_alpha", 32),
        lora_dropout = kwargs.get("lora_dropout", 0.05),
        lora_target_modules = kwargs.get("lora_target_modules", None),
        lora_bias = kwargs.get("lora_bias", "none"),
    )
    assert cfg.dataset_train and cfg.dataset_val and cfg.dataset_test, \
        "必须提供 dataset_train / dataset_val / dataset_test（支持逗号多源）"
    return _run_imbd(cfg, **kwargs)

# ============== /： lora/  LoRA ==============
def detect_imbd(**kwargs) -> Dict[str, Any]:
    """
    关键参数：
      - dataset
      - checkpoint
      - limit
      - output_dir
      - seed
      - cache_dir
      - 可覆盖 beta/base/ref（但常规用法：直接用 checkpoint 目录）
    逻辑：只检测 checkpoint 下是否存在 lora/ 目录。有则加载 LoRA；否则全参加载。
    """
    import os, json
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    dataset    = kwargs.get("dataset", None)
    checkpoint = kwargs.get("checkpoint", None)
    limit      = kwargs.get("limit", None)
    output_dir = kwargs.get("output_dir", "runs_imbd_detect")
    seed       = kwargs.get("seed", 42)
    cache_dir  = kwargs.get("cache_dir", "./models")
    beta       = kwargs.get("beta", 0.05)

    assert dataset or kwargs.get("text", None), "detect_imbd: 需要提供 dataset 或单条 text"
    assert checkpoint, "detect_imbd: 需要提供 checkpoint（train/save_pretrained 导出的目录）"

    _seed_everything(seed)
    device = _device()

    out_root = f"{output_dir}_{_timestamp()}"
    os.makedirs(out_root, exist_ok=True)

    with open(os.path.join(out_root, "detect_args.json"), "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset, "checkpoint": checkpoint, "limit": limit}, f, ensure_ascii=False, indent=2)

    # bundle （）
    model = ComputeScore.from_bundle(checkpoint, device=str(device), cache_dir=cache_dir)
    model.eval()

    # ： pairs； single
    mode = None
    pairs: List[Tuple[str, str]] = []
    single_examples: List[Dict[str, Any]] = []

    try:
        pairs, _ = load_dataset_unified_pairs(dataset, pair_by=None, sample_k_pairs=limit)
        if len(pairs) > 0:
            mode = "pairs"
    except Exception as e:
        logger.warning(f"[detect] as pairs failed: {e}")

    if mode is None:
        try:
            single_examples, _ = load_dataset_unified(dataset, sample_k=limit)
            if len(single_examples) > 0:
                mode = "single"
        except Exception as e:
            logger.warning(f"[detect] as single failed: {e}")

    if mode is None:
        raise RuntimeError("数据集既不符合配对格式、也不能解析为单文本格式，请检查输入。")

    if mode == "pairs":
        logger.info(f"[detect] mode=pairs | #pairs={len(pairs)}")
        result = _evaluate_model_SPO(model, pairs, device, epoch=1, epochs=1)
        out_json = os.path.join(out_root, "detect_pairs_results.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"[detect] pairs results saved to: {out_json}")
        try:
            import gc
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                for _dev in range(torch.cuda.device_count()):
                    torch.cuda.set_device(_dev)
                    torch.cuda.empty_cache()
        except Exception as _e:
            logger.warning(f"[detect] free model before unified-eval failed: {_e}")
        # （）
        try:
            eval_base_name = "unified_eval"
            eval_dir = os.path.join(out_root, eval_base_name); os.makedirs(eval_dir, exist_ok=True)
            eval_examples = []
            for o, r in pairs:
                eval_examples.append({"text": str(o), "label": 0})
                eval_examples.append({"text": str(r), "label": 1})
            eval_jsonl = _dump_jsonl(os.path.join(eval_dir, "pairs_as_single.jsonl"), eval_examples)

            calibrator_path = _ensure_calibrator(checkpoint, eval_jsonl, detector_name="imbd")
            adapter = _ImBDEvalAdapter(
                checkpoint_dir=checkpoint,
                beta=beta,
                device=str(device),
                cache_dir=cache_dir,
                calibrator_path=calibrator_path,
            )
            _ = _evaluate_detector(
                detector=adapter,
                dataset=eval_examples,
                batch_size=8,
                threshold=0.5,
                out_dir=eval_dir,
                save_curves=True,
                show_progress=True,
            )
            eval_dir = _collapse_eval_dir(out_root, eval_base_name)
            logger.info(f"[detect-unified] results saved under: {eval_dir}")
            return {
                "mode": "pairs",
                "output_root": out_json.rsplit("/",1)[0],
                "metrics": result,
                "calibrator_path": calibrator_path,
                "unified_eval_dir": eval_dir,
            }
        except Exception as e:
            logger.warning(f"[detect-unified] skipped due to error: {e}")
            return {"mode": "pairs", "output_root": out_json.rsplit("/",1)[0], "metrics": result}

    # single
    logger.info(f"[detect] mode=single | #examples={len(single_examples)}")
    scores: List[float] = []
    labels: List[Optional[int]] = []
    preds_path = os.path.join(out_root, "predictions.jsonl")
    t0 = time.time()

    with open(preds_path, "w", encoding="utf-8") as fout:
        for ex in tqdm(single_examples, desc="Scoring(single)", dynamic_ncols=True, leave=False):
            text = str(ex.get("text", "")).strip()
            if not text:
                continue
            s = model.score_single(text)
            rec_out = {"text": text, "score": s["score"], "loglik": s["loglik"]}
            if "label" in ex:
                try:
                    y = int(ex["label"])
                except Exception:
                    y = None
                rec_out["label"] = y
                labels.append(y)
            else:
                labels.append(None)
            scores.append(float(s["score"]))
            fout.write(json.dumps(rec_out, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    metrics: Dict[str, Any] = {"count": len(scores), "time_sec": elapsed}
    has_label = any(l is not None for l in labels)
    if has_label:
        y_true, y_score = [], []
        for y, s in zip(labels, scores):
            if y is None:
                continue
            y_true.append(int(y)); y_score.append(float(s))
        if len(set(y_true)) >= 2:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            metrics.update({"ROC_AUC": float(auc(fpr, tpr)), "PR_AUC": float(auc(rec, prec)), "n_labeled": int(len(y_true))})
        else:
            metrics.update({"warning": "标签仅含单一类别或数量不足，无法计算ROC/PR。"})

    out_json = os.path.join(out_root, "detect_single_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "predictions_path": preds_path}, f, ensure_ascii=False, indent=2)
    logger.info(f"[detect] single results saved to: {out_json}")
    try:
        eval_base_name = "unified_eval"
        eval_dir = os.path.join(out_root, eval_base_name); os.makedirs(eval_dir, exist_ok=True)
        eval_examples = []
        for ex in single_examples:
            rec = {"text": str(ex.get("text", "") or "")}
            if "label" in ex and ex["label"] is not None:
                try: rec["label"] = int(ex["label"])
                except Exception: pass
            eval_examples.append(rec)
        eval_jsonl = _dump_jsonl(os.path.join(eval_dir, "single_examples.jsonl"), eval_examples)

        calibrator_path = _ensure_calibrator(checkpoint, eval_jsonl, detector_name="imbd")
        adapter = _ImBDEvalAdapter(
            checkpoint_dir=checkpoint,
            beta=beta,
            device=str(device),
            cache_dir=cache_dir,
            calibrator_path=calibrator_path,
        )
        _ = _evaluate_detector(
            detector=adapter,
            dataset=eval_examples,
            batch_size=8,
            threshold=0.5,
            out_dir=eval_dir,
            save_curves=True,
            show_progress=True,
        )
        eval_dir = _collapse_eval_dir(out_root, eval_base_name)
        logger.info(f"[detect-unified] results saved under: {eval_dir}")
        return {
            "mode": "single",
            "output_root": out_root,
            "metrics": metrics,
            "predictions_path": preds_path,
            "calibrator_path": calibrator_path,
            "unified_eval_dir": eval_dir,
        }
    except Exception as e:
        logger.warning(f"[detect-unified] skipped due to error: {e}")

    return {"mode": "single", "output_root": out_root, "metrics": metrics, "predictions_path": preds_path}



def _device_of(device: Optional[str]) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def _mean_ll(model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    平均 token log-likelihood（和 metric.baseline 里的实现一致/等价）。
    返回形状：(B,)
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits                      # (B, T, V)
    shift_logits = logits[:, :-1, :].contiguous()
    labels = input_ids[:, 1:].contiguous()   # (B, T-1)
    if shift_logits.size(1) == 0:
        return torch.zeros(input_ids.size(0), device=input_ids.device)
    lprobs = torch.log_softmax(shift_logits, dim=-1)
    tok_ll = lprobs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    return tok_ll.mean(dim=1)  # (B,)

@register("imbd")
@register("imbd_spo")  # ，
class ImBDSPODetector(DetectorBase):
    """
    统一评测的注册适配器：
    - 从 ImBD 的 bundle（best/ 目录）加载干净基座 + LoRA（scoring）与 reference 基座
    - 对单文本返回 ΔLL = mean_ll(scoring) - mean_ll(reference)
    - 至于阈值与方向，由统一评测/校准器处理
    """
    CITATION_TITLE = "Aligning Machine Stylistic Preference for Machine-Revised Text Detection"
    CITATION_AUTHORS = "Community Implementation"
    CITATION_LINK = "N/A"

    def __init__(self,
             checkpoint: Optional[str] = None,   # ：best/
             model1: Optional[str] = None,
             score_model: Optional[str] = None,
             tokenizer: Optional[str] = None,
             device: Optional[str] = None,
             fp16: bool = True,
             max_length: int = 1024,
             **kwargs):
        cp = checkpoint or model1 or score_model
        if not cp:
            raise RuntimeError("ImBDSPODetector: require 'checkpoint' (or alias 'model1' / 'score_model') to point to best/ directory.")
        if not os.path.isdir(cp):
            raise RuntimeError(f"ImBDSPODetector: checkpoint dir not found: {cp}")

        super().__init__(score_model=cp,
                        tokenizer=tokenizer or cp,
                        device=device,
                        max_length=max_length,
                        fp16=fp16,
                        detector_type="Finetuned",
                        name="ImBD-SPO")
        self.checkpoint = cp
        self.device = _device_of(device)
        self.fp16 = bool(fp16)
        self.max_length = int(max_length)

        # NEW:
        self.ref_device = ("cuda:1" if (torch.cuda.is_available() and torch.cuda.device_count() >= 2) else self.device)

        self._tok = None
        self._score_m = None   # scoring: base + LoRA
        self._ref_m = None     # reference:
        self.is_loaded = False

    def _load_bundle(self):
        meta_path = os.path.join(self.checkpoint, "meta.json")
        meta = json.load(open(meta_path, "r", encoding="utf-8"))

        layout = meta.get("bundle_layout", "single")
        has_lora = bool(meta.get("has_lora", False))
        lora_dir = os.path.join(self.checkpoint, meta.get("lora_path", "lora"))

        if layout == "single":
            model_rel = (meta.get("paths", {}) or {}).get("model", "model")
            model_dir = os.path.join(self.checkpoint, model_rel)
            if not os.path.isdir(model_dir):
                raise RuntimeError(f"ImBDSPODetector: 'single' bundle missing model dir: {model_dir}")

            tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=True)
            if tok.pad_token is None and getattr(tok, "eos_token", None) is not None:
                tok.pad_token = tok.eos_token
            tok.padding_side = "right"

            score_m = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
            if has_lora and os.path.isdir(lora_dir):
                try:
                    from peft import PeftModel
                except Exception as e:
                    raise RuntimeError("ImBDSPODetector: bundle contains LoRA but 'peft' is not installed.") from e
                score_m = PeftModel.from_pretrained(score_m, lora_dir)
            try: score_m.config.use_cache = False
            except Exception: pass
            if self.fp16 and str(self.device).startswith("cuda"):
                try: score_m.half()
                except Exception: pass
            score_m.to(self.device).eval()

            # reference  scoring “meta.reference_model_path”
            base_rf = meta.get("reference_model_path") or meta.get("reference_model_name") or model_dir
            ref_m = AutoModelForCausalLM.from_pretrained(base_rf, trust_remote_code=True)
            try: ref_m.config.use_cache = False
            except Exception: pass
            if self.fp16 and str(self.ref_device).startswith("cuda"):
                try: ref_m.half()
                except Exception: pass
            ref_m.to(self.ref_device).eval()

        else:
            base_sc = meta.get("base_model_path") or meta.get("scoring_model_name")
            base_rf = meta.get("reference_model_path") or meta.get("reference_model_name") or base_sc
            if not (isinstance(base_sc, str) and os.path.isdir(base_sc)):
                raise RuntimeError(f"ImBDSPODetector: ref-by-path requires local base at base_model_path: {base_sc}")
            if not (isinstance(base_rf, str) and os.path.isdir(base_rf)):
                raise RuntimeError(f"ImBDSPODetector: ref-by-path requires local reference at reference_model_path: {base_rf}")

            tok = AutoTokenizer.from_pretrained(base_sc, use_fast=True, trust_remote_code=True)
            if tok.pad_token is None and getattr(tok, "eos_token", None) is not None:
                tok.pad_token = tok.eos_token
            tok.padding_side = "right"

            score_m = AutoModelForCausalLM.from_pretrained(base_sc, trust_remote_code=True)
            if has_lora and os.path.isdir(lora_dir):
                try:
                    from peft import PeftModel
                except Exception as e:
                    raise RuntimeError("ImBDSPODetector: bundle contains LoRA but 'peft' is not installed.") from e
                score_m = PeftModel.from_pretrained(score_m, lora_dir)
            try: score_m.config.use_cache = False
            except Exception: pass
            if self.fp16 and str(self.device).startswith("cuda"):
                try: score_m.half()
                except Exception: pass
            score_m.to(self.device).eval()

            ref_m = AutoModelForCausalLM.from_pretrained(base_rf, trust_remote_code=True)
            try: ref_m.config.use_cache = False
            except Exception: pass
            if self.fp16 and str(self.ref_device).startswith("cuda"):
                try: ref_m.half()
                except Exception: pass
            ref_m.to(self.ref_device).eval()

        self._tok, self._score_m, self._ref_m = tok, score_m, ref_m

    def load(self):
        self._load_bundle()
        super().load()

    @torch.no_grad()
    def score_batch(self, texts: List[str]) -> np.ndarray:
        if not self.is_loaded:
            self.load()
        enc = self._tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length)
        enc.pop("token_type_ids", None)

        # （）
        enc_score = {k: v.to(self.device) for k, v in enc.items()}
        enc_ref   = {k: v.to(self.ref_device) for k, v in enc.items()}

        ll_score = _mean_ll(self._score_m, enc_score["input_ids"], enc_score["attention_mask"]).to("cpu")
        ll_ref   = _mean_ll(self._ref_m,   enc_ref["input_ids"],   enc_ref["attention_mask"]).to("cpu")
        score = (ll_score - ll_ref).float().numpy()
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return score
