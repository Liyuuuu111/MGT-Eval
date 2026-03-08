# mgt_eval/train/train.py
from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
import os
import math
import time
import json
from datetime import datetime
import platform

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

# ：（）
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    plt = None
    _HAS_MPL = False
W_EPOCH = 8
W_MEM   = 8
W_NUM   = 7
W_STEP  = 8
SEP     = " "
# =========================
# （）
# =========================
def _bytes_to_gib(x: int) -> float:
    return float(x) / (1024.0 ** 3)

def _cuda_devices() -> List[int]:
    if (not hasattr(torch, "cuda")) or (not torch.cuda.is_available()):
        return []
    try:
        return list(range(torch.cuda.device_count()))
    except Exception:
        return []

def _reset_and_mark_cuda_peaks() -> Dict[str, Any]:
    """
    训练开始前调用：重置所有可见 CUDA 设备的峰值统计，并记录设备名。
    """
    ctx: Dict[str, Any] = {"cuda_available": bool(_cuda_devices()), "devices": []}
    devs = _cuda_devices()
    for idx in devs:
        try:
            torch.cuda.reset_peak_memory_stats(idx)
        except Exception:
            pass
        name = None
        try:
            name = torch.cuda.get_device_name(idx)
        except Exception:
            name = f"cuda:{idx}"
        ctx["devices"].append({"index": idx, "name": name})
    return ctx

def _collect_cuda_peaks(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    训练结束后调用：读取每张卡的峰值显存（allocated/reserved）。
    """
    out: Dict[str, Any] = {
        "cuda_available": bool(ctx.get("cuda_available", False)),
        "per_device": [],
        "total_peak_allocated_gib": 0.0,
        "total_peak_reserved_gib": 0.0,
    }
    if not out["cuda_available"]:
        return out

    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    total_alloc = 0
    total_res = 0
    for d in ctx.get("devices", []):
        idx = int(d["index"])
        name = d.get("name", f"cuda:{idx}")
        try:
            peak_alloc = torch.cuda.max_memory_allocated(idx)
        except Exception:
            peak_alloc = 0
        try:
            peak_reserved = torch.cuda.max_memory_reserved(idx)
        except Exception:
            peak_reserved = 0
        out["per_device"].append({
            "device": f"cuda:{idx}",
            "name": name,
            "peak_allocated_bytes": int(peak_alloc),
            "peak_reserved_bytes": int(peak_reserved),
            "peak_allocated_gib": _bytes_to_gib(int(peak_alloc)),
            "peak_reserved_gib": _bytes_to_gib(int(peak_reserved)),
        })
        total_alloc += int(peak_alloc)
        total_res += int(peak_reserved)

    out["total_peak_allocated_gib"] = _bytes_to_gib(total_alloc)
    out["total_peak_reserved_gib"] = _bytes_to_gib(total_res)
    return out
# =========================

def _default_device(dev: Optional[str]) -> str:
    if dev:
        return dev
    return "cuda" if torch.cuda.is_available() else "cpu"

def _collate_fn(tokenizer, max_length: int):
    def _fn(batch):
        texts = [b["text"] for b in batch]
        labels = torch.tensor([int(b["label"]) for b in batch], dtype=torch.long)
        toks = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        global_attention_mask = torch.zeros_like(toks["input_ids"])
        global_attention_mask[:, 0] = 1  # token  global
        toks["global_attention_mask"] = global_attention_mask
        toks["labels"] = labels
        return toks
    return _fn

import inspect

def _filter_forward_kwargs(model, batch: dict) -> dict:
    """
    仅将模型 forward 支持的 kwargs 传入，避免如 global_attention_mask 等参数导致 TypeError。
    - 对支持 **kwargs 的 forward：仍只传常见字段，避免把 labels 之类误传。
    - 对无法 introspect 的情况：退化为 input_ids/attention_mask。
    """
    # （：token_type_ids / position_ids / head_mask / etc.）
    candidate_keys = ("input_ids", "attention_mask", "token_type_ids", "position_ids", "global_attention_mask")

    sig = getattr(model, "_mgt_eval_forward_sig", None)
    if sig is None:
        try:
            sig = inspect.signature(model.forward)
        except Exception:
            sig = None
        setattr(model, "_mgt_eval_forward_sig", sig)

    if sig is None:
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch.get("attention_mask", None),
        }

    params = sig.parameters
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    allowed = set(params.keys())

    out = {}
    for k in candidate_keys:
        v = batch.get(k, None)
        if v is None:
            continue
        if has_varkw or (k in allowed):
            out[k] = v
    return out

def _compute_loss(logits, labels, smoothing: float = 0.0):
    """
    标签平滑的交叉熵；smoothing=0 即普通 CE。
    """
    if smoothing and smoothing > 0.0:
        num_classes = logits.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(smoothing / (num_classes - 1))
            true_dist.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(true_dist * log_probs).sum(dim=-1).mean()
        return loss
    else:
        return torch.nn.functional.cross_entropy(logits, labels)

@torch.no_grad()
def _evaluate_loop(model, dataloader, device: str, smoothing_eval: float = 0.0, fp16: bool = True):
    """
    评估：返回 (acc, avg_loss)。
    """
    model.eval()
    correct, total = 0, 0
    loss_sum = 0.0
    it = tqdm(
        dataloader,
        desc="Valid",
        leave=False,
        dynamic_ncols=True,
        disable=(len(dataloader) == 0),
    )
    for batch in it:
        batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=(fp16 and device.startswith("cuda"))):
            out = model(**_filter_forward_kwargs(model, batch))
            loss = _compute_loss(out.logits, batch["labels"], smoothing=smoothing_eval)
        preds = torch.argmax(out.logits, dim=-1)
        correct += int((preds == batch["labels"]).sum().item())
        total += int(batch["labels"].numel())
        loss_sum += float(loss.item())
    acc = correct / max(1, total)
    avg_loss = loss_sum / max(1, len(dataloader))
    return acc, avg_loss

def _is_oom_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("out of memory" in msg) or ("cuda error" in msg) or ("cudnn" in msg)

def _split_and_backward(
    model,
    batch: dict,
    *,
    device: str,
    fp16: bool,
    grad_accum: int,
    scaler: "torch.cuda.amp.GradScaler",
):
    B = batch["input_ids"].size(0)
    chunk = max(1, B // 2)
    while chunk >= 1:
        try:
            start = 0
            while start < B:
                end = min(B, start + chunk)

                # ✅  micro-batch（ B  tensor）
                cur = {}
                for k, v in batch.items():
                    if hasattr(v, "shape") and getattr(v, "size", lambda *_: 0)(0) == B:
                        cur[k] = v[start:end]
                    else:
                        cur[k] = v

                with torch.amp.autocast("cuda", enabled=(fp16 and device.startswith("cuda"))):
                    out = model(**_filter_forward_kwargs(model, cur))
                    loss = _compute_loss(out.logits, cur["labels"], smoothing=0.0)
                    scale = (end - start) / float(B)

                scaler.scale(loss * scale / max(1, grad_accum)).backward()
                start = end
            return True
        except RuntimeError as e:
            if not _is_oom_error(e):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            chunk //= 2
    return False

def _moving_average(seq: List[float], window: int) -> List[float]:
    if window is None or window <= 1:
        return list(seq)
    out: List[float] = []
    s = 0.0
    q: List[float] = []
    for v in seq:
        q.append(float(v))
        s += float(v)
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out

from collections import Counter

def _build_data_info(dataset_spec, train_dataset, val_dataset, test_dataset=None):
    info = {
        "dataset_spec": dataset_spec,
        "num_examples": {
            "train": int(len(train_dataset)) if hasattr(train_dataset, "__len__") else None,
            "val": int(len(val_dataset)) if (val_dataset is not None and hasattr(val_dataset, "__len__")) else 0,
            "test": int(len(test_dataset)) if (test_dataset is not None and hasattr(test_dataset, "__len__")) else 0,
        },
    }
    try:
        cnt = Counter(int(train_dataset[i]["label"]) for i in range(len(train_dataset)))
        info["label_distribution_train"] = {int(k): int(v) for k, v in sorted(cnt.items())}
    except Exception:
        pass
    try:
        import hashlib, os
        if isinstance(dataset_spec, str) and os.path.isfile(dataset_spec):
            h = hashlib.sha256()
            with open(dataset_spec, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            info["dataset_sha256"] = h.hexdigest()
    except Exception:
        pass
    return info

def _save_loss_plot(
    steps: List[int],
    losses: List[float],
    out_dir: str,
    filename: str = "train_loss.png",
    smooth_window: int = 0,
):
    if not _HAS_MPL or not losses:
        return None
    os.makedirs(out_dir, exist_ok=True)
    x = list(steps) if steps else list(range(1, len(losses) + 1))
    y = list(losses)
    y_plot = _moving_average(y, smooth_window)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y_plot)
    ax.set_title("Training Loss")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", linewidth=0.5)
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path

def train_model(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    cfg,
    dataset_spec: Optional[str] = None,
    test_dataset=None,                 # ✅ NEW
) -> Dict[str, Any]:
    """
    通用训练管线（带 tqdm 进度条 & 周期日志）：

    - 支持 AMP、梯度累积、线性 warmup、OOM 回退切分。
    - eval/save 支持 'epoch' 或 'steps' 策略。
    - 保存到 output_dir：train_args.json、train_summary.json、train_loss.png（可关闭）。
    - 返回:
        {
          "best_dir": str|None,
          "last_dir": str,
          "best_val_acc": float|None,
          "history": List[Dict],
          "memory": Dict,
          "timing": Dict,
          "artifacts": Dict
        }
    """
    torch.set_grad_enabled(True)
    device = _default_device(getattr(cfg, "device", None))
    mem_ctx = _reset_and_mark_cuda_peaks()
    model.to(device)

    # ====== （） ======
    max_length       = int(getattr(cfg, "max_length", 512))
    lr               = float(getattr(cfg, "lr", 5e-5))
    weight_decay     = float(getattr(cfg, "weight_decay", 0.0))
    epochs           = int(getattr(cfg, "epochs", 3))
    train_bs         = int(getattr(cfg, "train_batch_size", 32))
    eval_bs          = int(getattr(cfg, "eval_batch_size", 64))
    warmup_ratio     = float(getattr(cfg, "warmup_ratio", 0.06))
    grad_accum       = int(getattr(cfg, "grad_accum_steps", 1))
    fp16             = bool(getattr(cfg, "fp16", True))
    label_smoothing  = float(getattr(cfg, "label_smoothing", 0.0))
    output_dir       = str(getattr(cfg, "output_dir", "runs"))
    progress         = bool(getattr(cfg, "progress", True))
    log_interval     = int(getattr(cfg, "log_interval", 50))

    eval_strategy    = str(getattr(cfg, "eval_strategy", "epoch")).lower()     # 'epoch' | 'steps' | 'no'
    eval_interval    = int(getattr(cfg, "eval_interval", 1000))
    save_strategy    = str(getattr(cfg, "save_strategy", "epoch")).lower()     # 'epoch' | 'steps'
    save_interval    = int(getattr(cfg, "save_interval", 1000))
    save_every_epoch = bool(getattr(cfg, "save_every_epoch", False))

    max_grad_norm    = float(getattr(cfg, "max_grad_norm", 1.0))
    num_workers      = int(getattr(cfg, "num_workers", 0))

    early_patience   = getattr(cfg, "early_stopping_patience", None)  # None
    metric_for_best  = str(getattr(cfg, "metric_for_best", "val_acc"))
    greater_is_better= bool(getattr(cfg, "greater_is_better", True))

    # ：
    save_loss_plot           = bool(getattr(cfg, "save_loss_plot", True))
    loss_plot_filename       = str(getattr(cfg, "loss_plot_filename", "train_loss.png"))
    loss_plot_smooth_window  = int(getattr(cfg, "loss_plot_smooth_window", 0))

    os.makedirs(output_dir, exist_ok=True)

    # ---- （） ----
    try:
        if hasattr(cfg, "__dict__"):
            cfg_dict = dict(cfg.__dict__)
        else:
            cfg_dict = {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_") and not callable(getattr(cfg, k))}
    except Exception:
        cfg_dict = {}

    env_info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
    }
    data_info = _build_data_info(dataset_spec, train_dataset, val_dataset, test_dataset)
    args_json_path = os.path.join(output_dir, "train_args.json")
    with open(args_json_path, "w", encoding="utf-8") as f:
        json.dump({"args": cfg_dict, "env": env_info, "data": data_info}, f, ensure_ascii=False, indent=2)
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_fn(tokenizer, max_length),
        pin_memory=True,
    )
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=eval_bs,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate_fn(tokenizer, max_length),
            pin_memory=True,
        )
    test_loader = None
    if test_dataset is not None and hasattr(test_dataset, "__len__") and len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=eval_bs,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate_fn(tokenizer, max_length),
            pin_memory=True,
        )
    t_updates_per_epoch = math.ceil(len(train_loader) / max(1, grad_accum))
    t_total_updates = epochs * t_updates_per_epoch
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(grouped, lr=lr)
    num_warmup = int(warmup_ratio * t_total_updates)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup, t_total_updates)
    scaler = torch.amp.GradScaler("cuda", enabled=(fp16 and device.startswith("cuda")))
    best_dir = None
    last_dir = None
    best_metric_val = -float("inf") if greater_is_better else float("inf")
    no_improve_cnt = 0
    history = []
    global_step = 0

    # ： step （ update ）
    step_indices: List[int] = []
    step_losses: List[float] = []

    # ======  ======
    total_wall_start = time.perf_counter()
    total_train_time_accum = 0.0
    per_epoch_train_times: List[float] = []
    per_epoch_wall_times: List[float] = []

    def _is_better(new_val: float) -> bool:
        return (new_val > best_metric_val) if greater_is_better else (new_val < best_metric_val)
    for ep in range(1, epochs + 1):
        epoch_wall_start = time.perf_counter()

        model.train()
        running_loss = 0.0
        loss_since_log = 0.0
        st = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        # ---- （ 1  epoch ）----
        if progress:
            print("\n" +
                  f"{'Epoch':>{W_EPOCH}}{SEP}"
                  f"{'GPU_mem':>{W_MEM}}{SEP}"
                  f"{'L':>{W_NUM}}{SEP}"
                  f"{'avg':>{W_NUM}}{SEP}"
                  f"{'lr':>{W_NUM}}{SEP}"
                  f"{'step':>{W_STEP}}")
        train_iter = tqdm(
            train_loader,
            desc=f"Train [ep {ep}/{epochs}]",
            leave=False,
            dynamic_ncols=True,
            disable=(not progress),
        )

        for step, batch in enumerate(train_iter, start=1):
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}

            try:
                with torch.amp.autocast("cuda", enabled=(fp16 and device.startswith("cuda"))):
                    out = model(**_filter_forward_kwargs(model, batch))
                    loss = _compute_loss(out.logits, batch["labels"], smoothing=label_smoothing)
                scaler.scale(loss / max(1, grad_accum)).backward()
                oom_fallback_used = False
            except RuntimeError as e:
                if not _is_oom_error(e):
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                ok = _split_and_backward(
                    model, batch, device=device, fp16=fp16, grad_accum=grad_accum, scaler=scaler
                )
                if not ok:
                    raise
                oom_fallback_used = True

            if step % grad_accum == 0:
                if max_grad_norm and max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                # （ step  loss  update）
                step_indices.append(global_step)
                step_losses.append(float(loss.item()))

            # &
            loss_val = float(loss.item())
            running_loss += loss_val
            loss_since_log += loss_val

            if progress:
                avg_loss = running_loss / max(1, step)
                try:
                    cur_lr = scheduler.get_last_lr()[0]
                except Exception:
                    cur_lr = optimizer.param_groups[0]["lr"]

                # GPU （reserved）
                if torch.cuda.is_available():
                    mem = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'
                else:
                    mem = "0G"

                # style （ one_loss ）
                desc = (
                    f"{f'{ep}/{epochs}':>{W_EPOCH}}{SEP}"
                    f"{mem:>{W_MEM}}{SEP}"
                    f"{loss_val:>{W_NUM}.4f}{SEP}"
                    f"{avg_loss:>{W_NUM}.4f}{SEP}"
                    f"{cur_lr:>{W_NUM}.2e}{SEP}"
                    f"{global_step:>{W_STEP}d}"
                )
                train_iter.set_description(desc)
            if (val_loader is not None) and (eval_strategy == "steps") and (eval_interval > 0) \
               and (global_step > 0) and (global_step % eval_interval == 0) and (step % grad_accum == 0):
                val_acc, val_loss = _evaluate_loop(model, val_loader, device=device, smoothing_eval=0.0, fp16=fp16)
                metric_val = val_acc if metric_for_best == "val_acc" else (-val_loss if greater_is_better else val_loss)
                print(f"[eval @ step {global_step}] val_acc={val_acc:.4f} val_loss={val_loss:.4f}")

                if _is_better(metric_val):
                    best_metric_val = metric_val
                    no_improve_cnt = 0
                    best_dir = os.path.join(output_dir, "best")
                    os.makedirs(best_dir, exist_ok=True)
                    model.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)
                else:
                    no_improve_cnt += 1

                if isinstance(early_patience, int) and early_patience > 0 and no_improve_cnt >= early_patience:
                    print(f"[early stop] no improvement for {early_patience} evals (steps).")
                    break
            if False and (save_strategy == "steps") and (save_interval > 0) and (global_step > 0) \
               and (global_step % save_interval == 0) and (step % grad_accum == 0):
                step_dir = os.path.join(output_dir, f"step{global_step}")
                os.makedirs(step_dir, exist_ok=True)
                model.save_pretrained(step_dir)
                tokenizer.save_pretrained(step_dir)

        train_time = time.perf_counter() - st
        total_train_time_accum += train_time
        per_epoch_train_times.append(train_time)
        avg_train_loss = running_loss / max(1, len(train_loader))

        # epoch
        val_acc = None
        val_loss = None
        if val_loader is not None and (eval_strategy == "epoch"):
            val_acc, val_loss = _evaluate_loop(model, val_loader, device=device, smoothing_eval=0.0, fp16=fp16)
            metric_val = val_acc if metric_for_best == "val_acc" else (-val_loss if greater_is_better else val_loss)
            print(f"[Epoch {ep}] train_loss={avg_train_loss:.4f}  val_acc={val_acc:.4f}  val_loss={val_loss:.4f}  train_time={train_time:.1f}s")

            if _is_better(metric_val):
                best_metric_val = metric_val
                no_improve_cnt = 0
                best_dir = os.path.join(output_dir, "best")
                os.makedirs(best_dir, exist_ok=True)
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
            else:
                no_improve_cnt += 1

            if isinstance(early_patience, int) and early_patience > 0 and no_improve_cnt >= early_patience:
                print(f"[early stop] no improvement for {early_patience} evals (epochs).")
                pass

        elif val_loader is None:
            print(f"[Epoch {ep}] train_loss={avg_train_loss:.4f}  train_time={train_time:.1f}s")

        # epoch
        # epoch （； best/last）
        if save_every_epoch and (save_strategy == "epoch"):
            ep_dir = os.path.join(output_dir, f"ep{ep}")
            os.makedirs(ep_dir, exist_ok=True)
            model.save_pretrained(ep_dir)
            tokenizer.save_pretrained(ep_dir)

        # last
        last_dir = os.path.join(output_dir, "last")
        os.makedirs(last_dir, exist_ok=True)
        model.save_pretrained(last_dir)
        tokenizer.save_pretrained(last_dir)

        epoch_wall_time = time.perf_counter() - epoch_wall_start
        per_epoch_wall_times.append(epoch_wall_time)

        history.append({
            "epoch": ep,
            "avg_train_loss": avg_train_loss,
            "train_time_sec": train_time,
            "epoch_wall_time_sec": epoch_wall_time,
            "val_acc": val_acc,
            "val_loss": val_loss,
            "global_step": global_step,
        })

        if isinstance(early_patience, int) and early_patience > 0 and no_improve_cnt >= early_patience:
            break

    # ====== ： &  ======
    mem_stats = _collect_cuda_peaks(mem_ctx)
    total_wall_time = time.perf_counter() - total_wall_start

    # ====== （） ======
    loss_plot_path = None
    if save_loss_plot:
        loss_plot_path = _save_loss_plot(
            steps=step_indices,
            losses=step_losses,
            out_dir=output_dir,
            filename=loss_plot_filename,
            smooth_window=loss_plot_smooth_window,
        )
        if loss_plot_path:
            print(f"[plot] saved training loss curve -> {loss_plot_path}")

    # ====== （） ======
    best_val_acc = None
    if val_loader is not None and metric_for_best == "val_acc" and best_metric_val not in (-float("inf"), float("inf")):
        best_val_acc = best_metric_val if greater_is_better else -best_metric_val
    # ====== (optional) Test evaluation at end ======
    test_acc = None
    test_loss = None
    test_used = None  # "best" or "last"
    import torch.nn.functional as F

    @torch.no_grad()
    def _evaluate_loop_debug(model, dataloader, device: str, ai_label_id: int = 1, fp16: bool = True):
        model.eval()
        tot = 0
        c_argmax = 0
        c_prob05 = 0
        c_logitai0 = 0
        c_logitdiff0 = 0

        for batch in dataloader:
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=(fp16 and device.startswith("cuda"))):
                out = model(**_filter_forward_kwargs(model, batch))
            logits = out.logits.float()  # float32 ， fp16
            y = batch["labels"]

            # 1) argmax
            pred_argmax = logits.argmax(dim=-1)
            c_argmax += int((pred_argmax == y).sum().item())

            # 2) softmax p(ai)>0.5（ argmax ）
            probs = F.softmax(logits, dim=-1)
            p_ai = probs[:, ai_label_id]
            other = 1 - ai_label_id
            pred_prob05 = torch.where(p_ai >= 0.5, torch.full_like(y, ai_label_id), torch.full_like(y, other))
            c_prob05 += int((pred_prob05 == y).sum().item())

            # 3)  logit_ai > 0
            logit_ai = logits[:, ai_label_id]
            pred_logitai0 = torch.where(logit_ai > 0, torch.full_like(y, ai_label_id), torch.full_like(y, other))
            c_logitai0 += int((pred_logitai0 == y).sum().item())

            # 4)  logit_ai - logit_other > 0（ argmax ）
            logit_other = logits[:, other]
            diff = logit_ai - logit_other
            pred_diff0 = torch.where(diff > 0, torch.full_like(y, ai_label_id), torch.full_like(y, other))
            c_logitdiff0 += int((pred_diff0 == y).sum().item())

            tot += int(y.numel())

        def _fmt(x): return f"{x / max(1, tot):.4f}"

    # ： best checkpoint （ cfg.eval_test_on_best ）
    eval_test_on_best = bool(getattr(cfg, "eval_test_on_best", True))

    if test_loader is not None:
        eval_model = model
        if eval_test_on_best and best_dir:
            try:
                eval_model = model.__class__.from_pretrained(best_dir)
                eval_model.to(device)
                test_used = "best"
            except Exception as e:
                print(f"[test] reload best from {best_dir} failed, fallback to last. err={e}")
                eval_model = model
                test_used = "last"
        else:
            test_used = "last"
        if test_loader is not None:
            _evaluate_loop_debug(eval_model, test_loader, device=device, ai_label_id=1, fp16=fp16)
        test_acc, test_loss = _evaluate_loop(eval_model, test_loader, device=device, smoothing_eval=0.0, fp16=fp16)

        if test_acc is not None and test_loss is not None:
            print(f"[test] ({test_used}) test_acc={test_acc:.4f} test_loss={test_loss:.4f}")

        if (eval_model is not model):
            try:
                del eval_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    summary = {
        "best_dir": best_dir,
        "last_dir": last_dir or output_dir,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,            # ✅ NEW
        "test_loss": test_loss,          # ✅ NEW
        "test_used": test_used,          # ✅ NEW: best/last
        "data": data_info, 
        "history": history,
        "step_indices": step_indices,
        "step_losses": step_losses,
        "memory": mem_stats,
        "timing": {
            "total_train_time_sec": total_train_time_accum,
            "total_wall_time_sec": total_wall_time,
            "per_epoch_train_time_sec": per_epoch_train_times,
            "per_epoch_wall_time_sec": per_epoch_wall_times,
            "avg_epoch_train_time_sec": (sum(per_epoch_train_times) / len(per_epoch_train_times)) if per_epoch_train_times else 0.0,
            "avg_epoch_wall_time_sec": (sum(per_epoch_wall_times) / len(per_epoch_wall_times)) if per_epoch_wall_times else 0.0,
        },
        "artifacts": {
            "args_json": args_json_path,
            "summary_json": os.path.join(output_dir, "train_summary.json"),
            "loss_plot": loss_plot_path,
        },
    }

    summary_json_path = summary["artifacts"]["summary_json"]
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    def _dump_meta(meta_dir: Optional[str], tag: str):
        if not meta_dir:
            return
        try:
            os.makedirs(meta_dir, exist_ok=True)
            meta = {
                "tag": tag,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "best_val_acc": best_val_acc,
            }
            with open(os.path.join(meta_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    _dump_meta(best_dir, "best")
    _dump_meta(last_dir or output_dir, "last")

    return {
        "best_dir": best_dir,
        "last_dir": last_dir or output_dir,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,            # ✅ NEW
        "test_loss": test_loss,          # ✅ NEW
        "test_used": test_used,          # ✅ NEW
        "history": history,
        "step_indices": step_indices,
        "step_losses": step_losses,
        "memory": mem_stats,
        "timing": summary["timing"],
        "artifacts": summary["artifacts"],
    }
