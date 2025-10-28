# mgt_eval/cli.py
import argparse, json, os, time
from pathlib import Path

from mgt_eval.data_utils.load import load_dataset_unified
from mgt_eval.eval.evaluator import evaluate_detector as _eval
from mgt_eval.detectors.registry import get_detector_cls
from mgt_eval.calibration.runner import Calibrate as _calibrate, _build_detector

def _now():
    return time.strftime("%Y%m%d-%H%M%S")

def _import_all_detectors():
    """动态 import mgt_eval.detectors 包下所有子模块，触发 @register。"""
    import pkgutil, importlib
    import mgt_eval.detectors as pkg
    for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        try:
            importlib.import_module(m.name)
        except Exception as e:
            print(f"[MGTEval][list] skip {m.name}: {e}")

def cmd_list(_args):
    # 1) 预加载：尝试导入所有 detectors 子模块（触发 @register）
    from mgt_eval.detectors import ensure_all_detectors_registered
    ensure_all_detectors_registered()  # 轻量：内部自行 walk_packages

    # 2) 读取注册表
    try:
        from mgt_eval.detectors.registry import list_registered_detectors
        dets = list_registered_detectors()
    except Exception:
        # 兜底：直接访问 REGISTRY
        from mgt_eval.detectors.registry import REGISTRY as _REG
        dets = sorted(_REG.keys())

    if dets:
        print("\n".join(sorted(dets)))
    else:
        print("(no detectors found)")

def cmd_run(args):
    _import_all_detectors()
    # 载入数据（可抽样 sample_k）
    examples, _ = load_dataset_unified(
        dataset=args.data,
        sample_k=(None if (args.sample_k is None or args.sample_k <= 0) else int(args.sample_k)),
        sample_seed=int(args.seed),
        group_cols=None,
    )

    # 构造 detector（尽量复用已有的 _build_detector，以对齐你各家 detector 的入参映射）
    det = _build_detector(
        detector_name=args.detector,
        model1=args.model1,
        model2=args.model2,
        device=args.device,
        use_bfloat16=bool(args.bf16),
        detector_kwargs=(json.loads(args.detector_kwargs) if args.detector_kwargs else None),
        basemodel=args.basemodel,
        bart_ckpt=args.bart_ckpt,
    )

    res = _eval(
        detector=det,
        dataset=examples,
        batch_size=int(args.batch_size),
        threshold=float(args.threshold),
        show_progress=(not args.no_progress),
        out_dir=args.out,           # evaluator 内部会落盘曲线/结果
        save_curves=args.save_curves,
        k_runs=args.k_runs,                   # 单次
    )

def cmd_calibrate(args):
    _import_all_detectors()
    _ = _calibrate(
        detector=args.detector,
        model1=args.model1,
        model2=args.model2,
        data=args.data,
        batch_size=int(args.batch_size),
        sample_k=int(args.sample_k),
        seed=int(args.seed),
        device=args.device,
        bf16=bool(args.bf16),
        detector_kwargs=(json.loads(args.detector_kwargs) if args.detector_kwargs else None),
        basemodel=args.basemodel,
        bart_ckpt=args.bart_ckpt,
        calibrator_name=args.calibrator_name,
        l2=float(args.l2),
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        standardize=not args.no_standardize,
        out=args.out,
        out_dir=args.out_dir,
        show_progress=(not args.no_progress),
    )

def main(argv=None):
    ap = argparse.ArgumentParser(prog="mgt-eval", description="Unified CLI for MGT-Eval")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # list
    ap_list = sub.add_parser("list", help="List available detectors")
    ap_list.set_defaults(_fn=cmd_list)

    # run
    ap_run = sub.add_parser("run", help="Run detector on a dataset")
    ap_run.add_argument("--detector", required=True,
                        help="e.g. lastde, lastdepp, gltr, fastdetectgpt, likelihood, rank, logrank, entropy, binoculars...")
    ap_run.add_argument("--data", required=True)
    ap_run.add_argument("--model1", "--model", dest="model1", required=True)
    ap_run.add_argument("--model2", default=None)
    ap_run.add_argument("--batch_size", type=int, default=8)
    ap_run.add_argument("--threshold", type=float, default=0.5)
    ap_run.add_argument("--seed", type=int, default=114514)
    ap_run.add_argument("--sample_k", type=int, default=1000)
    ap_run.add_argument("--device", default=None)
    ap_run.add_argument("--bf16", type=bool, default=True)
    ap_run.add_argument("--detector_kwargs", default=None,
                        help='JSON string, e.g. \'{"max_length":512}\'')
    ap_run.add_argument("--basemodel", default=None)
    ap_run.add_argument("--bart_ckpt", default=None)
    ap_run.add_argument("--out", default=None)
    ap_run.add_argument("--save_curves", type=bool, default=True)
    ap_run.add_argument("--no_progress", action="store_true")
    ap_run.add_argument("--k_runs", type=int, default=1)
    ap_run.set_defaults(_fn=cmd_run)

    # calibrate
    ap_cal = sub.add_parser("calibrate", help="Fit and save a calibrator JSON")
    ap_cal.add_argument("--detector", required=True)
    ap_cal.add_argument("--data", required=True)
    ap_cal.add_argument("--model1", required=True)
    ap_cal.add_argument("--model2", default=None)
    ap_cal.add_argument("--batch_size", type=int, default=32)
    ap_cal.add_argument("--sample_k", type=int, default=10000)
    ap_cal.add_argument("--seed", type=int, default=114514)
    ap_cal.add_argument("--device", default=None)
    ap_cal.add_argument("--bf16", action="store_true")
    ap_cal.add_argument("--detector_kwargs", default=None,
                        help='JSON string for detector extra kwargs')
    ap_cal.add_argument("--basemodel", default=None)
    ap_cal.add_argument("--bart_ckpt", default=None)
    ap_cal.add_argument("--calibrator_name", default="platt_lr")
    ap_cal.add_argument("--l2", type=float, default=1e-2)
    ap_cal.add_argument("--max_iter", type=int, default=200)
    ap_cal.add_argument("--tol", type=float, default=1e-6)
    ap_cal.add_argument("--no_standardize", action="store_true")
    ap_cal.add_argument("--out", default=None)
    ap_cal.add_argument("--out_dir", default=None)
    ap_cal.add_argument("--no_progress", action="store_true")
    ap_cal.set_defaults(_fn=cmd_calibrate)

    args = ap.parse_args(argv)
    args._fn(args)

if __name__ == "__main__":
    main()
