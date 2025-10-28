# quickstart_binoculars.py
import argparse
from mgt_eval.detectors import Binoculars  # 这就是“同名运行器”

def parse_args():
    ap = argparse.ArgumentParser()
    # ===== 数据与评测参数 =====
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--sample_k", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.5)

    # ===== Binoculars detector 参数 =====
    ap.add_argument("--observer", type=str, required=True)
    ap.add_argument("--performer", type=str, required=True)
    ap.add_argument("--mode", type=str, default="low-fpr", choices=["low-fpr", "accuracy"])
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--prob_slope", type=float, default=8.0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--use_bfloat16", action="store_true", default=True)

    # ===== 新增：加载已训练好的校准器 JSON =====
    ap.add_argument(
        "--calibrator",
        type=str,
        default=None,
        help="Path to a calibrator JSON (e.g., output of Calibrate(...)). "
             "If provided, Binoculars scores will be mapped to probabilities via the learned LR."
    )
    ap.add_argument(
        "--calibrator_name",
        type=str,
        default="platt_lr",
        choices=["platt_lr", "linear_lr"],
        help="Calibrator type used in the JSON. Usually 'platt_lr' for 1D scores."
    )
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()

    Binoculars(
        # —— 评测参数 —— #
        data=args.data,
        out=args.out,
        save_curves=True,
        sample_k=args.sample_k,
        batch_size=args.batch_size,
        threshold=args.threshold,

        # —— detector 参数（透传到检测器构造器）—— #
        observer=args.observer,
        performer=args.performer,
        max_length=args.max_len,
        mode=args.mode,
        prob_slope=args.prob_slope,
        device=args.device,
        use_bfloat16=args.use_bfloat16,
        name=f"Binoculars[{args.mode}]",

        # —— 关键：指定校准器（runner 会把 calibrator → calibrator_path）—— #
        calibrator=args.calibrator,               # e.g. calibrators/calibrator_binoculars_neo_squad.json
        calibrator_name=args.calibrator_name,     # 默认 platt_lr
    )
