# quickstart.py
import argparse
from mgt_eval.detectors import LRR  # 这就是“同名运行器”

def parse_args():
    ap = argparse.ArgumentParser()
    # —— 评测参数 —— #
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--sample_k", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.5)  # 若为 None，由评测框架自行搜索/绘制曲线
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--save_curves", action="store_true", default=True)

    # —— detector 参数（仅保留 LRR 需要的接口）—— #
    ap.add_argument("--score_model", type=str, required=True, help="评分模型 id/路径（如 gpt2 或本地目录）")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--device", type=str, default=None, help='如 "cuda:0" 或 "cpu"')
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--k_runs", type=str, default=1)
    # ===== 新增：加载已训练好的校准器 JSON =====
    ap.add_argument(
        "--calibrator",
        type=str,
        default=None,
        help="Path to a calibrator JSON (e.g., output of Calibrate(...)). "
             "If provided, Binoculars scores will be mapped to probabilities via the learned LR."
    )
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()

    LRR(
        # —— 评测参数 —— #
        data=args.data,
        out=args.out,
        save_curves=args.save_curves,
        sample_k=args.sample_k,
        batch_size=args.batch_size,
        threshold=args.threshold,

        # —— detector 参数（透传到 LRRDetector.__init__）—— #
        score_model=args.score_model,
        max_length=args.max_len,
        device=args.device,
        use_bfloat16=True,
        name=args.name or "LRR",
        k_runs=args.k_runs,
        calibrator_path=args.calibrator
    )
