# quickstart_gltr_named.py
import argparse
from mgt_eval.detectors import GLTR  # 同名运行器

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--score_model", type=str, required=True)          # GLTR 只需单模型
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.5)      # 框架需要的统一参数；GLTR+LR 后会输出概率
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)

    # —— 新增：测试集抽样（与 Binoculars 一致）——
    ap.add_argument("--sample_k", type=int, default=2000, help="评测时从数据集中抽样的条数（None 表示全量）")

    # —— 新增：小样本标定 —— #
    ap.add_argument("--calib_k", type=int, default=1000, help="用于逻辑回归标定的样本数")
    ap.add_argument("--calib_seed", type=int, default=42, help="标定样本抽样随机种子")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()

    GLTR(
        # —— 评测参数 —— #
        data=args.data,
        out=args.out,
        save_curves=True,
        batch_size=args.batch_size,
        threshold=args.threshold,
        sample_k=args.sample_k,  # 评测抽样：如 Binoculars 一样

        # —— detector 参数（透传）—— #
        score_model=args.score_model,
        max_token_observed=args.max_len,
        device=args.device,
        use_bfloat16=True,
        name="GLTR",

        # —— 小样本标定参数（传给 GLTRDetector，在 calibrate() 中生效）—— #
        calibrate_k=args.calib_k,
        calibrate_seed=args.calib_seed,
    )
