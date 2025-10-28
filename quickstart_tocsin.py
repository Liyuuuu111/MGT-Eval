# quickstart_tocsin.py
import argparse
from mgt_eval.detectors import TOCSIN  # 这就是“同名运行器”

def parse_args():
    ap = argparse.ArgumentParser()
    # —— 评测数据与通用参数 —— #
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--sample_k", type=int, default=1000)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", type=str, default=None)

    # —— TOCSIN 模型参数（对应 TOCSINDetector 的入参）—— #
    ap.add_argument("--score_model", type=str, required=True, help="score_model")
    ap.add_argument("--reference_model", type=str, required=True, help="reference_model")
    ap.add_argument("--basemodel", type=str, default="Fast",
                    choices=["Fast", "lrr", "likelihood", "logrank", "standalone"])
    ap.add_argument("--max_len", type=int, default=512)  # max_token_observed
    ap.add_argument("--mask_pct", type=float, default=0.015)
    ap.add_argument("--perturb_k", type=int, default=10)  # 每条文本生成的扰动样本数
    ap.add_argument("--bart_ckpt", type=str, default="facebook/bart-base")
    ap.add_argument("--dataset_file", type=str, default=None,
                    help="仅用于复现脚本中的特殊分支（例如包含 'gemini' 且包含 'pubmed' 时）")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()

    TOCSIN(
        # —— 评测参数 —— #
        data=args.data,
        out=args.out,
        save_curves=True,
        sample_k=args.sample_k,
        batch_size=args.batch_size,
        threshold=args.threshold,

        # —— detector 参数（透传）—— #
        score_model=args.score_model,
        reference_model=args.reference_model,
        basemodel=args.basemodel,
        max_token_observed=args.max_len,
        mask_pct=args.mask_pct,
        perturb_per_text=args.perturb_k,
        bart_checkpoint=args.bart_ckpt,
        dataset_file=args.dataset_file,
        device=args.device,
        use_bfloat16=True,
        name=f"TOCSIN[{args.basemodel}]",
    )
