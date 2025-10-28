# quickstart.py
import argparse
from mgt_eval.detectors import NPR  # 这就是“同名运行器”

def parse_args():
    ap = argparse.ArgumentParser()
    # —— 评测参数 —— #
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--sample_k", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--save_curves", action="store_true", default=True)

    # —— detector 参数（仅保留 NPR 需要的接口）—— #
    ap.add_argument("--score_model", type=str, required=True, help="评分模型 id/路径（CausalLM）")
    ap.add_argument("--mask_model", type=str, default="t5-small", help="掩码填充模型（T5）")
    ap.add_argument("--pct_words_masked", type=float, default=0.3)
    ap.add_argument("--span_length", type=int, default=2)
    ap.add_argument("--n_perturbation", type=int, default=100)
    ap.add_argument("--chunk_size", type=int, default=20)
    ap.add_argument("--buffer_size", type=int, default=1)
    ap.add_argument("--mask_top_p", type=float, default=1.0)
    ap.add_argument("--max_len", type=int, default=400)
    ap.add_argument("--device", type=str, default=None, help='如 "cuda:0" 或 "cpu"')
    ap.add_argument("--name", type=str, default=None)

    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()

    NPR(
        # —— 评测参数 —— #
        data=args.data,
        out=args.out,
        save_curves=args.save_curves,
        sample_k=args.sample_k,
        batch_size=args.batch_size,
        threshold=args.threshold,

        # —— detector 参数（透传到 NPRDetector.__init__）—— #
        score_model=args.score_model,
        mask_model=args.mask_model,
        pct_words_masked=args.pct_words_masked,
        span_length=args.span_length,
        n_perturbation=args.n_perturbation,
        chunk_size=args.chunk_size,
        buffer_size=args.buffer_size,
        mask_top_p=args.mask_top_p,
        max_len=args.max_len,
        device=args.device,
        use_bfloat16=True,
        name=args.name or "NPR",
    )
