# quickstart_dna_gpt.py
import argparse
from mgt_eval.detectors import DNAGPT  # 构造即评测

def parse_args():
    ap = argparse.ArgumentParser()
    # —— 评测级参数 —— #
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--save_curves", action="store_true")
    ap.add_argument("--sample_k", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.5)

    # —— 检测器参数（全部作为 detector kwargs 透传；不要用名字 dataset）—— #
    ap.add_argument("--score_model", type=str, default="gpt2")                 # base_name_or_path
    ap.add_argument("--dataset_name", type=str, default="squad")       # ← 用 dataset_name
    ap.add_argument("--truncate_ratio", type=float, default=0.5)
    ap.add_argument("--regen_number", type=int, default=10)
    ap.add_argument("--max_len", type=int, default=200)                 # max_length（总长度语义）
    ap.add_argument("--min_len_pubmed", type=int, default=50)
    ap.add_argument("--min_len_non_pubmed", type=int, default=150)

    ap.add_argument("--do_top_k", action="store_true")
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--do_top_p", action="store_true")
    ap.add_argument("--top_p", type=float, default=0.96)
    ap.add_argument("--temperature", type=float, default=1.0)

    ap.add_argument("--device", type=str, default=None)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()

    DNAGPT(
        # —— 评测级 —— #
        data=args.data,
        out=args.out,
        save_curves=args.save_curves,
        sample_k=args.sample_k,
        batch_size=args.batch_size,
        threshold=args.threshold,

        # —— 检测器级 —— #
        score_model=args.score_model,
        dataset_name=args.dataset_name,      # ← 只传 dataset_name，避免与评测器重名
        truncate_ratio=args.truncate_ratio,
        regen_number=args.regen_number,
        max_length=args.max_len,
        min_length_pubmed=args.min_len_pubmed,
        min_length_non_pubmed=args.min_len_non_pubmed,

        do_top_k=args.do_top_k,
        top_k=args.top_k,
        do_top_p=args.do_top_p,
        top_p=args.top_p,
        temperature=args.temperature,

        device=args.device,
        use_bfloat16=True,
        name=f"DNAGPT[{args.dataset_name}]",
    )
