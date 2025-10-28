# quickstart_detectgpt.py
import argparse
from mgt_eval.detectors import DetectGPT  # 同名运行器

def parse_args():
    ap = argparse.ArgumentParser()
    # —— 评测参数 —— #
    ap.add_argument("--data", type=str, required=True, help="待评测数据文件路径（如 .jsonl）")
    ap.add_argument("--out", type=str, default=None, help="结果输出文件")
    ap.add_argument("--batch_size", type=int, default=4, help="评测时的 batch 大小")
    ap.add_argument("--sample_k", type=int, default=100, help="从数据中抽样评测的样本数上限")
    ap.add_argument("--threshold", type=float, default=0.5, help="概率阈值（>threshold 判为 AI）")

    # —— DetectGPT detector 参数（透传到 DetectGPTDetector.__init__）—— #
    ap.add_argument("--score_model", type=str, required=True, help="用于对数似然评估的 CausalLM（如 gpt2-medium）")
    ap.add_argument("--mask_model", type=str, default="t5-large", help="用于掩码填空扰动的 Seq2Seq（如 t5-large）")
    ap.add_argument("--pct", type=float, default=0.3, help="扰动比例（近似为被替换 token 占比）")
    ap.add_argument("--span_len", type=int, default=2, help="每个掩码片段的 token 数")
    ap.add_argument("--n_perturb", type=int, default=5, help="每条文本生成的扰动样本数")
    ap.add_argument("--buffer_size", type=int, default=1, help="掩码邻域缓冲，避免过密")
    ap.add_argument("--mask_top_p", type=float, default=1.0, help="T5 生成的 top-p")
    ap.add_argument("--max_len", type=int, default=256, help="最大截断长度（送入 base_model）")
    ap.add_argument("--use_zscore", action="store_true", help="使用 z 分数（d/std），默认使用 d=LL(x)-mean(LL(pert))")
    ap.add_argument("--prob_slope", type=float, default=8.0, help="logistic 映射斜率（越大越接近硬阈值）")
    ap.add_argument("--device", type=str, default=None, help="优先设备，如 cuda:0 / cpu；留空自动选择")
    ap.add_argument("--chunk_size", type=int, default=20, help="扰动生成分块大小（面向大 T5 可调小）")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()

    DetectGPT(
        # —— 评测参数 —— #
        data=args.data,
        out=args.out,
        save_curves=True,            # 若框架支持，将保存 ROC/PR 曲线等
        sample_k=args.sample_k,
        batch_size=args.batch_size,
        threshold=args.threshold,

        # —— detector 参数（透传）—— #
        score_model=args.score_model,
        mask_model=args.mask_model,
        pct_words_masked=args.pct,
        span_length=args.span_len,
        n_perturbations=args.n_perturb,
        buffer_size=args.buffer_size,
        mask_top_p=args.mask_top_p,
        max_token_observed=args.max_len,
        use_zscore=args.use_zscore,
        prob_slope=args.prob_slope,
        device=args.device,
        chunk_size=args.chunk_size,
        use_bfloat16=True,
        name=f"DetectGPT[{'z' if args.use_zscore else 'd'}]",
    )
