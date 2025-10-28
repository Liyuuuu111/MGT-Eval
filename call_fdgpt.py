# quickstart_fastdetectgpt.py
"""
Quickstart:
  python quickstart_fastdetectgpt.py \
    --data /hpc_stor03/sjtu_home/yuanfan.li/taste/hc3/hc3_en.jsonl \
    --model /hpc_stor03/sjtu_home/yuanfan.li/detect/model/gpt-neo-2.7B \
    --sample_k 100 \
    --out ./runs_fastdetectgpt
"""
import argparse
from mgt_eval.detectors import FastDetectGPT

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="数据集路径（可为 HC3 json/jsonl 或标准 {'text','label'}）")
    ap.add_argument("--score_model", type=str, required=True, help="评分模型目录或 HF ID")
    ap.add_argument("--sample_model", type=str, required=True, help="采样模型目录或 HF ID")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--fp16", action="store_true", help="CUDA 上使用半精度")
    ap.add_argument("--sample_k", type=int, default=100, help="抽样条数；<=0 表示全量")
    ap.add_argument("--out", type=str, default=None, help="输出目录")
    ap.add_argument("--group_cols", type=str, default=None,
                    help="逗号分隔的分组列（如: lang,source,model,sub_source）；缺省则自动探测存在列")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # 解析分组列
    group_cols = None
    if args.group_cols:
        group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]

    # 直接调用通用评测接口；统计/可视化会在 evaluator 内自动落盘
    FastDetectGPT(
        data=args.data,
        sample_k=args.sample_k,
        batch_size=args.batch_size,
        threshold=args.threshold,
        out_dir=args.out,
        save_curves=True,
        group_cols=group_cols,
        # fast-detect-gpt 专属参数
        scoring_model_name=args.score_model,
        sampling_model_name=args.sample_model,
        tokenizer_name=args.sample_model,
        fp16=args.fp16,
        use_analytic=True,
        # 可选：自定义经验参数（未提供则使用内置/自动回退）
        # distrib_params={ "gpt-neo-2.7B_gpt-neo-2.7B": {...} },
        max_length=args.max_length,
    )
