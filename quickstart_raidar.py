# quickstart_raidar_named.py
import argparse
from mgt_eval.detectors import RAIDAR  # 同名运行器

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)

    # —— 评测控制（与 GLTR 对齐）——
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--sample_k", type=int, default=50, help="评测抽样（None=全量）")

    # —— 小样本标定 —— #
    ap.add_argument("--calib_k", type=int, default=32, help="用于 MLP 标定的样本数")
    ap.add_argument("--calib_seed", type=int, default=42, help="标定样本抽样随机种子")

    # —— 改写器参数 —— 
    ap.add_argument("--rewrite_model", type=str, required=True, help="LLaMA 路径或 HF 名称（优先使用）")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--max_new_tokens_factor", type=float, default=0.1)

    # —— 新增：输入截断 k（包含提示词 + 原文） —— #
    ap.add_argument("--rewrite_k", type=int, default=400, help="改写时传入模型的最大 token 数（含提示词）")

    # 可选：OpenAI 回退
    ap.add_argument("--use_openai", action="store_true")
    ap.add_argument("--openai_model", type=str, default="gpt-3.5-turbo")

    ap.add_argument("--name", type=str, default="RAIDAR")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()

    RAIDAR(
        # —— 评测参数 —— #
        data=args.data,
        out=args.out,
        save_curves=True,
        batch_size=args.batch_size,
        threshold=args.threshold,
        sample_k=args.sample_k,

        # —— detector 参数（后端完成数据读取/训练/评估）—— #
        rewrite_model=args.rewrite_model,
        device=args.device,
        max_new_tokens_factor=args.max_new_tokens_factor,
        rewrite_input_max_tokens=args.rewrite_k,  # <<<<<< 新增：传入 k
        use_openai=args.use_openai,
        openai_model=args.openai_model,
        name=args.name,

        # —— 小样本标定 —— #
        calibrate_k=args.calib_k,
        calibrate_seed=args.calib_seed,
    )
