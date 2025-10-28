# quickstart_lastdepp.py
# 用法示例：
#   Lastde（单模型）：
#     python quickstart_lastde.py --method lastde --data /path/to/dataset_prefix \
#       --model /path/to/scoring_model --out /path/to/out.json
#
#   Lastde++（双模型+采样）：
#     python quickstart_lastde.py --method lastde++ --data /path/to/dataset_prefix \
#       --scoring_model /path/to/scoring_model --reference_model /path/to/reference_model \
#       --n_samples 100 --out /path/to/out.json
#
# 说明：
# - data 采用与你示例一致的“前缀”格式：程序内部会读取  <data>.raw_data.json
# - 脚本一次只运行一种方法；通过 --method 在 {lastde, lastde++} 中选择
# - 评测侧参数（data/out/save_curves/sample_k/batch_size/threshold）与 Binoculars quickstart 对齐
# - 检测器参数按你提供实现：Lastde 使用单模型；Lastde++ 使用 scoring+reference 两模型与采样

import argparse
from mgt_eval.detectors import Lastde, LastdePP  # “同名运行器”：已在注册表中对应 lastde / lastde++
                                                  # 注意：Lastde++ 的 Python 标识为 LastdePP

def parse_args():
    ap = argparse.ArgumentParser()
    # —— 通用评测参数 —— #
    ap.add_argument("--data", type=str, required=True, help="数据前缀路径")
    ap.add_argument("--out", type=str, default=None, help="结果输出路径（JSON）")
    ap.add_argument("--save_curves", action="store_true", help="保存 ROC/PR 曲线数据")
    ap.add_argument("--sample_k", type=int, default=100, help="评测时从数据集中抽样的条数")
    ap.add_argument("--batch_size", type=int, default=4, help="评测 batch size")
    ap.add_argument("--threshold", type=float, default=0.5, help="将连续分数映射为二分类的阈值（用于曲线/统计）")

    # —— 通用 detector 参数 —— #
    ap.add_argument("--max_len", type=int, default=512, help="最大截断长度")
    ap.add_argument("--prob_slope", type=float, default=-6.0, help="logistic 概率映射斜率")
    ap.add_argument("--device", type=str, default=None, help="优先设备，如 cuda:0 / cpu")

    # —— Lastde++（双模型+采样）专属 —— #
    ap.add_argument("--score_model", type=str, default=None, help="Lastde++ 的 scoring 模型路径/ID")
    ap.add_argument("--reference_model", type=str, default=None, help="Lastde++ 的 reference 模型路径/ID（可不填=同 scoring）")
    ap.add_argument("--n_samples", type=int, default=100, help="Lastde++ 参考采样数")
    ap.add_argument("--tau_prime_pp", type=int, default=15, help="Lastde++ 多尺度上限 tau'（缺省 15）")
    ap.add_argument("--embed_size_pp", type=int, default=4, help="Lastde++ 的 embedding 尺度（缺省 4）")
    ap.add_argument("--epsilon_pp", type=float, default=8.0, help="Lastde++ 的 epsilon 系数（缺省 8.0）")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    LastdePP(
            # —— 评测侧参数 —— #
            data=args.data,
            out=args.out,
            save_curves=args.save_curves,
            sample_k=args.sample_k,
            batch_size=args.batch_size,
            threshold=args.threshold,

            # —— detector 参数（透传）—— #
            score_model=args.score_model,
            reference_model=args.reference_model,
            max_token_observed=args.max_len,
            prob_slope=args.prob_slope,
            embed_size=args.embed_size_pp,
            epsilon_mult=args.epsilon_pp,
            tau_prime=args.tau_prime_pp,
            n_samples=args.n_samples,
            device=args.device,
            use_bfloat16=False,
            name=f"Lastde++",
    )
