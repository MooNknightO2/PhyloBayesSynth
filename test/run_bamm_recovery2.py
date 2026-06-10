"""BAMM 模型恢复实验: 异质灭绝率设计 (BAMM 信号强)。

真实模型: (BAMM 0.03 (CRBD 0.5 0.35) (CRBD 0.5 0.05))
  bg = CRBD(λ=0.5, μ=0.35)  高灭绝, 净速率 r=0.15
  fg = CRBD(λ=0.5, μ=0.05)  低灭绝, 净速率 r=0.45
  η  = 0.03                 稀少切换

运行:
  python test/run_bamm_recovery2.py [--n_iter 5000]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.algorithm.data_utils import load_phylojson, get_tree_data
from train.algorithm.expression import parse_expression
from train.algorithm.likelihood import (
    log_likelihood_crb, log_likelihood_crbd, log_likelihood_bamm,
)
from train.algorithm.inference import bayesian_synthesis_mcmc

DATA_PATH = ROOT / "data" / "generated" / "BAMM_hetero_extinction_n80_seed314.phylojson"
TRUE_EXPR_STR = "(BAMM 0.03 (CRBD 0.5 0.35) (CRBD 0.5 0.05))"

parser = argparse.ArgumentParser()
parser.add_argument("--n_iter", type=int, default=5000)
parser.add_argument("--seed", type=int, default=777)
args = parser.parse_args()

# ============================================================
print("=" * 65)
print("BAMM 模型恢复实验 — 异质灭绝率")
print("=" * 65)

print(f"\n真实模型: {TRUE_EXPR_STR}")
print("  bg = CRBD(λ=0.5, μ=0.35)  高灭绝, 净速率 r=0.15")
print("  fg = CRBD(λ=0.5, μ=0.05)  低灭绝, 净速率 r=0.45")
print("  η  = 0.03                 稀少切换")

root = load_phylojson(str(DATA_PATH))
td = get_tree_data(root)
print(f"\n树统计: n={td['n']}, T={td['tree_height']:.4f}, S={td['total_length']:.4f}")

# ============================================================
print("\n--- 基线似然对比 ---")
true_expr = parse_expression(TRUE_EXPR_STR)
ll_true = log_likelihood_bamm(true_expr, td)
lam_mle = (td["n"] - 1) / td["total_length"]
ll_crb_mle = log_likelihood_crb(lam_mle, td)
ll_bg = log_likelihood_crbd(0.5, 0.35, td)
ll_fg = log_likelihood_crbd(0.5, 0.05, td)

print(f"  BAMM 真实参数              log L = {ll_true:.4f}")
print(f"  CRB(λ_MLE={lam_mle:.4f})          log L = {ll_crb_mle:.4f}")
print(f"  CRBD(0.5, 0.35) bg-only       log L = {ll_bg:.4f}")
print(f"  CRBD(0.5, 0.05) fg-only       log L = {ll_fg:.4f}")
delta = ll_true - ll_crb_mle
print(f"  Δ(BAMM - CRB_MLE) = {delta:+.4f}  "
      f"{'BAMM wins' if delta > 0 else 'CRB wins'}")

# ============================================================
print(f"\n{'='*65}")
print(f"运行 MCMC ({args.n_iter} 轮, seed={args.seed})...")
print("=" * 65)

np.random.seed(args.seed)
result = bayesian_synthesis_mcmc(td, n_iter=args.n_iter, verbose=True)

# ============================================================
print(f"\n{'='*65}")
print("MCMC 结果汇总")
print("=" * 65)

n_samples = len(result["samples"])
print("\n模型后验频率:")
for tag, cnt in sorted(result["model_counts"].items(), key=lambda x: -x[1]):
    pct = 100 * cnt / n_samples
    bar = "█" * int(pct / 2)
    print(f"  {tag:6s}: {cnt:4d}/{n_samples} ({pct:5.1f}%) {bar}")

print("\n各模型类别最优表达式:")
for tag in sorted(result["best_per_model"].keys()):
    expr, ll = result["best_per_model"][tag]
    model_node = expr.children[1]
    sigma = expr.children[0].params[0]
    print(f"\n  [{tag}] log L = {ll:.4f}, σ={sigma:.4f}")
    print(f"    {model_node}")

best_expr = result["best_expr"]
best_ll = result["best_log_L"]
print(f"\n{'—'*65}")
print(f"全局最优: log L = {best_ll:.4f}")
print(f"  {best_expr}")
print(f"\n真实模型: log L = {ll_true:.4f}")
print(f"  {TRUE_EXPR_STR}")
print(f"\nCRB MLE:  log L = {ll_crb_mle:.4f}  (λ={lam_mle:.4f})")
