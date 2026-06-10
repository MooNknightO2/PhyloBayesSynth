"""BAMM 模型恢复实验: 低切换率, 强对比度。"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.algorithm.data_utils import load_phylojson, get_tree_data
from train.algorithm.expression import Expression, parse_expression
from train.algorithm.likelihood import (
    log_likelihood_crb, log_likelihood_crbd, log_likelihood_tdb,
    log_likelihood_tdbd, log_likelihood_bamm, evaluate_likelihood,
)
from train.algorithm.inference import bayesian_synthesis_mcmc

DATA_PATH = "data/generated/BAMM_strong_n80_seed314.phylojson"
TRUE_EXPR_STR = "(BAMM 0.03 (CRBD 0.15 0.03) (CRBD 0.6 0.05))"

print("=" * 65)
print("BAMM 模型恢复实验 (低 η, 强对比)")
print("=" * 65)

print(f"\n真实模型: {TRUE_EXPR_STR}")
print("  bg = CRBD(λ=0.15, μ=0.03)  — 慢速多样化")
print("  fg = CRBD(λ=0.60, μ=0.05)  — 4x 快速多样化")
print("  η  = 0.03                  — 稀少切换 (平均 ~33 时间单位)")

root = load_phylojson(DATA_PATH)
td = get_tree_data(root)
print(f"\n树统计: n={td['n']}, T={td['tree_height']:.4f}, S={td['total_length']:.4f}")

# --- 各模型在真实参数下的似然 ---
print("\n--- 似然对比 ---")
true_expr = parse_expression(TRUE_EXPR_STR)
ll_true = log_likelihood_bamm(true_expr, td)
print(f"  BAMM 真实参数          log L = {ll_true:.4f}")

lam_mle = (td["n"] - 1) / td["total_length"]
ll_crb_mle = log_likelihood_crb(lam_mle, td)
print(f"  CRB(λ_MLE={lam_mle:.4f})      log L = {ll_crb_mle:.4f}")

ll_crbd_bg = log_likelihood_crbd(0.15, 0.03, td)
print(f"  CRBD(0.15, 0.03) bg-only  log L = {ll_crbd_bg:.4f}")

ll_crbd_fg = log_likelihood_crbd(0.6, 0.05, td)
print(f"  CRBD(0.60, 0.05) fg-only  log L = {ll_crbd_fg:.4f}")

delta = ll_true - ll_crb_mle
print(f"\n  Δ(BAMM - CRB_MLE) = {delta:+.4f}  {'BAMM wins' if delta > 0 else 'CRB wins'}")

# --- MCMC ---
print("\n" + "=" * 65)
N_ITER = 5000
print(f"运行 MCMC ({N_ITER} 轮)...")
print("=" * 65)

np.random.seed(314)
samples, best_expr, best_ll = bayesian_synthesis_mcmc(td, n_iter=N_ITER, verbose=True)

# --- 结果 ---
print("\n" + "=" * 65)
print("MCMC 结果")
print("=" * 65)

model_counts = {}
model_best = {}
for expr in samples:
    tag = expr.children[1].tag
    model_counts[tag] = model_counts.get(tag, 0) + 1
    ll = evaluate_likelihood(expr, td)
    if tag not in model_best or ll > model_best[tag][1]:
        model_best[tag] = (expr, ll)

print("\n模型后验频率:")
for tag, cnt in sorted(model_counts.items(), key=lambda x: -x[1]):
    print(f"  {tag:6s}: {cnt:4d}/{len(samples)} ({100*cnt/len(samples):5.1f}%)")

print("\n各模型类别最优表达式:")
for tag in sorted(model_best.keys()):
    expr, ll = model_best[tag]
    model_node = expr.children[1]
    print(f"  {tag:6s}: log L = {ll:.4f}")
    print(f"          {model_node}")

print(f"\n全局最优: log L = {best_ll:.4f}")
print(f"  {best_expr}")
print(f"\n真实模型: log L = {ll_true:.4f}")
print(f"  {TRUE_EXPR_STR}")
