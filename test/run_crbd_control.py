"""对照实验: 纯 CRBD 树, 验证不会被错误建模为 PWBD."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

import numpy as np
from pathlib import Path

from algorithm.expression import Expression, parse_expression
from algorithm.generate_data import generate_tree_from_expression
from algorithm.data_utils import load_phylojson, get_tree_data
from algorithm.inference import bayesian_synthesis_mcmc, bayesian_synthesis_smc_cumulative
from algorithm.likelihood import evaluate_likelihood

TRUE_EXPR_STR = "(CRBD 0.3 0.05)"
N_TIPS = 100
N_TREES = 8
N_PARTICLES = 60
N_MOVE = 25
SEED = 4000
OUT_DIR = Path("data/generated/crbd_control")

print("=" * 60)
print(f"CRBD Control Experiment")
print(f"  TRUE: {TRUE_EXPR_STR}")
print(f"  {N_TIPS} tips x {N_TREES} trees")
print("=" * 60)

true_expr = parse_expression(TRUE_EXPR_STR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

observations = []
for i in range(N_TREES):
    seed_i = SEED + i
    out_path = OUT_DIR / f"tree_{i}_seed{seed_i}.phylojson"
    if not out_path.exists():
        data = generate_tree_from_expression(
            true_expr, n_tips=N_TIPS, seed=seed_i,
            max_time=500.0, max_retries=2000)
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    root = load_phylojson(str(out_path))
    td = get_tree_data(root)
    observations.append(td)
    print(f"  Tree {i}: n={td['n']}, T={td['tree_height']:.2f}")

noise = Expression("Noise", [0.001])
true_full = Expression("NoisyPhylo", [], [noise, true_expr])
true_cum = sum(evaluate_likelihood(true_full, td) for td in observations)
print(f"\nTrue CRBD cum-logL: {true_cum:.2f}")

# Single-tree MCMC
print(f"\n--- MCMC on tree 0 (500 iters) ---")
np.random.seed(SEED)
r1 = bayesian_synthesis_mcmc(observations[0], n_iter=500, verbose=False)
print(f"  Models: {r1['model_counts']}")
print(f"  Best: {r1['best_log_L']:.4f} {r1['best_expr']}")

# SMC-cum
print(f"\n--- SMC-cum ({N_PARTICLES} particles, {N_MOVE} moves, {N_TREES} trees) ---")
np.random.seed(SEED)
result = bayesian_synthesis_smc_cumulative(
    observations, n_move=N_MOVE, M=N_PARTICLES, verbose=True)

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"Model counts: {result['model_counts']}")
print(f"Best cum-logL: {result['best_log_L']:.4f}")
print(f"Best expr: {result['best_expr']}")
print(f"True CRBD cum-logL: {true_cum:.2f}")
if result["best_per_model"]:
    print("\nPer-model best:")
    for tag, (expr, ll) in sorted(result["best_per_model"].items(), key=lambda x: -x[1][1]):
        print(f"  {tag:6s}: {ll:.4f}  {expr}")
