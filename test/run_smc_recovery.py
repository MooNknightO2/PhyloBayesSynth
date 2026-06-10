"""SMC 累积后验恢复实验: 多棵同源树联合推断。

通过生成多棵来自同一 BAMM 表达式的树, 利用累积后验 SMC
逐步加入观测以累积信号, 尝试恢复真实模型。

默认真实模型: (BAMM 0.15 (CRBD 0.4 0.05) (CRBD 1.0 0.65))
  bg = CRBD(λ=0.4, μ=0.05)  低周转, 净速率 r=0.35
  fg = CRBD(λ=1.0, μ=0.65)  高周转, 净速率 r=0.35
  η  = 0.15                 中等切换

设计要点: 两个 regime 净多样化率相同 (r=0.35),
避免幸存者偏差导致快速 regime 支配整棵树;
但 λ 和 ε=μ/λ 差距极大, 产生 CRB/CRBD 无法解释的异质树形。

运行:
  python test/run_smc_recovery.py [--J 5] [--M 100] [--n_move 15]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.algorithm.data_utils import load_phylojson, get_tree_data
from train.algorithm.expression import Expression, parse_expression
from train.algorithm.generate_data import generate_tree_from_expression
from train.algorithm.likelihood import evaluate_likelihood
from train.algorithm.inference import bayesian_synthesis_smc_cumulative

# ===================================================================
# 配置
# ===================================================================
TRUE_EXPR_STR = "(BAMM 0.15 (CRBD 0.4 0.05) (CRBD 1.0 0.65))"
N_TIPS = 80
DATA_DIR = ROOT / "data" / "generated" / "smc_recovery"


def generate_trees(expr_str, J, n_tips, base_seed):
    """生成 J 棵同源树, 返回 tree_data 列表。"""
    expr = parse_expression(expr_str)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    observations = []
    for j in range(J):
        seed = base_seed + j
        fpath = DATA_DIR / f"tree_{j}_seed{seed}.phylojson"
        if fpath.exists():
            print(f"  [树 {j+1}/{J}] 已存在, 直接加载: {fpath.name}")
            root = load_phylojson(str(fpath))
        else:
            print(f"  [树 {j+1}/{J}] 生成中 (seed={seed}) ...", end=" ", flush=True)
            phylo = generate_tree_from_expression(
                expr, n_tips=n_tips, seed=seed, max_time=500.0, max_retries=2000
            )
            fpath.write_text(json.dumps(phylo, indent=2), encoding="utf-8")
            root = load_phylojson(str(fpath))
            print("OK")
        td = get_tree_data(root)
        observations.append(td)
        print(f"        n={td['n']}, T={td['tree_height']:.4f}, "
              f"S={td['total_length']:.4f}")
    return observations


SIGMA = 0.01


def _wrap_noisy(model_expr):
    return Expression("NoisyPhylo", [], [
        Expression("Noise", [SIGMA]), model_expr,
    ], nonterminal="N1")


def _cum_ll(expr, observations):
    total = 0.0
    for td in observations:
        ll = evaluate_likelihood(expr, td)
        if ll == -np.inf:
            return -np.inf
        total += ll
    return total


def _fit_crbd_global(observations):
    """联合 MLE: 单组 (λ, μ) 拟合所有树。"""
    n1 = sum(td["n"] - 1 for td in observations)
    S = sum(td["total_length"] for td in observations)
    lam0 = max(n1 / S, 0.1)

    def neg(x):
        lam, mu = float(x[0]), float(x[1])
        if lam <= 0 or mu < 0:
            return 1e300
        v = _cum_ll(_wrap_noisy(Expression("CRBD", [lam, mu])), observations)
        return -v if np.isfinite(v) else 1e300

    best = (-np.inf, None)
    for l0, m0 in [(lam0, 0.05*lam0), (lam0*0.6, lam0*0.25),
                    (0.8, 0.15), (0.4, 0.08), (1.2, 0.5)]:
        res = minimize(neg, [l0, m0], method="L-BFGS-B",
                       bounds=[(1e-4, 50), (0, 25)], options={"maxiter": 300})
        if res.fun < 1e299:
            ll = -res.fun
            if np.isfinite(ll) and ll > best[0]:
                best = (ll, (float(res.x[0]), float(res.x[1])))
    return best


def _fit_tdbd_global(observations):
    """联合 MLE: 单组 (λ₀, z, ε) 拟合所有树。"""
    n1 = sum(td["n"] - 1 for td in observations)
    S = sum(td["total_length"] for td in observations)
    lam0 = max(n1 / S, 0.1)

    def neg(x):
        l, z, e = float(x[0]), float(x[1]), float(x[2])
        if l <= 0 or e <= 0 or e >= 1:
            return 1e300
        v = _cum_ll(_wrap_noisy(Expression("TDBD", [l, z, e])), observations)
        return -v if np.isfinite(v) else 1e300

    best = (-np.inf, None)
    for l0, z0, e0 in [(lam0, 0.05, 0.25), (lam0, 0.0, 0.4),
                        (lam0*1.2, -0.05, 0.5), (0.6, 0.1, 0.3),
                        (0.5, 0.0, 0.6), (1.0, -0.1, 0.7)]:
        res = minimize(neg, [l0, z0, e0], method="L-BFGS-B",
                       bounds=[(1e-4, 50), (-4, 4), (1e-4, 1-1e-4)],
                       options={"maxiter": 400})
        if res.fun < 1e299:
            ll = -res.fun
            if np.isfinite(ll) and ll > best[0]:
                best = (ll, (float(res.x[0]), float(res.x[1]), float(res.x[2])))
    return best


def baseline_comparison(observations, true_expr_str):
    """计算真实模型和多种基线在所有树上的累积似然。"""
    true_expr = parse_expression(true_expr_str)
    noisy_true = _wrap_noisy(true_expr)

    total_n1 = sum(td["n"] - 1 for td in observations)
    total_S = sum(td["total_length"] for td in observations)
    lam_global = total_n1 / total_S
    noisy_crb_global = _wrap_noisy(Expression("CRB", [lam_global]))

    cum_true = 0.0
    cum_crb_per_tree = 0.0
    cum_crb_global = 0.0
    for td in observations:
        cum_true += evaluate_likelihood(noisy_true, td)
        cum_crb_global += evaluate_likelihood(noisy_crb_global, td)
        lam_j = (td["n"] - 1) / td["total_length"]
        cum_crb_per_tree += evaluate_likelihood(
            _wrap_noisy(Expression("CRB", [lam_j])), td)

    print("  (正在拟合 CRBD 全局 MLE ...)", flush=True)
    cum_crbd, crbd_p = _fit_crbd_global(observations)
    print("  (正在拟合 TDBD 全局 MLE ...)", flush=True)
    cum_tdbd, tdbd_p = _fit_tdbd_global(observations)

    return {
        "BAMM_true": cum_true,
        "CRB_global_MLE": (cum_crb_global, lam_global),
        "CRB_per_tree_MLE": cum_crb_per_tree,
        "CRBD_global_MLE": (cum_crbd, crbd_p),
        "TDBD_global_MLE": (cum_tdbd, tdbd_p),
    }


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="SMC cumulative posterior recovery experiment"
    )
    parser.add_argument("--J", type=int, default=5,
                        help="同源树数量 (默认 5)")
    parser.add_argument("--M", type=int, default=100,
                        help="SMC 粒子数 (默认 100)")
    parser.add_argument("--n_move", type=int, default=15,
                        help="每轮 rejuvenation MH 步数 (默认 15)")
    parser.add_argument("--n_tips", type=int, default=N_TIPS,
                        help=f"每棵树的叶数 (默认 {N_TIPS})")
    parser.add_argument("--seed", type=int, default=100,
                        help="数据生成 base seed")
    parser.add_argument("--smc_seed", type=int, default=777,
                        help="SMC 推断的随机种子")
    parser.add_argument("--expr", type=str, default=TRUE_EXPR_STR,
                        help=f"真实模型表达式 (默认: {TRUE_EXPR_STR})")
    parser.add_argument("--regen", action="store_true",
                        help="强制重新生成所有树")
    args = parser.parse_args()

    print("=" * 65)
    print("SMC 累积后验恢复实验 — 多棵同源树联合推断")
    print("=" * 65)
    print(f"\n真实模型: {args.expr}")
    print(f"参数: J={args.J} 棵树, n_tips={args.n_tips}, "
          f"M={args.M} 粒子, n_move={args.n_move}")

    # ---- 清理旧数据 (如果 --regen) ----
    if args.regen and DATA_DIR.exists():
        for f in DATA_DIR.glob("tree_*.phylojson"):
            f.unlink()
        print("已清除旧数据")

    # ---- 生成 / 加载树 ----
    print(f"\n--- 生成 {args.J} 棵同源树 ---")
    observations = generate_trees(args.expr, args.J, args.n_tips, args.seed)

    # ---- 基线对比 ----
    print("\n--- 基线似然对比 (累积) ---")
    bl = baseline_comparison(observations, args.expr)
    cum_true = bl["BAMM_true"]
    cum_crb_g, lam_g = bl["CRB_global_MLE"]
    cum_crb_pt = bl["CRB_per_tree_MLE"]
    cum_crbd, crbd_p = bl["CRBD_global_MLE"]
    cum_tdbd, tdbd_p = bl["TDBD_global_MLE"]

    print(f"  BAMM 真实参数 (σ={SIGMA})        cum-logL = {cum_true:.4f}")
    print(f"  CRB  全局 MLE (λ={lam_g:.4f})    cum-logL = {cum_crb_g:.4f}")
    if crbd_p is not None:
        print(f"  CRBD 全局 MLE (λ={crbd_p[0]:.4f}, μ={crbd_p[1]:.4f})"
              f"  cum-logL = {cum_crbd:.4f}")
    if tdbd_p is not None:
        print(f"  TDBD 全局 MLE (λ₀={tdbd_p[0]:.4f}, z={tdbd_p[1]:.4f}, "
              f"ε={tdbd_p[2]:.4f})  cum-logL = {cum_tdbd:.4f}")
    print(f"  CRB  逐树 MLE                cum-logL = {cum_crb_pt:.4f}  (作弊)")

    print(f"\n  --- Δ (正 = BAMM 更好) ---")
    print(f"  Δ(BAMM_true - CRB_global)  = {cum_true - cum_crb_g:+.4f}")
    if crbd_p is not None:
        print(f"  Δ(BAMM_true - CRBD_global) = {cum_true - cum_crbd:+.4f}")
    if tdbd_p is not None:
        print(f"  Δ(BAMM_true - TDBD_global) = {cum_true - cum_tdbd:+.4f}")

    # ---- 运行累积后验 SMC ----
    print(f"\n{'='*65}")
    print(f"运行累积后验 SMC (M={args.M}, n_move={args.n_move}, "
          f"J={args.J}, seed={args.smc_seed})")
    print("=" * 65)

    np.random.seed(args.smc_seed)
    t0 = time.time()
    result = bayesian_synthesis_smc_cumulative(
        observations,
        n_move=args.n_move,
        M=args.M,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\nSMC 完成, 耗时 {elapsed:.1f} 秒")

    # ---- 结果 ----
    print(f"\n{'='*65}")
    print("SMC 结果汇总")
    print("=" * 65)

    print("\n后验模型分布:")
    for tag, cnt in sorted(result["model_counts"].items(), key=lambda x: -x[1]):
        pct = 100 * cnt / args.M
        bar = "█" * int(pct / 2)
        print(f"  {tag:6s}: {cnt:4d}/{args.M} ({pct:5.1f}%) {bar}")

    print("\n各模型类别最优:")
    for tag in sorted(result["best_per_model"].keys()):
        expr, ll = result["best_per_model"][tag]
        print(f"  [{tag}] cum-logL = {ll:.4f}")
        print(f"    {expr}")

    best = result["best_expr"]
    print(f"\n{'—'*65}")
    print(f"SMC 全局最优:   cum-logL = {result['best_log_L']:.4f}")
    print(f"  {best}")
    print(f"\nBAMM 真实参数:  cum-logL = {cum_true:.4f}")
    print(f"  {args.expr}")
    print(f"CRB  全局 MLE:  cum-logL = {cum_crb_g:.4f}  (λ={lam_g:.4f})")
    if crbd_p is not None:
        print(f"CRBD 全局 MLE:  cum-logL = {cum_crbd:.4f}  "
              f"(λ={crbd_p[0]:.4f}, μ={crbd_p[1]:.4f})")
    if tdbd_p is not None:
        print(f"TDBD 全局 MLE:  cum-logL = {cum_tdbd:.4f}  "
              f"(λ₀={tdbd_p[0]:.4f}, z={tdbd_p[1]:.4f}, ε={tdbd_p[2]:.4f})")

    # ---- 逐轮收敛轨迹 ----
    print(f"\n--- 逐轮收敛轨迹 ---")
    for h in result["history"]:
        print(f"  轮 {h['step']:2d}: ESS={h['ess']:.1f}, "
              f"best cum-logL={h['best_cum_log_L']:.4f}, "
              f"E={h['best_expr']}")


if __name__ == "__main__":
    main()
