"""主入口: 加载数据并运行 MCMC / SMC 推断。"""

import argparse
import numpy as np

from .data_utils import load_phylojson, get_tree_data
from .inference import (
    bayesian_synthesis_mcmc,
    bayesian_synthesis_smc,
    bayesian_synthesis_smc_cumulative,
)
from .likelihood import evaluate_likelihood


def _print_best_expression(best_expr, best_ll, label="最高似然"):
    print(f"\n=== 最优表达式 ({label}) ===")
    print(f"  log-likelihood = {best_ll:.6f}")
    print(f"  表达式: {best_expr}")
    if best_expr.tag == "NoisyPhylo" and len(best_expr.children) > 1:
        model_node = best_expr.children[1]
        sigma = best_expr.children[0].params[0]
        print(f"  模型: {model_node.tag}")
        print(f"  噪声 sigma: {sigma:.6f}")
        print(f"  模型参数: {model_node.params}")
    else:
        print(f"  模型: {best_expr.tag}")
        print(f"  参数: {best_expr.params}")


def run_mcmc(data_path, n_iter=500, seed=42):
    """运行 MCMC 推断。"""
    np.random.seed(seed)
    print(f"加载数据: {data_path}")
    root = load_phylojson(data_path)
    tree_data = get_tree_data(root)
    print(f"  叶节点数: {tree_data['n']}")
    print(f"  树高: {tree_data['tree_height']:.4f}")
    print(f"  总枝长: {tree_data['total_length']:.4f}")
    print(f"\n运行 MCMC ({n_iter} 轮)...")
    result = bayesian_synthesis_mcmc(tree_data, n_iter, verbose=True)

    print("\n=== MCMC 结果统计 ===")
    for tag, cnt in sorted(result["model_counts"].items(), key=lambda x: -x[1]):
        n_samples = len(result["samples"])
        print(f"  {tag}: {cnt}/{n_samples} ({100*cnt/n_samples:.1f}%)")

    _print_best_expression(result["best_expr"], result["best_log_L"])
    return result


def run_smc(data_paths, n_move=10, M=50, seed=42, cumulative=False):
    """运行 SMC 推断 (多棵树)。"""
    np.random.seed(seed)
    observations = []
    for p in data_paths:
        print(f"加载数据: {p}")
        root = load_phylojson(p)
        td = get_tree_data(root)
        observations.append(td)
        print(f"  叶节点数: {td['n']}, 树高: {td['tree_height']:.4f}")

    mode_str = "累积后验" if cumulative else "顺序切换"
    print(f"\n运行 SMC-{mode_str} ({M} 粒子, {n_move} rejuvenation 步, "
          f"{len(observations)} 观测)...")

    if cumulative:
        result = bayesian_synthesis_smc_cumulative(
            observations, n_move, M, verbose=True
        )
        print("\n=== SMC 后验模型分布 ===")
        for tag, cnt in sorted(result["model_counts"].items(), key=lambda x: -x[1]):
            print(f"  {tag}: {cnt}/{M} ({100*cnt/M:.1f}%)")
        _print_best_expression(
            result["best_expr"], result["best_log_L"], label="最高累积似然"
        )
        if result["best_per_model"]:
            print("\n--- 各模型类别最优 ---")
            for tag, (expr, ll) in sorted(
                result["best_per_model"].items(), key=lambda x: -x[1][1]
            ):
                print(f"  {tag}: cum-logL = {ll:.4f}  E = {expr}")
        return result
    else:
        particles, weights, ancestors = bayesian_synthesis_smc(
            observations, n_move, M, verbose=True
        )
        last_obs = observations[-1]
        final_particles = particles[-1]
        final_weights = weights[-1]
        best_idx = int(np.argmax(final_weights))
        best_expr = final_particles[best_idx]
        best_ll = evaluate_likelihood(best_expr, last_obs)
        _print_best_expression(best_expr, best_ll)
        return particles, weights, ancestors


def main():
    parser = argparse.ArgumentParser(description="PhyloBayesSynth: 概率程序合成")
    parser.add_argument("--mode", choices=["mcmc", "smc", "smc-cum"], default="mcmc")
    parser.add_argument("--data", nargs="+", required=True,
                        help="phylojson 数据文件路径")
    parser.add_argument("--n_iter", type=int, default=500,
                        help="MCMC 迭代次数")
    parser.add_argument("--n_move", type=int, default=10,
                        help="SMC rejuvenation 步数")
    parser.add_argument("--M", type=int, default=50,
                        help="SMC 粒子数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.mode == "mcmc":
        run_mcmc(args.data[0], args.n_iter, args.seed)
    elif args.mode == "smc":
        run_smc(args.data, args.n_move, args.M, args.seed, cumulative=False)
    else:
        run_smc(args.data, args.n_move, args.M, args.seed, cumulative=True)


if __name__ == "__main__":
    main()
