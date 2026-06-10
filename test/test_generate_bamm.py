"""测试 BAMM 数据生成和似然计算。"""

import sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.algorithm.expression import Expression, parse_expression
from train.algorithm.generate_data import generate_tree_from_expression, generate_tree
from train.algorithm.data_utils import _parse_node, _compute_ages, get_tree_data
from train.algorithm.likelihood import (
    log_likelihood_crb, log_likelihood_crbd, log_likelihood_bamm, evaluate_likelihood,
)

PASS = 0
FAIL = 0

def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {msg}")
    else:
        FAIL += 1
        print(f"  [FAIL] {msg}")


def _load(data):
    root = _parse_node(data["trees"][0]["root"])
    _compute_ages(root)
    return get_tree_data(root)


def test_parse():
    print("=== 表达式解析 ===")
    e = parse_expression("(CRBD 0.5 0.1)")
    check(e.tag == "CRBD" and e.params == [0.5, 0.1], f"CRBD: {e}")

    e2 = parse_expression("(BAMM 0.3 (CRBD 0.5 0.1) (CRBD 1.0 0.3))")
    check(e2.tag == "BAMM" and len(e2.children) == 2, f"BAMM: {e2}")

    e3 = parse_expression("(BAMM 0.2 (BAMM 0.1 (CRB 0.8) (CRBD 1.0 0.2)) (CRBD 0.6 0.15))")
    check(e3.children[0].tag == "BAMM", f"nested: {e3}")

    e4 = parse_expression(repr(e2))
    check(repr(e4) == repr(e2), "roundtrip parse(repr(x)) == x")


def test_simple_models():
    print("\n=== 简单模型生成 ===")
    for tag, params in [("CRB", [0.3]), ("CRBD", [0.3, 0.05])]:
        expr = Expression(tag, params)
        t0 = time.time()
        data = generate_tree_from_expression(expr, n_tips=20, seed=42, max_retries=1000)
        td = _load(data)
        check(td["n"] == 20 and td["tree_height"] > 0,
              f"{tag}: n={td['n']}, T={td['tree_height']:.3f} ({time.time()-t0:.2f}s)")


def test_bamm_crbd():
    print("\n=== BAMM(CRBD, CRBD) ===")
    expr = parse_expression("(BAMM 0.3 (CRBD 0.5 0.1) (CRBD 1.0 0.2))")
    t0 = time.time()
    data = generate_tree_from_expression(expr, n_tips=20, seed=1, max_retries=2000)
    td = _load(data)
    check(td["n"] == 20, f"n={td['n']}, T={td['tree_height']:.3f} ({time.time()-t0:.2f}s)")

    ll = log_likelihood_bamm(expr, td)
    check(np.isfinite(ll), f"likelihood finite: {ll:.4f}")

    ll_bg = log_likelihood_crbd(0.5, 0.1, td)
    check(np.isfinite(ll_bg), f"bg-only likelihood: {ll_bg:.4f}")


def test_bamm_mixed():
    print("\n=== BAMM(CRBD, CRB) ===")
    expr = parse_expression("(BAMM 0.2 (CRBD 0.4 0.08) (CRB 0.8))")
    data = generate_tree_from_expression(expr, n_tips=20, seed=99, max_retries=2000)
    td = _load(data)
    ll = log_likelihood_bamm(expr, td)
    check(td["n"] == 20 and np.isfinite(ll),
          f"mixed BAMM: n={td['n']}, logL={ll:.4f}")


def test_bamm_tdbd():
    print("\n=== BAMM(TDBD, CRBD) 时变 ===")
    expr = parse_expression("(BAMM 0.15 (TDBD 0.3 -0.2 0.2) (CRBD 0.6 0.1))")
    t0 = time.time()
    data = generate_tree_from_expression(expr, n_tips=20, seed=77, max_retries=3000)
    td = _load(data)
    ll = log_likelihood_bamm(expr, td)
    check(td["n"] == 20 and np.isfinite(ll),
          f"TV BAMM: n={td['n']}, logL={ll:.4f} ({time.time()-t0:.2f}s)")


def test_nested_bamm():
    print("\n=== 嵌套 BAMM ===")
    expr = parse_expression(
        "(BAMM 0.1 (BAMM 0.15 (CRB 0.5) (CRBD 0.8 0.15)) (CRBD 0.6 0.1))"
    )
    data = generate_tree_from_expression(expr, n_tips=20, seed=55, max_retries=3000)
    td = _load(data)
    ll = log_likelihood_bamm(expr, td)
    check(td["n"] == 20 and np.isfinite(ll),
          f"nested: n={td['n']}, logL={ll:.4f}")


def test_backward_compat():
    print("\n=== 向后兼容 generate_tree ===")
    data = generate_tree("CRBD", 20, 0.3, 0.05, 0.0, 0.0,
                          seed=42, max_time=500, max_retries=1000)
    td = _load(data)
    check(td["n"] == 20, f"old API: n={td['n']}")


def test_evaluate_with_noise():
    print("\n=== evaluate_likelihood (含噪声包装) ===")
    model_expr = parse_expression("(BAMM 0.3 (CRBD 0.5 0.1) (CRBD 1.0 0.2))")
    data = generate_tree_from_expression(model_expr, n_tips=20, seed=1, max_retries=2000)
    td = _load(data)

    full = Expression("NoisyPhylo", [], [Expression("Noise", [0.0]), model_expr])
    ll_full = evaluate_likelihood(full, td)
    ll_bamm = log_likelihood_bamm(model_expr, td)
    check(abs(ll_full - ll_bamm) < 1e-8,
          f"σ=0: evaluate={ll_full:.4f} vs bamm={ll_bamm:.4f}")


if __name__ == "__main__":
    test_parse()
    test_simple_models()
    test_bamm_crbd()
    test_bamm_mixed()
    test_bamm_tdbd()
    test_nested_bamm()
    test_backward_compat()
    test_evaluate_with_noise()

    print(f"\n{'='*40}")
    print(f"通过: {PASS}, 失败: {FAIL}")
    if FAIL:
        print(f"有 {FAIL} 项失败!")
        sys.exit(1)
    else:
        print("全部通过!")
