"""likelihood.py 专项验证脚本。

运行方式:
    python test/test_likelihood_validation.py
"""

from math import lgamma, log
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.algorithm.data_utils import TreeNode, get_tree_data
from train.algorithm.expression import Expression
from train.algorithm.likelihood import (
    _log_likelihood_from_rate_system,
    _build_rate_system,
    _lambda_at,
    _log_observation_noise_baseline,
    evaluate_likelihood,
    log_likelihood_bamm,
    log_likelihood_crb,
    log_likelihood_crbd,
    log_likelihood_tdb,
    log_likelihood_tdbd,
)


def _build_balanced_ultrametric_tree():
    """构造一棵小型超度量二叉树:
        root(2.0)
        ├── i1(1.0) -> tipA(0), tipB(0)
        └── i2(1.0) -> tipC(0), tipD(0)
    """
    root = TreeNode()
    i1 = TreeNode()
    i2 = TreeNode()
    a = TreeNode()
    b = TreeNode()
    c = TreeNode()
    d = TreeNode()

    i1.branch_length = 1.0
    i2.branch_length = 1.0
    a.branch_length = 1.0
    b.branch_length = 1.0
    c.branch_length = 1.0
    d.branch_length = 1.0

    root.children = [i1, i2]
    i1.children = [a, b]
    i2.children = [c, d]

    root.age = 2.0
    i1.age = 1.0
    i2.age = 1.0
    a.age = b.age = c.age = d.age = 0.0
    return get_tree_data(root)


def _assert_close(v1, v2, tol=1e-6, msg=""):
    if not np.isfinite(v1) or not np.isfinite(v2) or abs(v1 - v2) > tol:
        raise AssertionError(f"{msg} | {v1} vs {v2}, tol={tol}")


def _legacy_crbd_closed_form(lam, mu, tree_data):
    """历史闭式实现（用于回归测试，验证其与 ODE 分歧）。"""
    n = tree_data["n"]
    branches = tree_data["branches"]
    T = tree_data["tree_height"]
    diff = lam - mu
    ratio = mu / lam
    out = lgamma(n) + (n - 1) * log(lam)
    for x_i, y_i, _ in branches:
        out += -diff * x_i
        denom = 1.0 - ratio * np.exp(-diff * y_i)
        out -= 2.0 * log(denom)
    e_root = mu * (1.0 - np.exp(-diff * T)) / (lam - mu * np.exp(-diff * T))
    out -= 2.0 * log(1.0 - e_root)
    return out


def test_crb_matches_closed_form(tree_data):
    lam = 0.7
    got = log_likelihood_crb(lam, tree_data)
    expected = lgamma(tree_data["n"]) + (tree_data["n"] - 1) * log(lam) - lam * tree_data["total_length"]
    _assert_close(got, expected, tol=2e-5, msg="CRB 与闭式不一致")


def test_crbd_matches_ode_kernel(tree_data):
    lam, mu = 0.9, 0.3
    model = Expression("CRBD", [lam, mu], [], "N3")
    system = _build_rate_system(model, tree_data["tree_height"])
    if system is None:
        raise AssertionError("CRBD 速率系统构建失败")
    expected = _log_likelihood_from_rate_system(system, tree_data)
    got = log_likelihood_crbd(lam, mu, tree_data)
    _assert_close(got, expected, tol=1e-7, msg="CRBD 与 ODE 内核值不一致")


def test_crbd_fixes_legacy_high_extinction_blowup(tree_data):
    lam = 1.0
    mu = 0.8
    fixed = log_likelihood_crbd(lam, mu, tree_data)
    legacy = _legacy_crbd_closed_form(lam, mu, tree_data)
    if not (np.isfinite(fixed) and np.isfinite(legacy)):
        raise AssertionError("CRBD 比较值必须为有限数")
    # 旧闭式在高灭绝率时会异常偏大；修复后应显著更小。
    if not (fixed < legacy - 1.0):
        raise AssertionError("CRBD 未修复高灭绝率下的闭式偏大问题")


def test_tdbd_reduces_to_tdb_at_epsilon_zero(tree_data):
    lam0, z = 0.8, -0.2
    got = log_likelihood_tdbd(lam0, z, 0.0, tree_data)
    expected = log_likelihood_tdb(lam0, z, tree_data)
    _assert_close(got, expected, tol=3e-5, msg="TDBD 在 epsilon=0 时未退化到 TDB")


def test_tdb_z_direction_semantics(_tree_data):
    lam0, z, T = 1.0, 0.5, 2.0
    lam_root = _lambda_at(lam0, z, T, T)
    lam_present = _lambda_at(lam0, z, T, 0.0)
    # 按 λ(t)=λ0*exp(z*(t0-t)): z>0 时越接近现世(τ 越小)出生率越高。
    if not (lam_present > lam_root):
        raise AssertionError("TDB 的 z 方向语义错误: z>0 时现世出生率应高于根部")


def test_bamm_equals_base_when_bg_fg_identical(tree_data):
    base = Expression("CRBD", [1.1, 0.35], [], "N3")
    bamm = Expression("BAMM", [0.6], [base.copy(), base.copy()], "N3")

    got = log_likelihood_bamm(bamm, tree_data)
    expected = log_likelihood_crbd(1.1, 0.35, tree_data)
    _assert_close(got, expected, tol=1e-4, msg="bg=fg 时 BAMM 未退化到基础模型")


def test_bamm_eta_zero_equals_bg(tree_data):
    bg = Expression("CRBD", [1.0, 0.25], [], "N3")
    fg = Expression("TDBD", [1.5, -0.4, 0.2], [], "N3")
    bamm = Expression("BAMM", [0.0], [bg, fg], "N3")

    got = log_likelihood_bamm(bamm, tree_data)
    expected = log_likelihood_crbd(1.0, 0.25, tree_data)
    _assert_close(got, expected, tol=1e-4, msg="eta=0 时 BAMM 未退化为 bg 模型")


def test_bamm_three_level_nesting_allowed(tree_data):
    b1 = Expression(
        "BAMM",
        [0.2],
        [Expression("CRB", [0.8], [], "N3"), Expression("CRBD", [1.1, 0.25], [], "N3")],
        "N3",
    )
    b2 = Expression(
        "BAMM",
        [0.15],
        [b1, Expression("TDBD", [1.0, -0.2, 0.2], [], "N3")],
        "N3",
    )
    b3 = Expression(
        "BAMM",
        [0.1],
        [b2, Expression("CRB", [0.9], [], "N3")],
        "N3",
    )
    got = log_likelihood_bamm(b3, tree_data)
    if not np.isfinite(got):
        raise AssertionError("三层嵌套 BAMM 应可计算且为有限值")


def test_bamm_fourth_level_rejected(tree_data):
    b1 = Expression(
        "BAMM",
        [0.2],
        [Expression("CRB", [0.8], [], "N3"), Expression("CRBD", [1.1, 0.25], [], "N3")],
        "N3",
    )
    b2 = Expression(
        "BAMM",
        [0.15],
        [b1, Expression("TDBD", [1.0, -0.2, 0.2], [], "N3")],
        "N3",
    )
    b3 = Expression(
        "BAMM",
        [0.1],
        [b2, Expression("CRB", [0.9], [], "N3")],
        "N3",
    )
    b4 = Expression(
        "BAMM",
        [0.12],
        [b3, Expression("CRBD", [1.0, 0.2], [], "N3")],
        "N3",
    )
    got = log_likelihood_bamm(b4, tree_data)
    if got != -np.inf:
        raise AssertionError("超过三层的 BAMM 应返回 -inf")


def test_bamm_transition_is_irreversible(tree_data):
    bg = Expression("CRBD", [1.0, 0.2], [], "N3")
    fg = Expression("TDBD", [1.3, -0.3, 0.25], [], "N3")
    bamm = Expression("BAMM", [0.5], [bg, fg], "N3")
    system = _build_rate_system(bamm, tree_data["tree_height"])
    if system is None:
        raise AssertionError("BAMM 速率系统构建失败")
    Q = system["Q"]
    if Q.shape != (2, 2):
        raise AssertionError(f"一层 BAMM 状态数应为 2，实际 {Q.shape}")
    # bg -> fg 正向切换 > 0；fg -> bg 应为 0（不可逆）
    if Q[0, 1] <= 0.0:
        raise AssertionError("BAMM 应允许 bg->fg 正向切换")
    if abs(Q[1, 0]) > 1e-12:
        raise AssertionError("BAMM 不应允许 fg->bg 回切")


def test_nested_bamm_outer_shift_no_reverse(tree_data):
    inner_bg = Expression("BAMM", [0.2], [Expression("CRB", [0.8], [], "N3"), Expression("CRBD", [1.1, 0.2], [], "N3")], "N3")
    inner_fg = Expression("BAMM", [0.15], [Expression("TDB", [1.0, -0.2], [], "N3"), Expression("TDBD", [1.0, -0.2, 0.2], [], "N3")], "N3")
    outer = Expression("BAMM", [0.3], [inner_bg, inner_fg], "N3")
    system = _build_rate_system(outer, tree_data["tree_height"])
    if system is None:
        raise AssertionError("嵌套 BAMM 速率系统构建失败")
    Q = system["Q"]
    n_bg = _build_rate_system(inner_bg, tree_data["tree_height"])["n_states"]
    n_total = Q.shape[0]
    # 外层语义: 任意 fg 块状态（后半块）不应有回到 bg 根状态的外层反向跳转。
    # 这里只检查 fg 块到 bg 根（索引 0）是否存在额外正跳率。
    for idx in range(n_bg, n_total):
        if Q[idx, 0] > 1e-12:
            raise AssertionError("外层 BAMM 不应从 fg 块回切到 bg 根状态")


def test_nested_bamm_shift_preserves_compatible_substate(tree_data):
    inner = Expression(
        "BAMM",
        [0.2],
        [Expression("CRB", [0.8], [], "N3"), Expression("CRBD", [1.1, 0.2], [], "N3")],
        "N3",
    )
    outer = Expression("BAMM", [0.3], [inner, inner.copy()], "N3")
    system = _build_rate_system(outer, tree_data["tree_height"])
    if system is None:
        raise AssertionError("嵌套 BAMM 速率系统构建失败")
    Q = system["Q"]
    # 外层 bg/fg 块各 2 个状态；兼容标签下应发生一一映射:
    # bg[0] -> fg[0], bg[1] -> fg[1]，而不是都跳到 fg_root。
    if Q.shape[0] < 4:
        raise AssertionError(f"状态空间维度异常: {Q.shape}")
    if Q[0, 2] <= 0.0 or Q[1, 3] <= 0.0:
        raise AssertionError("外层 shift 未保持兼容子状态映射")
    if Q[1, 2] > 1e-12:
        raise AssertionError("外层 shift 不应把所有状态都重置到 fg_root")


def test_noise_tempering(tree_data):
    model = Expression("CRB", [0.7], [], "N3")
    expr0 = Expression("NoisyPhylo", [], [Expression("Noise", [0.0], [], "N2"), model], "N1")
    expr1 = Expression("NoisyPhylo", [], [Expression("Noise", [1.0], [], "N2"), model], "N1")
    expr_big = Expression("NoisyPhylo", [], [Expression("Noise", [1e6], [], "N2"), model], "N1")

    ll_model = log_likelihood_crb(0.7, tree_data)
    ll_noise = _log_observation_noise_baseline(tree_data)
    ll0 = evaluate_likelihood(expr0, tree_data)
    ll1 = evaluate_likelihood(expr1, tree_data)
    ll_big = evaluate_likelihood(expr_big, tree_data)

    # sigma=0 时应严格退化到结构化模型似然
    _assert_close(ll0, ll_model, tol=1e-8, msg="sigma=0 时未退化到模型似然")

    # sigma=1 -> pi=1/2，等权混合
    expected = float(np.logaddexp(log(0.5) + ll_model, log(0.5) + ll_noise))
    _assert_close(ll1, expected, tol=1e-8, msg="Noise 混合模型计算错误")

    # sigma->∞ 时应逼近噪声基线
    _assert_close(ll_big, ll_noise, tol=2e-5, msg="sigma 很大时未逼近噪声基线")


def run_all_tests():
    tree_data = _build_balanced_ultrametric_tree()
    tests = [
        test_crb_matches_closed_form,
        test_crbd_matches_ode_kernel,
        test_crbd_fixes_legacy_high_extinction_blowup,
        test_tdbd_reduces_to_tdb_at_epsilon_zero,
        test_tdb_z_direction_semantics,
        test_bamm_equals_base_when_bg_fg_identical,
        test_bamm_eta_zero_equals_bg,
        test_bamm_three_level_nesting_allowed,
        test_bamm_fourth_level_rejected,
        test_bamm_transition_is_irreversible,
        test_nested_bamm_outer_shift_no_reverse,
        test_nested_bamm_shift_preserves_compatible_substate,
        test_noise_tempering,
    ]
    for t in tests:
        t(tree_data)
        print(f"[PASS] {t.__name__}")
    print("\n全部 likelihood 专项验证通过。")


if __name__ == "__main__":
    run_all_tests()
