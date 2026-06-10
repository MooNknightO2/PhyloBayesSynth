"""闭式解数学正确性验证: 逐公式对比 ODE 数值解。

运行方式:
    python test/test_closed_form_verification.py
"""

from math import exp, lgamma, log
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.algorithm.data_utils import TreeNode, get_tree_data
from train.algorithm.expression import Expression
from train.algorithm.likelihood import (
    _crbd_E_val,
    _crbd_log_one_minus_E,
    _crbd_log_branch_factor,
    _crbd_branch_factor_val,
    _tdbd_Lambda,
    _tdbd_E_val,
    _tdbd_log_one_minus_E,
    _tdbd_log_branch_factor,
    _tdbd_branch_factor_val,
    _build_rate_system,
    _log_likelihood_from_rate_system,
    log_likelihood_crb,
    log_likelihood_crbd,
    log_likelihood_tdb,
    log_likelihood_tdbd,
    log_likelihood_bamm,
    evaluate_likelihood,
    _log_observation_noise_baseline,
    _lambda_at,
)

PASS = 0
FAIL = 0

def check(condition, msg):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {msg}")
    else:
        FAIL += 1
        print(f"  [FAIL] {msg}")


def close(a, b, tol=1e-6):
    if not np.isfinite(a) or not np.isfinite(b):
        return False
    return abs(a - b) < tol


def _ode_E_crbd(lam, mu, tau_max):
    """用 ODE 直接求 E(τ) 作为 ground truth."""
    def rhs(tau, y):
        E = y[0]
        return [mu - (lam + mu) * E + lam * E ** 2]
    sol = solve_ivp(rhs, (0, tau_max), [0.0], dense_output=True,
                    method="BDF", rtol=1e-12, atol=1e-14)
    return sol.sol


def _ode_D_crbd(lam, mu, e_sol, tau_c, tau_p, d0=1.0):
    """用 ODE 直接求 D 传播作为 ground truth."""
    def rhs(tau, y):
        E = float(e_sol(tau)[0])
        return [(-(lam + mu) + 2 * lam * E) * y[0]]
    sol = solve_ivp(rhs, (tau_c, tau_p), [d0], method="BDF",
                    rtol=1e-12, atol=1e-14)
    return sol.y[0, -1]


def _ode_E_tdbd(lam0, z, epsilon, T, tau_max):
    """用 ODE 直接求 TDBD E(τ) 作为 ground truth."""
    def rhs(tau, y):
        E = y[0]
        lam = lam0 * exp(z * (T - tau))
        mu = epsilon * lam
        return [mu - (lam + mu) * E + lam * E ** 2]
    sol = solve_ivp(rhs, (0, tau_max), [0.0], dense_output=True,
                    method="BDF", rtol=1e-12, atol=1e-14)
    return sol.sol


def _ode_D_tdbd(lam0, z, epsilon, T, e_sol, tau_c, tau_p, d0=1.0):
    """用 ODE 直接求 TDBD D 传播作为 ground truth."""
    def rhs(tau, y):
        E = float(e_sol(tau)[0])
        lam = lam0 * exp(z * (T - tau))
        mu = epsilon * lam
        return [(-(lam + mu) + 2 * lam * E) * y[0]]
    sol = solve_ivp(rhs, (tau_c, tau_p), [d0], method="BDF",
                    rtol=1e-12, atol=1e-14)
    return sol.y[0, -1]


def _build_tree():
    root = TreeNode()
    i1 = TreeNode()
    i2 = TreeNode()
    a, b, c, d = TreeNode(), TreeNode(), TreeNode(), TreeNode()
    i1.branch_length = i2.branch_length = 1.0
    a.branch_length = b.branch_length = c.branch_length = d.branch_length = 1.0
    root.children = [i1, i2]
    i1.children = [a, b]
    i2.children = [c, d]
    root.age = 2.0
    i1.age = i2.age = 1.0
    a.age = b.age = c.age = d.age = 0.0
    return get_tree_data(root)


# ===================================================================
# 1. CRBD E(τ) 闭式 vs ODE
# ===================================================================
def test_crbd_E():
    print("\n=== CRBD E(τ) 闭式 vs ODE ===")
    cases = [
        (0.5, 0.1, "supercritical, low mu"),
        (1.0, 0.8, "near-critical, high mu"),
        (0.3, 0.3, "critical, lambda=mu"),
        (0.2, 0.5, "subcritical, mu>lambda"),
        (2.0, 0.01, "fast speciation, low extinction"),
    ]
    for lam, mu, label in cases:
        e_ode = _ode_E_crbd(lam, mu, 3.0)
        for tau in [0.0, 0.5, 1.0, 2.0, 3.0]:
            E_closed = _crbd_E_val(lam, mu, tau)
            E_ode = float(e_ode(tau)[0])
            check(close(E_closed, E_ode, 1e-7),
                  f"E({tau}) λ={lam} μ={mu} ({label}): closed={E_closed:.8f} ode={E_ode:.8f}")


# ===================================================================
# 2. CRBD 1-E(τ) 闭式 vs ODE
# ===================================================================
def test_crbd_survival():
    print("\n=== CRBD log(1-E(τ)) 闭式 vs ODE ===")
    cases = [(0.5, 0.1), (1.0, 0.8), (0.3, 0.3), (0.2, 0.5)]
    for lam, mu in cases:
        e_ode = _ode_E_crbd(lam, mu, 3.0)
        for tau in [0.5, 1.0, 2.0]:
            log_surv_closed = _crbd_log_one_minus_E(lam, mu, tau)
            E_ode = float(e_ode(tau)[0])
            log_surv_ode = log(1.0 - E_ode) if E_ode < 1.0 else -np.inf
            check(close(log_surv_closed, log_surv_ode, 1e-6),
                  f"log(1-E({tau})) λ={lam} μ={mu}: closed={log_surv_closed:.8f} ode={log_surv_ode:.8f}")


# ===================================================================
# 3. CRBD D 枝传播 闭式 vs ODE
# ===================================================================
def test_crbd_branch_propagation():
    print("\n=== CRBD D 传播 闭式 vs ODE ===")
    cases = [(0.5, 0.1), (1.0, 0.8), (0.3, 0.3), (0.2, 0.5)]
    for lam, mu in cases:
        e_ode = _ode_E_crbd(lam, mu, 3.0)
        for tau_c, tau_p in [(0.0, 1.0), (0.5, 1.5), (1.0, 2.0)]:
            D_closed = _crbd_branch_factor_val(lam, mu, tau_c, tau_p)
            D_ode = _ode_D_crbd(lam, mu, e_ode, tau_c, tau_p)
            check(close(D_closed, D_ode, 1e-6),
                  f"D({tau_c}->{tau_p}) λ={lam} μ={mu}: closed={D_closed:.8f} ode={D_ode:.8f}")


# ===================================================================
# 4. CRBD 完整似然 闭式 vs ODE rate_system
# ===================================================================
def test_crbd_full_likelihood():
    print("\n=== CRBD 完整似然 闭式 vs rate_system ===")
    td = _build_tree()
    cases = [(0.5, 0.1), (0.9, 0.3), (1.0, 0.8), (0.3, 0.3), (0.2, 0.5)]
    for lam, mu in cases:
        ll_closed = log_likelihood_crbd(lam, mu, td)
        model = Expression("CRBD", [lam, mu], [], "N3")
        system = _build_rate_system(model, td["tree_height"])
        ll_system = _log_likelihood_from_rate_system(system, td)
        check(close(ll_closed, ll_system, 1e-6),
              f"logL λ={lam} μ={mu}: closed={ll_closed:.6f} system={ll_system:.6f}")


# ===================================================================
# 5. CRBD μ→0 退化到 CRB
# ===================================================================
def test_crbd_reduces_to_crb():
    print("\n=== CRBD μ→0 退化到 CRB ===")
    td = _build_tree()
    for lam in [0.3, 0.7, 1.5]:
        ll_crb = log_likelihood_crb(lam, td)
        ll_crbd = log_likelihood_crbd(lam, 1e-15, td)
        check(close(ll_crb, ll_crbd, 1e-5),
              f"λ={lam}: CRB={ll_crb:.6f} CRBD(μ≈0)={ll_crbd:.6f}")


# ===================================================================
# 6. TDBD E(τ) 闭式 vs ODE
# ===================================================================
def test_tdbd_E():
    print("\n=== TDBD E(τ) 闭式 vs ODE ===")
    T = 2.0
    cases = [
        (0.5, 0.0, 0.3, "constant rate"),
        (0.5, 0.5, 0.3, "increasing lambda"),
        (0.5, -0.3, 0.2, "decreasing lambda"),
        (1.0, 0.0, 0.8, "high turnover"),
        (0.8, -0.2, 0.5, "moderate"),
    ]
    for lam0, z, eps, label in cases:
        e_ode = _ode_E_tdbd(lam0, z, eps, T, T)
        for tau in [0.0, 0.5, 1.0, 1.5, 2.0]:
            E_closed = _tdbd_E_val(lam0, z, eps, T, tau)
            E_ode = float(e_ode(tau)[0])
            check(close(E_closed, E_ode, 1e-6),
                  f"E({tau}) ({label}): closed={E_closed:.8f} ode={E_ode:.8f}")


# ===================================================================
# 7. TDBD D 枝传播 闭式 vs ODE
# ===================================================================
def test_tdbd_branch_propagation():
    print("\n=== TDBD D 传播 闭式 vs ODE ===")
    T = 2.0
    cases = [
        (0.5, 0.0, 0.3),
        (0.5, 0.5, 0.3),
        (0.5, -0.3, 0.2),
        (1.0, 0.0, 0.8),
    ]
    for lam0, z, eps in cases:
        e_ode = _ode_E_tdbd(lam0, z, eps, T, T)
        for tau_c, tau_p in [(0.0, 1.0), (0.5, 1.5), (1.0, 2.0)]:
            D_closed = _tdbd_branch_factor_val(lam0, z, eps, T, tau_c, tau_p)
            D_ode = _ode_D_tdbd(lam0, z, eps, T, e_ode, tau_c, tau_p)
            check(close(D_closed, D_ode, 1e-5),
                  f"D({tau_c}->{tau_p}) λ₀={lam0} z={z} ε={eps}: closed={D_closed:.8f} ode={D_ode:.8f}")


# ===================================================================
# 8. TDBD 完整似然 闭式 vs ODE rate_system
# ===================================================================
def test_tdbd_full_likelihood():
    print("\n=== TDBD 完整似然 闭式 vs rate_system ===")
    td = _build_tree()
    cases = [
        (0.5, 0.0, 0.3),
        (0.5, 0.5, 0.3),
        (0.5, -0.3, 0.2),
        (0.8, -0.2, 0.5),
    ]
    for lam0, z, eps in cases:
        ll_closed = log_likelihood_tdbd(lam0, z, eps, td)
        model = Expression("TDBD", [lam0, z, eps], [], "N3")
        system = _build_rate_system(model, td["tree_height"])
        ll_system = _log_likelihood_from_rate_system(system, td)
        check(close(ll_closed, ll_system, 1e-5),
              f"logL λ₀={lam0} z={z} ε={eps}: closed={ll_closed:.6f} system={ll_system:.6f}")


# ===================================================================
# 9. TDBD ε→0 退化到 TDB
# ===================================================================
def test_tdbd_reduces_to_tdb():
    print("\n=== TDBD ε→0 退化到 TDB ===")
    td = _build_tree()
    for lam0, z in [(0.8, -0.2), (0.5, 0.5), (1.0, 0.0)]:
        ll_tdb = log_likelihood_tdb(lam0, z, td)
        ll_tdbd = log_likelihood_tdbd(lam0, z, 1e-14, td)
        check(close(ll_tdb, ll_tdbd, 1e-4),
              f"λ₀={lam0} z={z}: TDB={ll_tdb:.6f} TDBD(ε≈0)={ll_tdbd:.6f}")


# ===================================================================
# 10. TDBD z=0 退化到 CRBD
# ===================================================================
def test_tdbd_z0_reduces_to_crbd():
    print("\n=== TDBD z=0 退化到 CRBD ===")
    td = _build_tree()
    for lam, mu in [(0.5, 0.1), (1.0, 0.8), (0.3, 0.15)]:
        eps = mu / lam
        ll_crbd = log_likelihood_crbd(lam, mu, td)
        ll_tdbd = log_likelihood_tdbd(lam, 0.0, eps, td)
        check(close(ll_crbd, ll_tdbd, 1e-5),
              f"λ={lam} μ={mu}: CRBD={ll_crbd:.6f} TDBD(z=0)={ll_tdbd:.6f}")


# ===================================================================
# 11. BAMM 退化: η=0 → bg, bg=fg → base
# ===================================================================
def test_bamm_degeneracies():
    print("\n=== BAMM 退化验证 ===")
    td = _build_tree()

    bg = Expression("CRBD", [0.5, 0.1], [], "N3")
    fg = Expression("CRBD", [1.0, 0.3], [], "N3")

    bamm_eta0 = Expression("BAMM", [0.0], [bg.copy(), fg.copy()], "N3")
    ll_eta0 = log_likelihood_bamm(bamm_eta0, td)
    ll_bg = log_likelihood_crbd(0.5, 0.1, td)
    check(close(ll_eta0, ll_bg, 1e-5),
          f"BAMM(η=0) = bg: {ll_eta0:.6f} vs {ll_bg:.6f}")

    bamm_same = Expression("BAMM", [0.5], [bg.copy(), bg.copy()], "N3")
    ll_same = log_likelihood_bamm(bamm_same, td)
    check(close(ll_same, ll_bg, 1e-4),
          f"BAMM(bg=fg) = base: {ll_same:.6f} vs {ll_bg:.6f}")


# ===================================================================
# 12. BAMM 混合求解: fg 用闭式, bg 用 ODE
# ===================================================================
def test_bamm_hybrid():
    print("\n=== BAMM hybrid 求解验证 ===")
    td = _build_tree()
    T = td["tree_height"]

    bg = Expression("CRBD", [0.5, 0.1], [], "N3")
    fg = Expression("CRBD", [1.0, 0.3], [], "N3")
    bamm = Expression("BAMM", [0.3], [bg, fg], "N3")

    ll = log_likelihood_bamm(bamm, td)
    check(np.isfinite(ll), f"BAMM(CRBD,CRBD) 有限: {ll:.6f}")

    bg2 = Expression("TDBD", [0.5, -0.2, 0.3], [], "N3")
    fg2 = Expression("CRBD", [1.0, 0.2], [], "N3")
    bamm2 = Expression("BAMM", [0.2], [bg2, fg2], "N3")
    ll2 = log_likelihood_bamm(bamm2, td)
    check(np.isfinite(ll2), f"BAMM(TDBD,CRBD) 有限: {ll2:.6f}")

    # 三层嵌套
    b1 = Expression("BAMM", [0.2],
        [Expression("CRB", [0.8], [], "N3"),
         Expression("CRBD", [1.1, 0.25], [], "N3")], "N3")
    b2 = Expression("BAMM", [0.15],
        [b1, Expression("TDBD", [1.0, -0.2, 0.2], [], "N3")], "N3")
    b3 = Expression("BAMM", [0.1],
        [b2, Expression("CRB", [0.9], [], "N3")], "N3")
    ll3 = log_likelihood_bamm(b3, td)
    check(np.isfinite(ll3), f"三层嵌套 BAMM 有限: {ll3:.6f}")


# ===================================================================
# 13. 噪声混合模型退化
# ===================================================================
def test_noise_mixing():
    print("\n=== 噪声混合退化 ===")
    td = _build_tree()
    model = Expression("CRB", [0.7], [], "N3")

    expr0 = Expression("NoisyPhylo", [], [Expression("Noise", [0.0], [], "N2"), model], "N1")
    expr1 = Expression("NoisyPhylo", [], [Expression("Noise", [1.0], [], "N2"), model], "N1")
    exprbig = Expression("NoisyPhylo", [], [Expression("Noise", [1e6], [], "N2"), model], "N1")

    ll_model = log_likelihood_crb(0.7, td)
    ll_noise = _log_observation_noise_baseline(td)

    check(close(evaluate_likelihood(expr0, td), ll_model, 1e-8),
          "σ=0 退化到模型似然")
    expected_mix = float(np.logaddexp(log(0.5) + ll_model, log(0.5) + ll_noise))
    check(close(evaluate_likelihood(expr1, td), expected_mix, 1e-8),
          "σ=1 等权混合")
    check(close(evaluate_likelihood(exprbig, td), ll_noise, 2e-5),
          "σ→∞ 逼近噪声基线")


# ===================================================================
# 14. BAMM 不可逆验证
# ===================================================================
def test_bamm_irreversible():
    print("\n=== BAMM 不可逆切换 ===")
    td = _build_tree()
    bg = Expression("CRBD", [1.0, 0.2], [], "N3")
    fg = Expression("TDBD", [1.3, -0.3, 0.25], [], "N3")
    bamm = Expression("BAMM", [0.5], [bg, fg], "N3")
    system = _build_rate_system(bamm, td["tree_height"])
    Q = system["Q"]
    check(Q[0, 1] > 0, "bg->fg 正向切换")
    check(abs(Q[1, 0]) < 1e-12, "fg->bg 无回切")


# ===================================================================
# 15. 原始测试套件回归
# ===================================================================
def test_original_suite():
    print("\n=== 原始测试套件回归 ===")
    td = _build_tree()

    # CRB
    lam = 0.7
    expected = lgamma(td["n"]) + (td["n"] - 1) * log(lam) - lam * td["total_length"]
    check(close(log_likelihood_crb(lam, td), expected, 2e-5), "CRB 闭式")

    # CRBD vs legacy (修复了偏大问题)
    def legacy_crbd(lam, mu, tree_data):
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

    fixed = log_likelihood_crbd(1.0, 0.8, td)
    legacy = legacy_crbd(1.0, 0.8, td)
    check(fixed < legacy - 1.0, f"CRBD 修复高灭绝率偏差: {fixed:.4f} << {legacy:.4f}")

    # TDBD ε=0 → TDB
    check(close(log_likelihood_tdbd(0.8, -0.2, 0.0, td),
                log_likelihood_tdb(0.8, -0.2, td), 3e-5),
          "TDBD(ε=0) = TDB")

    # TDB z 方向
    T = 2.0
    check(_lambda_at(1.0, 0.5, T, 0.0) > _lambda_at(1.0, 0.5, T, T),
          "TDB z>0: 现世出生率 > 根部")


if __name__ == "__main__":
    test_crbd_E()
    test_crbd_survival()
    test_crbd_branch_propagation()
    test_crbd_full_likelihood()
    test_crbd_reduces_to_crb()
    test_tdbd_E()
    test_tdbd_branch_propagation()
    test_tdbd_full_likelihood()
    test_tdbd_reduces_to_tdb()
    test_tdbd_z0_reduces_to_crbd()
    test_bamm_degeneracies()
    test_bamm_hybrid()
    test_noise_mixing()
    test_bamm_irreversible()
    test_original_suite()

    print(f"\n{'='*50}")
    print(f"通过: {PASS}, 失败: {FAIL}")
    if FAIL == 0:
        print("全部验证通过!")
    else:
        print(f"有 {FAIL} 项失败, 请检查。")
        sys.exit(1)
