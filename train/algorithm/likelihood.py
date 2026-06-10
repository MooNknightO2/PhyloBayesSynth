"""似然函数: 对每种系统发育模型计算给定树数据的对数似然。

关键设计:
1) 单状态 CRBD / TDBD 使用 Riccati 方程闭式解,
   避免 ODE 开销, 同时保持与多状态框架完全一致。
2) BAMM 对吸收态（fg 状态）使用闭式解, 仅对耦合态数值积分 (降维 ODE)。
3) 所有含灭绝模型统一进行根存活条件化。
4) BAMM 采用单向 change-point 机制（仅 bg -> fg，不允许回切）。
"""

from math import exp, lgamma, log

import numpy as np
from scipy.integrate import solve_ivp
from .grammar import DEFAULT_CONFIG


_EPS = 1e-12
_MAX_BAMM_NESTING = 3
_EXP_UPPER = 700.0
_EXP_LOWER = -745.0
_RATE_CAP = 1e4
_EXTINCTION_STATE_CAP = 10.0
_D_STATE_CAP = 1e120


def _lambda_at(lam0, z, T, tau):
    expo = z * (T - tau)
    if expo > _EXP_UPPER:
        return float("inf")
    if expo < _EXP_LOWER:
        return 0.0
    return lam0 * exp(expo)


def _lambda_integral(lam0, z, T, tau_c, tau_p):
    if abs(z) < 1e-12:
        return lam0 * (tau_p - tau_c)
    return lam0 / z * (exp(z * (T - tau_c)) - exp(z * (T - tau_p)))


# ===================================================================
# 闭式解辅助函数
# ===================================================================
#
# 对于 μ/λ 恒定的出生-死亡过程, E 的 Riccati 方程可解析求解:
#   dE/dτ = λ(τ)(E - 1)(E - ε),  E(0)=0,  ε = μ/λ
# 解: E(τ) = ε(1-ψ)/(ε-ψ),  ψ = exp((1-ε)Λ(τ)),  Λ(τ)=∫₀^τ λ(s)ds
#
# D 传播为线性 ODE, 其因子也有闭式:
#   D(τ_p)/D(τ_c) = ψ_p(ψ_c-ε)² / (ψ_c(ψ_p-ε)²)


def _g_crbd(lam, mu, r, tau):
    """log|λe^{rτ}-μ| for CRBD (r = λ-μ, constant rates)."""
    rt = r * tau
    if rt > _EXP_UPPER:
        return log(lam) + rt
    if rt < _EXP_LOWER:
        return log(mu) if mu > _EPS else -np.inf
    return log(abs(lam * exp(rt) - mu))


def _crbd_E_val(lam, mu, tau):
    """CRBD 灭绝概率 E(τ) 闭式, E(0)=0."""
    if tau < _EPS or mu < _EPS:
        return 0.0
    r = lam - mu
    if abs(r) < _EPS:
        return lam * tau / (1.0 + lam * tau)
    rt = r * tau
    if rt > _EXP_UPPER:
        return mu / lam
    if rt < _EXP_LOWER:
        return 1.0
    ert = exp(rt)
    denom = lam * ert - mu
    if abs(denom) < _EPS:
        return 1.0
    return mu * (ert - 1.0) / denom


def _crbd_log_one_minus_E(lam, mu, tau):
    """log(1-E(τ)) for CRBD,  1-E = r·e^{rτ}/(λe^{rτ}-μ)."""
    if tau < _EPS or mu < _EPS:
        return 0.0
    r = lam - mu
    if abs(r) < _EPS:
        return -log(1.0 + lam * tau)
    rt = r * tau
    if rt > _EXP_UPPER:
        return log(abs(r)) - log(lam)
    if rt < _EXP_LOWER:
        return -np.inf
    return log(abs(r)) + rt - _g_crbd(lam, mu, r, tau)


def _crbd_log_branch_factor(lam, mu, tau_c, tau_p):
    """log(D(τ_p)/D(τ_c)),  = r(τ_p-τ_c) + 2g(τ_c) - 2g(τ_p)."""
    if tau_p <= tau_c + _EPS:
        return 0.0
    if mu < _EPS:
        return -lam * (tau_p - tau_c)
    r = lam - mu
    if abs(r) < _EPS:
        return 2.0 * (log(1.0 + lam * tau_c) - log(1.0 + lam * tau_p))
    return (r * (tau_p - tau_c)
            + 2.0 * _g_crbd(lam, mu, r, tau_c)
            - 2.0 * _g_crbd(lam, mu, r, tau_p))


def _crbd_branch_factor_val(lam, mu, tau_c, tau_p):
    """D(τ_p)/D(τ_c) 数值."""
    lf = _crbd_log_branch_factor(lam, mu, tau_c, tau_p)
    if lf > _EXP_UPPER:
        return _D_STATE_CAP
    if lf < _EXP_LOWER:
        return 0.0
    return exp(lf)


# ===================================================================
# 通用 Riccati 解 (任意初始条件 E₀)
# ===================================================================
#
# dE/dτ = λ(E-1)(E-ε),  E(τ₀)=E₀,  ε=μ/λ
# 定义 V(s) = (E₀-ε) - (E₀-1)·exp(r·s),  s = τ-τ₀
# 则 E(s) = W(s)/V(s),  W(s) = (E₀-ε) - ε(E₀-1)·exp(r·s)
# D 传播: log D(s₁)/D(s₀) = r(s₁-s₀) + 2 log|V(s₀)| - 2 log|V(s₁)|


def _log_abs_V(E0, eps, r, s):
    """Compute log|V(s)| stably where V(s) = (E0-eps) - (E0-1)*exp(r*s)."""
    rs = r * s
    if rs > _EXP_UPPER:
        one_m_E0 = 1.0 - E0
        if one_m_E0 <= 0:
            return -np.inf
        return log(one_m_E0) + rs
    if rs < _EXP_LOWER:
        val = E0 - eps
        return log(max(abs(val), _EPS))
    V = (E0 - eps) - (E0 - 1.0) * exp(rs)
    if abs(V) < _EPS:
        return log(_EPS)
    return log(abs(V))


def _general_riccati_E_val(lam, mu, E0, s):
    """通用 Riccati E(s): 常数 (λ,μ), 初始条件 E(0)=E₀."""
    if s < _EPS:
        return E0
    if mu < _EPS:
        if E0 < _EPS:
            return 0.0
        ls = lam * s
        if ls > _EXP_UPPER:
            return 1.0
        denom = E0 - (E0 - 1.0) * exp(ls)
        if abs(denom) < _EPS:
            return 1.0
        return E0 / denom
    r = lam - mu
    eps = mu / lam
    if abs(r) < _EPS:
        one_m_E0 = 1.0 - E0
        if one_m_E0 < _EPS:
            return 1.0
        return 1.0 - one_m_E0 / (1.0 + one_m_E0 * lam * s)
    rs = r * s
    if rs > _EXP_UPPER:
        return eps if E0 < 1.0 else 1.0
    if rs < _EXP_LOWER:
        val = E0 - eps
        if abs(val) < _EPS:
            return E0
        return 1.0 if E0 > eps else E0
    ers = exp(rs)
    V = (E0 - eps) - (E0 - 1.0) * ers
    W = (E0 - eps) - eps * (E0 - 1.0) * ers
    if abs(V) < _EPS:
        return 1.0
    return W / V


def _general_log_one_minus_E(lam, mu, E0, s):
    """log(1-E(s)) for general Riccati.

    1-E = (1-ε)(1-E₀)·exp(r·s) / V(s).
    """
    if s < _EPS:
        if E0 >= 1.0:
            return -np.inf
        return log(1.0 - E0)
    if mu < _EPS:
        if E0 < _EPS:
            return 0.0
        ls = lam * s
        if ls > _EXP_UPPER:
            return -np.inf
        denom = E0 - (E0 - 1.0) * exp(ls)
        val = (1.0 - E0) * exp(ls) / denom
        if val <= 0:
            return -np.inf
        return log(val)
    r = lam - mu
    eps = mu / lam
    if abs(r) < _EPS:
        one_m_E0 = 1.0 - E0
        if one_m_E0 < _EPS:
            return -np.inf
        return log(one_m_E0) - log(1.0 + one_m_E0 * lam * s)
    one_m_eps = 1.0 - eps
    one_m_E0 = 1.0 - E0
    if one_m_E0 < _EPS or one_m_eps < _EPS:
        return -np.inf
    rs = r * s
    log_V = _log_abs_V(E0, eps, r, s)
    if not np.isfinite(log_V):
        return -np.inf
    return log(one_m_eps) + log(one_m_E0) + rs - log_V


def _general_log_branch_factor(lam, mu, E0, s_c, s_p):
    """log D(s_p)/D(s_c) for constant (λ,μ) interval with E(0)=E₀.

    s_c, s_p are measured from the interval start (τ₀).

    Formula: r(s_p-s_c) + 2 log|V(s_c)| - 2 log|V(s_p)|
    """
    if s_p <= s_c + _EPS:
        return 0.0
    if mu < _EPS:
        if E0 < _EPS:
            return -lam * (s_p - s_c)
        ls_c = lam * s_c
        ls_p = lam * s_p
        V_c = E0 - (E0 - 1.0) * (exp(ls_c) if ls_c < _EXP_UPPER else float("inf"))
        V_p = E0 - (E0 - 1.0) * (exp(ls_p) if ls_p < _EXP_UPPER else float("inf"))
        log_Vc = log(max(abs(V_c), _EPS))
        log_Vp = log(max(abs(V_p), _EPS))
        return lam * (s_p - s_c) + 2.0 * log_Vc - 2.0 * log_Vp
    r = lam - mu
    if abs(r) < _EPS:
        one_m_E0 = max(1.0 - E0, _EPS)
        return 2.0 * (log(1.0 + one_m_E0 * lam * s_c)
                       - log(1.0 + one_m_E0 * lam * s_p))
    eps = mu / lam
    log_Vc = _log_abs_V(E0, eps, r, s_c)
    log_Vp = _log_abs_V(E0, eps, r, s_p)
    if not np.isfinite(log_Vc) or not np.isfinite(log_Vp):
        return -np.inf
    return r * (s_p - s_c) + 2.0 * log_Vc - 2.0 * log_Vp


# --- TDBD 闭式 (推广 CRBD: λ(τ)=λ₀exp(z(T-τ)), μ=ε·λ) ---

def _tdbd_Lambda(lam0, z, T, tau):
    """Λ(τ) = ∫₀^τ λ(s) ds."""
    return _lambda_integral(lam0, z, T, 0.0, tau)


def _tdbd_E_val(lam0, z, epsilon, T, tau):
    """TDBD E(τ) = ε(1-ψ)/(ε-ψ),  ψ = exp((1-ε)Λ(τ))."""
    if tau < _EPS or epsilon < _EPS:
        return 0.0
    Lam = _tdbd_Lambda(lam0, z, T, tau)
    if Lam < _EPS:
        return 0.0
    if abs(1.0 - epsilon) < _EPS:
        return Lam / (1.0 + Lam)
    log_psi = (1.0 - epsilon) * Lam
    if log_psi > _EXP_UPPER:
        return epsilon
    if log_psi < _EXP_LOWER:
        return 1.0
    psi = exp(log_psi)
    denom = epsilon - psi
    if abs(denom) < _EPS:
        return 1.0
    return epsilon * (1.0 - psi) / denom


def _tdbd_log_one_minus_E(lam0, z, epsilon, T, tau):
    """log(1-E(τ)) = log|1-ε| + (1-ε)Λ - log|ψ-ε|."""
    if tau < _EPS or epsilon < _EPS:
        return 0.0
    Lam = _tdbd_Lambda(lam0, z, T, tau)
    if Lam < _EPS:
        return 0.0
    if abs(1.0 - epsilon) < _EPS:
        return -log(1.0 + Lam)
    log_psi = (1.0 - epsilon) * Lam
    if log_psi < _EXP_LOWER:
        return -np.inf
    log_d = _log_abs_psi_minus_eps(log_psi, epsilon)
    if not np.isfinite(log_d):
        return -np.inf
    return log(abs(1.0 - epsilon)) + log_psi - log_d


def _log_abs_psi_minus_eps(log_psi, epsilon):
    """Compute log|ψ-ε| stably given log(ψ), avoiding exp overflow."""
    if log_psi > 50.0:
        return log_psi + log(1.0 - epsilon * exp(-log_psi))
    if log_psi < -50.0:
        return log(epsilon)
    psi = exp(log_psi)
    d = psi - epsilon
    if abs(d) < _EPS:
        return log(_EPS)
    return log(abs(d))


def _tdbd_log_branch_factor(lam0, z, epsilon, T, tau_c, tau_p):
    """log(D(τ_p)/D(τ_c)) = (1-ε)(Λ_p-Λ_c) + 2log|ψ_c-ε| - 2log|ψ_p-ε|."""
    if tau_p <= tau_c + _EPS:
        return 0.0
    Lam_c = _tdbd_Lambda(lam0, z, T, tau_c)
    Lam_p = _tdbd_Lambda(lam0, z, T, tau_p)
    if epsilon < _EPS:
        return -(Lam_p - Lam_c)
    if abs(1.0 - epsilon) < _EPS:
        return 2.0 * (log(1.0 + Lam_c) - log(1.0 + Lam_p))
    one_m_eps = 1.0 - epsilon
    log_psi_c = one_m_eps * Lam_c
    log_psi_p = one_m_eps * Lam_p
    log_vc = _log_abs_psi_minus_eps(log_psi_c, epsilon)
    log_vp = _log_abs_psi_minus_eps(log_psi_p, epsilon)
    if not np.isfinite(log_vc) or not np.isfinite(log_vp):
        return -np.inf
    return one_m_eps * (Lam_p - Lam_c) + 2.0 * log_vc - 2.0 * log_vp


def _tdbd_branch_factor_val(lam0, z, epsilon, T, tau_c, tau_p):
    """D(τ_p)/D(τ_c) 数值."""
    lf = _tdbd_log_branch_factor(lam0, z, epsilon, T, tau_c, tau_p)
    if lf > _EXP_UPPER:
        return _D_STATE_CAP
    if lf < _EXP_LOWER:
        return 0.0
    return exp(lf)


# ===================================================================
# 速率系统构建
# ===================================================================

def _prepare_single_regime(tag, params):
    """构造单个 regime 的 λ(τ), μ(τ) 函数与状态元信息。"""
    if tag == "CRB":
        lam = params[0]
        if lam <= 0:
            return None

        def lam_fn(_tau, _T):
            return lam

        def mu_fn(_tau, _T):
            return 0.0

        info = {"type": "constant", "lam": lam, "mu": 0.0}
        return lam_fn, mu_fn, info

    if tag == "CRBD":
        lam, mu = params
        if lam <= 0 or mu < 0:
            return None

        def lam_fn(_tau, _T):
            return lam

        def mu_fn(_tau, _T):
            return mu

        info = {"type": "constant", "lam": lam, "mu": mu}
        return lam_fn, mu_fn, info

    if tag == "TDB":
        lam0, z = params
        if lam0 <= 0:
            return None

        def lam_fn(tau, T):
            return _lambda_at(lam0, z, T, tau)

        def mu_fn(_tau, _T):
            return 0.0

        info = {"type": "tdbd", "lam0": lam0, "z": z, "epsilon": 0.0}
        return lam_fn, mu_fn, info

    if tag == "TDBD":
        lam0, z, epsilon = params
        if lam0 <= 0 or epsilon < 0 or epsilon >= 1.0:
            return None

        def lam_fn(tau, T):
            return _lambda_at(lam0, z, T, tau)

        def mu_fn(tau, T):
            return epsilon * _lambda_at(lam0, z, T, tau)

        info = {"type": "tdbd", "lam0": lam0, "z": z, "epsilon": epsilon}
        return lam_fn, mu_fn, info

    if tag == "PCBD":
        lam1, mu1, lam2, mu2, tau_frac = params
        if lam1 <= 0 or lam2 <= 0 or mu1 < 0 or mu2 < 0:
            return None
        if tau_frac <= 0 or tau_frac >= 1.0:
            return None

        def lam_fn(tau, T):
            return lam1 if tau <= tau_frac * T else lam2

        def mu_fn(tau, T):
            return mu1 if tau <= tau_frac * T else mu2

        info = {"type": "pcbd", "lam1": lam1, "mu1": mu1,
                "lam2": lam2, "mu2": mu2, "tau_frac": tau_frac}
        return lam_fn, mu_fn, info

    return None


def _build_rate_system(model_expr, T, bamm_depth=0, max_bamm_nesting=_MAX_BAMM_NESTING):
    """返回系统定义:
    - n_states, rate_fn, Q, root_state, labels, state_info, _T
    """
    tag = model_expr.tag
    if tag != "BAMM":
        prepared = _prepare_single_regime(tag, model_expr.params)
        if prepared is None:
            return None
        lam_fn, mu_fn, state_info_single = prepared

        def rate_fn(tau):
            return np.array([lam_fn(tau, T)]), np.array([mu_fn(tau, T)])

        return {
            "n_states": 1,
            "rate_fn": rate_fn,
            "Q": np.zeros((1, 1), dtype=float),
            "root_state": 0,
            "labels": [()],
            "state_info": [state_info_single],
            "_T": T,
        }

    depth_now = bamm_depth + 1
    if depth_now > max_bamm_nesting:
        return None

    eta = model_expr.params[0]
    if eta < 0 or len(model_expr.children) != 2:
        return None
    bg_model = model_expr.children[0]
    fg_model = model_expr.children[1]
    bg_system = _build_rate_system(bg_model, T, depth_now, max_bamm_nesting)
    fg_system = _build_rate_system(fg_model, T, depth_now, max_bamm_nesting)
    if bg_system is None or fg_system is None:
        return None

    n_bg = bg_system["n_states"]
    n_fg = fg_system["n_states"]
    n_total = n_bg + n_fg
    bg_root = int(bg_system["root_state"])
    bg_labels = list(bg_system.get("labels", [()] * n_bg))
    fg_labels = list(fg_system.get("labels", [()] * n_fg))
    fg_label_to_idx = {lbl: idx for idx, lbl in enumerate(fg_labels)}

    def rate_fn(tau):
        lam_bg, mu_bg = bg_system["rate_fn"](tau)
        lam_fg, mu_fg = fg_system["rate_fn"](tau)
        return (
            np.concatenate([np.asarray(lam_bg, dtype=float), np.asarray(lam_fg, dtype=float)]),
            np.concatenate([np.asarray(mu_bg, dtype=float), np.asarray(mu_fg, dtype=float)]),
        )

    Q = np.zeros((n_total, n_total), dtype=float)
    Q[:n_bg, :n_bg] = bg_system["Q"]
    Q[n_bg:, n_bg:] = fg_system["Q"]
    for i in range(n_bg):
        target_fg = fg_label_to_idx.get(bg_labels[i], int(fg_system["root_state"]))
        Q[i, i] -= eta
        Q[i, n_bg + target_fg] += eta

    labels = [("bg",) + tuple(lbl) for lbl in bg_labels] + [
        ("fg",) + tuple(lbl) for lbl in fg_labels
    ]

    state_info = list(bg_system.get("state_info", [])) + list(fg_system.get("state_info", []))

    return {
        "n_states": n_total,
        "rate_fn": rate_fn,
        "Q": Q,
        "root_state": bg_root,
        "labels": labels,
        "state_info": state_info,
        "_T": T,
    }


def _rate_system_is_numerically_safe(rate_system, T):
    """快速筛掉明显数值病态的速率系统，避免 ODE 积分器长时间僵死。"""
    Q = np.asarray(rate_system["Q"], dtype=float)
    if np.any(~np.isfinite(Q)):
        return False
    if np.any(np.diag(Q) > _EPS):
        return False
    offdiag = Q.copy()
    np.fill_diagonal(offdiag, 0.0)
    if np.any(offdiag < -_EPS):
        return False
    if np.max(np.abs(Q)) > _RATE_CAP:
        return False

    n_probe = 5
    if T <= _EPS:
        taus = [0.0]
    else:
        taus = np.linspace(0.0, T, n_probe)
    for tau in taus:
        lam, mu = rate_system["rate_fn"](float(tau))
        lam = np.asarray(lam, dtype=float)
        mu = np.asarray(mu, dtype=float)
        if np.any(~np.isfinite(lam)) or np.any(~np.isfinite(mu)):
            return False
        if np.any(lam < 0.0) or np.any(mu < 0.0):
            return False
        if np.max(lam) > _RATE_CAP or np.max(mu) > _RATE_CAP:
            return False
    return True


# ===================================================================
# 灭绝概率求解 (闭式 + ODE 混合)
# ===================================================================

def _identify_analytical_E_states(n_states, Q, state_info, T):
    """识别可用闭式解的吸收态, 返回 {state_idx: E_func(tau)}。"""
    if state_info is None:
        return {}
    analytical = {}
    for i in range(n_states):
        off_diag = sum(abs(Q[i, j]) for j in range(n_states) if j != i)
        if off_diag > _EPS:
            continue
        si = state_info[i]
        if si["type"] == "constant":
            lam_i, mu_i = si["lam"], si["mu"]
            analytical[i] = lambda tau, l=lam_i, m=mu_i: _crbd_E_val(l, m, tau)
        elif si["type"] == "tdbd":
            l0, z_i, eps_i = si["lam0"], si["z"], si["epsilon"]
            analytical[i] = lambda tau, l=l0, z=z_i, e=eps_i, TT=T: _tdbd_E_val(l, z, e, TT, tau)
    return analytical


def _integrate_extinction(rate_system, T):
    """求解灭绝概率向量 E(τ), 边界条件 E(0)=0.

    吸收态使用闭式 Riccati 解, 其余状态缩减维度后数值积分。
    """
    n_states = rate_system["n_states"]
    rate_fn = rate_system["rate_fn"]
    Q = rate_system["Q"]
    state_info = rate_system.get("state_info")

    analytical = _identify_analytical_E_states(n_states, Q, state_info, T)
    active = [i for i in range(n_states) if i not in analytical]
    n_active = len(active)

    if n_active == 0:
        def e_sol(tau):
            r = np.zeros(n_states, dtype=float)
            for i, fn in analytical.items():
                r[i] = fn(float(tau))
            return r
        return e_sol

    def _build_full_E(tau, y_active):
        E = np.zeros(n_states, dtype=float)
        for i, fn in analytical.items():
            E[i] = fn(tau)
        for idx, i in enumerate(active):
            E[i] = y_active[idx]
        return E

    def rhs(tau, y):
        E_full = _build_full_E(tau, y)
        if np.any(~np.isfinite(E_full)) or np.max(np.abs(E_full)) > _EXTINCTION_STATE_CAP:
            raise FloatingPointError("extinction state diverged")
        lam, mu = rate_fn(tau)
        lam = np.asarray(lam, dtype=float)
        mu = np.asarray(mu, dtype=float)
        if np.any(~np.isfinite(lam)) or np.any(~np.isfinite(mu)):
            raise FloatingPointError("non-finite rates")
        if np.max(lam) > _RATE_CAP or np.max(mu) > _RATE_CAP:
            raise FloatingPointError("rates too large")
        dy = np.zeros(n_active, dtype=float)
        for idx, i in enumerate(active):
            shift_term = 0.0
            for j in range(n_states):
                if i != j:
                    shift_term += Q[i, j] * E_full[j]
            dy[idx] = (
                mu[i]
                - (lam[i] + mu[i] - Q[i, i]) * E_full[i]
                + lam[i] * (E_full[i] ** 2)
                + shift_term
            )
        return dy

    try:
        sol = solve_ivp(
            rhs,
            (0.0, T),
            np.zeros(n_active, dtype=float),
            dense_output=True,
            method="BDF",
            rtol=1e-7,
            atol=1e-9,
            max_step=max(1e-3, T / 300.0),
        )
    except (FloatingPointError, OverflowError, ValueError):
        return None
    if not sol.success:
        return None

    dense = sol.sol

    def e_sol(tau):
        r = np.zeros(n_states, dtype=float)
        for i, fn in analytical.items():
            r[i] = fn(float(tau))
        reduced = np.asarray(dense(tau), dtype=float).reshape(-1)
        for idx, i in enumerate(active):
            r[i] = reduced[idx]
        return r

    return e_sol


# ===================================================================
# D 向量枝上传播 (闭式 + ODE 混合)
# ===================================================================

def _identify_analytical_D_states(n_states, Q, state_info, T_tree, tau_c, d_child):
    """识别可用闭式传播 D 的吸收态.

    返回:
      result_d: {state_idx: D_i(tau_p)}
      analytical_d_fn: {state_idx: callable(tau) -> D_i(tau)}
    """
    if state_info is None:
        return {}, {}
    result_d = {}
    analytical_d_fn = {}
    for i in range(n_states):
        off_diag = sum(abs(Q[i, j]) for j in range(n_states) if j != i)
        if off_diag > _EPS:
            continue
        si = state_info[i]
        d0 = float(d_child[i])
        if si["type"] == "constant":
            lam_i, mu_i = si["lam"], si["mu"]
            analytical_d_fn[i] = lambda tau, d=d0, l=lam_i, m=mu_i, tc=tau_c: (
                d * _crbd_branch_factor_val(l, m, tc, tau)
            )
        elif si["type"] == "tdbd":
            l0, z_i, eps_i = si["lam0"], si["z"], si["epsilon"]
            analytical_d_fn[i] = lambda tau, d=d0, l=l0, z=z_i, e=eps_i, TT=T_tree, tc=tau_c: (
                d * _tdbd_branch_factor_val(l, z, e, TT, tc, tau)
            )
        else:
            continue
        result_d[i] = None  # placeholder; 实际值由 analytical_d_fn 在 tau_p 处求得
    return result_d, analytical_d_fn


def _propagate_branch_vector(d_child, tau_c, tau_p, rate_system, e_sol):
    """沿枝 [tau_c, tau_p] 传播 D 向量.

    吸收态闭式传播, 耦合态缩减 ODE。
    """
    if tau_p <= tau_c + _EPS:
        return d_child.copy()

    n_states = rate_system["n_states"]
    rate_fn = rate_system["rate_fn"]
    Q = rate_system["Q"]
    state_info = rate_system.get("state_info")
    T_tree = rate_system.get("_T", tau_p)

    closed_set, analytical_d_fn = _identify_analytical_D_states(
        n_states, Q, state_info, T_tree, tau_c, d_child
    )

    active = [i for i in range(n_states) if i not in closed_set]
    n_active = len(active)

    if n_active == 0:
        d_result = np.zeros(n_states, dtype=float)
        for i, fn in analytical_d_fn.items():
            d_result[i] = fn(tau_p)
        return d_result

    def rhs(tau, d):
        D_full = np.zeros(n_states, dtype=float)
        for i, fn in analytical_d_fn.items():
            D_full[i] = fn(tau)
        for idx, i in enumerate(active):
            D_full[i] = d[idx]
        if np.any(~np.isfinite(D_full)) or np.max(np.abs(D_full)) > _D_STATE_CAP:
            raise FloatingPointError("D state diverged")
        lam, mu = rate_fn(tau)
        lam = np.asarray(lam, dtype=float)
        mu = np.asarray(mu, dtype=float)
        if np.any(~np.isfinite(lam)) or np.any(~np.isfinite(mu)):
            raise FloatingPointError("non-finite rates")
        if np.max(lam) > _RATE_CAP or np.max(mu) > _RATE_CAP:
            raise FloatingPointError("rates too large")
        e = np.asarray(e_sol(tau), dtype=float).reshape(-1)
        if np.any(~np.isfinite(e)) or np.max(np.abs(e)) > _EXTINCTION_STATE_CAP:
            raise FloatingPointError("extinction state invalid")
        dd = np.zeros(n_active, dtype=float)
        for idx, i in enumerate(active):
            dd[idx] = (-(lam[i] + mu[i] - Q[i, i]) + 2.0 * lam[i] * e[i]) * D_full[i]
            for j in range(n_states):
                if i != j:
                    dd[idx] += Q[i, j] * D_full[j]
        return dd

    d0 = np.array([float(d_child[i]) for i in active], dtype=float)
    try:
        sol = solve_ivp(
            rhs,
            (tau_c, tau_p),
            d0,
            method="BDF",
            rtol=1e-7,
            atol=1e-9,
            max_step=max(1e-3, (tau_p - tau_c) / 80.0),
        )
    except (FloatingPointError, OverflowError, ValueError):
        return None
    if not sol.success:
        return None

    d_result = np.zeros(n_states, dtype=float)
    for i, fn in analytical_d_fn.items():
        d_result[i] = fn(tau_p)
    for idx, i in enumerate(active):
        d_result[i] = sol.y[idx, -1]
    return d_result


def _compute_d_vector(node, rate_system, e_sol, T):
    """后序递归计算节点处 D 向量。"""
    n_states = rate_system["n_states"]
    rate_fn = rate_system["rate_fn"]

    if node.is_leaf():
        if node.age < -1e-8:
            return None
        return np.ones(n_states, dtype=float)

    child_vectors = []
    for child in node.children:
        d_child_node = _compute_d_vector(child, rate_system, e_sol, T)
        if d_child_node is None:
            return None
        d_up = _propagate_branch_vector(
            d_child_node, child.age, node.age, rate_system, e_sol
        )
        if d_up is None or np.any(~np.isfinite(d_up)) or np.any(d_up < 0):
            return None
        child_vectors.append(d_up)

    lam, _ = rate_fn(node.age)
    lam = np.asarray(lam, dtype=float)
    if np.any(lam <= 0):
        return None

    m = len(child_vectors)
    prod = np.ones(n_states, dtype=float)
    for vec in child_vectors:
        prod *= vec
    return (lam ** max(0, m - 1)) * prod


def _log_likelihood_from_rate_system(rate_system, tree_data):
    n = tree_data["n"]
    T = tree_data["tree_height"]
    root = tree_data["root"]
    if T < 0:
        return -np.inf
    if not _rate_system_is_numerically_safe(rate_system, T):
        return -np.inf

    e_sol = _integrate_extinction(rate_system, T)
    if e_sol is None:
        return -np.inf

    d_root = _compute_d_vector(root, rate_system, e_sol, T)
    if d_root is None:
        return -np.inf
    root_state = rate_system["root_state"]
    root_density = float(d_root[root_state])
    if root_density <= 0 or not np.isfinite(root_density):
        return -np.inf

    e_root = float(np.asarray(e_sol(T), dtype=float).reshape(-1)[root_state])
    survival = 1.0 - e_root
    if survival <= 0 or not np.isfinite(survival):
        return -np.inf

    return lgamma(n) + log(root_density) - 2.0 * log(survival)


# ===================================================================
# 各模型似然函数
# ===================================================================

def log_likelihood_crb(lam, tree_data):
    n = tree_data["n"]
    S = tree_data["total_length"]
    if lam <= 0:
        return -np.inf
    return lgamma(n) + (n - 1) * log(lam) - lam * S


def log_likelihood_crbd(lam, mu, tree_data):
    """CRBD 闭式似然.

    利用 Riccati 方程解析解直接计算, 无需 ODE 积分。
    """
    n = tree_data["n"]
    T = tree_data["tree_height"]
    branches = tree_data["branches"]
    if lam <= 0 or mu < 0 or T < 0:
        return -np.inf
    if mu < _EPS:
        return log_likelihood_crb(lam, tree_data)

    log_L = lgamma(n) + (n - 1) * log(lam)
    for _, tau_c, tau_p in branches:
        log_L += _crbd_log_branch_factor(lam, mu, tau_c, tau_p)

    log_surv = _crbd_log_one_minus_E(lam, mu, T)
    if not np.isfinite(log_surv):
        return -np.inf
    log_L -= 2.0 * log_surv
    return log_L


def log_likelihood_tdb(lam0, z, tree_data):
    n = tree_data["n"]
    T = tree_data["tree_height"]
    branching_times = tree_data["branching_times"]
    branches = tree_data["branches"]
    if lam0 <= 0:
        return -np.inf

    log_L = lgamma(n)
    for tau_k in branching_times:
        lam_k = _lambda_at(lam0, z, T, tau_k)
        if lam_k <= 0:
            return -np.inf
        log_L += log(lam_k)
    for _, tau_c, tau_p in branches:
        log_L -= _lambda_integral(lam0, z, T, tau_c, tau_p)
    return log_L


def log_likelihood_tdbd(lam0, z, epsilon, tree_data):
    """TDBD 闭式似然.

    μ(τ)/λ(τ) = ε 为常数, Riccati 方程可分离变量求解。
    """
    n = tree_data["n"]
    T = tree_data["tree_height"]
    branching_times = tree_data["branching_times"]
    branches = tree_data["branches"]
    if lam0 <= 0 or epsilon < 0 or epsilon >= 1.0 or T < 0:
        return -np.inf
    if epsilon < _EPS:
        return log_likelihood_tdb(lam0, z, tree_data)

    log_L = lgamma(n)
    for tau_k in branching_times:
        lam_k = _lambda_at(lam0, z, T, tau_k)
        if lam_k <= 0:
            return -np.inf
        log_L += log(lam_k)

    for _, tau_c, tau_p in branches:
        log_L += _tdbd_log_branch_factor(lam0, z, epsilon, T, tau_c, tau_p)

    log_surv = _tdbd_log_one_minus_E(lam0, z, epsilon, T, T)
    if not np.isfinite(log_surv):
        return -np.inf
    log_L -= 2.0 * log_surv
    return log_L


def log_likelihood_pcbd(lam1, mu1, lam2, mu2, tau_frac, tree_data):
    """PCBD (分段常数 BD) 闭式似然。

    将时间 [0, T] 分为两个区间:
      区间 1: [0, τ_b]  速率 (λ₁, μ₁)  — 近现世
      区间 2: [τ_b, T]  速率 (λ₂, μ₂)  — 近根部
    其中 τ_b = tau_frac · T。

    利用通用 Riccati 解 (含非零初始条件) 对灭绝概率 E 与 D 传播
    在两个 CRBD 子区间上分别求解析解, 在断点处衔接。
    """
    n = tree_data["n"]
    T = tree_data["tree_height"]
    branches = tree_data["branches"]
    branching_times = tree_data["branching_times"]
    if (lam1 <= 0 or lam2 <= 0 or mu1 < 0 or mu2 < 0
            or tau_frac <= 0 or tau_frac >= 1.0 or T < _EPS):
        return -np.inf
    tau_b = tau_frac * T

    E_b = _crbd_E_val(lam1, mu1, tau_b)
    if not np.isfinite(E_b):
        return -np.inf

    log_L = lgamma(n)
    for tau_k in branching_times:
        lam_k = lam1 if tau_k <= tau_b else lam2
        if lam_k <= 0:
            return -np.inf
        log_L += log(lam_k)

    for _, tau_c, tau_p in branches:
        lbf = _pcbd_log_branch_factor(
            lam1, mu1, lam2, mu2, tau_b, E_b, tau_c, tau_p
        )
        if not np.isfinite(lbf):
            return -np.inf
        log_L += lbf

    s_root = T - tau_b
    log_surv = _general_log_one_minus_E(lam2, mu2, E_b, s_root)
    if not np.isfinite(log_surv):
        return -np.inf
    log_L -= 2.0 * log_surv
    return log_L


def _pcbd_log_branch_factor(lam1, mu1, lam2, mu2, tau_b, E_b,
                             tau_c, tau_p):
    """计算 PCBD 模型中一条枝的 log D(τ_p)/D(τ_c)。

    根据枝与断点 τ_b 的关系拆分为至多两段, 每段使用对应区间
    的通用闭式 branch factor。
    """
    if tau_p <= tau_c + _EPS:
        return 0.0
    if tau_p <= tau_b + _EPS:
        return _crbd_log_branch_factor(lam1, mu1, tau_c, tau_p)
    if tau_c >= tau_b - _EPS:
        s_c = tau_c - tau_b
        s_p = tau_p - tau_b
        return _general_log_branch_factor(lam2, mu2, E_b, s_c, s_p)
    part1 = _crbd_log_branch_factor(lam1, mu1, tau_c, tau_b)
    part2 = _general_log_branch_factor(lam2, mu2, E_b, 0.0, tau_p - tau_b)
    if not np.isfinite(part1) or not np.isfinite(part2):
        return -np.inf
    return part1 + part2


def _model_struct_equal(a, b):
    if a.tag != b.tag:
        return False
    if len(a.params) != len(b.params):
        return False
    if len(a.children) != len(b.children):
        return False
    for x, y in zip(a.params, b.params):
        if abs(x - y) > 1e-12:
            return False
    for ca, cb in zip(a.children, b.children):
        if not _model_struct_equal(ca, cb):
            return False
    return True


def _base_model_log_likelihood(model_expr, tree_data):
    tag = model_expr.tag
    if tag == "CRB":
        return log_likelihood_crb(model_expr.params[0], tree_data)
    if tag == "CRBD":
        return log_likelihood_crbd(model_expr.params[0], model_expr.params[1], tree_data)
    if tag == "TDB":
        return log_likelihood_tdb(model_expr.params[0], model_expr.params[1], tree_data)
    if tag == "TDBD":
        return log_likelihood_tdbd(model_expr.params[0], model_expr.params[1], model_expr.params[2], tree_data)
    if tag == "PCBD":
        p = model_expr.params
        return log_likelihood_pcbd(p[0], p[1], p[2], p[3], p[4], tree_data)
    if tag == "BAMM":
        return log_likelihood_bamm(model_expr, tree_data)
    return -np.inf


def _log_observation_noise_baseline(tree_data):
    """观测噪声基线分布 log p_noise(D)。

    严格概率版本:
    使用 CRB 基线并对 lambda 做先验边际化，而不是用数据驱动的 plug-in 估计。
    设 lambda ~ Exponential(scale=s), 则 rate beta=1/s。
    对应边际似然:
      p_noise(D) = ∫ p(D|lambda, CRB) p(lambda) d lambda
                 = beta * Gamma(n) * Gamma(n) / (S + beta)^n
    其中 n 为 tips 数, S 为总枝长。
    """
    n = tree_data["n"]
    s = tree_data["total_length"]
    if n < 2 or s <= 0 or not np.isfinite(s):
        return -np.inf

    scale = float(DEFAULT_CONFIG["param_priors"]["lambda"][1]["scale"])
    if scale <= 0 or not np.isfinite(scale):
        return -np.inf
    beta = 1.0 / scale

    return log(beta) + lgamma(n) + lgamma(n) - n * log(s + beta)


def _mix_with_observation_noise(log_l_model, sigma, tree_data):
    """log p(D|E,sigma) = log[(1-pi)*p_model + pi*p_noise], pi=sigma/(1+sigma)."""
    if sigma < 0 or not np.isfinite(sigma):
        return -np.inf
    if sigma <= _EPS:
        return log_l_model

    pi = sigma / (1.0 + sigma)
    log_l_noise = _log_observation_noise_baseline(tree_data)
    return float(np.logaddexp(log(1.0 - pi) + log_l_model, log(pi) + log_l_noise))


def log_likelihood_bamm(model_expr, tree_data):
    eta = model_expr.params[0]
    bg_model = model_expr.children[0]
    fg_model = model_expr.children[1]

    if eta <= _EPS:
        return _base_model_log_likelihood(bg_model, tree_data)
    if _model_struct_equal(bg_model, fg_model):
        return _base_model_log_likelihood(bg_model, tree_data)

    system = _build_rate_system(model_expr, tree_data["tree_height"])
    if system is None:
        return -np.inf
    return _log_likelihood_from_rate_system(system, tree_data)


# ===================================================================
# 总入口
# ===================================================================

def evaluate_likelihood(expression, tree_data):
    """计算表达式 E 在给定树数据 D 下的对数似然。

    表达式结构: (NoisyPhylo (Noise ρ) M)
    """
    noise_expr = expression.children[0]
    model_expr = expression.children[1]
    sigma = noise_expr.params[0]

    tag = model_expr.tag
    if tag == "CRB":
        log_L = log_likelihood_crb(model_expr.params[0], tree_data)
    elif tag == "CRBD":
        log_L = log_likelihood_crbd(model_expr.params[0], model_expr.params[1], tree_data)
    elif tag == "TDB":
        log_L = log_likelihood_tdb(model_expr.params[0], model_expr.params[1], tree_data)
    elif tag == "TDBD":
        log_L = log_likelihood_tdbd(
            model_expr.params[0], model_expr.params[1], model_expr.params[2], tree_data
        )
    elif tag == "PCBD":
        p = model_expr.params
        log_L = log_likelihood_pcbd(p[0], p[1], p[2], p[3], p[4], tree_data)
    elif tag == "BAMM":
        log_L = log_likelihood_bamm(model_expr, tree_data)
    else:
        return -np.inf

    return _mix_with_observation_noise(log_L, sigma, tree_data)
