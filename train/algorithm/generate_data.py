"""按指定模型参数生成合成系统发育树数据 (phylojson)。

支持模型:
  - CRB   : 常数出生率
  - CRBD  : 常数出生-死亡率
  - TDB   : 时间依赖出生率, lambda(t)=lambda0*exp(z*t)
  - TDBD  : 时间依赖出生-死亡率, mu(t)=epsilon*lambda(t)
  - BAMM  : 多状态切换模型 (Poisson change-point, bg→fg 单向)

说明:
  t 是从根到现世的前向时间（root=0, present=T）。
  生成的是仅含现生物种的 reconstructed tree，便于直接喂给现有推断流程。
  BAMM 通过表达式字符串指定, 如: "(BAMM 0.3 (CRBD 0.5 0.1) (CRBD 1.0 0.3))"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .expression import Expression, parse_expression


_EXP_UPPER = 700.0
_EXP_LOWER = -745.0


@dataclass
class SimNode:
    time: float
    children: List["SimNode"] = field(default_factory=list)
    is_extant: bool = False
    taxon: Optional[int] = None


@dataclass
class _ActiveLineage:
    node: SimNode
    state: int


def _safe_exp(x: float) -> float:
    if x > _EXP_UPPER:
        return float("inf")
    if x < _EXP_LOWER:
        return 0.0
    return float(np.exp(x))


# ===================================================================
# Simulation rate system (forward-time)
# ===================================================================

@dataclass
class _SimSystem:
    """Forward-time multi-state rate system for simulation."""
    n_states: int
    rate_fns: List  # rate_fns[s](t) -> (lam, mu)
    Q: np.ndarray   # switching rate matrix
    root_state: int
    is_constant: bool


def _build_sim_system(expr: Expression, _depth: int = 0,
                      max_time: float = 500.0) -> _SimSystem:
    """Convert a model Expression to a forward-time simulation system."""
    tag = expr.tag
    if _depth > 3:
        raise ValueError("BAMM nesting exceeds 3 levels")

    if tag == "CRB":
        lam = expr.params[0]
        if lam <= 0:
            raise ValueError(f"CRB: lambda must be > 0, got {lam}")
        return _SimSystem(
            n_states=1,
            rate_fns=[lambda t, l=lam: (l, 0.0)],
            Q=np.zeros((1, 1)),
            root_state=0,
            is_constant=True,
        )

    if tag == "CRBD":
        lam, mu = expr.params
        if lam <= 0:
            raise ValueError(f"CRBD: lambda must be > 0, got {lam}")
        if mu < 0:
            raise ValueError(f"CRBD: mu must be >= 0, got {mu}")
        return _SimSystem(
            n_states=1,
            rate_fns=[lambda t, l=lam, m=mu: (l, m)],
            Q=np.zeros((1, 1)),
            root_state=0,
            is_constant=True,
        )

    if tag == "TDB":
        lam0, z = expr.params
        if lam0 <= 0:
            raise ValueError(f"TDB: lambda0 must be > 0, got {lam0}")
        return _SimSystem(
            n_states=1,
            rate_fns=[lambda t, l=lam0, zz=z: (l * _safe_exp(zz * t), 0.0)],
            Q=np.zeros((1, 1)),
            root_state=0,
            is_constant=abs(z) < 1e-12,
        )

    if tag == "TDBD":
        lam0, z, epsilon = expr.params
        if lam0 <= 0:
            raise ValueError(f"TDBD: lambda0 must be > 0, got {lam0}")
        if epsilon < 0:
            raise ValueError(f"TDBD: epsilon must be >= 0, got {epsilon}")

        def _rf(t, l=lam0, zz=z, e=epsilon):
            lv = l * _safe_exp(zz * t)
            return (lv, e * lv)

        return _SimSystem(
            n_states=1,
            rate_fns=[_rf],
            Q=np.zeros((1, 1)),
            root_state=0,
            is_constant=abs(z) < 1e-12,
        )

    if tag == "PCBD":
        lam1, mu1, lam2, mu2, tau_frac = expr.params
        if lam1 <= 0:
            raise ValueError(f"PCBD: lambda1 must be > 0, got {lam1}")
        if lam2 <= 0:
            raise ValueError(f"PCBD: lambda2 must be > 0, got {lam2}")
        if mu1 < 0:
            raise ValueError(f"PCBD: mu1 must be >= 0, got {mu1}")
        if mu2 < 0:
            raise ValueError(f"PCBD: mu2 must be >= 0, got {mu2}")
        if not (0 < tau_frac < 1):
            raise ValueError(f"PCBD: tau_frac must be in (0,1), got {tau_frac}")
        t_break = max_time * (1.0 - tau_frac)

        def _rf(t, l1=lam1, m1=mu1, l2=lam2, m2=mu2, tb=t_break):
            return (l1, m1) if t >= tb else (l2, m2)

        return _SimSystem(
            n_states=1,
            rate_fns=[_rf],
            Q=np.zeros((1, 1)),
            root_state=0,
            is_constant=False,
        )

    if tag == "BAMM":
        eta = expr.params[0]
        if eta < 0:
            raise ValueError(f"BAMM: eta must be >= 0, got {eta}")
        if len(expr.children) != 2:
            raise ValueError("BAMM must have exactly 2 children (bg, fg)")
        bg = _build_sim_system(expr.children[0], _depth + 1, max_time)
        fg = _build_sim_system(expr.children[1], _depth + 1, max_time)

        n_bg, n_fg = bg.n_states, fg.n_states
        n_total = n_bg + n_fg

        rate_fns = list(bg.rate_fns) + list(fg.rate_fns)

        Q = np.zeros((n_total, n_total))
        Q[:n_bg, :n_bg] = bg.Q
        Q[n_bg:, n_bg:] = fg.Q
        for i in range(n_bg):
            target = n_bg + min(i, n_fg - 1)
            Q[i, i] -= eta
            Q[i, target] += eta

        return _SimSystem(
            n_states=n_total,
            rate_fns=rate_fns,
            Q=Q,
            root_state=bg.root_state,
            is_constant=bg.is_constant and fg.is_constant,
        )

    raise ValueError(f"Unsupported model tag: {tag}")


# ===================================================================
# Next-event sampling
# ===================================================================

def _aggregate_rate(active: List[_ActiveLineage], sys: _SimSystem, t: float) -> float:
    R = 0.0
    for lin in active:
        s = lin.state
        lam_s, mu_s = sys.rate_fns[s](t)
        eta_s = max(0.0, -sys.Q[s, s])
        R += lam_s + mu_s + eta_s
    return R


def _next_event_constant(R_total: float, rng: np.random.Generator) -> float:
    if R_total <= 0 or not np.isfinite(R_total):
        return float("inf")
    return float(rng.exponential(1.0 / R_total))


def _next_event_thinning(
    t: float,
    active: List[_ActiveLineage],
    sys: _SimSystem,
    max_time: float,
    rng: np.random.Generator,
) -> Optional[float]:
    """Lewis-Shedler thinning for time-varying aggregate rate."""
    R_lo = _aggregate_rate(active, sys, t)
    R_hi = _aggregate_rate(active, sys, max_time)
    R_bound = max(R_lo, R_hi) * 1.5 + 1e-6
    if R_bound <= 0:
        return None
    t_cur = t
    for _ in range(100_000):
        dt = float(rng.exponential(1.0 / R_bound))
        t_cur += dt
        if t_cur >= max_time:
            return None
        R_actual = _aggregate_rate(active, sys, t_cur)
        if rng.random() * R_bound <= R_actual:
            return t_cur
    return None


# ===================================================================
# Multi-state simulation core
# ===================================================================

def _simulate_once_multistate(
    sys: _SimSystem,
    n_tips: int,
    max_time: float,
    rng: np.random.Generator,
) -> Optional[SimNode]:
    if n_tips < 2:
        return None

    root = SimNode(time=0.0)
    active: List[_ActiveLineage] = [_ActiveLineage(node=root, state=sys.root_state)]
    t = 0.0

    while 0 < len(active) < n_tips and t < max_time:
        if sys.is_constant:
            R = _aggregate_rate(active, sys, t)
            dt = _next_event_constant(R, rng)
            t_next = t + dt
        else:
            t_next_opt = _next_event_thinning(t, active, sys, max_time, rng)
            if t_next_opt is None:
                return None
            t_next = t_next_opt

        if not np.isfinite(t_next) or t_next <= t:
            return None
        t = t_next
        if t >= max_time:
            break

        # Compute per-lineage rates at event time
        k = len(active)
        rates = []
        for lin in active:
            s = lin.state
            lam_s, mu_s = sys.rate_fns[s](t)
            eta_s = max(0.0, -sys.Q[s, s])
            rates.append((lam_s, mu_s, eta_s, lam_s + mu_s + eta_s))

        R_total = sum(r[3] for r in rates)
        if R_total <= 0 or not np.isfinite(R_total):
            return None

        probs = np.array([r[3] for r in rates])
        probs /= probs.sum()
        chosen_idx = int(rng.choice(k, p=probs))
        lin = active[chosen_idx]
        lam_c, mu_c, eta_c, total_c = rates[chosen_idx]

        u = rng.random() * total_c
        if u < lam_c:
            # --- Birth ---
            active.pop(chosen_idx)
            event_node = SimNode(time=t)
            lin.node.children.append(event_node)
            active.append(_ActiveLineage(node=event_node, state=lin.state))
            active.append(_ActiveLineage(node=event_node, state=lin.state))

        elif u < lam_c + mu_c:
            # --- Death ---
            active.pop(chosen_idx)
            event_node = SimNode(time=t)
            lin.node.children.append(event_node)

        else:
            # --- Regime switch ---
            s_old = lin.state
            switch_targets = []
            switch_rates = []
            for j in range(sys.n_states):
                if j != s_old and sys.Q[s_old, j] > 0:
                    switch_targets.append(j)
                    switch_rates.append(sys.Q[s_old, j])
            if not switch_targets:
                continue
            sr = np.array(switch_rates, dtype=float)
            sr /= sr.sum()
            new_state = int(rng.choice(switch_targets, p=sr))
            lin.state = new_state

    if len(active) != n_tips:
        return None

    for lin in active:
        leaf = SimNode(time=t, is_extant=True)
        lin.node.children.append(leaf)

    pruned = _prune_extant(root)
    return pruned


# ===================================================================
# Tree pruning / output helpers (unchanged logic)
# ===================================================================

def _prune_extant(node: SimNode) -> Optional[SimNode]:
    """仅保留现生叶，并压缩单子节点路径。"""
    if not node.children:
        return node if node.is_extant else None

    kept = []
    for c in node.children:
        k = _prune_extant(c)
        if k is not None:
            kept.append(k)

    if len(kept) == 0:
        return None
    if len(kept) == 1:
        return kept[0]
    return SimNode(time=node.time, children=kept, is_extant=False, taxon=None)


def _assign_taxa_and_collect_names(node: SimNode, names: List[dict], next_id: List[int]) -> None:
    if not node.children:
        tid = next_id[0]
        next_id[0] += 1
        node.taxon = tid
        names.append({"id": tid, "name": f"Taxon_{tid}"})
        return
    for c in node.children:
        _assign_taxa_and_collect_names(c, names, next_id)


def _to_phylojson_node(node: SimNode, parent_time: Optional[float]) -> dict:
    if parent_time is None:
        branch_length = 0.0
    else:
        branch_length = max(0.0, node.time - parent_time)
    if not node.children:
        return {"branch_length": float(branch_length), "children": [], "taxon": int(node.taxon)}
    return {
        "branch_length": float(branch_length),
        "children": [_to_phylojson_node(c, node.time) for c in node.children],
    }


# ===================================================================
# Public API
# ===================================================================

def generate_tree_from_expression(
    expr: Expression,
    n_tips: int = 100,
    seed: int = 42,
    max_time: float = 500.0,
    max_retries: int = 500,
) -> dict:
    """Generate a phylojson tree from a model Expression.

    Supports all models including BAMM.

    Args:
        expr: Model expression (e.g. Expression("CRBD", [0.5, 0.1])
              or Expression("BAMM", [0.3], [crbd_bg, crbd_fg]))
        n_tips: Target number of extant tips.
        seed: Random seed.
        max_time: Maximum forward-time allowed before giving up.
        max_retries: Number of simulation attempts.

    Returns:
        phylojson dict.
    """
    sys = _build_sim_system(expr, max_time=max_time)
    rng = np.random.default_rng(seed)

    for _ in range(max_retries):
        root = _simulate_once_multistate(sys, n_tips, max_time, rng)
        if root is None:
            continue
        taxa: List[dict] = []
        _assign_taxa_and_collect_names(root, taxa, [0])
        root_dict = _to_phylojson_node(root, parent_time=None)
        return {
            "format": "phylojson",
            "taxa": taxa,
            "trees": [{"name": f"{expr.tag}_synthetic", "root": root_dict}],
            "version": "1.0",
            "model_expr": repr(expr),
        }

    raise RuntimeError(
        f"Failed to simulate extant tree after {max_retries} retries. "
        "Try increasing --max_retries or adjusting parameters."
    )


def generate_tree(
    model: str,
    n_tips: int,
    lam: float,
    mu: float,
    z: float,
    epsilon: float,
    seed: int,
    max_time: float,
    max_retries: int,
) -> dict:
    """Backward-compatible API for simple (non-BAMM) models."""
    if model == "CRB":
        expr = Expression("CRB", [lam])
    elif model == "CRBD":
        expr = Expression("CRBD", [lam, mu])
    elif model == "TDB":
        expr = Expression("TDB", [lam, z])
    elif model == "TDBD":
        expr = Expression("TDBD", [lam, z, epsilon])
    else:
        raise ValueError(f"Unsupported model: {model}. Use --expr for BAMM.")
    return generate_tree_from_expression(expr, n_tips, seed, max_time, max_retries)


# ===================================================================
# CLI
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic phylojson tree data.",
        epilog='Example: --expr "(BAMM 0.3 (CRBD 0.5 0.1) (CRBD 1.0 0.3))"',
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--model", choices=["CRB", "CRBD", "TDB", "TDBD"],
                      help="Simple model type (for backward compatibility).")
    grp.add_argument("--expr", type=str,
                      help='Model expression, e.g. "(BAMM 0.3 (CRBD 0.5 0.1) (CRBD 1.0 0.3))"')

    parser.add_argument("--n_tips", type=int, default=100)
    parser.add_argument("--lam", type=float, default=0.0, help="lambda (--model mode only).")
    parser.add_argument("--mu", type=float, default=0.0, help="mu (CRBD --model mode only).")
    parser.add_argument("--z", type=float, default=0.0, help="z (TDB/TDBD --model mode only).")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon (TDBD --model mode only).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_time", type=float, default=500.0)
    parser.add_argument("--max_retries", type=int, default=500)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    if args.expr:
        expr = parse_expression(args.expr)
        data = generate_tree_from_expression(
            expr, args.n_tips, args.seed, args.max_time, args.max_retries
        )
        label = expr.tag
    else:
        if args.lam <= 0:
            raise ValueError("--lam must be > 0")
        data = generate_tree(
            model=args.model,
            n_tips=args.n_tips,
            lam=args.lam,
            mu=args.mu,
            z=args.z,
            epsilon=args.epsilon,
            seed=args.seed,
            max_time=args.max_time,
            max_retries=args.max_retries,
        )
        label = args.model

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path("data") / "generated" / f"{label}_n{args.n_tips}_seed{args.seed}.phylojson"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print(f"Generated: {out_path}")
    print(f"  Expression: {data.get('model_expr', label)}")
    print(f"  Tips: {len(data['taxa'])}")


if __name__ == "__main__":
    main()
