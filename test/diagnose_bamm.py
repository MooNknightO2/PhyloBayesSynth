"""诊断 BAMM 树的 regime 构成。"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.algorithm.expression import parse_expression
from train.algorithm.generate_data import (
    _build_sim_system, _ActiveLineage, SimNode, _safe_exp
)

def simulate_with_tracking(expr_str, n_tips, seed):
    """模拟 BAMM 树并追踪各 regime 的贡献。"""
    expr = parse_expression(expr_str)
    sys_ = _build_sim_system(expr)
    rng = np.random.default_rng(seed)

    for attempt in range(5000):
        root = SimNode(time=0.0)
        active = [_ActiveLineage(node=root, state=sys_.root_state)]
        t = 0.0
        birth_in_state = [0] * sys_.n_states
        death_in_state = [0] * sys_.n_states
        switch_count = 0
        time_in_state = [0.0] * sys_.n_states

        while 0 < len(active) < n_tips and t < 500.0:
            state_counts = [0] * sys_.n_states
            for lin in active:
                state_counts[lin.state] += 1

            rates = []
            for lin in active:
                s = lin.state
                lam_s, mu_s = sys_.rate_fns[s](t)
                eta_s = max(0, -sys_.Q[s, s])
                rates.append((lam_s, mu_s, eta_s, lam_s + mu_s + eta_s))
            R = sum(r[3] for r in rates)
            if R <= 0:
                break
            dt = float(rng.exponential(1.0 / R))
            t_old = t
            t += dt
            if t >= 500.0:
                break

            for s_idx in range(sys_.n_states):
                time_in_state[s_idx] += state_counts[s_idx] * (t - t_old)

            probs = np.array([r[3] for r in rates])
            probs /= probs.sum()
            chosen = int(rng.choice(len(active), p=probs))
            lin = active[chosen]
            lam_c, mu_c, eta_c, total_c = rates[chosen]
            u = rng.random() * total_c

            if u < lam_c:
                active.pop(chosen)
                ev = SimNode(time=t)
                lin.node.children.append(ev)
                active.append(_ActiveLineage(node=ev, state=lin.state))
                active.append(_ActiveLineage(node=ev, state=lin.state))
                birth_in_state[lin.state] += 1
            elif u < lam_c + mu_c:
                active.pop(chosen)
                ev = SimNode(time=t)
                lin.node.children.append(ev)
                death_in_state[lin.state] += 1
            else:
                targets = []
                sw_rates = []
                for j in range(sys_.n_states):
                    if j != lin.state and sys_.Q[lin.state, j] > 0:
                        targets.append(j)
                        sw_rates.append(sys_.Q[lin.state, j])
                if targets:
                    sr = np.array(sw_rates); sr /= sr.sum()
                    new_s = int(rng.choice(targets, p=sr))
                    lin.state = new_s
                    switch_count += 1

        if len(active) == n_tips:
            final_state_counts = [0] * sys_.n_states
            for lin in active:
                final_state_counts[lin.state] += 1
            return {
                "births": birth_in_state,
                "deaths": death_in_state,
                "switches": switch_count,
                "lineage_time": time_in_state,
                "final_tips_per_state": final_state_counts,
                "T": t,
                "attempt": attempt,
            }
    return None

# ============================================================
print("=" * 65)
print("实验 1: 原始参数 (η=0.03, bg 慢, fg 快)")
print("=" * 65)
r1 = simulate_with_tracking("(BAMM 0.03 (CRBD 0.15 0.03) (CRBD 0.6 0.05))", 80, 314)
if r1:
    print(f"  树高 T = {r1['T']:.2f}")
    print(f"  regime 切换次数: {r1['switches']}")
    print(f"  各 state 出生事件: {r1['births']}")
    print(f"  各 state 死亡事件: {r1['deaths']}")
    print(f"  各 state 谱系累积时间: [{r1['lineage_time'][0]:.1f}, {r1['lineage_time'][1]:.1f}]")
    pct = [100*t/sum(r1['lineage_time']) for t in r1['lineage_time']]
    print(f"  时间占比: [{pct[0]:.1f}%, {pct[1]:.1f}%]")
    print(f"  最终 tips 所在 state: {r1['final_tips_per_state']}")

# ============================================================
print("\n" + "=" * 65)
print("实验 2: 修正参数 (η=0.03, bg/fg 速率相近但灭绝率差异大)")
print("  bg = CRBD(0.5, 0.35) 高灭绝, 净速率=0.15")
print("  fg = CRBD(0.5, 0.05) 低灭绝, 净速率=0.45")
print("=" * 65)
r2 = simulate_with_tracking("(BAMM 0.03 (CRBD 0.5 0.35) (CRBD 0.5 0.05))", 80, 314)
if r2:
    print(f"  树高 T = {r2['T']:.2f}")
    print(f"  regime 切换次数: {r2['switches']}")
    print(f"  各 state 出生事件: {r2['births']}")
    print(f"  各 state 死亡事件: {r2['deaths']}")
    print(f"  各 state 谱系累积时间: [{r2['lineage_time'][0]:.1f}, {r2['lineage_time'][1]:.1f}]")
    pct = [100*t/sum(r2['lineage_time']) for t in r2['lineage_time']]
    print(f"  时间占比: [{pct[0]:.1f}%, {pct[1]:.1f}%]")
    print(f"  最终 tips 所在 state: {r2['final_tips_per_state']}")

# ============================================================
print("\n" + "=" * 65)
print("实验 3: 更大 η 但 λ 相同 (异质灭绝)")
print("  bg = CRBD(0.3, 0.25) 高灭绝, 净速率=0.05")
print("  fg = CRBD(0.3, 0.02) 低灭绝, 净速率=0.28")
print("  η = 0.05")
print("=" * 65)
r3 = simulate_with_tracking("(BAMM 0.05 (CRBD 0.3 0.25) (CRBD 0.3 0.02))", 80, 314)
if r3:
    print(f"  树高 T = {r3['T']:.2f}")
    print(f"  regime 切换次数: {r3['switches']}")
    print(f"  各 state 出生事件: {r3['births']}")
    print(f"  各 state 死亡事件: {r3['deaths']}")
    print(f"  各 state 谱系累积时间: [{r3['lineage_time'][0]:.1f}, {r3['lineage_time'][1]:.1f}]")
    pct = [100*t/sum(r3['lineage_time']) for t in r3['lineage_time']]
    print(f"  时间占比: [{pct[0]:.1f}%, {pct[1]:.1f}%]")
    print(f"  最终 tips 所在 state: {r3['final_tips_per_state']}")

# === 对三组参数计算似然对比 ===
print("\n" + "=" * 65)
print("似然对比")
print("=" * 65)
from train.algorithm.data_utils import load_phylojson, get_tree_data
from train.algorithm.generate_data import generate_tree_from_expression
from train.algorithm.data_utils import _parse_node, _compute_ages
from train.algorithm.likelihood import log_likelihood_bamm, log_likelihood_crb, log_likelihood_crbd

configs = [
    ("(BAMM 0.03 (CRBD 0.15 0.03) (CRBD 0.6 0.05))", 80, 314, "原始(慢bg/快fg)"),
    ("(BAMM 0.03 (CRBD 0.5 0.35) (CRBD 0.5 0.05))", 80, 314, "修正(同λ异μ)"),
    ("(BAMM 0.05 (CRBD 0.3 0.25) (CRBD 0.3 0.02))", 80, 314, "异质灭绝"),
]

for expr_str, ntips, seed, label in configs:
    expr = parse_expression(expr_str)
    data = generate_tree_from_expression(expr, ntips, seed, max_retries=5000)
    root = _parse_node(data["trees"][0]["root"])
    _compute_ages(root)
    td = get_tree_data(root)
    ll_bamm = log_likelihood_bamm(expr, td)
    lam_mle = (td["n"]-1) / td["total_length"]
    ll_crb = log_likelihood_crb(lam_mle, td)
    delta = ll_bamm - ll_crb
    print(f"\n{label}: {expr_str}")
    print(f"  n={td['n']}, T={td['tree_height']:.2f}, S={td['total_length']:.1f}")
    print(f"  BAMM true:  {ll_bamm:.2f}")
    print(f"  CRB MLE:    {ll_crb:.2f}  (λ={lam_mle:.4f})")
    print(f"  Δ = {delta:+.2f}  {'*** BAMM wins ***' if delta > 0 else '(CRB wins)'}")
