"""诊断: 多棵树的累积似然对比 — PWBD vs 最优单一 CRBD.

关键: SMC-cum 中同一个表达式必须解释所有树。
单棵树上 CRBD 可以"过拟合"赢过 PWBD，
但跨多棵树时，CRBD 必须妥协，PWBD 则持续稳定。
"""
import sys, os, json, tempfile
sys.path.insert(0, "train")
import numpy as np
from algorithm.expression import Expression, parse_expression
from algorithm.generate_data import generate_tree_from_expression
from algorithm.data_utils import load_phylojson, get_tree_data
from algorithm.likelihood import evaluate_likelihood

noise = Expression("Noise", [0.001])
def ll(model, td):
    return evaluate_likelihood(Expression("NoisyPhylo", [], [noise, model]), td)

# 真实数据参考: Accipitridae n=175, T=59.6, mean_rate=0.084
print("真实数据参考: Accipitridae n=175, T=59.6, mean_rate=0.084\n")

# ====================================================================
# 生成多棵同源树
# ====================================================================
CONFIGS = [
    # 相同 net rate, 不同 turnover — CRBD 最难模仿
    ("(PWBD 0.5 (CRB 0.06) (CRBD 0.12 0.06))", 150, "same-net-rate"),
    # 速率对比强
    ("(PWBD 0.5 (CRBD 0.04 0.005) (CRBD 0.12 0.04))", 150, "strong-contrast"),
    # 三段
    ("(PWBD 0.4 (CRB 0.05) (PWBD 0.5 (CRBD 0.10 0.03) (CRBD 0.04 0.01)))", 150, "3-segment"),
]

for expr_str, n_tips, label in CONFIGS:
    print("=" * 65)
    print(f"模型: {expr_str}")
    print(f"      {n_tips} tips × 10 棵树")
    print("=" * 65)
    true_expr = parse_expression(expr_str)
    
    trees = []
    for i in range(10):
        try:
            tree_json = generate_tree_from_expression(
                true_expr, n_tips=n_tips, seed=3000+i, max_time=8000.0, max_retries=3000)
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".phylojson", delete=False)
            json.dump(tree_json, tmp); tmp.close()
            root = load_phylojson(tmp.name)
            td = get_tree_data(root)
            os.unlink(tmp.name)
            trees.append(td)
        except RuntimeError:
            pass
    
    if len(trees) < 5:
        print(f"  *** 只生成了 {len(trees)} 棵树, 跳过 ***\n")
        continue
    
    N = len(trees)
    Ts = [td["tree_height"] for td in trees]
    print(f"  生成 {N} 棵树, T范围=[{min(Ts):.1f}, {max(Ts):.1f}], 中位={np.median(Ts):.1f}")
    
    # 真实模型累积似然
    true_cum = sum(ll(true_expr, td) for td in trees)
    print(f"\n  TRUE PWBD cum-logL = {true_cum:.4f}  (per-tree avg = {true_cum/N:.4f})")
    
    # 网格搜索: 在所有树上联合优化的最优单一 CRBD
    best_crbd_cum = -np.inf
    best_crbd_p = None
    for lam in np.linspace(0.02, 0.20, 20):
        for mu in np.linspace(0.0, 0.10, 12):
            if mu >= lam: continue
            crbd = Expression("CRBD", [lam, mu])
            cum = sum(ll(crbd, td) for td in trees)
            if cum > best_crbd_cum:
                best_crbd_cum = cum
                best_crbd_p = (round(lam,4), round(mu,4))
    print(f"  Best CRBD cum-logL = {best_crbd_cum:.4f}  params={best_crbd_p}")
    
    # 最优 CRB
    best_crb_cum = -np.inf
    for lam in np.linspace(0.02, 0.15, 20):
        crb = Expression("CRB", [lam])
        cum = sum(ll(crb, td) for td in trees)
        if cum > best_crb_cum:
            best_crb_cum = cum
    print(f"  Best CRB  cum-logL = {best_crb_cum:.4f}")
    
    # 最优 TDB
    best_tdb_cum = -np.inf
    for lam in np.linspace(0.02, 0.12, 10):
        for z in np.linspace(-0.05, 0.05, 10):
            tdb = Expression("TDB", [lam, z])
            cum = sum(ll(tdb, td) for td in trees)
            if cum > best_tdb_cum:
                best_tdb_cum = cum
    print(f"  Best TDB  cum-logL = {best_tdb_cum:.4f}")
    
    gap_crbd = true_cum - best_crbd_cum
    gap_crb = true_cum - best_crb_cum
    print(f"\n  >>> PWBD vs CRBD gap = {gap_crbd:+.4f}  {'PWBD WINS!' if gap_crbd > 0 else 'CRBD still wins'}")
    print(f"  >>> PWBD vs CRB  gap = {gap_crb:+.4f}")
    print(f"  >>> PWBD vs TDB  gap = {true_cum - best_tdb_cum:+.4f}")
    print()
