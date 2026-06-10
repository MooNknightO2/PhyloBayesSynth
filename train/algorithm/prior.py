"""先验采样: Expand 与 Generate_Expression_From_Prior。

Expand[(t_ik, θ_1…θ_{h_ik}, E_1…E_{n_ik})](N_i)
  := p_ik · γ_ik(θ_1,…,θ_{h_ik}) · Π Expand[E_j](Ñ_{ik}^j)

Prior[E] := Expand[E](N^start)   其中 N^start = N1
"""

import numpy as np
from .expression import Expression
from .grammar import GRAMMAR, DEFAULT_CONFIG, sample_parameter


def expand(nonterminal, config=None, depth=0, bamm_depth=0, pwbd_depth=0):
    """递归展开非终结符，从先验中采样得到表达式子树。

    Args:
        nonterminal: 要展开的非终结符 ("N1", "N2", "N3")
        config: 先验配置字典
        depth: 当前递归深度
        bamm_depth: 当前路径上 BAMM 嵌套层数
        pwbd_depth: 当前路径上 PWBD 嵌套层数

    Returns:
        Expression 对象
    """
    cfg = config or DEFAULT_CONFIG
    rules = GRAMMAR[nonterminal]
    if nonterminal == "N3":
        probs = cfg["N3_probs"]
        tags = [r[0] for r in rules]
        p = np.array([probs[t] for t in tags], dtype=float)
        max_bamm = cfg.get("max_bamm_nesting", 3)
        max_pwbd = cfg.get("max_pwbd_nesting", 3)
        if bamm_depth >= max_bamm:
            p[tags.index("BAMM")] = 0.0
        if pwbd_depth >= max_pwbd:
            p[tags.index("PWBD")] = 0.0
        if p.sum() <= 0:
            for t in ("CRB", "CRBD"):
                p[tags.index(t)] = 1.0
        p /= p.sum()
        choice = np.random.choice(len(rules), p=p)
    else:
        choice = 0
    tag, param_names, child_nonterminals = rules[choice]
    params = [sample_parameter(name, cfg) for name in param_names]
    child_bamm = bamm_depth + (1 if tag == "BAMM" else 0)
    child_pwbd = pwbd_depth + (1 if tag == "PWBD" else 0)
    children = [
        expand(nt, cfg, depth + 1,
               bamm_depth=child_bamm, pwbd_depth=child_pwbd)
        for nt in child_nonterminals
    ]
    return Expression(tag, params, children, nonterminal)

def generate_expression_from_prior(config=None):
    """
    Prior[E] := Expand[E](N1)
    """
    return expand("N1", config)
