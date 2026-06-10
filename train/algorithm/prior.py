"""先验采样: Expand 与 Generate_Expression_From_Prior。

Expand[(t_ik, θ_1…θ_{h_ik}, E_1…E_{n_ik})](N_i)
  := p_ik · γ_ik(θ_1,…,θ_{h_ik}) · Π Expand[E_j](Ñ_{ik}^j)

Prior[E] := Expand[E](N^start)   其中 N^start = N1
"""

import numpy as np
from .expression import Expression
from .grammar import GRAMMAR, DEFAULT_CONFIG, sample_parameter


def expand(nonterminal, config=None, depth=0, bamm_depth=0):
    """递归展开非终结符，从先验中采样得到表达式子树。

    Args:
        nonterminal: 要展开的非终结符 ("N1", "N2", "N3")
        config: 先验配置字典
        depth: 当前递归深度
        bamm_depth: 当前路径上 BAMM 嵌套层数

    Returns:
        Expression 对象
    """
    cfg = config or DEFAULT_CONFIG
    rules = GRAMMAR[nonterminal]
    if nonterminal == "N3":
        probs = cfg["N3_probs"]
        tags = [r[0] for r in rules]
        p = np.array([probs[t] for t in tags], dtype=float)
        max_bamm_nesting = cfg.get("max_bamm_nesting", cfg.get("max_depth", 3))
        if bamm_depth >= max_bamm_nesting:
            p[tags.index("BAMM")] = 0.0
        p /= p.sum()
        choice = np.random.choice(len(rules), p=p)
    else:
        choice = 0  # N1, N2 各只有一条产生式
    tag, param_names, child_nonterminals = rules[choice]
    params = [sample_parameter(name, cfg) for name in param_names]
    child_bamm_depth = bamm_depth + (1 if tag == "BAMM" else 0)
    children = [
        expand(nt, cfg, depth + 1, bamm_depth=child_bamm_depth)
        for nt in child_nonterminals
    ]
    return Expression(tag, params, children, nonterminal)

def generate_expression_from_prior(config=None):
    """
    Prior[E] := Expand[E](N1)
    """
    return expand("N1", config)

