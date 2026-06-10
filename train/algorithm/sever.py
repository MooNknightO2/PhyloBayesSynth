"""剪枝函数 Sever 与转移算子 Generate_New_Expression。

Sever_a[E]  → (N_i, E_sev)
  在解析树位置 a 处剪断，返回该位置的非终结符及带空洞的表达式。

Generate-New-Expression(O, E)
  对表达式 E 做一次 Metropolis-Hastings 提议并决定接受/拒绝。
"""

import numpy as np
from .prior import expand
from .likelihood import evaluate_likelihood
from .grammar import param_log_prior, DEFAULT_CONFIG


# ===================================================================
# Sever
# ===================================================================

def sever(expression, path):
    """在解析树的 path 位置剪枝。

    Args:
        expression: 原始 Expression
        path: 要剪除节点的路径 (子索引列表)

    Returns:
        (nonterminal, severed_expr)
        - nonterminal: 被剪除节点对应的非终结符
        - severed_expr: 带空洞的 Expression (空洞用 None 标记在该位置)
    """
    target = expression.get_node_at(path)
    nonterminal = target.nonterminal

    # 构造带空洞的表达式
    severed = expression.set_node_at(path, None)
    return nonterminal, severed


def fill_hole(severed_expr, path, sub_expr):
    """将子表达式填入空洞。
    """
    if not path:
        # 根节点被剪枝，直接返回新子表达式
        return sub_expr
    return severed_expr.set_node_at(path, sub_expr)

def _bamm_depth_at(expression, path):
    """统计从根到 path 位置途经的 BAMM 节点数。"""
    depth = 0
    node = expression
    for idx in path:
        if node.tag == "BAMM":
            depth += 1
        node = node.children[idx]
    return depth


_TAG_PARAM_NAMES = {
    "Noise": ["sigma"],
    "CRB": ["lambda"],
    "CRBD": ["lambda", "mu"],
    "TDB": ["lambda", "z"],
    "TDBD": ["lambda", "z", "epsilon"],
    "PCBD": ["lambda", "mu", "lambda", "mu", "tau_frac"],
    "BAMM": ["eta"],
}


def _iter_parameter_sites(expression):
    """枚举表达式中所有可扰动参数位置: (path, param_idx, param_name)。"""
    sites = []
    for path, node in expression.get_all_nodes():
        if not node.params:
            continue
        names = _TAG_PARAM_NAMES.get(node.tag, [])
        for i in range(len(node.params)):
            pname = names[i] if i < len(names) else "default"
            sites.append((path, i, pname))
    return sites


def _propose_local_parameter_move(expression, config=None):
    """对单个参数做 Gaussian 随机游走，小步提议。

    Returns:
        (new_expr, log_prior_ratio) 或 None（无可扰动参数时）。
    """
    cfg = config or DEFAULT_CONFIG
    scales = cfg.get("local_proposal_scales", {})
    default_scale = float(scales.get("default", 0.05))
    sites = _iter_parameter_sites(expression)
    if not sites:
        return None
    path, param_idx, pname = sites[np.random.randint(len(sites))]
    step = float(scales.get(pname, default_scale))
    if step <= 0:
        return None

    new_expr = expression.copy()
    target = new_expr.get_node_at(path)
    old_value = float(target.params[param_idx])
    new_value = float(old_value + np.random.normal(0.0, step))
    target.params[param_idx] = new_value

    log_prior_old = param_log_prior(pname, old_value, cfg)
    log_prior_new = param_log_prior(pname, new_value, cfg)
    log_prior_ratio = log_prior_new - log_prior_old

    return new_expr, log_prior_ratio


# ===================================================================
# Generate-New-Expression (转移算子 / MH 提议)
# ===================================================================

def _eval_likelihood(expr, tree_data):
    """单树或多树累积似然。tree_data 可以是 dict 或 list[dict]。"""
    if isinstance(tree_data, list):
        total = 0.0
        for td in tree_data:
            ll = evaluate_likelihood(expr, td)
            if ll == -np.inf:
                return -np.inf
            total += ll
        return total
    return evaluate_likelihood(expr, tree_data)


def generate_new_expression(expression, tree_data, config=None, log_L_cached=None):
    """对表达式执行一步 MH 提议。

    Args:
        expression: 当前表达式 E
        tree_data: 树数据 (单棵 dict 或多棵 list[dict])
        config: 先验配置
        log_L_cached: 当前表达式的已知对数似然 (避免重复计算)。
                      传 None 则内部重新计算。
    Returns:
        (new_expr, new_log_L):
          new_expr — 接受后的 Expression (可能是 E 或 E')
          new_log_L — 对应的对数似然
    """
    cfg = config or DEFAULT_CONFIG
    local_move_prob = float(cfg.get("local_move_prob", 0.5))
    local_move_prob = min(max(local_move_prob, 0.0), 1.0)

    all_nodes = expression.get_all_nodes()
    n_nodes = len(all_nodes)

    do_local = np.random.random() < local_move_prob
    log_prior_ratio = 0.0
    if do_local:
        result = _propose_local_parameter_move(expression, cfg)
        if result is None:
            do_local = False
        else:
            new_expr, log_prior_ratio = result

    if not do_local:
        idx = np.random.randint(n_nodes)
        path, _ = all_nodes[idx]
        nonterminal, severed = sever(expression, path)
        depth = _bamm_depth_at(expression, path)
        sub_expr = expand(nonterminal, config, depth=depth, bamm_depth=depth)
        new_expr = fill_hole(severed, path, sub_expr)

    log_L = log_L_cached if log_L_cached is not None else _eval_likelihood(expression, tree_data)
    log_L_prime = _eval_likelihood(new_expr, tree_data)
    n_nodes_new = new_expr.count_nodes()

    if log_L_prime == -np.inf:
        return expression, log_L
    if log_L == -np.inf:
        return new_expr, log_L_prime

    size_correction = 0.0 if do_local else (np.log(n_nodes) - np.log(n_nodes_new))
    prior_correction = log_prior_ratio if do_local else 0.0
    log_ratio = log_L_prime - log_L + size_correction + prior_correction
    p_accept = 1.0 if log_ratio > 0 else np.exp(log_ratio)
    if np.random.random() < p_accept:
        return new_expr, log_L_prime
    return expression, log_L

