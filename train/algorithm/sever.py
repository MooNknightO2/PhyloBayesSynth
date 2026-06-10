"""剪枝函数 Sever 与转移算子 Generate_New_Expression。

Sever_a[E]  → (N_i, E_sev)
  在解析树位置 a 处剪断，返回该位置的非终结符及带空洞的表达式。

Generate-New-Expression(O, E)
  对表达式 E 做一次 Metropolis-Hastings 提议并决定接受/拒绝。
  包含数据驱动的 PWBD 提议: 从树的分枝时间分布估计断点与分段速率。
"""

import numpy as np
from .expression import Expression
from .prior import expand
from .likelihood import evaluate_likelihood, log_likelihood_crbd
from .grammar import param_log_prior, DEFAULT_CONFIG, sample_parameter


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

def _nesting_depths_at(expression, path):
    """统计从根到 path 位置途经的 BAMM/PWBD 节点数。"""
    bamm_d = 0
    pwbd_d = 0
    node = expression
    for idx in path:
        if node.tag == "BAMM":
            bamm_d += 1
        elif node.tag == "PWBD":
            pwbd_d += 1
        node = node.children[idx]
    return bamm_d, pwbd_d


_TAG_PARAM_NAMES = {
    "Noise": ["sigma"],
    "CRB": ["lambda"],
    "CRBD": ["lambda", "mu"],
    "TDB": ["lambda", "z"],
    "TDBD": ["lambda", "z", "epsilon"],
    "PWBD": ["tau_frac"],
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


def _hill_climb_params(expression, tree_data, config, n_steps):
    """对表达式做若干步贪心参数扰动 (仅改善, 不恶化).

    用于结构提议后的参数预调: 让新结构有机会调优参数,
    避免因随机初始参数而被立即拒绝。
    """
    cfg = config or DEFAULT_CONFIG
    best_expr = expression
    best_ll = _eval_likelihood(best_expr, tree_data)
    if best_ll == -np.inf:
        return best_expr, best_ll
    for _ in range(n_steps):
        result = _propose_local_parameter_move(best_expr, cfg)
        if result is None:
            break
        candidate, _ = result
        ll_c = _eval_likelihood(candidate, tree_data)
        if ll_c > best_ll:
            best_expr = candidate
            best_ll = ll_c
    return best_expr, best_ll


# ===================================================================
# BIC 复杂度惩罚辅助函数
# ===================================================================

def _count_free_params(expression):
    """统计表达式中的自由参数总数."""
    return sum(len(node.params) for _, node in expression.get_all_nodes())


def _get_n_tips(tree_data):
    """从 tree_data (单棵或多棵) 获取总 tips 数."""
    if isinstance(tree_data, list):
        return sum(td["n"] for td in tree_data)
    return tree_data["n"]


# ===================================================================
# 数据驱动 PWBD 提议
# ===================================================================

def _estimate_crbd_for_segment(branching_times_seg, branches_seg, T_seg):
    """对一段区间快速估计 CRBD(λ, μ) 的近似 MLE。

    利用 CRB 的 MLE (λ_hat = (n-1)/S) 作为 λ 的起点,
    然后用几个 μ 候选值取似然最高的。
    """
    n_events = len(branching_times_seg)
    total_length = sum(bl for bl, _, _ in branches_seg) if branches_seg else 0.0
    if n_events < 1 or total_length < 1e-10:
        return 0.1, 0.01
    lam_hat = n_events / total_length
    if lam_hat < 1e-6:
        lam_hat = 0.1
    best_mu = 0.0
    return max(lam_hat, 0.01), max(best_mu, 0.0)


def _propose_pwbd_data_driven(tree_data, config=None):
    """从树数据估计一个 PWBD 表达式。

    算法:
    1. 在若干候选断点 τ_frac 处将分枝时间分成两组
    2. 对每组快速估计 CRBD 参数
    3. 计算该 PWBD 的似然, 取最优断点
    4. 对参数加随机扰动 (增加探索性)

    Returns:
        Expression (PWBD with estimated params) 或 None
    """
    cfg = config or DEFAULT_CONFIG

    if isinstance(tree_data, list):
        td = tree_data[0]
    else:
        td = tree_data

    T = td["tree_height"]
    bts = td["branching_times"]
    branches = td["branches"]
    n = td["n"]
    S = td["total_length"]

    if T < 1e-8 or n < 5:
        return None

    lam_global = max((n - 1) / S, 0.01)

    best_ll = -np.inf
    best_expr = None

    for tf_candidate in np.linspace(0.15, 0.85, 8):
        tau_b = tf_candidate * T

        bts_recent = [t for t in bts if t <= tau_b]
        bts_ancient = [t for t in bts if t > tau_b]
        br_recent = [(bl, tc, tp) for bl, tc, tp in branches if tp <= tau_b + 1e-10]
        br_ancient = [(bl, tc, tp) for bl, tc, tp in branches if tc >= tau_b - 1e-10]

        n_recent = len(bts_recent)
        n_ancient = len(bts_ancient)
        S_recent = sum(bl for bl, _, _ in br_recent) if br_recent else 0.0
        S_ancient = sum(bl for bl, _, _ in br_ancient) if br_ancient else 0.0

        if n_recent < 2 or n_ancient < 2:
            continue

        lam1 = max(n_recent / max(S_recent, 1e-8), 0.01)
        lam2 = max(n_ancient / max(S_ancient, 1e-8), 0.01)

        for mu1_frac in [0.0, 0.1, 0.3]:
            for mu2_frac in [0.0, 0.1, 0.3, 0.5]:
                mu1 = mu1_frac * lam1
                mu2 = mu2_frac * lam2
                child1 = Expression("CRBD", [lam1, mu1], nonterminal="N3")
                child2 = Expression("CRBD", [lam2, mu2], nonterminal="N3")
                pwbd = Expression("PWBD", [tf_candidate], [child1, child2],
                                  nonterminal="N3")
                noise = Expression("Noise", [0.01], nonterminal="N2")
                full = Expression("NoisyPhylo", [], [noise, pwbd], nonterminal="N1")
                ll = _eval_likelihood(full, tree_data)
                if ll > best_ll:
                    best_ll = ll
                    best_expr = full

    if best_expr is None:
        return None

    jittered = best_expr.copy()
    pwbd_node = jittered.children[1]
    pwbd_node.params[0] = np.clip(
        pwbd_node.params[0] + np.random.normal(0, 0.05), 0.05, 0.95
    )
    for child in pwbd_node.children:
        for i in range(len(child.params)):
            child.params[i] = max(
                child.params[i] * np.exp(np.random.normal(0, 0.15)),
                1e-5
            )
        if child.tag == "CRBD" and child.params[1] >= child.params[0]:
            child.params[1] = child.params[0] * 0.8

    noise_node = jittered.children[0]
    noise_node.params[0] = max(sample_parameter("sigma", cfg), 1e-6)

    return jittered


def generate_new_expression(expression, tree_data, config=None, log_L_cached=None):
    """对表达式执行一步 MH 提议。

    结构提议后会做若干步贪心参数调优 (hill-climb), 让复杂模型
    有机会在被 MH 判决前找到更好的参数。这是单向提议 (从先验
    采样 → 调优), MH 比率中的 proposal 修正通过 size_correction
    近似处理。

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
    struct_tune_steps = int(cfg.get("struct_tune_steps", 8))

    all_nodes = expression.get_all_nodes()
    n_nodes = len(all_nodes)

    pwbd_smart_prob = float(cfg.get("pwbd_smart_prob", 0.15))

    u = np.random.random()
    do_local = u < local_move_prob
    do_pwbd_smart = (not do_local) and (u < local_move_prob + pwbd_smart_prob)
    log_prior_ratio = 0.0

    if do_local:
        result = _propose_local_parameter_move(expression, cfg)
        if result is None:
            do_local = False
            do_pwbd_smart = False
        else:
            new_expr, log_prior_ratio = result

    if do_pwbd_smart:
        smart = _propose_pwbd_data_driven(tree_data, config)
        if smart is not None:
            new_expr = smart
        else:
            do_pwbd_smart = False

    if not do_local and not do_pwbd_smart:
        idx = np.random.randint(n_nodes)
        path, _ = all_nodes[idx]
        nonterminal, severed = sever(expression, path)
        bamm_d, pwbd_d = _nesting_depths_at(expression, path)
        sub_expr = expand(nonterminal, config,
                          depth=bamm_d + pwbd_d,
                          bamm_depth=bamm_d,
                          pwbd_depth=pwbd_d)
        new_expr = fill_hole(severed, path, sub_expr)
        if struct_tune_steps > 0:
            new_expr, _ = _hill_climb_params(
                new_expr, tree_data, cfg, struct_tune_steps
            )

    log_L = log_L_cached if log_L_cached is not None else _eval_likelihood(expression, tree_data)
    log_L_prime = _eval_likelihood(new_expr, tree_data)
    n_nodes_new = new_expr.count_nodes()

    if log_L_prime == -np.inf:
        return expression, log_L
    if log_L == -np.inf:
        return new_expr, log_L_prime

    size_correction = 0.0 if do_local else (np.log(n_nodes) - np.log(n_nodes_new))
    prior_correction = log_prior_ratio if do_local else 0.0

    bic_weight = float(cfg.get("bic_penalty_weight", 0.5))
    if bic_weight > 0 and not do_local:
        n_data = _get_n_tips(tree_data)
        if n_data > 1:
            k_old = _count_free_params(expression)
            k_new = _count_free_params(new_expr)
            complexity_penalty = -bic_weight * (k_new - k_old) * np.log(n_data)
        else:
            complexity_penalty = 0.0
    else:
        complexity_penalty = 0.0

    log_ratio = (log_L_prime - log_L + size_correction
                 + prior_correction + complexity_penalty)
    p_accept = 1.0 if log_ratio > 0 else np.exp(log_ratio)
    if np.random.random() < p_accept:
        return new_expr, log_L_prime
    return expression, log_L

