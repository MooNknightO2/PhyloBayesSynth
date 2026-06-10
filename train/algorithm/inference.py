"""推断算法: MCMC 与 SMC。"""

import numpy as np
from .prior import generate_expression_from_prior
from .likelihood import evaluate_likelihood
from .sever import generate_new_expression


# ===================================================================
# MCMC: Bayesian-Synthesis-MCMC
# ===================================================================

def bayesian_synthesis_mcmc(tree_data, n_iter, config=None, verbose=True):
    """基于 MCMC 的贝叶斯程序合成。

    似然值在 MH 步骤间缓存传递, 每步仅计算一次似然 (提议表达式)。

    Args:
        tree_data: 观测数据 (get_tree_data 返回的 dict)
        n_iter: 迭代次数
        config: 先验配置
        verbose: 是否打印进度
    Returns:
        dict with keys:
          samples:       list of Expression (长度 n_iter)
          log_likelihoods: 对应的对数似然 (长度 n_iter)
          best_expr:     全局最优表达式
          best_log_L:    全局最优对数似然
          best_per_model: {tag: (expr, log_L)} 各模型类别最优
          model_counts:  {tag: count} 后验频率
    """
    max_init_tries = 10000
    for attempt in range(max_init_tries):
        E = generate_expression_from_prior(config)
        log_L = evaluate_likelihood(E, tree_data)
        if log_L > -np.inf:
            break
    else:
        raise RuntimeError(f"无法在 {max_init_tries} 次尝试内找到似然非零的初始表达式")

    best_expr = E.copy()
    best_log_L = log_L
    best_per_model = {}
    model_counts = {}
    if verbose:
        print(f"初始表达式: {E}")
        print(f"初始对数似然: {log_L:.4f}")

    samples = []
    log_likelihoods = []
    log_interval = max(1, n_iter // 20)

    for i in range(1, n_iter + 1):
        E, log_L = generate_new_expression(E, tree_data, config, log_L_cached=log_L)
        samples.append(E)
        log_likelihoods.append(log_L)

        tag = E.children[1].tag
        model_counts[tag] = model_counts.get(tag, 0) + 1

        if log_L > best_log_L:
            best_log_L = log_L
            best_expr = E.copy()
        if tag not in best_per_model or log_L > best_per_model[tag][1]:
            best_per_model[tag] = (E.copy(), log_L)

        if verbose and (i % log_interval == 0 or i == n_iter):
            print(f"  迭代 {i}/{n_iter}, log-likelihood = {log_L:.4f}, "
                  f"best = {best_log_L:.4f}, E = {E}")

    return {
        "samples": samples,
        "log_likelihoods": log_likelihoods,
        "best_expr": best_expr,
        "best_log_L": best_log_L,
        "best_per_model": best_per_model,
        "model_counts": model_counts,
    }

# ===================================================================
# SMC: Bayesian-Synthesis-SMC
# ===================================================================

def _effective_sample_size(log_weights):
    """计算有效样本量 ESS。"""
    max_w = np.max(log_weights)
    # 全部为 -inf 时，表示该轮所有粒子都不可行；返回 0 触发重采样分支。
    if max_w == -np.inf:
        return 0.0
    w = np.exp(log_weights - max_w)
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        return 0.0
    w /= s
    denom = np.sum(w ** 2)
    if denom <= 0 or not np.isfinite(denom):
        return 0.0
    return 1.0 / denom

def _systematic_resample(weights, M):
    """系统重采样, 返回祖先索引数组。"""
    w = np.asarray(weights, dtype=float)
    w /= w.sum()
    positions = (np.arange(M) + np.random.random()) / M
    cumsum = np.cumsum(w)
    indices = np.searchsorted(cumsum, positions)
    return indices

def bayesian_synthesis_smc(observations, n_move, M, ess_min=None,
                           config=None, verbose=True):
    """基于 SMC 的贝叶斯程序合成（顺序单观测切换方案）。

    注意：本算法使用的权重更新方式为顺序切换——第 j 步的增量权重为
        w_j(E) ∝ p(O_j | E) / p(O_{j-1} | E)
    即从以 p(E | O_{j-1}) 为目标的粒子群，过渡到以 p(E | O_j) 为目标。
    这与标准累积后验 SMC（目标 p(E | O_1, ..., O_j)，增量权重仅为
    p(O_j | E)）不同。当前方案适用于"多棵不同的树逐个分析"的场景，
    每步只聚焦于当前观测，前序观测的信息通过粒子群隐式传递。

    Args:
        observations: list of tree_data dicts, 长度 J
        n_move: 每轮的 rejuvenation 迭代次数
        M: 粒子数
        ess_min: ESS 阈值 (默认 M/2)
        config: 先验配置
        verbose: 是否打印进度
    Returns:
        (particles, weights, ancestors)
        - particles[j][ℓ]: 第 j 轮第 ℓ 个粒子的 Expression
        - weights[j][ℓ]:   对应的归一化权重
        - ancestors[j][ℓ]: 第 j 轮第 ℓ 个粒子的祖先索引
    """
    J = len(observations)
    if ess_min is None:
        ess_min = M / 2.0
    particles = [[None] * M]        # particles[0][ℓ]
    log_w = np.zeros(M)             # log weights w_0
    log_w_hat = np.zeros(M)         # log ŵ_{j-1}
    for ell in range(M):
        particles[0][ell] = generate_expression_from_prior(config)
    all_weights = [np.ones(M) / M]
    all_ancestors = []
    # prev_log_L[ℓ] = log p(O_{j-1} | E_ℓ)，用于顺序切换的增量权重分母。
    # 初始 O_0 为空观测 (log L = 0)，因此第 1 步退化为标准增量 p(O_1 | E)。
    prev_log_L = np.zeros(M)
    for j in range(1, J + 1):
        obs_j = observations[j - 1]
        curr_log_L = np.array([
            evaluate_likelihood(particles[j - 1][ell], obs_j)
            for ell in range(M)
        ])
        # 顺序切换增量: p(O_j | E) / p(O_{j-1} | E)
        log_w = log_w_hat + curr_log_L - prev_log_L
        # 归一化权重
        max_lw = np.max(log_w)
        if max_lw == -np.inf:
            norm_w = np.ones(M) / M
        else:
            w = np.exp(log_w - max_lw)
            norm_w = w / w.sum()
        # resample (adaptive)
        ess = _effective_sample_size(log_w)
        ancestors_j = np.arange(M)
        if ess < ess_min and j < J:
            ancestors_j = _systematic_resample(norm_w, M)
            log_w_hat = np.zeros(M)
            if verbose:
                print(f"  SMC 轮 {j}: ESS={ess:.1f} < {ess_min:.1f}, 重采样")
        else:
            log_w_hat = log_w.copy()
        # rejuvenate
        new_particles = [None] * M
        new_log_L = np.empty(M)
        for ell in range(M):
            E = particles[j - 1][ancestors_j[ell]].copy()
            ll = curr_log_L[ancestors_j[ell]]
            for _ in range(n_move):
                E, ll = generate_new_expression(E, obs_j, config, log_L_cached=ll)
            new_particles[ell] = E
            new_log_L[ell] = ll
        particles.append(new_particles)
        all_weights.append(norm_w.copy())
        all_ancestors.append(ancestors_j.copy())
        prev_log_L = new_log_L
        if verbose:
            best = np.argmax(prev_log_L)
            print(f"  SMC 轮 {j}/{J}: best log-L = {prev_log_L[best]:.4f}, "
                  f"E = {new_particles[best]}")
    return particles, all_weights, all_ancestors


# ===================================================================
# Cumulative-posterior SMC: 多棵同源树联合推断
# ===================================================================

def bayesian_synthesis_smc_cumulative(
    observations, n_move, M, ess_min=None, config=None, verbose=True
):
    """累积后验 SMC: 多棵同源树联合推断。

    目标序列:
        π_j(E) = p(E) · Π_{k=1}^{j} p(O_k | E)   (j = 1 … J)
    增量权重:
        w_j(E) = p(O_j | E)
    Rejuvenation:
        MH 步骤以 π_j 为目标, 使用观测 O_1…O_j 的累积似然。

    相比单观测切换 SMC, 累积方案会随着更多树的加入不断收紧后验,
    而非每步仅关注当前观测。这在"多棵树来自同一模型"的场景下能
    更有效地累积信号, 恢复出更准确的表达式。

    Args:
        observations: list of tree_data dicts (J 棵同源树)
        n_move:  每轮每粒子的 MH rejuvenation 步数
        M:       粒子数
        ess_min: ESS 阈值 (默认 M/2)
        config:  先验配置 dict
        verbose: 是否打印进度

    Returns:
        dict with keys:
          particles:    最终粒子列表 (length M)
          cum_log_L:    各粒子的累积对数似然 (shape M)
          best_expr:    全局最优表达式
          best_log_L:   全局最优累积对数似然
          best_per_model: {tag: (expr, cum_log_L)}
          model_counts:   {tag: count}
          history:      每轮的快照 list[dict]
    """
    J = len(observations)
    if J == 0:
        raise ValueError("observations 不能为空")
    if ess_min is None:
        ess_min = M / 2.0

    # ---- 初始化粒子 (从先验采样) ----
    particles = [generate_expression_from_prior(config) for _ in range(M)]
    cum_log_L = np.zeros(M)
    log_w_hat = np.zeros(M)

    history = []

    for j in range(J):
        obs_j = observations[j]
        obs_so_far = observations[: j + 1]

        # ---- 增量权重: p(O_j | E_ell) ----
        incr = np.array([
            evaluate_likelihood(particles[ell], obs_j) for ell in range(M)
        ])
        cum_log_L += incr
        log_w = log_w_hat + incr

        # ---- 归一化 & ESS ----
        max_lw = np.max(log_w)
        if max_lw == -np.inf:
            norm_w = np.ones(M) / M
        else:
            w = np.exp(log_w - max_lw)
            norm_w = w / w.sum()

        ess = _effective_sample_size(log_w)

        # ---- 自适应重采样 ----
        if ess < ess_min:
            ancestors = _systematic_resample(norm_w, M)
            particles = [particles[a].copy() for a in ancestors]
            cum_log_L = cum_log_L[ancestors].copy()
            log_w_hat = np.zeros(M)
            if verbose:
                print(f"  SMC 轮 {j+1}/{J}: ESS={ess:.1f} < {ess_min:.1f}, 重采样")
        else:
            log_w_hat = log_w.copy()

        # ---- Rejuvenation: MH 以 π_j = p(E)·Π_{k≤j} p(O_k|E) 为目标 ----
        for ell in range(M):
            E = particles[ell]
            ll = cum_log_L[ell]
            for _ in range(n_move):
                E, ll = generate_new_expression(
                    E, obs_so_far, config, log_L_cached=ll
                )
            particles[ell] = E
            cum_log_L[ell] = ll

        # ---- 记录本轮 ----
        best_j = int(np.argmax(cum_log_L))
        if verbose:
            print(
                f"  SMC 轮 {j+1}/{J}: best cum-logL = {cum_log_L[best_j]:.4f}, "
                f"E = {particles[best_j]}"
            )
        history.append({
            "step": j + 1,
            "ess": ess,
            "best_cum_log_L": cum_log_L[best_j],
            "best_expr": particles[best_j].copy(),
        })

    # ---- 汇总结果 ----
    best_idx = int(np.argmax(cum_log_L))
    best_per_model = {}
    model_counts = {}
    for ell in range(M):
        tag = particles[ell].tag
        if tag == "NoisyPhylo":
            tag = particles[ell].children[1].tag if len(particles[ell].children) > 1 else tag
        model_counts[tag] = model_counts.get(tag, 0) + 1
        if tag not in best_per_model or cum_log_L[ell] > best_per_model[tag][1]:
            best_per_model[tag] = (particles[ell].copy(), float(cum_log_L[ell]))

    return {
        "particles": particles,
        "cum_log_L": cum_log_L,
        "best_expr": particles[best_idx].copy(),
        "best_log_L": float(cum_log_L[best_idx]),
        "best_per_model": best_per_model,
        "model_counts": model_counts,
        "history": history,
    }

