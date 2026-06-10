"""上下文无关文法定义与先验配置。

文法规则:
  N1 := (NoisyPhylo N2 N3)
  N2 := (Noise P)
  N3 := (CRB P) | (CRBD P P) | (TDB P P) | (TDBD P P P)
      | (PCBD P P P P P) | (BAMM P N3 N3)

PCBD 是分段常数出生-死亡模型 (Piecewise-Constant Birth-Death),
参数为 (lambda1, mu1, lambda2, mu2, tau_frac), 其中 tau_frac ∈ (0,1)
表示断点在树高中的比例位置。
"""

import numpy as np
from scipy import stats

GRAMMAR = {
    "N1": [
        ("NoisyPhylo", [], ["N2", "N3"]),
    ],
    "N2": [
        ("Noise", ["sigma"], []),
    ],
    "N3": [
        ("CRB",  ["lambda"],              []),
        ("CRBD", ["lambda", "mu"],         []),
        ("TDB",  ["lambda", "z"],          []),
        ("TDBD", ["lambda", "z", "epsilon"], []),
        ("PCBD", ["lambda", "mu", "lambda", "mu", "tau_frac"], []),
        ("BAMM", ["eta"],                  ["N3", "N3"]),
    ],
}

DEFAULT_CONFIG = {
    # N3 各产生式的选择概率
    "N3_probs": {
        "CRB":  0.20,
        "CRBD": 0.20,
        "TDB":  0.15,
        "TDBD": 0.15,
        "PCBD": 0.20,
        "BAMM": 0.10,
    },
    # 各参数的先验分布: (分布名, 参数字典)
    "param_priors": {
        "lambda":    ("exponential", {"scale": 0.15}),
        "mu":        ("exponential", {"scale": 0.08}),
        "z":         ("normal",      {"loc": 0.0, "scale": 0.5}),
        "epsilon":   ("beta",        {"a": 2, "b": 6}),
        "eta":       ("exponential", {"scale": 0.03}),
        "sigma":     ("exponential", {"scale": 0.08}),
        "tau_frac":  ("beta",        {"a": 2, "b": 2}),
    },
    # BAMM 递归展开的最大嵌套层数
    "max_bamm_nesting": 3,
    # MH 提议: 局部参数随机游走 vs 结构剪枝重采样
    "local_move_prob": 0.5,
    # 各参数的局部提议步长 (Gaussian random walk std)
    "local_proposal_scales": {
        "lambda": 0.05,
        "mu": 0.03,
        "z": 0.15,
        "epsilon": 0.05,
        "eta": 0.02,
        "sigma": 0.03,
        "tau_frac": 0.05,
        "default": 0.05,
    },
}

def sample_parameter(name, config=None):
    """从先验分布中采样一个参数值。"""
    cfg = config or DEFAULT_CONFIG
    dist_name, params = cfg["param_priors"][name]
    if dist_name == "exponential":
        return np.random.exponential(params["scale"])
    if dist_name == "normal":
        return np.random.normal(params["loc"], params["scale"])
    if dist_name == "beta":
        return np.random.beta(params["a"], params["b"])
    if dist_name == "uniform":
        return np.random.uniform(params["low"], params["high"])
    raise ValueError(f"不支持的先验分布类型: {dist_name!r} (param={name})")

def param_log_prior(name, value, config=None):
    """计算参数值的对数先验概率密度。"""
    cfg = config or DEFAULT_CONFIG
    dist_name, params = cfg["param_priors"][name]
    if dist_name == "exponential":
        return stats.expon.logpdf(value, scale=params["scale"])
    if dist_name == "normal":
        return stats.norm.logpdf(value, loc=params["loc"], scale=params["scale"])
    if dist_name == "beta":
        return stats.beta.logpdf(value, params["a"], params["b"])
    if dist_name == "uniform":
        return stats.uniform.logpdf(value, params["low"], params["high"] - params["low"])
    raise ValueError(f"不支持的先验分布类型: {dist_name!r} (param={name})")

