"""上下文无关文法定义与先验配置。

文法规则:
  N1 := (NoisyPhylo N2 N3)
  N2 := (Noise P)
  N3 := (CRB P) | (CRBD P P) | (TDB P P) | (TDBD P P P)
      | (PWBD P N3 N3) | (BAMM P N3 N3)

PWBD (Piecewise Birth-Death) 是嵌套分段模型。参数 tau_frac ∈ (0,1)
表示断点在当前区间中的相对位置，两个子表达式分别描述近现世端与
近根端的模型。嵌套 PWBD 可组合出任意多段、每段不同模型类型的
分段 BD 过程（如第一段 CRBD、第二段 TDBD、第三段 CRB）。

BAMM 保留但默认先验概率为 0，用户需要时可手动开启。
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
        ("PWBD", ["tau_frac"],             ["N3", "N3"]),
        ("BAMM", ["eta"],                  ["N3", "N3"]),
    ],
}

DEFAULT_CONFIG = {
    "N3_probs": {
        "CRB":  0.20,
        "CRBD": 0.25,
        "TDB":  0.10,
        "TDBD": 0.15,
        "PWBD": 0.30,
        "BAMM": 0.00,
    },
    "param_priors": {
        "lambda":    ("exponential", {"scale": 0.15}),
        "mu":        ("exponential", {"scale": 0.08}),
        "z":         ("normal",      {"loc": 0.0, "scale": 0.5}),
        "epsilon":   ("beta",        {"a": 2, "b": 6}),
        "eta":       ("exponential", {"scale": 0.03}),
        "sigma":     ("exponential", {"scale": 0.08}),
        "tau_frac":  ("beta",        {"a": 2, "b": 2}),
    },
    "max_bamm_nesting": 3,
    "max_pwbd_nesting": 3,
    "local_move_prob": 0.65,
    "pwbd_smart_prob": 0.15,
    "struct_tune_steps": 30,
    "bic_penalty_weight": 0.0,
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

