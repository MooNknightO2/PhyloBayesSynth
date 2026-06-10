"""数据约束与 SMC 数值稳定性验证。

运行方式:
    python test/test_data_and_smc_validation.py
"""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.algorithm.data_utils import TreeNode, get_tree_data
from train.algorithm.inference import _effective_sample_size
from train.algorithm.grammar import sample_parameter, param_log_prior


def _build_tree_with_fossil_tip():
    root = TreeNode()
    a = TreeNode()
    b = TreeNode()
    root.children = [a, b]
    a.branch_length = 1.0
    b.branch_length = 0.5
    root.age = 1.0
    a.age = 0.0
    b.age = 0.5  # 化石/灭绝叶
    return root


def _build_valid_extant_tree():
    root = TreeNode()
    a = TreeNode()
    b = TreeNode()
    root.children = [a, b]
    a.branch_length = 1.0
    b.branch_length = 1.0
    root.age = 1.0
    a.age = 0.0
    b.age = 0.0
    return root


def test_fossil_tree_rejected():
    root = _build_tree_with_fossil_tip()
    try:
        get_tree_data(root)
    except ValueError:
        return
    raise AssertionError("应拒绝含化石叶节点的树")


def test_extant_tree_allowed():
    root = _build_valid_extant_tree()
    td = get_tree_data(root)
    if td["n"] != 2:
        raise AssertionError("现生重构树解析结果异常")


def test_ess_all_negative_inf_returns_zero():
    ess = _effective_sample_size(np.array([-np.inf, -np.inf, -np.inf], dtype=float))
    if ess != 0.0:
        raise AssertionError(f"全 -inf 权重时 ESS 应为 0，实际为 {ess}")


def test_unknown_prior_distribution_raises():
    bad_cfg = {
        "param_priors": {
            "lambda": ("not-a-dist", {}),
        }
    }
    try:
        sample_parameter("lambda", bad_cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("sample_parameter 遇到未知分布类型应抛 ValueError")

    try:
        param_log_prior("lambda", 1.0, bad_cfg)
    except ValueError:
        return
    raise AssertionError("param_log_prior 遇到未知分布类型应抛 ValueError")


def run_all():
    tests = [
        test_fossil_tree_rejected,
        test_extant_tree_allowed,
        test_ess_all_negative_inf_returns_zero,
        test_unknown_prior_distribution_raises,
    ]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}")
    print("\n数据约束与 SMC 稳定性验证通过。")


if __name__ == "__main__":
    run_all()
