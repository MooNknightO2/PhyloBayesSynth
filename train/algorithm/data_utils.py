"""加载 phylojson 格式的系统发育树数据。"""

import json


class TreeNode:
    """系统发育树的节点。"""

    def __init__(self):
        self.branch_length = 0.0   # 从父节点到本节点的枝长
        self.children = []         # 子节点列表
        self.taxon = None          # 叶节点的分类单元标识
        self.age = 0.0             # 距今时间 (present = 0)

    def is_leaf(self):
        return len(self.children) == 0

    def num_tips(self):
        if self.is_leaf():
            return 1
        return sum(c.num_tips() for c in self.children)

    def get_all_branches(self):
        """返回所有枝的 (枝长, 子节点age, 父节点age) 列表。"""
        branches = []
        for child in self.children:
            branches.append((child.branch_length, child.age, self.age))
            branches.extend(child.get_all_branches())
        return branches

    def get_branching_times(self):
        """返回所有内部节点的 age (从大到小，即从根到最近的分支事件)。"""
        times = []
        if not self.is_leaf():
            times.append(self.age)
            for child in self.children:
                times.extend(child.get_branching_times())
        return sorted(times, reverse=True)

    def total_branch_length(self):
        """计算树的总枝长 (不含根的茎)。"""
        total = 0.0
        for child in self.children:
            total += child.branch_length + child.total_branch_length()
        return total

def _parse_node(node_dict):
    node = TreeNode()
    node.branch_length = max(node_dict.get("branch_length", 0.0), 0.0)
    node.taxon = node_dict.get("taxon", None)
    for child_dict in node_dict.get("children", []):
        node.children.append(_parse_node(child_dict))
    return node

def _max_root_to_tip(node):
    """从 node 到其下最远叶节点的距离（子树高度）。"""
    if node.is_leaf():
        return 0.0
    return max(c.branch_length + _max_root_to_tip(c) for c in node.children)

def _compute_ages(node, parent_age=None):
    """为每个节点计算 age（距今时间）。

    根据枝长逐层计算：
      root.age = max root-to-tip distance  （定义 "现世" = 最远叶节点）
      child.age = parent.age - child.branch_length

    * 对于超度量树（重构树，只含现存物种）：所有叶节点 age ≈ 0。
    * 对于非超度量树（含化石/灭绝物种）：化石叶节点 age > 0，
      代表该物种在 age 时刻已灭绝，不在现世。
    * 叶节点 age 出现极小负值（< -1e-8）时截断为 0，视为浮点误差。
    """
    if parent_age is None:
        node.age = _max_root_to_tip(node)
    else:
        node.age = parent_age - node.branch_length
        # 叶节点浮点误差修正
        if node.is_leaf() and node.age < 0:
            node.age = max(node.age, 0.0)
    for child in node.children:
        _compute_ages(child, node.age)


def _validate_extant_only_tree(node, tol=1e-6):
    """校验是否为仅含现生物种的重构树（所有叶子 age≈0）。"""
    if node.is_leaf():
        if node.age > tol:
            raise ValueError(
                f"检测到化石/灭绝叶节点（age={node.age:.6g} > {tol}）。"
                "当前算法仅支持现生重构树（extant-only reconstructed tree）。"
            )
        return
    for child in node.children:
        _validate_extant_only_tree(child, tol=tol)

def load_phylojson(filepath):
    """加载 phylojson 文件，返回 TreeNode 根节点。"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    tree_data = data["trees"][0]
    root = _parse_node(tree_data["root"])
    _compute_ages(root)
    return root

def get_tree_data(root):
    """
    Returns:
        dict with keys: n, branches, branching_times, total_length, tree_height, root
    """
    _validate_extant_only_tree(root)
    return {
        "n": root.num_tips(),
        "branches": root.get_all_branches(),
        "branching_times": root.get_branching_times(),
        "total_length": root.total_branch_length(),
        "tree_height": root.age,
        "root": root,
    }