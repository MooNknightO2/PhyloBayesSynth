"""表达式树数据结构，用于表示系统发育模型的语法树。

解析格式: (TAG p1 p2 ... (CHILD1 ...) (CHILD2 ...))
例如:
  (CRBD 0.5 0.1)
  (BAMM 0.3 (CRBD 0.5 0.1) (CRBD 1.0 0.3))
"""

import copy

class Expression:
    """语法树中的一个节点。
    Attributes:
        tag: 标识符，如 "NoisyPhylo", "CRB", "CRBD" 等
        params: 参数列表 (float)
        children: 子表达式列表
        nonterminal: 该节点由哪个非终结符展开 (N1/N2/N3)
    """
    def __init__(self, tag, params=None, children=None, nonterminal=None):
        self.tag = tag
        self.params = params or []
        self.children = children or []
        self.nonterminal = nonterminal

    def __repr__(self):
        parts = [self.tag]
        for p in self.params:
            parts.append(f"{p:.4f}")
        for c in self.children:
            parts.append(repr(c))
        return "(" + " ".join(parts) + ")"

    def copy(self):
        return copy.deepcopy(self)

    def count_nodes(self):
        """统计表达式树中所有节点的数量。"""
        return 1 + sum(c.count_nodes() for c in self.children)

    def get_all_nodes(self, path=None):
        """获取所有节点及其路径 (从根到该节点的子索引列表)。

        Returns:
            list of (path, node) tuples
        """
        if path is None:
            path = []
        result = [(list(path), self)]
        for i, child in enumerate(self.children):
            result.extend(child.get_all_nodes(path + [i]))
        return result

    def get_node_at(self, path):
        """根据路径获取节点。"""
        node = self
        for idx in path:
            node = node.children[idx]
        return node

    def set_node_at(self, path, new_node):
        """在指定路径替换节点，返回新的表达式 (不修改原表达式)。"""
        if not path:
            return new_node
        expr = self.copy()
        parent = expr
        for idx in path[:-1]:
            parent = parent.children[idx]
        parent.children[path[-1]] = new_node
        return expr


def _tokenize(s):
    return s.replace("(", " ( ").replace(")", " ) ").split()


def _parse_at(tokens, pos):
    if pos >= len(tokens):
        raise ValueError("unexpected end of expression")
    if tokens[pos] != "(":
        raise ValueError(f"expected '(' at position {pos}, got '{tokens[pos]}'")
    pos += 1
    if pos >= len(tokens):
        raise ValueError("unexpected end after '('")
    tag = tokens[pos]
    pos += 1
    params = []
    children = []
    while pos < len(tokens) and tokens[pos] != ")":
        if tokens[pos] == "(":
            child, pos = _parse_at(tokens, pos)
            children.append(child)
        else:
            try:
                params.append(float(tokens[pos]))
            except ValueError:
                raise ValueError(f"cannot parse '{tokens[pos]}' as float")
            pos += 1
    if pos >= len(tokens):
        raise ValueError("missing closing ')'")
    pos += 1
    return Expression(tag, params, children), pos


def parse_expression(s):
    """Parse an S-expression string into an Expression tree.

    >>> parse_expression("(CRBD 0.5 0.1)")
    (CRBD 0.5000 0.1000)
    >>> parse_expression("(BAMM 0.3 (CRBD 0.5 0.1) (CRBD 1.0 0.3))")
    (BAMM 0.3000 (CRBD 0.5000 0.1000) (CRBD 1.0000 0.3000))
    """
    tokens = _tokenize(s.strip())
    if not tokens:
        raise ValueError("empty expression")
    expr, pos = _parse_at(tokens, 0)
    if pos != len(tokens):
        raise ValueError(f"trailing tokens after position {pos}")
    return expr

