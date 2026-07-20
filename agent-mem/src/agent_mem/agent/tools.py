"""通用工具集（给 CLI agent 用）：``search``（canned）+ ``python``（安全算术）。

``python`` 工具用 AST 白名单求值（仅常量 + 四则/一元运算），不执行任意代码，
适合本地 demo。真实生产工具（联网检索、代码解释器）留扩展。
"""

from __future__ import annotations

import ast
import operator
from typing import Any

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"不支持的表达式: {ast.dump(node)}")


def python(expression: str) -> str:
    """安全算术求值（AST 白名单），返回结果字符串。"""
    tree = ast.parse(expression, mode="eval")
    return str(_safe_eval(tree.body))


def search(query: str) -> str:
    """canned 检索（无联网），返回固定占位结果。"""
    return f"(stub search) 未找到关于 {query!r} 的结果。"


# OpenAI function schema
TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "搜索信息（本地 stub，返回占位结果）。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": "对一个算术表达式安全求值（支持 + - * / % ** 与括号）。",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]


def execute_tool(name: str, args: dict[str, Any]) -> str:
    """按名分发到具体工具。"""
    if name == "search":
        return search(args.get("query", ""))
    if name == "python":
        return python(args.get("expression", ""))
    raise ValueError(f"未知工具: {name}")
