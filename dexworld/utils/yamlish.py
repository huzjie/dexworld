# -*- coding: utf-8 -*-
"""极简 YAML 子集解析器（零依赖回退）。

支持：缩进嵌套映射、内联列表 [a, b]、行尾注释、标量（str/int/float/bool/null）、
列表块（- item）。仅用于在没有 pyyaml 时兜底读取配置文件。
"""

from __future__ import annotations

from typing import Any, Dict


def _strip_comment(line: str) -> str:
    in_s = False
    for i, ch in enumerate(line):
        if ch in ("\'", "\""):
            in_s = not in_s
        elif ch == "#" and not in_s:
            return line[:i]
    return line


def _scalar(s: str):
    s = s.strip()
    if not s:
        return None
    if s.startswith(("\'", "\"")) and s.endswith(("\'", "\"")) and len(s) >= 2:
        return s[1:-1]
    low = s.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none", "~"):
        return None
    try:
        if s.startswith("0x") or s.startswith("0X"):
            return int(s, 16)
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _inline_list(s: str):
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    inner = s[1:-1].strip()
    if not inner:
        return []
    parts = []
    cur = ""
    depth = 0
    for ch in inner:
        if ch == "," and depth == 0:
            parts.append(_scalar(cur))
            cur = ""
        else:
            if ch in "[(\'\"":
                depth += 1
            elif ch in "])":
                depth -= 1
            cur += ch
    if cur.strip():
        parts.append(_scalar(cur))
    return parts


def parse_yaml(text: str) -> Dict[str, Any]:
    """解析 YAML 文本为 dict（仅顶层映射）。"""
    lines = [_strip_comment(ln).rstrip() for ln in text.splitlines()]
    # 预处理块列表（- item）成内联
    result: Dict[str, Any] = {}
    stack = [(result, -1)]  # (dict, indent)

    def _cur_dict() -> Dict[str, Any]:
        return stack[-1][0]

    for raw in lines:
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        content = raw.strip()
        # 列表块
        if content.startswith("- "):
            item = _scalar(content[2:])
            d = _cur_dict()
            if not isinstance(d.get("__list__"), list):
                d["__list__"] = []
            d["__list__"].append(item)
            continue
        if ":" not in content:
            continue
        key, _, val = content.partition(":")
        key = key.strip().strip("\"'\"")
        val = val.strip()
        while stack and stack[-1][1] >= indent:
            stack.pop()
        d = _cur_dict()
        if val == "":
            child: Dict[str, Any] = {}
            d[key] = child
            stack.append((child, indent))
        else:
            lst = _inline_list(val)
            d[key] = lst if lst is not None else _scalar(val)

    def _post(d: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in d.items():
            if k == "__list__":
                continue
            if isinstance(v, dict):
                v = _post(v)
            out[k] = v
        if "__list__" in d:
            return {"__list__": d["__list__"], **out}
        return out

    return _post(result)


__all__ = ["parse_yaml"]
