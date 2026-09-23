# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name="dexworld",
    version="1.0.0",
    description="多模态触觉世界模型训练与推理框架（受 ME-Dex-1.0 启发）",
    packages=find_packages(exclude=["tests", "examples"]),
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "dexworld=dexworld.cli:main",
        ],
    },
)
