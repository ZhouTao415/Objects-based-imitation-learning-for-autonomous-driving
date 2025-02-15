from setuptools import setup, find_packages

setup(
    name="autobrains_home_assignment",
    version="0.1",
    packages=find_packages(),  # 自动找到所有 `__init__.py` 标记的包
    install_requires=[
        "numpy",
        "torch"
    ],
)
