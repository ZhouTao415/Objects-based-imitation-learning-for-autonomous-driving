from setuptools import setup, find_packages

# 读取 requirements.txt 以避免重复维护依赖项
def parse_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="imitationLearning",
    version="0.1",
    packages=find_packages(),  # 自动查找包含 __init__.py 的包
    install_requires=parse_requirements(),
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
