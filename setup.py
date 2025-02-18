from setuptools import setup, find_packages

# Read requirements.txt to avoid maintaining dependencies in multiple places
def parse_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="imitationLearning",
    version="0.1",
    packages=find_packages(),  # Automatically find packages that contain __init__.py
    install_requires=parse_requirements(),
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
