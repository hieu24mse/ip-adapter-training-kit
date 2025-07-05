#!/usr/bin/env python3
"""
Setup script for IP-Adapter Training Kit
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        return f.read()

# Read requirements
def read_requirements(filename):
    req_path = os.path.join("requirements", filename)
    with open(req_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#") and not line.startswith("-r")]

setup(
    name="ip-adapter-training-kit",
    version="1.0.0",
    author="IP-Adapter Training Kit Team",
    author_email="contact@example.com",
    description="A complete toolkit for training custom IP-Adapter models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/ip-adapter-training-kit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
    },
    entry_points={
        "console_scripts": [
            "ip-adapter-train=scripts.train_mini_dataset:main",
            "ip-adapter-convert=scripts.convert_checkpoint:main",
            "ip-adapter-generate=scripts.use_trained_model_proper:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "examples/**/*", "docs/*.md"],
    },
    keywords="ip-adapter stable-diffusion training ai machine-learning computer-vision",
    project_urls={
        "Bug Reports": "https://github.com/your-username/ip-adapter-training-kit/issues",
        "Source": "https://github.com/your-username/ip-adapter-training-kit",
        "Documentation": "https://github.com/your-username/ip-adapter-training-kit/tree/main/docs",
    },
) 