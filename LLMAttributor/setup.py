#!/usr/bin/env python

"""The setup script."""

from json import loads, load, dump
from setuptools import setup, find_packages
from pathlib import Path

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["ipython", "tqdm", "torch", "transformers", "datasets", "peft"]

test_requirements = []

version = "0.1.1"

setup(
    author="Seongmin Lee",
    author_email="seongmin@gatech.edu",
    python_requires=">=3.6",
    platforms="Linux, Mac OS X, Windows",
    keywords=[
        "Jupyter",
        "JupyterLab",
        "JupyterLab3",
        "Machine Learning",
        "LLM",
        "ChatGPT",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Framework :: Jupyter",
        "Framework :: Jupyter :: JupyterLab",
        "Framework :: Jupyter :: JupyterLab :: 3",
    ],
    description="A Python package to run LLMAttributor in your computational notebooks.",
    install_requires=requirements,
    license="MIT",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    name="llm-attributor",
    packages=find_packages(include=["LLMAttributor", "LLMAttributor.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/poloclub/LLM-Attribution",
    version=version,
    zip_safe=False,
)
