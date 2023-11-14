#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

setup_requirements = ["pytest-runner"]

setup(
    author="author",
    author_email="email@email.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="Package for the paper: On the Importance of Step-wise Embeddings for Heterogeneous Clinical Time-Series",
    entry_points={"console_scripts": ["icu-benchmarks = icu_benchmarks.run:main"]},
    install_requires=[],  # dependencies managed via conda for the moment
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords="icu_benchmarks",
    name="icu_benchmarks",
    packages=find_packages(include=["icu_benchmarks"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=[],
    version="1.0.0",
    zip_safe=False,
)
