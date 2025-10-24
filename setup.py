"""
Time Series Library - Setup Configuration
A comprehensive time series analysis library with ARIMA implementation from scratch
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies
INSTALL_REQUIRES = [
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "pandas>=1.3.0",
    "matplotlib>=3.4.0",
]

# Optional PySpark dependencies
EXTRAS_REQUIRE = {
    "spark": ["pyspark>=3.2.0"],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "mypy>=0.950",
    ],
    "docs": [
        "sphinx>=4.5.0",
        "sphinx-rtd-theme>=1.0.0",
    ],
}

# All extras combined
EXTRAS_REQUIRE["all"] = sum(EXTRAS_REQUIRE.values(), [])

setup(
    name="tslib",
    version="0.1.0",
    author="Genaro Melgar",
    description="A time series analysis library with ARIMA implementation from scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/genaromelgar/time-series-library",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    include_package_data=True,
    zip_safe=False,
)


