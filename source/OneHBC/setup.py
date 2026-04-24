"""Installation script for the 'OneHBC' python package."""

from setuptools import setup

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # NOTE: Add dependencies
    "psutil",
    "joblib",
    "mjlab",
]

# Installation operation
setup(
    name="OneHBC",
    packages=["OneHBC"],
    author="Hongtu Zhou",
    maintainer="Hongtu Zhou",
    url="https://github.com/HongtuZ/OneHBC.git",
    version="1.0.0",
    description="OneHBC",
    keywords=["OneHBC"],
    install_requires=INSTALL_REQUIRES,
    license="Apache-2.0",
    include_package_data=True,
    python_requires=">=3.12",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.12",
    ],
    zip_safe=False,
)
