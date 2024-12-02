from setuptools import find_packages, setup

setup(
    name="sdmbc_v2",
    version="0.1.0",
    description="A Python package for bias correction of climate data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Youngil Kim",
    author_email="youngil.kim@unsw.edu.au",
    url="https://github.com/yourusername/sdmbc_v2",  # Update with your repository URL
    packages=find_packages(exclude=["tests", "docs"]),
    install_requires=[
        "numpy",
        "dask",
        "xarray",
        "pandas",
        "pyyaml",
        "tqdm",
        "matplotlib",
        "scipy",
        "cartopy",
    ],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
