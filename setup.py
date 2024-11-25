from setuptools import find_packages, setup

setup(
    name="sdmbc_v2",  # Your package name
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Add dependencies
    ],
    author="Youngil Kim",
    author_email="youngil.kim@unsw.edu.au",
    description="SDMBC v2",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/young-ccrc/sdmbc_v2",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
