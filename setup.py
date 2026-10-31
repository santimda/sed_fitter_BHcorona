from pathlib import Path

from setuptools import find_packages, setup


README = Path(__file__).parent / "README.md"

setup(
    name="smbh_corona",
    version="0.1.0",
    description="Code for millimeter emission from supermassive black hole coronae",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/santimda/sed_fitter_BHcorona",
    license="GPL-3.0-only",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "astropy",
        "bilby>=2.8.2",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
