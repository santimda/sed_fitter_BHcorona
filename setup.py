from setuptools import setup, find_packages

setup(
    name="smbh_corona",
    version="0.1.0",
    description="Code for millimeter emission from supermassive black hole coronae",
    url="https://github.com/santimda/sed_fitter_BHcorona",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "astropy",
        "bilby",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
