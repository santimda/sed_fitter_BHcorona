# sed_fitter_BHcorona

## Project

This repository contains the code and data used for the paper *Millimeter
emission from supermassive black hole coronae* (del Palacio et al., 2025:
https://arxiv.org/abs/2504.07762). The paper version is preserved by Git tag
`v0.1.0`; current development and integration work is performed on `main`.

The model combines four components: black-hole corona synchrotron emission,
modified dust emission, free-free emission, and extended synchrotron emission.
The main implementation is in `src/smbh_corona/`; source data and initial
parameters are in `data/`.

The repository also contains notebooks for fitting individual sources,
exploring model parameters, and producing correlation and derived-quantity
plots. Generated figures and sampler outputs are not required for installation.

If new sources are added, include a flux file and an initial-parameter file in
the corresponding folders within `data/`, following the existing conventions.
The data paths are defined in `src/smbh_corona/config.py`; edit that file if the
data are stored elsewhere.

Flux files use six columns:

```text
frequency_GHz flux_mJy flux_error_mJy beam_major_arcsec beam_minor_arcsec flag
```

Flags identify detections (`0`), upper limits (`1`), fake or diagnostic points
(`2`), and lower limits (`4`).

## Installation

```bash
git clone https://github.com/santimda/sed_fitter_BHcorona.git
cd sed_fitter_BHcorona
pip install -r requirements.txt
pip install -e .
```

The project can also be checked in the provided mamba environment:

```bash
mamba activate corona
python -m pytest -q
```

## Example

Run the example from the repository root:

```bash
python scripts/example.py
```

The script evaluates the NGC985 model and writes `NGC985_model_sed.txt` in the
current directory. Plotting may require a working LaTeX installation.

For a complete fit workflow, open `scripts/fit_galaxy.ipynb`. Other analysis
notebooks are located in `scripts/`.
