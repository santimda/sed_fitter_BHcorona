# sed_fitter_BHcorona

This package provides a framework to derive physical information of supermassive black hole (SMBH) coronae by modelling their spectral energy distribution (SED) from radio to far-infrared frequencies. We introduce a corona emission model based on a one-zone spherical region with a hybrid thermal and non-thermal plasma based on [Inoue et al. (2019)](https://ui.adsabs.harvard.edu/#abs/2019ApJ...880...40I/abstract). We use a slightly modified version of the numerical code from
[Margalit & Quataert (2021, 2024)](https://github.com/bmargalit/thermal-synchrotron) to compute the synchrotron SED of a hybrid plasma.
In addition to the physical coronal model, we include phenomenological continuum-emission components. To fit the SEDs in a flexible and robust way, we use the package Bilby [Ashton et al. 2019](https://ui.adsabs.harvard.edu/abs/2019ApJS..241...27A/abstract). Further details of the model are provided in [del Palacio et al. (2025)](https://ui.adsabs.harvard.edu/abs/2025A%26A...701A..41D/abstract).


## Project

This repository contains the code and data used for the paper *Millimeter
emission from supermassive black hole coronae* 
[del Palacio et al. (2025)](https://ui.adsabs.harvard.edu/abs/2025A%26A...701A..41D/abstract). 
The paper version is preserved by Git tag
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
The repository uses `data/` by default. To keep data outside the source checkout,
set `SMBH_CORONA_DATA_PATH` to a directory containing `parameters/` and `fluxes/`:

```bash
export SMBH_CORONA_DATA_PATH=/path/to/smbh-corona-data
```

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

For tests and notebooks, install the additional development dependencies:

```bash
pip install -r requirements-dev.txt
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

## Mock Validation

The deterministic `test_galaxy` fixture validates the current fitting workflow
with radio/mm, far-infrared, and high-frequency points. The validation script
regenerates the data file from the truth parameters with noiseless model fluxes
and 3% statistical errors, fits seven free parameters, and verifies that
the injected values are within their 1-sigma posterior intervals. The normal
data loader adds the configured systematic uncertainties before fitting. 

Run the default validation from the repository root:

```bash
MPLBACKEND=Agg python tests/validation_test/fit_mock_galaxy.py
```

The default run uses 32 walkers and 2000 steps and writes the regenerated flux
file to `data/fluxes/test_galaxy_flux.dat` and its summary, SED plot, and corner
plot to `validation_test_output/`. A shorter recovery check runs as part of the
default pytest suite.

The separate mixed-limit validation uses the same model but writes detections
between 100 and 300 GHz and paired lower/upper limits at the other frequencies
into the same six-column flux file. The validation runs once with the soft
(`w=1`) treatment and once with the strict (`w=10`) treatment:

```bash
MPLBACKEND=Agg python tests/validation_test/fit_mock_limits.py
```

Each run writes a fit plot, corner plot, and summary named for its `w` value.
Bilby sampler chains and pickle files are removed after the validation completes.

The limit widths use the loaded statistical and systematic errors. Larger `w`
values narrow the effective one-sided Gaussian width and therefore penalize model
predictions outside the limits more strongly.
