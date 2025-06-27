# sed_fitter_BHcorona

## Project
This repository provides code and data to reproduce and extend the results
from the paper *Millimeter emission from supermassive black hole coronae*
(del Palacio et al., 2025 - https://arxiv.org/abs/2504.07762)

Run the jupyter notebook in the ipynbs folder.

If new sources are added, include a new file with the fluxes and initial parameters in the respective folders, following the same convention as with the other sources.

## Installation

```bash
git clone https://github.com/santimda/sed_fitter_BHcorona.git
cd sed_fitter_BHcorona
pip install -r requirements.txt

# install the project as a package
pip install -e .
```

## Example

1. Run the example python script.
```bash
python ./scripts/example.py
```

2. See a full example with the Python Notebook.