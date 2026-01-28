# PLToolkit

**PLToolkit** is a Python package for computing **phonon-assisted photoluminescence (PL) spectra** of point defects in solids, based on the **Huang–Rhys framework** and its **anharmonic extensions**.

The toolkit supports:
- Analytical generating-function methods (harmonic approximation)
- Numerical Franck–Condon methods (harmonic and anharmonic)
- Monte Carlo sampling of phonon sidebands
- Generation of displaced structures for *ab initio* (VASP) anharmonic potential fitting

The code is designed for **research workflows**, **HPC environments**, and **reproducible studies**, and is accompanied by a full working example.

---

## Features

- **Analytical PL spectra**
  - Generating-function formalism
  - Gaussian or Lorentzian phonon broadening
  - Finite-temperature support

- **Numerical PL spectra**
  - Franck–Condon factors from explicit vibrational Hamiltonians
  - Anharmonic excited- and ground-state potentials
  - Numerical generating function
  - Monte Carlo spectrum generation

- **Anharmonicity**
  - Mode-selective cubic anharmonicity
  - PES fitting from displaced DFT calculations
  - Comparison against harmonic (Poisson) limit

- **Structure & phonon I/O**
  - POSCAR / CONTCAR parsing
  - Phonopy band.yaml
  - VASP OUTCAR phonons and forces

- **HPC-friendly**
  - Editable installation
  - Clean separation of library code and job scripts

---

## Repository Structure

pltoolkit/
├── pltoolkit/
│   ├── __init__.py
│   ├── displacements.py
│   └── photoluminescence.py
│
├── examples/
│   └── rhombohedral-Boron-Nitride/
│       ├── workflow.ipynb
│       ├── band_gs_scratch_hse.yaml
│       ├── contcar_gs_scratch_hse
│       ├── contcar_esmc_scratch_hse
│       └── disp_energies.npy
│
├── scripts/
│   └── disp_run.py
│
├── pyproject.toml
├── README.md
└── LICENSE

---

## Installation

### Requirements
- Python ≥ 3.8
- NumPy
- SciPy
- Matplotlib

### Recommended installation (editable mode)

Clone the repository:

git clone https://github.com/m-singhal/pltoolkit.git
cd pltoolkit

Install in editable mode:

pip install -e .

This ensures that:
- The package is importable from anywhere
- Any local code edits are immediately reflected
- No reinstallation is required after updates

---

### HPC usage (example)

module load python
python -m venv venv
source venv/bin/activate
pip install -e /path/to/pltoolkit

---

## Quick Start

from pltoolkit import (
    GenerateDisplacements,
    calculate_spectrum_analytical,
    calculate_spectrum_numerical,
)

---

## Example Workflow

A complete working example is provided in:

examples/rhombohedral-Boron-Nitride/workflow.ipynb

This notebook demonstrates:

1. Analytical PL spectra using the generating-function formalism
2. Numerical Franck–Condon spectra (harmonic and anharmonic)
3. Monte Carlo sampling of phonon sidebands
4. Mode selection using inverse participation ratio (IPR)
5. Displacement generation for anharmonic PES fitting
6. Comparison of harmonic and anharmonic photoluminescence

---

## Displacement Generation

Displaced structures for DFT calculations are generated using:

GenerateDisplacements(
    gs_poscar_path,
    normal_modes,
    atomic_masses,
    selected_indices,
    disp_list,
)

This produces a directory structure suitable for VASP single-point calculations.

---

## License

This project is released under the **MIT License**.
See the LICENSE file for details.

---

## Citation

If you use this code in published work, please cite:

PLToolkit – Photoluminescence Toolkit  
Author: Mridul  
GitHub: https://github.com/<your-username>/pltoolkit

---

## Contact

For questions, suggestions, or collaboration:
- Open an issue on GitHub
- Or contact the author directly

---

This package is intended for **research use**, with emphasis on **clarity, extensibility, and physical transparency**.
