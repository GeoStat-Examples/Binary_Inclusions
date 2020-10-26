[![GS-Frame](https://img.shields.io/badge/github-GeoStat_Framework-468a88?logo=github&style=flat)](https://github.com/GeoStat-Framework)
[![Gitter](https://badges.gitter.im/GeoStat-Examples/community.svg)](https://gitter.im/GeoStat-Examples/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4134626.svg)](https://doi.org/10.5281/zenodo.4134626)

# Overview

This project aims to use realizations of heterogeneous hydraulic conductivity with a binary inclusion structures for numerical subsurface flow and transport simulations.
Code is available to (i) generate and (ii) visualize random conductivity structures of binary inclusions. Furthermore, routines are provided (iii) to perform flow and transport simulations making use of the FEM-solver OpenGeoSys and (iv) to post-process results.

## Structure

The project is organized as follows:

- `README.md` - description of the project
- `results/` - folder with simulation results and plots
- `src/` - folder containing the Python scripts of the project:
  + `01_plot_binary_inclusions.py` - generate and visualize inclusion structures
  + `02_run_sim.py` - run OGS 5 simulation on inclusion structures according to MADE setting
  + `03_run_ensemble.py` - run OGS 5 ensemble of transport simulations
  + `binary_inclusions.py` - package containing inclusing structure classes
  + `sim_processing.py` - package containing post-processing routines

## Python environment

To make the example reproducible, we provide the following files:
- `requirements.txt` - requirements for [pip](https://pip.pypa.io/en/stable/user_guide/#requirements-files) to install all needed packages

## Contact

You can contact us via <a.zech@uu.nl>.

## License

MIT © 2020
