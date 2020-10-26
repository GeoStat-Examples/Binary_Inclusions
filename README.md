[![GS-Frame](https://img.shields.io/badge/github-GeoStat_Framework-468a88?logo=github&style=flat)](https://github.com/GeoStat-Framework)
[![Gitter](https://badges.gitter.im/GeoStat-Examples/community.svg)](https://gitter.im/GeoStat-Examples/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

# Overview

This project aims to use realizations of heterogeneous hydraulic conductivity with a binary inclusion structures for numerical subsurface flow and transport simulations. 
Code is available to (i) generate and (ii) visualize random conductivity structures of binary inclusions. Furthermore, routines are provided (iii) to perform flow and transport simulations making use of the FEM-solver OpenGeoSys and (iv) to post-process results. 

## Structure

The project is organized as follows
    • README.md - description of the project 
    • data/ - folder for data used in the project 
    • results/ 
    • src/ - folder containing the Python scripts of the project:	

Please try to organize your example in the given Structure
- `README.md` - description of the project 
- `data/` - folder for data used in the project 
- `results/` - folder with simulation results and plots 
- `src/` - folder containing the Python scripts of the project:	
  + 01_plot_binary_inclusions.py
  + 02_run_sim.py
  + 03_run_ensemble.py
  + binary_inclusions.py
  + sim_processing.py

## Python environment

To make the example reproducible, we provide the following files:
- `requirements.txt` - requirements for [pip](https://pip.pypa.io/en/stable/user_guide/#requirements-files) to install all needed packages
- `spec-file.txt` - specification file to create the original [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments)

## Contact

You can contact us via <a.zech@uu.nl>.

## License

MIT © 2020
