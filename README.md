This repo contains code for modeling collective rotations of octahedra in materials that have the perovkites structure.  The code goes along with the publication <reference when it is available>.
  
Perovskite structures are made of corner shared octahedra and common distortions in these materials involve mostly rigid collective rotations of the BO6 octahedra. In the 1970's by building macroscopic models of corner-shared rigid octahedra, Prof. Michael Glazer was able to describe all the 22 different patterns in which the rigid octahedra could collectively tilt.  This code uses the diffpy-cmi structure modeling framework to develop constraint equations for doing quantitative structure refinements to measured pair distribution function (PDF)  data for all the different Glazer patterns.
  
The resulting "Glazer model" is further fitted to experimental X-ray pair distribution function data of CaTiO3, which has a known Glazer tilt pattern of alpha+ beta- beta-.
  
How to run:
1. Install diffpy-CMI, following instructions here: https://www.diffpy.org/products/diffpycmi/  so it is installed in a conda environment
2. Clone this repository to your computer.
3. Open the Jupyter notebook in the conda environment where you installed diffpy-CMI (or select this environment as the kernel)
4. Run the cells.
