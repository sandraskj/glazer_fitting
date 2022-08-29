# -*- coding: utf-8 -*-

# ========== Packages ==========

import os

# ========== Shared variables ==========

class shared:

    forbidden_names = ['AUX','CLOCK','COM','CON','LPT','NUL','PRN']

    config = {
		'element_a': "Ca",
		'element_b': "Ti",
		'element_x': "O",
		'compound': "CaTiO3",
		'qmax': 0,
		'qmin': 31.414,
		'qdamp': 0.0255496541373,
		'qbroad': 0.0220472479685,
		'rmin': 1.6,
		'rmax': 15,
		'rstep': 0.01,
		'RUN_PARALLEL': True,
		'glazer_system': 10,
		'latpar_a_cub_i': 3.90,
		'glazer_tilts': {'alpha_i': 8, 'beta_i': 9, 'gamma_i': 9},
		'adisp': {'aydisp_i': 0.02, 'azdisp_i': 0.01},
		'uiso_a_i': 0.007,  # initial value for isotropic Uiso(A)
		'uiso_b_i': 0.007,  # initial value for isotropic Uiso(B)
		'uiso_x_i': 0.008,  # initial value for isotropic Uiso(X)
		'delta2_i': 2.05,  # initial value for nearest neighbor correlation
		'scale_i': 0.7,

		'dpath': os.getcwd().replace("\\", "/"),
		'gr_name': 'name_of_pdf_datafile.gr',

		'mpath': 'path_to_model_directory',
		'cif_name': "CaTiO3_2x2x2-supercell_0_0_0_tilt.cif",
    }
