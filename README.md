#  SimNIBS Memoslap utils
A. Thielscher, 23/05/2025

The uility functions are used in MemoSLAP (https://www.memoslap.de) project 9 to plan personalized transcranial electric stimulation montages.
They need a working SimNIBS installation (http://www.simnibs.org). You need to provide m2m-folders with the personal head models, created by SimNIBS charm.

## What the utils do:
1) Create a coarse cerebellum central gm surface and add it to the m2m-folder content (only for charm results)
2) Map project-specific mask to the central gm surfaces. The mask can be either in fsaverage surface space (most projects) or in MNI volume space (e.g. project 6)
3) Get the position of center electrode
4) Set up and run the FEM simulations (sets also the surround electrode positions)
5) Map E-field quantities (magn, normal, tangent) onto the middle GM surfaces, and results of lh and rh to fsaverage (optional)
6) Get field medians and focalities
7) Export electrode positions for use with neuronavigation (only simnibs4)
8) Save some key results

## Notes:
* example.py in the examples folder shows how to use it
* the mask files are in simnibs_memoslap_utils/masks
* project_settings.py in simnibs_memoslap_utils contains all project-relevant settings
* electrode properties are specified in simu_settings.py
* the utils support iteration over different radii (see also exemple.py), but not phis
* focality values will be different for add_cerebellum=True versus add_cerebellum=False, as for the former case the area of the cerebellum surface will be included in the calculations
* when running with simnibs3, set add_cerebellum=False
* it should run with simnibs3, but is not tested yet (good luck!)
