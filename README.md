## Summer Arctic melt-pond fraction predicted by advected Spring sea-ice roughness

MSci Geophysics  
Supervisor: Dr Michel Tsamados  
May 2024  

This repository develops a new model for Summer melt-pond fraction as a function of latitude and Spring sea-ice roughness. A process for advecting sea-ice to improve predictions is also included.

The main code for carrying out the advection is in `forward_sir.py`, which advects sea-ice by a set number of days, calculating melt-pond fractions and sea-ice roughnesses for advected cells. `forward_floes.ipynb` provides a more interactive version of this script, and plots the advection figures in Section 3.1.

`get_monthly_means.py` calculates mean monthly melt-pond fractions for the Arctic region for years included in this study.

`latitude_model_analysis.ipynb` develops the melt-pond fraction model and conducts a leave-one-out analysis.

`report_figures.ipynb` plots the figures included in the report.

`archive/` contains code used in the development of the final scripts and work done along the way which was not included in the final report.

