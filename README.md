# Analysis-of-Electric-Currents-and-Magnetic-Field-Distributions-in-Solar-Magnetic-Flux-Ropes
(Bachelor's thesis)
The list of files that were used for the thesis:
1. pvd_data_extraction.py <- this script extracts regional data from a MHD simulation for further analysis;
2. rainbow_radial_profiles_plot.py <- this script uses extracted data to build plot for selsected time steps;
3. model_fit.py <- this script fits the exctracted data to 4 shielded theoretical models (Solov'ev & Kirichek, 2021) and builds plots, the data is then saved to the "all_fit_results.csv";
4. all_fit_results.csv <- this file contains all fit data;
5. check_results.py <- this script takes data from all_fit_results.csv and presents it in a readable format ready for further analysis.



P.s. This repository also contains a visualisation pipeline for ParaView that was used for interpretation of the MHD simulation but did not make it to the final version of the methodology section. The file is called "VBJ_analysis.py". It's an earlier vesion of "pvd_data_exctraction.py". Data was being processed directly in ParaView without any additional exctractions but it was very resource-demanding (mostly RAM) so the visualisation was removed from the later version of the code to avoid crushes. 

