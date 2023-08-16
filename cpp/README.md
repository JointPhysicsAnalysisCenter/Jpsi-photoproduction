## C++ codes
The C++ codes to reproduce results of the analysis use the framework of the much larger project [`jpacPhoto`](https://github.com/dwinney/jpacPhoto). The initial release incorporating the $J/\psi$ photoproduction results is [v2.0](https://github.com/dwinney/jpacPhoto/releases/tag/v2.0), a copy of which is found here for posterity. 

Because jpacPhoto is in continued development, it is recommended that clone and use the latest release or most up-to-date version of the main branch. 

Installation and usage instructions can be found in the [README.md](./cpp/jpacPhoto-2.0/README.md). The scripts to reproduce the plots and results in the paper care all found in the [`scripts/jpsi_photoproduction`](./cpp/jpacPhoto-2.0/scripts/jpsi_photoproduction/) directory and after installation can be run with the included executable. For example:
```
jpacPhoto partial_waves.cpp
```
General information on each script and its output can be found at the top of each file and making changes them should (hopefully) be straightforward from comments throughout each file.

### Summary of provided scipts

- [**dynamics.cpp**](./cpp/jpacPhoto-2.0/scripts/jpsi_photoproduction/dynamics.cpp)
    Computes the quantities $\zeta_\text{th}$, $R_\text{VMD}$ and $a_{\psi p}$ for the  _best fit_  each of the four models (1C, 2C, 3C-NR, 3C-R) formatted as a table in the commandline.
- [**fit.cpp**](./cpp/jpacPhoto-2.0/scripts/jpsi_photoproduction/fit.cpp)
    For a desired model do $N$ fits to the complete data set with randomizly initialized parameters. After completetion, output parameter results in command line for the best fit found and plot the curves on top of data in .pdf format. 
- [**partial_waves.cpp**](./cpp/jpacPhoto-2.0/scripts/jpsi_photoproduction/partial_waves.cpp)
    Using the 2C model and best fit parameters, plot the individual partial wave contributions compared to data. Reproduces fig 5 of paper. 
- [**radius.cpp**](./cpp/jpacPhoto-2.0/scripts/jpsi_photoproduction/radius.cpp) 
    For each model and set of best fit parameters, compute the radius of convergence $r$ at threshold and at $E_\gamma = 12$ GeV. Output formatted in a table and $r$ given both in terms of GeV $^{-1}$ and fm.
- [**results.cpp**](./cpp/jpacPhoto-2.0/scripts/jpsi_photoproduction/results.cpp)
    Using the [boostrap results](./cpp/jpacPhoto-2.0/scripts/jpsi_photoproduction/bootstrap/), plot the resulting integrated and differential cross section results for all models with their associated $1\sigma$ error band. Reproduces figs. 2-4 of paper.
- [**sigma_tot.cpp**](./cpp/jpacPhoto-2.0/scripts/jpsi_photoproduction/sigma_tot.cpp) 
    Plots the total $J/\psi$ absorption cross section for each of the model curves. Reproduces fig. 6 of paper.